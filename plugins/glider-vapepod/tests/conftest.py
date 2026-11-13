"""Fixtures shared by the whole ``glider-vapepod`` suite.

Two things live here: the plugin's registration with the core registries (in
the app that is PluginManager's job), and a fake ``bleak`` for the device tests.
bleak is lazy-imported inside the connect path, so the fake goes into
``sys.modules``; its client records an ordered event log, because what matters
most here is the order of writes against the link going up and down.
"""

import asyncio
import sys
import types

import pytest

from glider_vapepod import DEVICE_TYPES, NODE_TYPES


@pytest.fixture(autouse=True)
def registered_plugin(request, monkeypatch):
    """Register the plugin's device and node, and unregister afterwards.

    Autouse: a leaked registration would let a later test pass for the wrong
    reason. test_packaging is marked to skip this, because a hand-registered
    ``VapePod`` would mask exactly what it checks.
    """
    if "real_registration" in request.keywords:
        return

    from glider.core.flow_engine import FlowEngine
    from glider.hal.base_device import DEVICE_REGISTRY
    from glider.plugins import plugin_manager as pm

    for name, cls in DEVICE_TYPES.items():
        monkeypatch.setitem(DEVICE_REGISTRY, name, cls)
        monkeypatch.setitem(pm._PLUGIN_COMPONENTS, ("device", name), "glider-vapepod")
    for name, cls in NODE_TYPES.items():
        monkeypatch.setitem(FlowEngine._node_registry, name, cls)
        monkeypatch.setitem(pm._PLUGIN_COMPONENTS, ("node", name), "glider-vapepod")


class _Characteristic:
    def __init__(self, properties):
        self.properties = properties


class _Services:
    """The firmware's command characteristic is write-with-response only."""

    def get_characteristic(self, uuid):
        return _Characteristic(["write"])


class FakeClient:
    """Async BleakClient stand-in: an event log, a notify hook, and faults."""

    def __init__(self, address, disconnected_callback=None):
        self.address = address
        self.is_connected = False
        self.disconnected_callback = disconnected_callback
        self.services = _Services()
        self.events = []  # ("write", b"...") / ("disconnect", None), in call order
        self.notify_handler = None
        self.notify_char = None
        self.connect_error = None
        self.write_error = None
        self.write_hangs = False
        self.last_write_char = None
        self.last_response = None

    @property
    def written(self):
        return [payload for kind, payload in self.events if kind == "write"]

    async def connect(self):
        if self.connect_error:
            raise self.connect_error
        self.is_connected = True

    async def disconnect(self):
        self.events.append(("disconnect", None))
        self.is_connected = False

    async def start_notify(self, char, handler):
        self.notify_char = char
        self.notify_handler = handler

    async def stop_notify(self, char):
        self.notify_handler = None

    async def write_gatt_char(self, char, data, response=False):
        if self.write_hangs:
            await asyncio.sleep(30)
        if self.write_error:
            raise self.write_error
        self.events.append(("write", bytes(data)))
        self.last_write_char = char
        self.last_response = response

    def push(self, text):
        """The pod notifying its status."""
        self.notify_handler(None, bytearray(text.encode()))

    def drop(self):
        """The link going away under the client, as bleak reports it."""
        self.is_connected = False
        if self.disconnected_callback is not None:
            self.disconnected_callback(self)


@pytest.fixture
def fake_bleak(monkeypatch):
    """Inject a fake ``bleak``; return the holder of every client it made.

    Faults armed on the holder (``connect_error`` and friends) apply to clients
    made later too: a failed write makes BLEDevice drop its client and build a
    new one, and a fault on one instance alone would silently go away.
    """
    created = {"clients": []}

    def make_client(address, *args, disconnected_callback=None, **kwargs):
        client = FakeClient(address, disconnected_callback)
        for key in ("connect_error", "write_error", "write_hangs"):
            if key in created:
                setattr(client, key, created[key])
        created["client"] = client
        created["clients"].append(client)
        return client

    async def find_device_by_name(name, timeout=8.0):
        created["scanned_name"] = name
        return types.SimpleNamespace(address="AA:BB:CC:DD:EE:FF", name=name)

    async def find_device_by_address(address, timeout=10.0):
        return address  # BleakClient takes the address or the device alike

    module = types.ModuleType("bleak")
    module.BleakClient = make_client
    module.BleakScanner = types.SimpleNamespace(
        find_device_by_name=find_device_by_name,
        find_device_by_address=find_device_by_address,
    )
    monkeypatch.setitem(sys.modules, "bleak", module)
    return created


@pytest.fixture
def arm(fake_bleak):
    """``arm(write_error=...)``: a fault on every client, live and future."""

    def _arm(**faults):
        fake_bleak.update(faults)
        for client in fake_bleak["clients"]:
            for key, value in faults.items():
                setattr(client, key, value)

    return _arm
