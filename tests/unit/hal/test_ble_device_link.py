"""BLEDevice's tracked link state.

The complaint this answers: GLIDER kept calling a peripheral connected after
it had gone. Nothing passed bleak's disconnected_callback, and
_ensure_connected only consulted client.is_connected lazily at I/O time, so
the drop was genuinely unknown to the process until the next write failed.
"""

import sys

import pytest

from glider.hal.base_board import ConnectionState
from glider.hal.base_device import DeviceConfig
from glider.hal.devices.ble_device import BLEDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


class _FakeClient:
    """A BleakClient stand-in that can be told to drop."""

    def __init__(self, address, disconnected_callback=None):
        self.address = address
        self.is_connected = False
        self.written = []
        self.read_value = bytearray(b"42")
        self.disconnected_callback = disconnected_callback
        self._handler = None

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def write_gatt_char(self, char, data, response=False):
        self.written.append((char, bytes(data), response))

    async def read_gatt_char(self, char):
        return bytearray(self.read_value)

    async def start_notify(self, char, handler):
        self._handler = handler

    async def stop_notify(self, char):
        self._handler = None

    def drop(self, *, notify=True):
        """Peripheral goes away. ``notify=False`` models a missed callback."""
        self.is_connected = False
        if notify and self.disconnected_callback is not None:
            self.disconnected_callback(self)


@pytest.fixture
def fake_bleak(monkeypatch):
    from unittest.mock import MagicMock

    created = {}

    def make_client(address, *a, disconnected_callback=None, **k):
        client = _FakeClient(address, disconnected_callback)
        created["client"] = client
        created.setdefault("clients", []).append(client)
        return client

    module = MagicMock(name="bleak")
    module.BleakClient = make_client
    monkeypatch.setitem(sys.modules, "bleak", module)
    return module, created


async def _initialized(**settings):
    settings.setdefault("address", "11:22:33:44:55:66")
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings=settings), name="ble")
    await device.initialize()
    return device


def test_ble_owns_its_link():
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings={"address": "AA"}), name="ble")
    assert device.owns_link is True


def test_link_is_disconnected_before_initialize():
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings={"address": "AA"}), name="ble")
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_initialize_reports_connected(fake_bleak):
    device = await _initialized()
    assert device.link_state is ConnectionState.CONNECTED


async def test_the_client_is_built_with_a_disconnect_callback(fake_bleak):
    _module, created = fake_bleak
    await _initialized()
    assert created["client"].disconnected_callback is not None


async def test_a_drop_moves_the_state(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_a_drop_fires_the_listener_once(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    seen = []
    device.set_link_state_callback(lambda dev: seen.append(dev.link_state))
    created["client"].drop()
    assert seen == [ConnectionState.DISCONNECTED]


async def test_repeating_a_state_does_not_renotify(fake_bleak):
    """The 2s poll runs forever; it must not spam the GUI with no news."""
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    seen = []
    device.set_link_state_callback(lambda dev: seen.append(dev.link_state))
    await device.poll_link()
    await device.poll_link()
    assert seen == []


async def test_poll_catches_a_drop_the_callback_missed(fake_bleak):
    """CoreBluetooth and WinRT both lose the callback often enough that this
    backstop is the mechanism, not a belt on top of one."""
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop(notify=False)
    assert device.link_state is ConnectionState.CONNECTED  # nobody has looked yet
    await device.poll_link()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_shutdown_reports_disconnected(fake_bleak):
    device = await _initialized()
    await device.shutdown()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_a_stale_client_callback_is_ignored(fake_bleak):
    """The old client's teardown callback must not knock over a live link."""
    _module, created = fake_bleak
    device = await _initialized()
    stale = created["client"]
    device._client = _FakeClient("other")
    device._client.is_connected = True
    device._set_link(ConnectionState.CONNECTED)
    stale.drop()
    assert device.link_state is ConnectionState.CONNECTED


async def test_poll_before_initialize_is_quiet(fake_bleak):
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings={"address": "AA"}), name="ble")
    await device.poll_link()
    assert device.link_state is ConnectionState.DISCONNECTED
