"""
Tests for BLEWriteDevice -- the generic "write a command to a BLE
characteristic" device (e.g. an optogenetic stimulator).

bleak is mocked via sys.modules so these run with no Bluetooth hardware and
even when bleak isn't installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from glider.hal.base_device import (
    BLEWriteDevice,
    DeviceConfig,
    create_device_from_dict,
)


class _FakeClient:
    """Stand-in for bleak.BleakClient that records writes."""

    instances: list[_FakeClient] = []

    def __init__(self, address):
        self.address = address
        self.connected = False
        self.writes: list[tuple] = []
        _FakeClient.instances.append(self)

    @property
    def is_connected(self) -> bool:
        return self.connected

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def write_gatt_char(self, uuid, data, response=False):
        if not self.connected:
            raise RuntimeError("not connected")
        self.writes.append((uuid, data, response))


@pytest.fixture
def fake_bleak(monkeypatch):
    mod = types.ModuleType("bleak")
    mod.BleakClient = _FakeClient
    _FakeClient.instances = []
    monkeypatch.setitem(sys.modules, "bleak", mod)
    return mod


def _make_device(**settings) -> BLEWriteDevice:
    cfg = DeviceConfig(
        pins={},
        settings={"address": "AA:BB:CC:DD:EE:FF", "char_uuid": "uuid-1", **settings},
    )
    return BLEWriteDevice(board=None, config=cfg, name="opto")


async def test_write_encodes_and_sends(fake_bleak):
    dev = _make_device()
    await dev.initialize()
    await dev.write("20,10")
    client = _FakeClient.instances[-1]
    # Default is write-without-response (response=False), UTF-8 bytes.
    assert client.writes == [("uuid-1", b"20,10", False)]


async def test_on_off_commands(fake_bleak):
    dev = _make_device()
    await dev.initialize()
    await dev.write("on")
    await dev.write("off")
    assert [w[1] for w in _FakeClient.instances[-1].writes] == [b"on", b"off"]


async def test_write_joins_multiple_args_as_hz_seconds(fake_bleak):
    # Two Number Input ports (floats) -> "20,10", whole floats rendered as ints.
    dev = _make_device()
    await dev.initialize()
    await dev.write(20.0, 10.0)
    assert _FakeClient.instances[-1].writes[-1] == ("uuid-1", b"20,10", False)


async def test_write_response_setting(fake_bleak):
    dev = _make_device(write_response=True)
    await dev.initialize()
    await dev.write("on")
    assert _FakeClient.instances[-1].writes[-1] == ("uuid-1", b"on", True)


class _FakeChar:
    def __init__(self, props):
        self.properties = props


class _FakeServices:
    def __init__(self, props):
        self._char = _FakeChar(props)

    def get_characteristic(self, uuid):
        return self._char


async def test_autodetect_forces_with_response_when_only_write(fake_bleak):
    # Characteristic advertises only "write" (the Zephyr LED firmware case):
    # even with the default preference (without-response), the write must use
    # response=True.
    dev = _make_device()  # write_response defaults to False
    await dev.initialize()
    dev._client.services = _FakeServices(["read", "write"])
    await dev.write("on")
    assert dev._client.writes[-1] == ("uuid-1", b"on", True)


async def test_autodetect_uses_without_response_when_supported(fake_bleak):
    dev = _make_device(write_response=True)  # prefers with-response...
    await dev.initialize()
    # ...but the characteristic only supports write-without-response.
    dev._client.services = _FakeServices(["write-without-response"])
    await dev.write("off")
    assert dev._client.writes[-1] == ("uuid-1", b"off", False)


async def test_preference_honored_when_both_modes_supported(fake_bleak):
    dev = _make_device(write_response=True)
    await dev.initialize()
    dev._client.services = _FakeServices(["write", "write-without-response"])
    await dev.write("on")
    assert dev._client.writes[-1] == ("uuid-1", b"on", True)


async def test_initialize_requires_address_and_char(fake_bleak):
    dev = BLEWriteDevice(None, DeviceConfig(pins={}, settings={"char_uuid": "x"}))
    with pytest.raises(ValueError):
        await dev.initialize()

    dev = BLEWriteDevice(None, DeviceConfig(pins={}, settings={"address": "AA"}))
    with pytest.raises(ValueError):
        await dev.initialize()


async def test_write_reconnects_after_drop(fake_bleak):
    dev = _make_device()
    await dev.initialize()
    first = dev._client
    first.connected = False  # simulate a dropped link
    await dev.write("off")
    # A fresh client was connected and the write went through.
    assert dev._client is not first
    assert dev._client.is_connected
    assert dev._client.writes[-1] == ("uuid-1", b"off", False)


async def test_write_requires_command(fake_bleak):
    dev = _make_device()
    await dev.initialize()
    with pytest.raises(ValueError):
        await dev.write(None)


async def test_shutdown_disconnects(fake_bleak):
    dev = _make_device()
    await dev.initialize()
    client = dev._client
    await dev.shutdown()
    assert client.connected is False
    assert dev._client is None


def test_actions_and_registry():
    dev = _make_device()
    assert "write" in dev.actions
    assert dev.device_type == "BLEWrite"
    # Factory / DEVICE_REGISTRY wiring.
    data = {
        "id": "opto1",
        "device_type": "BLEWrite",
        "name": "opto",
        "board_id": "ble",
        "config": {"pins": {}, "settings": {"address": "AA", "char_uuid": "u"}},
    }
    dev2 = create_device_from_dict(data, board=None)
    assert isinstance(dev2, BLEWriteDevice)
    assert dev2.address == "AA"
    assert dev2.char_uuid == "u"
