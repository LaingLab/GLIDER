"""Tests for the full BLEDevice (device_type "BLE": write / read / notify).

bleak is lazy-imported inside the connect/scan paths, so tests inject a fake
``bleak`` module with an async fake client, exactly as the serial/i2c device
tests inject fake ``serial``/``smbus2``. (BLEWriteDevice's own tests live in
test_ble_device.py; this file covers the read/notify superset.)
"""

import sys
import time

import pytest

from glider.hal.base_device import DeviceConfig
from glider.hal.devices.ble_device import BLEDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"


class _FakeClient:
    """Minimal async BleakClient stand-in."""

    def __init__(self, address):
        self.address = address
        self.is_connected = False
        self.written = []
        self.read_value = bytearray(b"42")
        self._handler = None
        self.stop_notify_called = False
        self.disconnect_error = None

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        if self.disconnect_error:
            raise self.disconnect_error
        self.is_connected = False

    async def write_gatt_char(self, char, data, response=False):
        self.written.append((char, bytes(data), response))

    async def read_gatt_char(self, char):
        return bytearray(self.read_value)

    async def start_notify(self, char, handler):
        self._handler = handler

    async def stop_notify(self, char):
        self.stop_notify_called = True
        self._handler = None

    def push(self, data: bytes):
        """Simulate a GATT notification from the peripheral."""
        assert self._handler is not None, "not subscribed"
        self._handler(object(), bytearray(data))


@pytest.fixture
def fake_bleak(monkeypatch):
    """Inject a fake ``bleak``; yield (module, created-client holder)."""
    from unittest.mock import MagicMock

    created = {}

    def make_client(address, *a, **k):
        c = _FakeClient(address)
        created["client"] = c
        return c

    module = MagicMock(name="bleak")
    module.BleakClient = make_client

    resolved = MagicMock()
    resolved.address = "AA:BB:CC:DD:EE:FF"

    async def find_device_by_name(name, timeout=8.0):
        created["scanned_name"] = name
        return resolved

    module.BleakScanner.find_device_by_name = find_device_by_name

    monkeypatch.setitem(sys.modules, "bleak", module)
    return module, created


def _make_device(settings=None, name=None):
    config = DeviceConfig(pins={}, settings=settings or {})
    return BLEDevice(_FakeBoard(), config, name=name or "ble")


async def _initialized(settings, *, name=None):
    device = _make_device(settings=settings, name=name)
    await device.initialize()
    return device


# --- identity / config --------------------------------------------------------


def test_device_type_is_ble():
    assert _make_device().device_type == "BLE"


def test_actions_surface():
    assert set(_make_device().actions) == {"write", "read"}


def test_requires_no_pins():
    assert _make_device().required_pins == []


def test_invalid_value_format_raises():
    with pytest.raises(ValueError, match="value_format"):
        _make_device(settings={"value_format": "float128"})


# --- lifecycle ----------------------------------------------------------------


async def test_initialize_requires_address_or_name():
    device = _make_device(settings={})
    with pytest.raises(ValueError, match="address"):
        await device.initialize()


async def test_initialize_connects(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized({"address": "11:22:33:44:55:66"})
    assert device.is_initialized
    assert created["client"].is_connected


async def test_notify_requires_read_char(fake_bleak):
    device = _make_device(settings={"address": "x", "notify": True})
    with pytest.raises(ValueError, match="read_char_uuid"):
        await device.initialize()


async def test_initialize_subscribes_when_notify(fake_bleak):
    _module, created = fake_bleak
    await _initialized({"address": "x", "notify": True, "read_char_uuid": "uuid-r"})
    assert created["client"]._handler is not None


async def test_shutdown_disconnects_and_clears(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized({"address": "x"})
    await device.shutdown()
    assert not device.is_initialized
    assert not created["client"].is_connected


async def test_shutdown_clears_initialized_even_if_disconnect_raises(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized({"address": "x"})
    created["client"].disconnect_error = RuntimeError("gone")
    await device.shutdown()  # must not propagate; must still clear state
    assert not device.is_initialized


async def test_shutdown_stops_notify(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized({"address": "x", "notify": True, "read_char_uuid": "r"})
    await device.shutdown()
    assert created["client"].stop_notify_called


# --- address resolution (Mac<->Win portability) -------------------------------


async def test_resolves_address_from_name_when_address_blank(fake_bleak):
    _module, created = fake_bleak
    await _initialized({"name": "GliderSensor", "read_char_uuid": "r"})
    assert created["scanned_name"] == "GliderSensor"
    assert created["client"].address == "AA:BB:CC:DD:EE:FF"


# --- write --------------------------------------------------------------------


async def test_write_sends_to_write_char(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized({"address": "x", "write_char_uuid": "uuid-w"})
    await device.write("ON")
    assert created["client"].written == [("uuid-w", b"ON", False)]


async def test_write_joins_multiple_args(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized({"address": "x", "write_char_uuid": "w"})
    await device.write(500, 10)  # floats from Number Inputs -> "500,10"
    assert created["client"].written[0][1] == b"500,10"


async def test_write_without_write_char_raises(fake_bleak):
    device = await _initialized({"address": "x"})
    with pytest.raises(ValueError, match="write_char_uuid"):
        await device.write("ON")


async def test_write_before_initialize_raises():
    device = _make_device(settings={"address": "x", "write_char_uuid": "w"})
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.write("ON")


# --- read (on-demand) ---------------------------------------------------------


async def test_read_decodes_text(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized({"address": "x", "read_char_uuid": "r", "value_format": "text"})
    created["client"].read_value = bytearray(b"98.6")
    assert await device.read() == "98.6"


async def test_read_decodes_int_le(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized({"address": "x", "read_char_uuid": "r", "value_format": "int"})
    created["client"].read_value = bytearray(b"\x2a\x00")  # 42 little-endian
    assert await device.read() == 42


async def test_read_decodes_hex(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized({"address": "x", "read_char_uuid": "r", "value_format": "hex"})
    created["client"].read_value = bytearray(b"\xde\xad")
    assert await device.read() == "dead"


# --- notify / streaming -------------------------------------------------------


async def test_notify_updates_get_state(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized(
        {"address": "x", "notify": True, "read_char_uuid": "r", "value_format": "int"}
    )
    assert await device.get_state() is None  # nothing pushed yet
    created["client"].push(b"\x64\x00")  # 100
    assert await device.get_state() == 100


async def test_read_on_notify_device_returns_latest_push(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized(
        {"address": "x", "notify": True, "read_char_uuid": "r", "value_format": "text"}
    )
    created["client"].push(b"hello")
    assert await device.read() == "hello"


async def test_get_state_none_when_stale(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized(
        {"address": "x", "notify": True, "read_char_uuid": "r", "value_format": "int"}
    )
    device._latest = (5, time.perf_counter() - 999)
    assert await device.get_state() is None


async def test_read_on_notify_times_out_without_push(fake_bleak, monkeypatch):
    import glider.hal.devices.ble_device as bd

    monkeypatch.setattr(bd, "READ_WAIT_S", 0.05)  # keep the test fast
    device = await _initialized({"address": "x", "notify": True, "read_char_uuid": "r"})
    with pytest.raises(RuntimeError, match="no notification"):
        await device.read()


# --- serialization / registry -------------------------------------------------


def test_to_dict_round_trips():
    device = _make_device(
        settings={"address": "AA", "read_char_uuid": "r", "notify": True}, name="hr"
    )
    data = device.to_dict()
    assert data["device_type"] == "BLE"
    rebuilt = BLEDevice.from_dict(data, _FakeBoard())
    assert rebuilt.address == "AA"
    assert rebuilt.is_streaming is True
    assert rebuilt.name == "hr"
    assert rebuilt.id == device.id


def test_registered_as_ble_and_blewrite_untouched():
    from glider.hal.base_device import DEVICE_REGISTRY

    assert DEVICE_REGISTRY.get("BLE") is BLEDevice
    # Backward compat: the original write-only device is still registered.
    assert "BLEWrite" in DEVICE_REGISTRY


def test_exported_from_devices_package():
    from glider.hal import devices

    assert devices.BLEDevice is BLEDevice
