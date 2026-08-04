"""Tests for the GenericSerialDevice HAL device type.

The device lazy-imports ``serial`` (pyserial) inside ``initialize()``. Tests
that need a working port inject a fake ``serial`` module into ``sys.modules`` so
the import resolves to a mock regardless of whether a real port exists; the
"missing library" test forces that import to fail.
"""

import asyncio
import sys
import time
from unittest.mock import MagicMock

import pytest

from glider.hal.base_device import DeviceConfig
from glider.hal.devices.serial_device import GenericSerialDevice


class _FakeBoard:
    """Minimal board stand-in exposing the bits the device touches."""

    def __init__(self):
        self.id = "fake_board"


def _make_device(settings=None, name=None):
    config = DeviceConfig(pins={}, settings=settings or {})
    return GenericSerialDevice(_FakeBoard(), config, name=name or "ser")


@pytest.fixture
def fake_serial(monkeypatch):
    """Inject a fake ``serial`` module; yield (module, mock Serial instance)."""
    module = MagicMock(name="serial")
    ser = MagicMock(name="Serial_instance")
    module.Serial.return_value = ser
    monkeypatch.setitem(sys.modules, "serial", module)
    return module, ser


async def _initialized(settings=None, *, name=None):
    device = _make_device(settings={"port": "/dev/ttyFAKE", **(settings or {})}, name=name)
    await device.initialize()
    return device


# --- Identity / configuration -------------------------------------------------


def test_device_type_is_generic_serial():
    assert _make_device().device_type == "GenericSerial"


def test_requires_no_gpio_pins():
    assert _make_device().required_pins == []


def test_settings_defaults():
    d = _make_device()
    assert d.port == ""
    assert d.baudrate == 9600
    assert d._bytesize == 8
    assert d._parity == "N"
    assert d._stopbits == 1
    assert d._terminator == "\n"
    assert d._encoding == "utf-8"
    assert d.is_streaming is False


def test_settings_parsed_from_config():
    d = _make_device(
        settings={
            "port": "COM3",
            "baudrate": 115200,
            "bytesize": 7,
            "parity": "E",
            "stopbits": 2,
            "terminator": "\r",
            "stream": True,
        }
    )
    assert d.port == "COM3"
    assert d.baudrate == 115200
    assert d._bytesize == 7
    assert d._parity == "E"
    assert d._stopbits == 2
    assert d._terminator == "\r"
    assert d.is_streaming is True


def test_actions_surface():
    assert set(_make_device().actions) == {"write", "read_line", "query", "read"}


# --- Validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "settings",
    [
        {"baudrate": 0},
        {"baudrate": -1},
        {"bytesize": 9},
        {"parity": "Z"},
        {"stopbits": 3},
        {"timeout": -0.5},
    ],
)
def test_invalid_settings_raise_valueerror(settings):
    with pytest.raises(ValueError):
        _make_device(settings=settings)


# --- Lifecycle ----------------------------------------------------------------


async def test_initialize_requires_port(fake_serial):
    device = _make_device(settings={"port": ""})
    with pytest.raises(ValueError, match="port"):
        await device.initialize()


async def test_initialize_opens_port_with_configured_params(fake_serial):
    module, _ser = fake_serial
    await _initialized(settings={"baudrate": 19200, "parity": "E", "stopbits": 2})
    module.Serial.assert_called_once()
    kwargs = module.Serial.call_args.kwargs
    assert kwargs["port"] == "/dev/ttyFAKE"
    assert kwargs["baudrate"] == 19200
    assert kwargs["parity"] == "E"
    assert kwargs["stopbits"] == 2


async def test_initialize_without_pyserial_raises_runtimeerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)
    device = _make_device(settings={"port": "/dev/ttyFAKE"})
    with pytest.raises(RuntimeError, match="pyserial"):
        await device.initialize()


async def test_shutdown_closes_port(fake_serial):
    _module, ser = fake_serial
    device = await _initialized()
    await device.shutdown()
    assert not device.is_initialized
    ser.close.assert_called_once()


async def test_shutdown_without_initialize_is_safe():
    device = _make_device()
    await device.shutdown()  # must not raise
    assert not device.is_initialized


async def test_shutdown_clears_initialized_even_if_close_raises(fake_serial):
    _module, ser = fake_serial
    ser.close.side_effect = OSError("port already gone")
    device = await _initialized()
    await device.shutdown()  # close() raising must not leave us initialized
    assert not device.is_initialized


# --- Request/response actions -------------------------------------------------


async def test_write_appends_terminator_and_flushes(fake_serial):
    _module, ser = fake_serial
    device = await _initialized()
    await device.write("MEAS?")
    ser.write.assert_called_once_with(b"MEAS?\n")
    ser.flush.assert_called_once()


async def test_write_does_not_double_terminate(fake_serial):
    _module, ser = fake_serial
    device = await _initialized()
    await device.write("ON\n")
    ser.write.assert_called_once_with(b"ON\n")


async def test_write_none_raises_valueerror(fake_serial):
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.write(None)


async def test_read_line_reads_until_terminator_and_strips(fake_serial):
    _module, ser = fake_serial
    ser.read_until.return_value = b"42.5\r\n"
    device = await _initialized()
    assert await device.read_line() == "42.5"
    ser.read_until.assert_called_once_with(b"\n")


async def test_query_writes_then_reads(fake_serial):
    _module, ser = fake_serial
    ser.read_until.return_value = b"OK\n"
    device = await _initialized()
    assert await device.query("STATUS?") == "OK"
    ser.write.assert_called_once_with(b"STATUS?\n")


async def test_action_before_initialize_raises_runtimeerror():
    device = _make_device(settings={"port": "/dev/ttyFAKE"})
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.write("x")


async def test_execute_action_write_zero_is_sent(fake_serial):
    _module, ser = fake_serial
    device = await _initialized()
    await device.execute_action("write", 0)  # a wired 0 must still be written
    ser.write.assert_called_once_with(b"0\n")


# --- Streaming ----------------------------------------------------------------


async def test_get_state_none_before_any_sample():
    device = _make_device(settings={"port": "/dev/ttyFAKE", "stream": True})
    assert await device.get_state() is None


async def test_get_state_returns_fresh_cached_line():
    device = _make_device(settings={"port": "/dev/ttyFAKE", "stream": True})
    device._latest = ("weight=12.3", time.perf_counter())
    assert await device.get_state() == "weight=12.3"


async def test_get_state_none_when_sample_is_stale():
    device = _make_device(settings={"port": "/dev/ttyFAKE", "stream": True})
    device._latest = ("old", time.perf_counter() - 999)
    assert await device.get_state() is None


async def test_streaming_reader_thread_starts_and_stops(fake_serial):
    _module, ser = fake_serial
    # read_until idles (returns empty after a short "timeout") so the loop keeps
    # re-checking the stop event without spinning the CPU.
    ser.read_until.side_effect = lambda *a, **k: (time.sleep(0.005) or b"")
    device = await _initialized(settings={"stream": True})
    assert device._thread is not None and device._thread.is_alive()
    await device.shutdown()
    assert device._thread is None
    ser.close.assert_called_once()


async def test_streaming_reader_caches_a_framed_line(fake_serial):
    _module, ser = fake_serial
    lines = [b"reading=7\n"]

    def _read_until(*_a, **_k):
        if lines:
            return lines.pop(0)
        time.sleep(0.005)
        return b""

    ser.read_until.side_effect = _read_until
    device = await _initialized(settings={"stream": True})
    # Give the reader thread a moment to consume the queued line.
    for _ in range(50):
        if await device.get_state() is not None:
            break
        await asyncio.sleep(0.01)
    assert await device.get_state() == "reading=7"
    await device.shutdown()


# --- apply_settings -----------------------------------------------------------


def test_apply_settings_updates_caches_and_config():
    d = _make_device(settings={"port": "COM1", "baudrate": 9600})
    d.apply_settings({"baudrate": 115200, "port": "COM7"})
    assert d.baudrate == 115200
    assert d.port == "COM7"
    assert d._config.settings["baudrate"] == 115200


def test_apply_settings_rejects_invalid_without_partial_apply():
    d = _make_device(settings={"baudrate": 9600})
    with pytest.raises(ValueError):
        d.apply_settings({"baudrate": -5})
    assert d.baudrate == 9600  # unchanged


async def test_apply_settings_while_initialized_saves_but_keeps_live_caches(fake_serial):
    # #B5: flipping stream (or any conn param) on a live device would desync the
    # reader thread from the flag, so a live edit only records to config.settings.
    device = await _initialized(settings={"stream": False, "baudrate": 9600})
    device.apply_settings({"stream": True, "baudrate": 115200})
    assert device._config.settings["baudrate"] == 115200  # saved to file
    assert device.baudrate == 9600  # live cache unchanged
    assert device.is_streaming is False  # reader-thread state not desynced


# --- review fixes -------------------------------------------------------------


async def test_read_line_raises_on_incomplete_frame(fake_serial):
    # #4: read_until returns partial bytes (no terminator) on a timeout; that
    # must raise, not be recorded as a truncated value.
    _module, ser = fake_serial
    ser.read_until.return_value = b"23.4"  # no trailing \n
    device = await _initialized()
    with pytest.raises(RuntimeError, match="incomplete read"):
        await device.read_line()


async def test_streaming_reader_discards_partial_frame(fake_serial):
    # #4: a partial frame in the reader loop is discarded, never cached.
    _module, ser = fake_serial
    ser.read_until.side_effect = [b"12.3"]  # partial, then StopIteration -> handled

    def _read_until(*_a, **_k):
        import time as _t

        _t.sleep(0.005)
        return b"12.3"  # always partial

    ser.read_until.side_effect = _read_until
    device = await _initialized(settings={"stream": True})
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert await device.get_state() is None  # nothing complete was cached
    await device.shutdown()


async def test_write_comma_joins_multiple_args(fake_serial):
    # #5: node comma-split "SET,1,2" -> write("SET",1,2) must round-trip.
    _module, ser = fake_serial
    device = await _initialized()
    await device.execute_action("write", "SET", 1, 2)
    ser.write.assert_called_once_with(b"SET,1,2\n")


async def test_streaming_reader_timeout_is_clamped(fake_serial):
    # #6/C5: a huge user timeout must not become the reader's stop-latency.
    _module, ser = fake_serial
    ser.read_until.side_effect = lambda *a, **k: (__import__("time").sleep(0.005) or b"")
    device = await _initialized(settings={"stream": True, "timeout": 60.0})
    assert ser.timeout <= 1.0  # MAX_READER_TIMEOUT_S
    await device.shutdown()


async def test_action_after_shutdown_raises(fake_serial):
    # #3: write re-checks state under the port lock, so it can't use a closed port.
    _module, ser = fake_serial
    device = await _initialized()
    await device.shutdown()
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.write("x")
    ser.write.assert_not_called()


# --- Serialization / registry -------------------------------------------------


def test_to_dict_round_trips_through_from_dict():
    device = _make_device(settings={"port": "COM3", "baudrate": 57600}, name="scale")
    data = device.to_dict()
    assert data["device_type"] == "GenericSerial"
    rebuilt = GenericSerialDevice.from_dict(data, _FakeBoard())
    assert rebuilt.port == "COM3"
    assert rebuilt.baudrate == 57600
    assert rebuilt.name == "scale"
    assert rebuilt.id == device.id


def test_registered_in_device_registry():
    from glider.hal.base_device import DEVICE_REGISTRY

    assert DEVICE_REGISTRY.get("GenericSerial") is GenericSerialDevice


def test_create_device_from_dict_builds_generic_serial():
    from glider.hal.base_device import create_device_from_dict

    data = {
        "id": "ser_1",
        "device_type": "GenericSerial",
        "name": "x",
        "board_id": "b",
        "config": {"pins": {}, "settings": {"port": "COM4"}},
    }
    device = create_device_from_dict(data, _FakeBoard())
    assert isinstance(device, GenericSerialDevice)
    assert device.port == "COM4"


def test_exported_from_devices_package():
    from glider.hal import devices

    assert devices.GenericSerialDevice is GenericSerialDevice
