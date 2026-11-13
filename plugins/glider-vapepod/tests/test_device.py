"""Tests for the VapePod device: identity, the commands it sends, and shutdown."""

import asyncio
import logging
import re
from pathlib import Path

import pytest

import glider_vapepod.device as vapepod_module
from glider.hal.base_device import DeviceConfig
from glider_vapepod.device import (
    DEFAULT_COMMAND_CHAR_UUID,
    DEFAULT_NAME,
    DEFAULT_SERVICE_UUID,
    DEFAULT_STATUS_CHAR_UUID,
    MAX_ATOM_PERCENT,
    MAX_PUFF_MS,
    VapePodDevice,
)

# The XIAO nRF52840 sketch the pod is built with, and the ESP32 alternative.
FIRMWARE_DIR = Path(__file__).parents[1] / "firmware"
SKETCHES = [
    FIRMWARE_DIR / "vapepod" / "vapepod.ino",
    FIRMWARE_DIR / "vapepod_esp32" / "vapepod_esp32.ino",
]


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"


def _make_device(settings=None):
    config = DeviceConfig(pins={}, settings=dict(settings or {}))
    return VapePodDevice(_FakeBoard(), config, name="pod")


async def _initialized(settings=None):
    device = _make_device(settings or {"address": "11:22:33:44:55:66"})
    await device.initialize()
    return device


# --- identity / config --------------------------------------------------------


def test_device_type_is_vapepod():
    assert _make_device().device_type == "VapePod"


def test_actions_are_the_command_set():
    assert set(_make_device().actions) == {
        "puff",
        "pump_on",
        "pump_off",
        "pump",
        "off",
        "read",
        "write",
    }


def test_requires_no_pins():
    assert _make_device().required_pins == []


def test_defaults_are_applied_and_saved():
    """A file saved from this device carries the defaults, not just the cache."""
    device = _make_device({"address": "x"})
    assert device._adv_name == DEFAULT_NAME
    assert device._write_char == DEFAULT_COMMAND_CHAR_UUID
    assert device._read_char == DEFAULT_STATUS_CHAR_UUID
    assert device._service_uuid == DEFAULT_SERVICE_UUID
    assert device.is_streaming
    assert device._config.settings["read_char_uuid"] == DEFAULT_STATUS_CHAR_UUID
    assert device._config.settings["notify"] is True


def test_an_explicit_setting_wins():
    device = _make_device({"address": "x", "name": "VapePod-2"})
    assert device._adv_name == "VapePod-2"


def test_from_dict_applies_defaults():
    device = VapePodDevice.from_dict(
        {"name": "pod", "config": {"settings": {"address": "x"}}}, _FakeBoard()
    )
    assert isinstance(device, VapePodDevice)
    assert device._write_char == DEFAULT_COMMAND_CHAR_UUID


@pytest.mark.parametrize("sketch", SKETCHES, ids=lambda p: p.parent.name)
def test_the_firmware_uses_the_same_uuids_and_puff_limit(sketch):
    """Copies of the protocol in C++: every sketch has to agree with device.py."""
    source = sketch.read_text()
    assert f'SERVICE_UUID = "{DEFAULT_SERVICE_UUID}"' in source
    assert f'COMMAND_UUID = "{DEFAULT_COMMAND_CHAR_UUID}"' in source
    assert f'STATUS_UUID = "{DEFAULT_STATUS_CHAR_UUID}"' in source
    assert f'DEVICE_NAME = "{DEFAULT_NAME}"' in source
    limit = re.search(r"MAX_PUFF_MS = (\d+);", source)
    assert limit is not None and int(limit.group(1)) == MAX_PUFF_MS


def test_the_panel_offers_no_puff_the_firmware_would_refuse():
    duration = VapePodDevice.ACTION_ARGS_SCHEMA["puff"][0]
    assert (duration["min"], duration["max"]) == (1, MAX_PUFF_MS)


# --- lifecycle ----------------------------------------------------------------


async def test_initialize_connects_and_subscribes_to_the_status(fake_bleak):
    device = await _initialized()
    client = fake_bleak["client"]
    assert device.is_initialized
    assert client.is_connected
    assert client.notify_char == DEFAULT_STATUS_CHAR_UUID


async def test_a_blank_address_finds_the_pod_by_its_default_name(fake_bleak):
    await _initialized({"address": ""})
    assert fake_bleak["scanned_name"] == DEFAULT_NAME
    assert fake_bleak["client"].address == "AA:BB:CC:DD:EE:FF"


# --- commands -----------------------------------------------------------------


async def test_puff_writes_duration_and_power(fake_bleak):
    device = await _initialized()
    await device.puff(3000, 15)
    assert fake_bleak["client"].written == [b"puff,3000,15"]


async def test_puff_defaults_to_the_power_cap(fake_bleak):
    device = await _initialized()
    await device.puff(2000)
    assert fake_bleak["client"].written == [f"puff,2000,{MAX_ATOM_PERCENT}".encode()]


async def test_the_strongest_puff_is_the_cap(fake_bleak):
    device = await _initialized()
    await device.puff(1000, MAX_ATOM_PERCENT)
    assert fake_bleak["client"].written == [f"puff,1000,{MAX_ATOM_PERCENT}".encode()]


async def test_puff_writes_whole_numbers_from_floats(fake_bleak):
    """Spin boxes and math nodes deliver floats; the firmware refuses "3000.0"."""
    device = await _initialized()
    await device.puff(3000.0, 10.0)
    assert fake_bleak["client"].written == [b"puff,3000,10"]


async def test_the_longest_puff_is_allowed(fake_bleak):
    device = await _initialized()
    await device.puff(MAX_PUFF_MS)
    assert fake_bleak["client"].written == [f"puff,{MAX_PUFF_MS},{MAX_ATOM_PERCENT}".encode()]


@pytest.mark.parametrize(
    "call, expected",
    [
        (lambda d: d.pump(60), b"pump,60"),
        (lambda d: d.pump(0), b"pump,0"),
        (lambda d: d.pump_on(), b"pump,100"),
        (lambda d: d.pump_off(), b"pump,0"),
        (lambda d: d.off(), b"off"),
    ],
)
async def test_pump_and_off_commands(fake_bleak, call, expected):
    device = await _initialized()
    await call(device)
    assert fake_bleak["client"].written == [expected]


async def test_commands_are_acknowledged_writes_to_the_command_characteristic(fake_bleak):
    """The firmware declares WRITE only, so GLIDER hears of a write that failed."""
    device = await _initialized()
    await device.puff(1000)
    client = fake_bleak["client"]
    assert client.last_write_char == DEFAULT_COMMAND_CHAR_UUID
    assert client.last_response is True


@pytest.mark.parametrize("bad", [0, -1, MAX_PUFF_MS + 1, 1.5, "long", None])
async def test_puff_rejects_a_bad_duration(fake_bleak, bad):
    device = await _initialized()
    with pytest.raises(ValueError, match="duration_ms"):
        await device.puff(bad)
    assert fake_bleak["client"].written == []


@pytest.mark.parametrize("bad", [0, MAX_ATOM_PERCENT + 1, 50, 100, 10.5, "max"])
async def test_puff_rejects_a_bad_power(fake_bleak, bad):
    device = await _initialized()
    with pytest.raises(ValueError, match="power"):
        await device.puff(1000, bad)
    assert fake_bleak["client"].written == []


@pytest.mark.parametrize("bad", [-1, 101, 50.5, None])
async def test_pump_rejects_a_bad_speed(fake_bleak, bad):
    device = await _initialized()
    with pytest.raises(ValueError, match="percent"):
        await device.pump(bad)
    assert fake_bleak["client"].written == []


async def test_execute_action_routes_to_puff(fake_bleak):
    """The nodes and the control panel call through execute_action."""
    device = await _initialized()
    await device.execute_action("puff", 2000, 12)
    assert fake_bleak["client"].written == [b"puff,2000,12"]


async def test_read_returns_the_pods_own_status(fake_bleak):
    device = await _initialized()
    fake_bleak["client"].push("atom=100 pump=60")
    assert await device.read() == "atom=100 pump=60"
    assert await device.get_state() == "atom=100 pump=60"


# --- shutdown -----------------------------------------------------------------


async def test_shutdown_sends_off_before_disconnecting(fake_bleak):
    device = await _initialized()
    await device.pump_on()
    client = fake_bleak["client"]

    await device.shutdown()

    assert client.events == [("write", b"pump,100"), ("write", b"off"), ("disconnect", None)]
    assert not device.is_initialized


async def test_shutdown_disconnects_even_when_off_fails(fake_bleak, arm):
    device = await _initialized()
    arm(write_error=RuntimeError("pod vanished"))

    await device.shutdown()

    assert ("disconnect", None) in fake_bleak["client"].events
    assert not device.is_initialized
    assert device._client is None


async def test_shutdown_disconnects_even_when_off_hangs(fake_bleak, arm, monkeypatch):
    """A wedged write must not eat the emergency-stop budget."""
    monkeypatch.setattr(vapepod_module, "OFF_ON_SHUTDOWN_S", 0.05)
    device = await _initialized()
    arm(write_hangs=True)

    await asyncio.wait_for(device.shutdown(), timeout=2.0)

    assert ("disconnect", None) in fake_bleak["client"].events
    assert not device.is_initialized


async def test_shutdown_before_initialize_is_safe_and_quiet(caplog):
    device = _make_device({"address": "x"})
    with caplog.at_level(logging.WARNING, logger="glider_vapepod.device"):
        await device.shutdown()
    assert not device.is_initialized
    assert caplog.records == []


@pytest.mark.parametrize("sketch", SKETCHES, ids=lambda p: p.parent.name)
def test_the_firmware_caps_the_atomizer_at_the_same_percent(sketch):
    """Both sketches refuse, and clamp, above the figure device.py checks."""
    source = sketch.read_text(encoding="utf-8")
    cap = re.search(r"MAX_ATOM_PERCENT = (\d+);", source)
    assert cap is not None and int(cap.group(1)) == MAX_ATOM_PERCENT
    assert "parseNumber(fields[2], 1, MAX_ATOM_PERCENT, b)" in source
    assert "long a = 0, b = MAX_ATOM_PERCENT;" in source  # the default puff power
    assert "&out == &atom && percent > MAX_ATOM_PERCENT" in source  # setOutput clamp


def test_the_panel_offers_no_power_above_the_cap():
    power = VapePodDevice.ACTION_ARGS_SCHEMA["puff"][1]
    assert (power["key"], power["max"], power["default"]) == (
        "power",
        MAX_ATOM_PERCENT,
        MAX_ATOM_PERCENT,
    )
