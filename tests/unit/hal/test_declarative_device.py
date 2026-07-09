"""
Tests for declarative ("no-code") custom devices: the interpreter
(DeclarativeDevice), definition validation, and the device-library
load/save/register round-trip.

smbus2 and the board are mocked so these run with no hardware.
"""

from __future__ import annotations

import sys
import types

import pytest

from glider.core import device_library
from glider.hal.base_device import DeviceConfig
from glider.hal.declarative_device import (
    build_device_class,
    revolution_settings,
    standard_settings,
    validate_definition,
)

# --- fakes ---------------------------------------------------------------


class _FakeSMBus:
    instances: list[_FakeSMBus] = []

    def __init__(self, busnum):
        self.busnum = busnum
        self.calls = []
        _FakeSMBus.instances.append(self)

    def read_i2c_block_data(self, addr, reg, length):
        self.calls.append(("read_block", addr, reg, length))
        return [0x12, 0x34][:length] + [0] * max(0, length - 2)

    def read_byte_data(self, addr, reg):
        self.calls.append(("read_byte", addr, reg))
        return 0x55

    def write_byte_data(self, addr, reg, val):
        self.calls.append(("write_byte", addr, reg, val))

    def write_i2c_block_data(self, addr, reg, data):
        self.calls.append(("write_block", addr, reg, list(data)))

    def close(self):
        pass


@pytest.fixture
def fake_smbus(monkeypatch):
    mod = types.ModuleType("smbus2")
    mod.SMBus = _FakeSMBus
    _FakeSMBus.instances = []
    monkeypatch.setitem(sys.modules, "smbus2", mod)
    return mod


class _MockBoard:
    def __init__(self):
        self.calls = []

    async def set_pin_mode(self, pin, mode, ptype):
        self.calls.append(("mode", pin, mode.name, ptype.name))

    async def write_digital(self, pin, val):
        self.calls.append(("write_digital", pin, val))

    async def read_digital(self, pin):
        self.calls.append(("read_digital", pin))
        return True

    async def read_analog(self, pin):
        self.calls.append(("read_analog", pin))
        return 512

    async def write_analog(self, pin, val):
        self.calls.append(("write_analog", pin, val))


def _device(definition, board=None, **settings):
    cls = build_device_class(definition)
    return cls(board, DeviceConfig(pins={}, settings=settings), name=definition["name"])


# --- validation ----------------------------------------------------------


def test_validate_catches_problems():
    assert "needs a name" in " ".join(
        validate_definition({"transport": "i2c", "actions": [{"name": "a", "op": "read_byte"}]})
    )
    assert "transport" in " ".join(validate_definition({"name": "X", "actions": []}))
    bad_op = validate_definition(
        {"name": "X", "transport": "i2c", "actions": [{"name": "a", "op": "set_high"}]}
    )
    assert any("not valid" in e for e in bad_op)  # gpio op on i2c device
    assert (
        validate_definition(
            {"name": "X", "transport": "i2c", "actions": [{"name": "a", "op": "read_byte"}]}
        )
        == []
    )


def test_build_class_carries_identity_and_schema():
    defn = {
        "name": "MySensor",
        "transport": "i2c",
        "settings": standard_settings("i2c"),
        "actions": [{"name": "v", "op": "read_byte", "params": {"register": 0}}],
    }
    dev = _device(defn)
    assert dev.device_type == "MySensor"
    assert {f["key"] for f in type(dev).SETTINGS_SCHEMA} == {"i2c_bus", "i2c_address"}


# --- i2c interpreter -----------------------------------------------------


async def test_i2c_read_word_big_endian_and_primary(fake_smbus):
    defn = {
        "name": "I2CDev",
        "transport": "i2c",
        "settings": standard_settings("i2c"),
        "actions": [
            {"name": "temp", "op": "read_word", "params": {"register": 0}, "primary": True},
            {
                "name": "cfg",
                "op": "write_byte",
                "params": {"register": 1},
                "runtime_args": ["value"],
            },
        ],
    }
    dev = _device(defn)  # defaults: bus 1, addr 0x48
    await dev.initialize()
    assert await dev.read() == 0x1234  # big-endian [0x12,0x34]
    await dev.execute_action("cfg", 42)
    bus = _FakeSMBus.instances[-1]
    assert ("write_byte", 0x48, 1, 42) in bus.calls


async def test_i2c_address_from_settings(fake_smbus):
    defn = {
        "name": "I2CAddr",
        "transport": "i2c",
        "settings": standard_settings("i2c"),
        "actions": [{"name": "v", "op": "read_byte", "params": {"register": 5}, "primary": True}],
    }
    dev = _device(defn, i2c_address=0x60)
    await dev.initialize()
    await dev.read()
    assert ("read_byte", 0x60, 5) in _FakeSMBus.instances[-1].calls


# --- gpio interpreter ----------------------------------------------------


async def test_gpio_output_ops_and_pin_mode():
    defn = {
        "name": "GpioOut",
        "transport": "gpio",
        "settings": standard_settings("gpio"),
        "actions": [
            {"name": "on", "op": "set_high"},
            {"name": "off", "op": "set_low"},
            {"name": "dim", "op": "write_pwm", "runtime_args": ["value"]},
        ],
    }
    board = _MockBoard()
    dev = _device(defn, board=board, pin=7)
    await dev.initialize()
    assert ("mode", 7, "OUTPUT", "DIGITAL") in board.calls
    await dev.execute_action("on")
    await dev.execute_action("dim", 200)
    assert ("write_digital", 7, True) in board.calls
    assert ("write_analog", 7, 200) in board.calls


async def test_gpio_analog_input():
    defn = {
        "name": "GpioIn",
        "transport": "gpio",
        "settings": standard_settings("gpio"),
        "actions": [{"name": "level", "op": "read_analog", "primary": True}],
    }
    board = _MockBoard()
    dev = _device(defn, board=board, pin=14)
    await dev.initialize()
    assert ("mode", 14, "INPUT", "ANALOG") in board.calls
    assert await dev.read() == 512


# --- revolution tracking -------------------------------------------------


def _rev_device(**settings):
    defn = {
        "name": "Enc",
        "transport": "i2c",
        "track_revolutions": True,
        "settings": standard_settings("i2c") + revolution_settings(),
        "actions": [
            {"name": "turns", "op": "read_revolutions", "primary": True},
            {"name": "angle", "op": "read_angle"},
            {"name": "reset", "op": "reset_revolutions"},
        ],
    }
    cls = build_device_class(defn)
    return cls(None, DeviceConfig(pins={}, settings=settings), name="Enc")


def test_revolution_ops_need_track_flag():
    base = {"name": "X", "transport": "i2c", "actions": [{"name": "a", "op": "read_revolutions"}]}
    assert any("not valid" in e for e in validate_definition(base))  # no track flag
    assert validate_definition({**base, "track_revolutions": True}) == []


async def test_revolution_accumulate_and_reset():
    dev = _rev_device()  # counts_per_turn default 4096, decimals 2
    dev._initialized = True  # execute_action now gates on initialization
    for raw in [0, 1024, 2048, 3072, 4095, 0]:  # one full forward turn
        dev._accumulate(raw)
    assert await dev.read() == 1.0  # primary = read_revolutions
    assert await dev.execute_action("angle") == 0
    await dev.execute_action("reset")
    assert await dev.read() == 0.0


# --- device library ------------------------------------------------------


def test_library_save_load_roundtrip(tmp_path):
    defn = {
        "name": "LibSensor",
        "transport": "i2c",
        "settings": standard_settings("i2c"),
        "actions": [{"name": "v", "op": "read_word", "params": {"register": 0}, "primary": True}],
    }
    path = device_library.save_definition(defn, tmp_path)
    assert path.exists()
    loaded = device_library.load_definitions(tmp_path)
    assert len(loaded) == 1 and loaded[0]["name"] == "LibSensor"


def test_library_skips_invalid(tmp_path):
    (tmp_path / "bad.gdevice").write_text(
        '{"name": "Bad"}', encoding="utf-8"
    )  # no transport/actions
    assert device_library.load_definitions(tmp_path) == []


def test_library_register_into_registry(tmp_path):
    from glider.hal.base_device import DEVICE_REGISTRY

    defn = {
        "name": "RegSensor",
        "transport": "gpio",
        "settings": standard_settings("gpio"),
        "actions": [{"name": "v", "op": "read_digital", "primary": True}],
    }
    device_library.save_definition(defn, tmp_path)
    try:
        names = device_library.load_and_register_all(tmp_path)
        assert "RegSensor" in names
        assert "RegSensor" in DEVICE_REGISTRY
    finally:
        DEVICE_REGISTRY.pop("RegSensor", None)
