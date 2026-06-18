"""Tests for the GenericI2CDevice HAL device type.

The device lazy-imports ``smbus2`` inside ``initialize()`` (smbus2 imports
``fcntl`` and is therefore not importable on Windows/macOS). Every test that
needs a working bus injects a fake ``smbus2`` module into ``sys.modules`` so the
device's lazy import resolves to a mock regardless of the host OS, and the
"missing library" test forces that import to fail.
"""

import asyncio
import sys
from unittest.mock import MagicMock

import pytest

from glider.hal.base_device import DeviceConfig, GenericI2CDevice


class _FakeBoard:
    """Minimal board stand-in exposing the bits the device touches."""

    def __init__(self):
        self.id = "fake_board"
        self.i2c_lock = asyncio.Lock()


def _make_device(settings=None, name=None):
    config = DeviceConfig(pins={}, settings=settings or {})
    return GenericI2CDevice(_FakeBoard(), config, name=name)


@pytest.fixture
def fake_smbus2(monkeypatch):
    """Inject a fake ``smbus2`` module; yield (module, mock SMBus instance)."""
    module = MagicMock(name="smbus2")
    bus = MagicMock(name="SMBus_instance")
    module.SMBus.return_value = bus
    monkeypatch.setitem(sys.modules, "smbus2", module)
    return module, bus


async def _initialized(settings=None, *, name=None):
    device = _make_device(settings=settings, name=name)
    await device.initialize()
    return device


# --- Identity / configuration -------------------------------------------------


def test_device_type_is_generic_i2c():
    assert _make_device().device_type == "GenericI2C"


def test_requires_no_gpio_pins():
    assert _make_device().required_pins == []


def test_settings_defaults():
    device = _make_device()
    assert device.i2c_bus == 1
    assert device.i2c_address == 0x40
    assert device.register is None


def test_settings_are_parsed_from_config():
    device = _make_device(settings={"i2c_bus": 0, "i2c_address": 0x53, "register": 0x10})
    assert device.i2c_bus == 0
    assert device.i2c_address == 0x53
    assert device.register == 0x10


def test_actions_expose_full_smbus_surface():
    actions = _make_device().actions
    assert set(actions) == {
        "read",
        "read_byte",
        "write_byte",
        "read_byte_data",
        "write_byte_data",
        "read_word_data",
        "write_word_data",
        "read_word_be",
        "read_block",
        "write_block",
    }


def test_read_word_setting_defaults_false():
    assert _make_device().read_word is False


def test_read_word_setting_parsed():
    assert _make_device(settings={"read_word": True}).read_word is True


# --- Lifecycle ----------------------------------------------------------------


async def test_initialize_opens_bus_on_configured_number(fake_smbus2):
    module, _bus = fake_smbus2
    device = await _initialized(settings={"i2c_bus": 1})
    assert device.is_initialized
    module.SMBus.assert_called_once_with(1)


async def test_shutdown_closes_bus(fake_smbus2):
    _module, bus = fake_smbus2
    device = await _initialized()
    await device.shutdown()
    assert not device.is_initialized
    bus.close.assert_called_once()


async def test_shutdown_without_initialize_is_safe():
    device = _make_device()
    await device.shutdown()  # must not raise
    assert not device.is_initialized


async def test_initialize_without_smbus2_raises_runtimeerror(monkeypatch):
    # sys.modules[name] = None makes ``import name`` raise ImportError.
    monkeypatch.setitem(sys.modules, "smbus2", None)
    device = _make_device()
    with pytest.raises(RuntimeError, match="smbus2"):
        await device.initialize()


# --- Transfers ----------------------------------------------------------------


async def test_read_byte(fake_smbus2):
    _module, bus = fake_smbus2
    bus.read_byte.return_value = 0x01
    device = await _initialized(settings={"i2c_address": 0x50})
    assert await device.read_byte() == 0x01
    bus.read_byte.assert_called_once_with(0x50)


async def test_write_byte(fake_smbus2):
    _module, bus = fake_smbus2
    device = await _initialized(settings={"i2c_address": 0x50})
    await device.write_byte(0x02)
    bus.write_byte.assert_called_once_with(0x50, 0x02)


async def test_read_byte_data(fake_smbus2):
    _module, bus = fake_smbus2
    bus.read_byte_data.return_value = 0xAB
    device = await _initialized(settings={"i2c_address": 0x50})
    assert await device.read_byte_data(0x10) == 0xAB
    bus.read_byte_data.assert_called_once_with(0x50, 0x10)


async def test_write_byte_data(fake_smbus2):
    _module, bus = fake_smbus2
    device = await _initialized(settings={"i2c_address": 0x50})
    await device.write_byte_data(0x10, 0x7F)
    bus.write_byte_data.assert_called_once_with(0x50, 0x10, 0x7F)


async def test_read_word_data(fake_smbus2):
    _module, bus = fake_smbus2
    bus.read_word_data.return_value = 0x1234
    device = await _initialized(settings={"i2c_address": 0x50})
    assert await device.read_word_data(0x20) == 0x1234
    bus.read_word_data.assert_called_once_with(0x50, 0x20)


async def test_write_word_data(fake_smbus2):
    _module, bus = fake_smbus2
    device = await _initialized(settings={"i2c_address": 0x50})
    await device.write_word_data(0x20, 0xBEEF)
    bus.write_word_data.assert_called_once_with(0x50, 0x20, 0xBEEF)


async def test_read_block(fake_smbus2):
    _module, bus = fake_smbus2
    bus.read_i2c_block_data.return_value = [1, 2, 3]
    device = await _initialized(settings={"i2c_address": 0x50})
    assert await device.read_block(0x30, 3) == [1, 2, 3]
    bus.read_i2c_block_data.assert_called_once_with(0x50, 0x30, 3)


async def test_write_block(fake_smbus2):
    _module, bus = fake_smbus2
    device = await _initialized(settings={"i2c_address": 0x50})
    await device.write_block(0x30, [4, 5, 6])
    bus.write_i2c_block_data.assert_called_once_with(0x50, 0x30, [4, 5, 6])


# --- The no-arg ``read`` alias ------------------------------------------------


async def test_read_alias_uses_default_register_when_set(fake_smbus2):
    _module, bus = fake_smbus2
    bus.read_byte_data.return_value = 0x09
    device = await _initialized(settings={"i2c_address": 0x50, "register": 0x05})
    assert await device.read() == 0x09
    bus.read_byte_data.assert_called_once_with(0x50, 0x05)


async def test_read_alias_raw_byte_when_no_default_register(fake_smbus2):
    _module, bus = fake_smbus2
    bus.read_byte.return_value = 0x07
    device = await _initialized(settings={"i2c_address": 0x50})
    assert await device.read() == 0x07
    bus.read_byte.assert_called_once_with(0x50)


# --- Big-endian 2-byte read (e.g. AS5600 12-bit angle) ------------------------


async def test_read_word_be_combines_two_registers_msb_first(fake_smbus2):
    _module, bus = fake_smbus2
    # AS5600 ANGLE_H=0x0A, ANGLE_L=0xBC  -> 0x0ABC (2748)
    bus.read_i2c_block_data.return_value = [0x0A, 0xBC]
    device = await _initialized(settings={"i2c_address": 0x36})
    assert await device.read_word_be(0x0E) == 0x0ABC
    bus.read_i2c_block_data.assert_called_once_with(0x36, 0x0E, 2)


async def test_read_alias_uses_big_endian_word_when_read_word_set(fake_smbus2):
    _module, bus = fake_smbus2
    bus.read_i2c_block_data.return_value = [0x0A, 0xBC]
    device = await _initialized(settings={"i2c_address": 0x36, "register": 0x0E, "read_word": True})
    assert await device.read() == 0x0ABC
    bus.read_i2c_block_data.assert_called_once_with(0x36, 0x0E, 2)
    # Must NOT fall back to the single-byte read.
    bus.read_byte_data.assert_not_called()


# --- Validation / error paths -------------------------------------------------


def test_address_above_range_raises_valueerror():
    with pytest.raises(ValueError):
        _make_device(settings={"i2c_address": 0x80})


def test_address_below_range_raises_valueerror():
    with pytest.raises(ValueError):
        _make_device(settings={"i2c_address": 0x02})


async def test_invalid_register_raises_valueerror(fake_smbus2):
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.read_byte_data(0x100)


async def test_invalid_byte_value_raises_valueerror(fake_smbus2):
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.write_byte_data(0x10, 256)


async def test_invalid_word_value_raises_valueerror(fake_smbus2):
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.write_word_data(0x10, 0x10000)


async def test_invalid_block_length_raises_valueerror(fake_smbus2):
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.read_block(0x10, 33)


async def test_invalid_block_data_byte_raises_valueerror(fake_smbus2):
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.write_block(0x10, [0, 256])


async def test_transfer_before_initialize_raises_runtimeerror():
    device = _make_device()
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.read_byte()


async def test_missing_required_arg_raises_valueerror_not_typeerror(fake_smbus2):
    # Mirrors a Device Action node leaving arg2 (value) unconnected: the action is
    # invoked with one positional. Must be a clear ValueError, never a raw TypeError.
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.execute_action("write_byte_data", 0x10)


async def test_wired_zero_value_is_written(fake_smbus2):
    _module, bus = fake_smbus2
    device = await _initialized(settings={"i2c_address": 0x50})
    await device.execute_action("write_byte_data", 0x10, 0)
    bus.write_byte_data.assert_called_once_with(0x50, 0x10, 0)


# --- Serialization ------------------------------------------------------------


def test_to_dict_round_trips_through_from_dict():
    device = _make_device(
        settings={"i2c_bus": 1, "i2c_address": 0x53, "register": 0x10}, name="sensor"
    )
    data = device.to_dict()
    assert data["device_type"] == "GenericI2C"

    rebuilt = GenericI2CDevice.from_dict(data, _FakeBoard())
    assert rebuilt.i2c_bus == 1
    assert rebuilt.i2c_address == 0x53
    assert rebuilt.register == 0x10
    assert rebuilt.name == "sensor"
    assert rebuilt.id == device.id


# --- Registry / package export ------------------------------------------------


def test_registered_in_device_registry():
    from glider.hal.base_device import DEVICE_REGISTRY

    assert DEVICE_REGISTRY.get("GenericI2C") is GenericI2CDevice


def test_create_device_from_dict_builds_generic_i2c():
    from glider.hal.base_device import create_device_from_dict

    data = {
        "id": "i2c_1",
        "device_type": "GenericI2C",
        "name": "x",
        "board_id": "b",
        "config": {"pins": {}, "settings": {"i2c_address": 0x40}},
    }
    device = create_device_from_dict(data, _FakeBoard())
    assert isinstance(device, GenericI2CDevice)
    assert device.i2c_address == 0x40


def test_exported_from_devices_package():
    from glider.hal import devices

    assert devices.GenericI2CDevice is GenericI2CDevice
