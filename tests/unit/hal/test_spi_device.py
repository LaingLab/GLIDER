"""Tests for the GenericSPIDevice HAL device type.

The device lazy-imports ``spidev`` inside ``initialize()`` (spidev wraps the
Linux spidev ioctl API and is not importable off-Linux). Tests inject a fake
``spidev`` module into ``sys.modules`` so the import resolves to a mock
regardless of host OS; the "missing library" test forces the import to fail.
"""

import sys
from unittest.mock import MagicMock

import pytest

from glider.hal.base_device import DeviceConfig
from glider.hal.devices.spi_device import GenericSPIDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"


def _make_device(settings=None, name=None):
    config = DeviceConfig(pins={}, settings=settings or {})
    return GenericSPIDevice(_FakeBoard(), config, name=name or "spi")


@pytest.fixture
def fake_spidev(monkeypatch):
    """Inject a fake ``spidev`` module; yield (module, mock SpiDev instance)."""
    module = MagicMock(name="spidev")
    spi = MagicMock(name="SpiDev_instance")
    module.SpiDev.return_value = spi
    monkeypatch.setitem(sys.modules, "spidev", module)
    return module, spi


async def _initialized(settings=None, *, name=None):
    device = _make_device(settings=settings, name=name)
    await device.initialize()
    return device


# --- identity / config --------------------------------------------------------


def test_device_type_is_generic_spi():
    assert _make_device().device_type == "GenericSPI"


def test_requires_no_pins():
    assert _make_device().required_pins == []


def test_settings_defaults():
    d = _make_device()
    assert d.spi_bus == 0
    assert d.spi_device == 0
    assert d._max_speed_hz == 500000
    assert d._mode == 0


def test_settings_parsed():
    d = _make_device(
        settings={"spi_bus": 1, "spi_device": 2, "max_speed_hz": 1_000_000, "spi_mode": 3}
    )
    assert d.spi_bus == 1
    assert d.spi_device == 2
    assert d._max_speed_hz == 1_000_000
    assert d._mode == 3


def test_actions_surface():
    assert set(_make_device().actions) == {"transfer", "write", "read", "read_register"}


@pytest.mark.parametrize(
    "settings",
    [
        {"spi_bus": -1},
        {"spi_device": -1},
        {"max_speed_hz": 0},
        {"spi_mode": 4},
    ],
)
def test_invalid_settings_raise(settings):
    with pytest.raises(ValueError):
        _make_device(settings=settings)


# --- lifecycle ----------------------------------------------------------------


async def test_initialize_opens_and_configures(fake_spidev):
    module, spi = fake_spidev
    await _initialized(settings={"spi_bus": 1, "spi_device": 0, "max_speed_hz": 2_000_000,
                                 "spi_mode": 2})
    spi.open.assert_called_once_with(1, 0)
    assert spi.max_speed_hz == 2_000_000
    assert spi.mode == 2


async def test_initialize_without_spidev_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "spidev", None)
    device = _make_device()
    with pytest.raises(RuntimeError, match="spidev"):
        await device.initialize()


async def test_shutdown_closes(fake_spidev):
    _module, spi = fake_spidev
    device = await _initialized()
    await device.shutdown()
    assert not device.is_initialized
    spi.close.assert_called_once()


async def test_shutdown_without_initialize_is_safe():
    device = _make_device()
    await device.shutdown()
    assert not device.is_initialized


async def test_shutdown_clears_even_if_close_raises(fake_spidev):
    _module, spi = fake_spidev
    spi.close.side_effect = OSError("gone")
    device = await _initialized()
    await device.shutdown()
    assert not device.is_initialized


# --- transfers ----------------------------------------------------------------


async def test_transfer_xfer2_and_returns_read_bytes(fake_spidev):
    _module, spi = fake_spidev
    spi.xfer2.return_value = [0x00, 0xAB]
    device = await _initialized()
    assert await device.transfer([0x01, 0x00]) == [0x00, 0xAB]
    spi.xfer2.assert_called_once_with([0x01, 0x00])


async def test_transfer_parses_hex_string(fake_spidev):
    _module, spi = fake_spidev
    spi.xfer2.return_value = [0, 0]
    device = await _initialized()
    await device.transfer("0x01,0x80")
    spi.xfer2.assert_called_once_with([0x01, 0x80])


async def test_write_writebytes(fake_spidev):
    _module, spi = fake_spidev
    device = await _initialized()
    await device.write([0xDE, 0xAD])
    spi.writebytes.assert_called_once_with([0xDE, 0xAD])


async def test_read_readbytes(fake_spidev):
    _module, spi = fake_spidev
    spi.readbytes.return_value = [1, 2, 3]
    device = await _initialized()
    assert await device.read(3) == [1, 2, 3]
    spi.readbytes.assert_called_once_with(3)


async def test_read_register_strips_command_bytes(fake_spidev):
    _module, spi = fake_spidev
    # Frame = [reg, 0, 0]; xfer2 echoes a byte for the command, then 2 data bytes.
    spi.xfer2.return_value = [0x00, 0x12, 0x34]
    device = await _initialized()
    assert await device.read_register(0x8F, 2) == [0x12, 0x34]
    spi.xfer2.assert_called_once_with([0x8F, 0x00, 0x00])


async def test_transfer_rejects_out_of_range_byte(fake_spidev):
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.transfer([0, 256])


async def test_read_rejects_bad_length(fake_spidev):
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.read(0)


async def test_action_before_initialize_raises():
    device = _make_device()
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.transfer([0x01])


async def test_execute_action_transfer_string_arg(fake_spidev):
    _module, spi = fake_spidev
    spi.xfer2.return_value = [0]
    device = await _initialized()
    await device.execute_action("write", "255")
    spi.writebytes.assert_called_once_with([255])


# --- review fixes -------------------------------------------------------------


async def test_get_state_returns_none_so_recorder_does_not_poll_the_bus(fake_spidev):
    # #1: without get_state the DataRecorder falls through to read() and clocks
    # an unsolicited readbytes() every tick. get_state() must short-circuit that.
    _module, spi = fake_spidev
    device = await _initialized()
    assert await device.get_state() is None
    spi.readbytes.assert_not_called()


async def test_transfer_accepts_multiple_args_from_node_comma_split(fake_spidev):
    # #5: DeviceAction splits "0x01,0x80" into two positional args.
    _module, spi = fake_spidev
    spi.xfer2.return_value = [0, 0]
    device = await _initialized()
    await device.execute_action("transfer", "0x01", "0x80")
    spi.xfer2.assert_called_once_with([0x01, 0x80])


async def test_write_accepts_multiple_args(fake_spidev):
    _module, spi = fake_spidev
    device = await _initialized()
    await device.execute_action("write", 1, 2, 3)
    spi.writebytes.assert_called_once_with([1, 2, 3])


def test_to_byte_list_hex_strings_in_a_list():
    # #9: a list of hex-string tokens parses like the comma-string form.
    assert GenericSPIDevice._to_byte_list(["0x01", "0x80"]) == [0x01, 0x80]


async def test_action_after_shutdown_raises_not_uses_closed_handle(fake_spidev):
    # #2: _call re-checks state under the lock, so a transfer dispatched after
    # shutdown raises cleanly instead of touching the closed handle.
    _module, spi = fake_spidev
    device = await _initialized()
    await device.shutdown()
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.transfer([0x01])
    spi.xfer2.assert_not_called()


async def test_apply_settings_while_initialized_saves_but_does_not_change_live_caches(fake_spidev):
    # #B-apply: a live edit is recorded to config.settings (for the file) but the
    # open handle's cached params stay put until reconnect.
    device = await _initialized(settings={"spi_bus": 0, "max_speed_hz": 500000})
    device.apply_settings({"max_speed_hz": 8_000_000})
    assert device._config.settings["max_speed_hz"] == 8_000_000  # saved
    assert device._max_speed_hz == 500000  # live cache unchanged until reconnect


# --- serialization / registry -------------------------------------------------


def test_to_dict_round_trips():
    device = _make_device(settings={"spi_bus": 1, "spi_device": 1, "spi_mode": 3}, name="adc")
    data = device.to_dict()
    assert data["device_type"] == "GenericSPI"
    rebuilt = GenericSPIDevice.from_dict(data, _FakeBoard())
    assert rebuilt.spi_bus == 1
    assert rebuilt.spi_device == 1
    assert rebuilt._mode == 3
    assert rebuilt.name == "adc"
    assert rebuilt.id == device.id


def test_registered_in_device_registry():
    from glider.hal.base_device import DEVICE_REGISTRY

    assert DEVICE_REGISTRY.get("GenericSPI") is GenericSPIDevice


def test_create_device_from_dict_builds_generic_spi():
    from glider.hal.base_device import create_device_from_dict

    data = {
        "id": "spi_1",
        "device_type": "GenericSPI",
        "name": "x",
        "board_id": "b",
        "config": {"pins": {}, "settings": {"spi_bus": 0, "spi_device": 1}},
    }
    device = create_device_from_dict(data, _FakeBoard())
    assert isinstance(device, GenericSPIDevice)
    assert device.spi_device == 1


def test_exported_from_devices_package():
    from glider.hal import devices

    assert devices.GenericSPIDevice is GenericSPIDevice


def test_apply_settings_validates_and_updates():
    d = _make_device(settings={"spi_bus": 0})
    d.apply_settings({"spi_bus": 1, "max_speed_hz": 8_000_000})
    assert d.spi_bus == 1
    assert d._max_speed_hz == 8_000_000
    with pytest.raises(ValueError):
        d.apply_settings({"spi_mode": 9})
    assert d._mode == 0  # unchanged after a rejected edit
