"""``add_device`` / ``add_device_multi_pin`` accept settings as a dict.

These take the display name as ``name=`` and forward everything else as
``**kwargs``, so a device setting called ``name`` -- which every BLE device has,
for the advertised local name -- could not be passed at all: splatting it raised
``TypeError: got multiple values for keyword argument 'name'``. That broke adding
a BLE or Maimu device from the dialog, and reloading a saved file containing one.

The ``settings=`` parameter is the way through. ``**kwargs`` still works, and
still wins on a conflict.
"""

import pytest

from glider.core.hardware_manager import HardwareManager
from glider.hal.mock_board import MockBoard


def _manager_with_board(board_id="b1"):
    manager = HardwareManager()
    manager._boards[board_id] = MockBoard()
    return manager


def test_settings_dict_carries_a_name_setting():
    manager = _manager_with_board()
    manager.add_device_multi_pin(
        "stim_1",
        "Maimu",
        "b1",
        pins={},
        name="Left stimulator",
        settings={"name": "Maimu-01", "address": ""},
    )

    device = manager.get_device("stim_1")
    assert device.name == "Left stimulator"  # display name
    assert device._config.settings["name"] == "Maimu-01"  # advertised name
    assert device._adv_name == "Maimu-01"


def test_splatting_a_name_setting_still_raises():
    """Documents why `settings=` exists -- the old call shape cannot work."""
    manager = _manager_with_board()
    with pytest.raises(TypeError, match="name"):
        manager.add_device_multi_pin(
            "stim_1", "Maimu", "b1", pins={}, name="Left stimulator", **{"name": "Maimu-01"}
        )


def test_kwargs_still_work():
    manager = _manager_with_board()
    manager.add_device_multi_pin("i2c_1", "GenericI2C", "b1", pins={}, i2c_address=0x40)
    assert manager.get_device("i2c_1").i2c_address == 0x40


def test_kwargs_merge_over_settings():
    manager = _manager_with_board()
    manager.add_device_multi_pin(
        "i2c_1", "GenericI2C", "b1", pins={}, settings={"i2c_address": 0x40}, i2c_address=0x53
    )
    assert manager.get_device("i2c_1").i2c_address == 0x53


def test_add_device_takes_a_settings_dict_too():
    manager = _manager_with_board()
    manager.add_device("led_1", "DigitalOutput", "b1", pin=5, settings={"inverted": True})
    assert manager.get_device("led_1")._config.settings["inverted"] is True
