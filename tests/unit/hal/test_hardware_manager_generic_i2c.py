"""HardwareManager integration for the GenericI2C device type.

The manager's ``add_device_multi_pin`` is already generic (it forwards ``**kwargs``
as device settings and allocates no pins when ``pins == {}``); these tests lock in
that the new ``GenericI2C`` type round-trips through it.
"""

from glider.core.hardware_manager import HardwareManager
from glider.hal.base_device import GenericI2CDevice
from glider.hal.mock_board import MockBoard


def _manager_with_board(board_id="b1"):
    manager = HardwareManager()
    manager._boards[board_id] = MockBoard()
    return manager


def test_add_generic_i2c_device_stores_settings():
    manager = _manager_with_board()
    manager.add_device_multi_pin(
        "i2c_1",
        "GenericI2C",
        "b1",
        pins={},
        name="My Sensor",
        i2c_bus=1,
        i2c_address=0x40,
        register=0,
    )

    device = manager.get_device("i2c_1")
    assert isinstance(device, GenericI2CDevice)
    assert device.name == "My Sensor"
    assert device.i2c_bus == 1
    assert device.i2c_address == 0x40
    assert device.register == 0


def test_add_generic_i2c_device_allocates_no_pins():
    manager = _manager_with_board()
    manager.add_device_multi_pin("i2c_1", "GenericI2C", "b1", pins={}, i2c_address=0x53)

    device = manager.get_device("i2c_1")
    assert device.required_pins == []
    assert device.pins == {}
