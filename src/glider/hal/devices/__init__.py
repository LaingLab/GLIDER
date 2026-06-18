"""
Device implementations for the GLIDER HAL.

Devices are higher-level abstractions that wrap board operations
into semantic actions.
"""

from glider.hal.base_device import (
    DEVICE_REGISTRY,
    ADS1115Device,
    AnalogInputDevice,
    BaseDevice,
    DeviceConfig,
    DigitalInputDevice,
    DigitalOutputDevice,
    GenericI2CDevice,
    MotorGovernorDevice,
    PWMOutputDevice,
    ServoDevice,
    create_device_from_dict,
)

__all__ = [
    "BaseDevice",
    "DeviceConfig",
    "DigitalOutputDevice",
    "DigitalInputDevice",
    "AnalogInputDevice",
    "PWMOutputDevice",
    "ServoDevice",
    "MotorGovernorDevice",
    "ADS1115Device",
    "GenericI2CDevice",
    "DEVICE_REGISTRY",
    "create_device_from_dict",
]
