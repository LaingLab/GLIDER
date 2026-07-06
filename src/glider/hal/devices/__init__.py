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
from glider.hal.devices.stepper_a4988 import StepperA4988Device

# Built-in devices that live in this package (rather than base_device.py)
# register here; this module is imported by glider.hal.__init__ so the
# registry is populated whenever any glider.hal.* module is imported.
DEVICE_REGISTRY["StepperA4988"] = StepperA4988Device

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
    "StepperA4988Device",
    "DEVICE_REGISTRY",
    "create_device_from_dict",
]
