"""
GLIDER Hardware Abstraction Layer (HAL)

Provides a uniform API for diverse hardware platforms, enabling the software
to treat hardware operations consistently across different boards and devices.
"""

# Importing the devices package registers package-local device types
# (e.g. StepperA4988) into DEVICE_REGISTRY as a side effect.
from glider.hal import devices  # noqa: F401
from glider.hal.base_board import BaseBoard, PinMode, PinType
from glider.hal.base_device import BaseDevice
from glider.hal.pin_manager import PinManager

__all__ = [
    "BaseBoard",
    "BaseDevice",
    "PinManager",
    "PinType",
    "PinMode",
]
