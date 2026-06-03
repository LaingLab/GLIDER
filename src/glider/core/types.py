"""
Centralized Type Definitions for GLIDER.

This module provides enums and type definitions that replace stringly-typed
logic throughout the codebase, improving type safety and maintainability.
"""

from enum import Enum


class NodeType(Enum):
    """
    Enumeration of all node types in GLIDER.

    Use this enum instead of string comparisons for node type checking.
    """

    # Experiment nodes
    START_EXPERIMENT = "StartExperiment"
    END_EXPERIMENT = "EndExperiment"
    DELAY = "Delay"
    OUTPUT = "Output"
    INPUT = "Input"
    MOTOR_GOVERNOR = "MotorGovernor"
    CUSTOM_DEVICE = "CustomDevice"
    CUSTOM_DEVICE_ACTION = "CustomDeviceAction"

    # Control nodes
    LOOP = "Loop"
    WAIT_FOR_INPUT = "WaitForInput"

    # Graph function nodes (inline StartFunction -> EndFunction chains)
    START_FUNCTION = "StartFunction"
    END_FUNCTION = "EndFunction"
    FUNCTION_CALL = "FunctionCall"

    # Hardware nodes
    DIGITAL_WRITE = "DigitalWrite"
    DIGITAL_READ = "DigitalRead"
    ANALOG_READ = "AnalogRead"
    PWM_WRITE = "PWMWrite"
    DEVICE_ACTION = "DeviceAction"
    DEVICE_READ = "DeviceRead"

    # Logic nodes - Math
    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    DIVIDE = "Divide"
    MAP_RANGE = "MapRange"
    CLAMP = "Clamp"

    # Logic nodes - Comparison
    THRESHOLD = "Threshold"
    IN_RANGE = "InRange"

    # Logic nodes - Control
    PID = "PID"
    TOGGLE = "Toggle"
    SEQUENCE = "Sequence"
    TIMER = "Timer"

    # Interface nodes - Display
    LABEL = "Label"
    GAUGE = "Gauge"
    CHART = "Chart"
    LED_INDICATOR = "LEDIndicator"

    # Interface nodes - Input
    BUTTON = "Button"
    TOGGLE_SWITCH = "ToggleSwitch"
    SLIDER = "Slider"
    NUMERIC_INPUT = "NumericInput"

    @classmethod
    def from_string(cls, type_str: str) -> "NodeType":
        """
        Convert a string to NodeType enum.

        Args:
            type_str: The node type string (e.g., "StartExperiment", "Delay")

        Returns:
            Corresponding NodeType enum value

        Raises:
            ValueError: If the string doesn't match any known node type
        """
        # Normalize the string (remove spaces, handle common variations)
        normalized = type_str.replace(" ", "").replace("Node", "")

        for member in cls:
            if member.value.replace(" ", "") == normalized:
                return member
            # Also check against the normalized member name
            if member.name.replace("_", "") == normalized.upper():
                return member

        raise ValueError(f"Unknown node type: {type_str}")

    @classmethod
    def from_string_safe(cls, type_str: str) -> "NodeType | None":
        """
        Convert a string to NodeType enum, returning None if not found.

        Args:
            type_str: The node type string

        Returns:
            Corresponding NodeType enum value or None
        """
        try:
            return cls.from_string(type_str)
        except ValueError:
            return None


class DeviceType(Enum):
    """
    Enumeration of all device types in GLIDER.

    Use this enum instead of string comparisons for device type checking.
    """

    DIGITAL_OUTPUT = "DigitalOutput"
    DIGITAL_INPUT = "DigitalInput"
    ANALOG_INPUT = "AnalogInput"
    PWM_OUTPUT = "PWMOutput"
    SERVO = "Servo"
    STEPPER = "Stepper"

    @classmethod
    def from_string(cls, type_str: str) -> "DeviceType":
        """
        Convert a string to DeviceType enum.

        Args:
            type_str: The device type string (e.g., "DigitalOutput")

        Returns:
            Corresponding DeviceType enum value

        Raises:
            ValueError: If the string doesn't match any known device type
        """
        for member in cls:
            if member.value == type_str:
                return member

        raise ValueError(f"Unknown device type: {type_str}")

    @classmethod
    def from_string_safe(cls, type_str: str) -> "DeviceType | None":
        """
        Convert a string to DeviceType enum, returning None if not found.

        Args:
            type_str: The device type string

        Returns:
            Corresponding DeviceType enum value or None
        """
        try:
            return cls.from_string(type_str)
        except ValueError:
            return None

    @property
    def is_input(self) -> bool:
        """Whether this device type is an input device."""
        return self in (DeviceType.DIGITAL_INPUT, DeviceType.ANALOG_INPUT)

    @property
    def is_output(self) -> bool:
        """Whether this device type is an output device."""
        return self in (
            DeviceType.DIGITAL_OUTPUT,
            DeviceType.PWM_OUTPUT,
            DeviceType.SERVO,
            DeviceType.STEPPER,
        )

    @property
    def is_analog(self) -> bool:
        """Whether this device type uses analog signals."""
        return self in (DeviceType.ANALOG_INPUT, DeviceType.PWM_OUTPUT)
