"""
Custom Device System - User-definable devices with configurable pins.

Custom devices allow users to define arbitrary hardware devices with
configurable pins. Flow logic is created via the node graph.
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard

logger = logging.getLogger(__name__)


class PinType(Enum):
    """Types of pins supported by custom devices."""

    DIGITAL_OUTPUT = "digital_output"
    DIGITAL_INPUT = "digital_input"
    ANALOG_INPUT = "analog_input"
    ANALOG_OUTPUT = "analog_output"
    PWM = "pwm"


@dataclass
class PinDefinition:
    """Definition of a pin for a custom device."""

    name: str
    pin_type: PinType
    pin_number: int | None = None  # Physical pin number on the board
    default_value: Any = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pin_type": self.pin_type.value,
            "pin_number": self.pin_number,
            "default_value": self.default_value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PinDefinition":
        return cls(
            name=data["name"],
            pin_type=PinType(data["pin_type"]),
            pin_number=data.get("pin_number"),
            default_value=data.get("default_value"),
            description=data.get("description", ""),
        )


@dataclass
class CustomDeviceDefinition:
    """Complete definition of a custom device type."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Custom Device"
    description: str = ""
    pins: list[PinDefinition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pins": [p.to_dict() for p in self.pins],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CustomDeviceDefinition":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Custom Device"),
            description=data.get("description", ""),
            pins=[PinDefinition.from_dict(p) for p in data.get("pins", [])],
        )

    def get_pin(self, name: str) -> PinDefinition | None:
        """Get a pin definition by name."""
        for pin in self.pins:
            if pin.name == name:
                return pin
        return None


class CustomDeviceRunner:
    """
    Runtime executor for custom devices.

    Takes a CustomDeviceDefinition and manages hardware pins
    through a BaseBoard.
    """

    def __init__(
        self,
        definition: CustomDeviceDefinition,
        board: "BaseBoard",
        pin_assignments: dict[str, int] | None = None,
        name: str | None = None,
    ):
        """
        Initialize the custom device runner.

        Args:
            definition: The device definition
            board: The board this device is connected to
            pin_assignments: Optional mapping of pin names to actual pin numbers.
                            If not provided, uses pin_number from each PinDefinition.
            name: Optional instance name
        """
        self._id = str(uuid.uuid4())
        self._definition = definition
        self._board = board
        self._name = name or definition.name
        self._initialized = False
        self._pin_states: dict[str, Any] = {}
        self._last_read_values: dict[str, Any] = {}

        # Build pin assignments from definitions if not provided
        if pin_assignments is not None:
            self._pin_assignments = pin_assignments
        else:
            self._pin_assignments = {}
            for pin_def in definition.pins:
                if pin_def.pin_number is not None:
                    self._pin_assignments[pin_def.name] = pin_def.pin_number

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> CustomDeviceDefinition:
        return self._definition

    @property
    def device_type(self) -> str:
        return f"Custom:{self._definition.name}"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def pins(self) -> list[PinDefinition]:
        """List of pin definitions."""
        return self._definition.pins

    async def initialize(self) -> None:
        """Initialize all pins to their configured modes and defaults."""
        from glider.hal.base_board import PinMode
        from glider.hal.base_board import PinType as BoardPinType

        for pin_def in self._definition.pins:
            pin_num = self._pin_assignments.get(pin_def.name)
            if pin_num is None:
                logger.warning(f"Pin '{pin_def.name}' not assigned")
                continue

            # Set pin mode based on type
            if pin_def.pin_type == PinType.DIGITAL_OUTPUT:
                await self._board.set_pin_mode(pin_num, PinMode.OUTPUT, BoardPinType.DIGITAL)
                default_val = bool(pin_def.default_value) if pin_def.default_value else False
                await self._board.write_digital(pin_num, default_val)
                self._pin_states[pin_def.name] = default_val

            elif pin_def.pin_type == PinType.DIGITAL_INPUT:
                await self._board.set_pin_mode(pin_num, PinMode.INPUT, BoardPinType.DIGITAL)
                self._pin_states[pin_def.name] = None

            elif pin_def.pin_type == PinType.ANALOG_INPUT:
                await self._board.set_pin_mode(pin_num, PinMode.INPUT, BoardPinType.ANALOG)
                self._pin_states[pin_def.name] = None

            elif pin_def.pin_type == PinType.ANALOG_OUTPUT:
                await self._board.set_pin_mode(pin_num, PinMode.OUTPUT, BoardPinType.ANALOG)
                default_val = int(pin_def.default_value) if pin_def.default_value else 0
                await self._board.write_analog(pin_num, default_val)
                self._pin_states[pin_def.name] = default_val

            elif pin_def.pin_type == PinType.PWM:
                await self._board.set_pin_mode(pin_num, PinMode.OUTPUT, BoardPinType.PWM)
                default_val = int(pin_def.default_value) if pin_def.default_value else 0
                await self._board.write_analog(pin_num, default_val)
                self._pin_states[pin_def.name] = default_val

        self._initialized = True
        logger.info(f"Custom device '{self._name}' initialized")

    async def shutdown(self) -> None:
        """Shutdown the device safely."""
        # Set all outputs to low/0
        for pin_def in self._definition.pins:
            pin_num = self._pin_assignments.get(pin_def.name)
            if pin_num is None:
                continue

            if pin_def.pin_type == PinType.DIGITAL_OUTPUT:
                await self._board.write_digital(pin_num, False)
            elif pin_def.pin_type in (PinType.ANALOG_OUTPUT, PinType.PWM):
                await self._board.write_analog(pin_num, 0)

        self._initialized = False
        logger.info(f"Custom device '{self._name}' shutdown")

    async def write_pin(self, pin_name: str, value: Any) -> None:
        """
        Write a value to a pin.

        Args:
            pin_name: Name of the pin
            value: Value to write (bool for digital, int for analog/PWM)
        """
        pin_num = self._pin_assignments.get(pin_name)
        if pin_num is None:
            raise ValueError(f"Pin '{pin_name}' not assigned")

        pin_def = self._definition.get_pin(pin_name)
        if pin_def is None:
            raise ValueError(f"Pin '{pin_name}' not defined")

        if pin_def.pin_type == PinType.DIGITAL_OUTPUT:
            await self._board.write_digital(pin_num, bool(value))
            self._pin_states[pin_name] = bool(value)
        elif pin_def.pin_type in (PinType.ANALOG_OUTPUT, PinType.PWM):
            await self._board.write_analog(pin_num, int(value))
            self._pin_states[pin_name] = int(value)
        else:
            raise ValueError(f"Cannot write to input pin '{pin_name}'")

        logger.debug(f"Wrote {value} to pin '{pin_name}'")

    async def read_pin(self, pin_name: str) -> Any:
        """
        Read a value from a pin.

        Args:
            pin_name: Name of the pin

        Returns:
            The read value (bool for digital, int for analog)
        """
        pin_num = self._pin_assignments.get(pin_name)
        if pin_num is None:
            raise ValueError(f"Pin '{pin_name}' not assigned")

        pin_def = self._definition.get_pin(pin_name)
        if pin_def is None:
            raise ValueError(f"Pin '{pin_name}' not defined")

        if pin_def.pin_type == PinType.DIGITAL_INPUT:
            value = await self._board.read_digital(pin_num)
        elif pin_def.pin_type == PinType.ANALOG_INPUT:
            value = await self._board.read_analog(pin_num)
        else:
            # For outputs, return the last written state
            value = self._pin_states.get(pin_name)

        self._last_read_values[pin_name] = value
        logger.debug(f"Read {value} from pin '{pin_name}'")
        return value

    def get_pin_state(self, pin_name: str) -> Any:
        """Get the last known state of a pin."""
        return self._pin_states.get(pin_name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize device instance configuration."""
        return {
            "id": self._id,
            "definition_id": self._definition.id,
            "name": self._name,
            "pin_assignments": self._pin_assignments,
        }
