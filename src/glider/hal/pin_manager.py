"""
Pin Manager for tracking resource allocation.

Prevents pin conflicts by tracking which pins are claimed by devices
and ensuring no duplicate assignments.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard, PinType
    from glider.hal.base_device import BaseDevice


@dataclass
class PinAllocation:
    """Represents a pin allocation to a device."""

    pin: int
    device_id: str
    device_name: str
    pin_role: str  # e.g., "output", "trigger_pin", "echo_pin"


class PinConflictError(Exception):
    """Raised when attempting to allocate an already-claimed pin."""

    def __init__(self, pin: int, existing_device: str, new_device: str):
        self.pin = pin
        self.existing_device = existing_device
        self.new_device = new_device
        super().__init__(
            f"Pin {pin} is already claimed by '{existing_device}', "
            f"cannot assign to '{new_device}'"
        )


class InvalidPinError(Exception):
    """Raised when attempting to use a pin for an unsupported operation."""

    def __init__(self, pin: int, requested_type: str, supported_types: set[str]):
        self.pin = pin
        self.requested_type = requested_type
        self.supported_types = supported_types
        super().__init__(
            f"Pin {pin} does not support '{requested_type}'. " f"Supported types: {supported_types}"
        )


class PinManager:
    """
    Manages pin allocations across a board.

    Tracks which pins are claimed by devices and validates new
    allocations to prevent conflicts.
    """

    def __init__(self, board: "BaseBoard"):
        """
        Initialize the pin manager for a board.

        Args:
            board: The board to manage pins for
        """
        self._board = board
        self._allocations: dict[int, PinAllocation] = {}

    @property
    def board(self) -> "BaseBoard":
        """The board being managed."""
        return self._board

    @property
    def allocated_pins(self) -> set[int]:
        """Set of currently allocated pins."""
        return set(self._allocations.keys())

    @property
    def available_pins(self) -> set[int]:
        """Set of available (unallocated) pins."""
        all_pins = set(self._board.capabilities.pins.keys())
        return all_pins - self.allocated_pins

    def get_allocation(self, pin: int) -> PinAllocation | None:
        """Get the allocation info for a pin, if any."""
        return self._allocations.get(pin)

    def is_pin_available(self, pin: int) -> bool:
        """Check if a pin is available for allocation."""
        return pin not in self._allocations

    def get_pins_for_device(self, device_id: str) -> list[PinAllocation]:
        """Get all pin allocations for a specific device."""
        return [alloc for alloc in self._allocations.values() if alloc.device_id == device_id]

    def get_compatible_pins(self, pin_type: "PinType") -> list[int]:
        """
        Get list of pins compatible with a specific type.

        Used by the GUI to filter available pins in dropdown menus.

        Args:
            pin_type: The type of operation needed

        Returns:
            List of pin numbers that support the requested type
        """
        compatible = []
        for pin_num, capability in self._board.capabilities.pins.items():
            if pin_type in capability.supported_types:
                compatible.append(pin_num)
        return sorted(compatible)

    def get_available_compatible_pins(self, pin_type: "PinType") -> list[int]:
        """Get available pins that are compatible with a specific type."""
        compatible = self.get_compatible_pins(pin_type)
        return [p for p in compatible if p not in self._allocations]

    def validate_pin_type(self, pin: int, pin_type: "PinType") -> None:
        """
        Validate that a pin supports the requested type.

        Args:
            pin: Pin number
            pin_type: Requested operation type

        Raises:
            InvalidPinError: If the pin doesn't support the type
        """
        capabilities = self._board.capabilities.pins.get(pin)
        if capabilities is None:
            raise InvalidPinError(pin, str(pin_type), set())

        if pin_type not in capabilities.supported_types:
            supported = {str(t) for t in capabilities.supported_types}
            raise InvalidPinError(pin, str(pin_type), supported)

    def allocate_pin(
        self,
        pin: int,
        device: "BaseDevice",
        pin_role: str,
    ) -> PinAllocation:
        """
        Allocate a pin to a device.

        Args:
            pin: Pin number to allocate
            device: Device claiming the pin
            pin_role: Role of the pin for this device

        Returns:
            The created allocation

        Raises:
            PinConflictError: If the pin is already allocated
        """
        if pin in self._allocations:
            existing = self._allocations[pin]
            raise PinConflictError(pin, existing.device_name, device.name)

        allocation = PinAllocation(
            pin=pin,
            device_id=device.id,
            device_name=device.name,
            pin_role=pin_role,
        )
        self._allocations[pin] = allocation
        return allocation

    def allocate_device_pins(self, device: "BaseDevice") -> list[PinAllocation]:
        """
        Allocate all pins for a device.

        Args:
            device: Device to allocate pins for

        Returns:
            List of created allocations

        Raises:
            PinConflictError: If any pin is already allocated
        """
        allocations = []

        # First, validate all pins are available (don't partially allocate)
        for _pin_role, pin in device.pins.items():
            if pin in self._allocations:
                existing = self._allocations[pin]
                raise PinConflictError(pin, existing.device_name, device.name)

        # All pins are available, proceed with allocation
        for pin_role, pin in device.pins.items():
            allocation = PinAllocation(
                pin=pin,
                device_id=device.id,
                device_name=device.name,
                pin_role=pin_role,
            )
            self._allocations[pin] = allocation
            allocations.append(allocation)

        return allocations

    def release_pin(self, pin: int) -> PinAllocation | None:
        """
        Release a pin allocation.

        Args:
            pin: Pin number to release

        Returns:
            The released allocation, or None if pin wasn't allocated
        """
        return self._allocations.pop(pin, None)

    def release_device_pins(self, device_id: str) -> list[PinAllocation]:
        """
        Release all pins allocated to a device.

        Args:
            device_id: ID of the device

        Returns:
            List of released allocations
        """
        released = []
        pins_to_release = [
            pin for pin, alloc in self._allocations.items() if alloc.device_id == device_id
        ]
        for pin in pins_to_release:
            released.append(self._allocations.pop(pin))
        return released

    def clear_all(self) -> None:
        """Clear all pin allocations."""
        self._allocations.clear()

    def to_dict(self) -> dict[int, dict]:
        """Serialize allocations to dictionary."""
        return {
            pin: {
                "device_id": alloc.device_id,
                "device_name": alloc.device_name,
                "pin_role": alloc.pin_role,
            }
            for pin, alloc in self._allocations.items()
        }

    def get_allocation_summary(self) -> str:
        """Get a human-readable summary of allocations."""
        if not self._allocations:
            return "No pins allocated"

        lines = ["Pin Allocations:"]
        for pin in sorted(self._allocations.keys()):
            alloc = self._allocations[pin]
            lines.append(f"  Pin {pin}: {alloc.device_name} ({alloc.pin_role})")
        return "\n".join(lines)
