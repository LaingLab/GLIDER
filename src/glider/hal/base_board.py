"""
Abstract Base Class for hardware boards.

Any hardware plugin must implement this interface to integrate with GLIDER.
This polymorphism allows the Core to iterate over a list of BaseBoard objects
and perform operations without knowing the specific hardware implementation.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class PinType(Enum):
    """Types of pin operations supported by the HAL."""

    DIGITAL = auto()
    ANALOG = auto()
    PWM = auto()
    I2C = auto()
    SPI = auto()
    SERVO = auto()


class PinMode(Enum):
    """Pin modes for configuration."""

    INPUT = auto()
    OUTPUT = auto()
    INPUT_PULLUP = auto()
    INPUT_PULLDOWN = auto()


@dataclass
class PinCapability:
    """Describes the capabilities of a specific pin."""

    pin: int
    supported_types: set[PinType] = field(default_factory=set)
    max_value: int = 1  # For analog/PWM, max value (e.g., 255 for 8-bit PWM)
    description: str = ""


@dataclass
class BoardCapabilities:
    """Describes the overall capabilities of a board."""

    name: str
    pins: dict[int, PinCapability] = field(default_factory=dict)
    supports_analog: bool = False
    analog_resolution: int = 10  # bits
    pwm_resolution: int = 8  # bits
    pwm_frequency: int = 490  # Hz
    i2c_buses: list[int] = field(default_factory=list)
    spi_buses: list[int] = field(default_factory=list)


class BoardConnectionState(Enum):
    """Connection states for the board."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()
    RECONNECTING = auto()


class BaseBoard(ABC):
    """
    Abstract Base Class defining the contract for hardware board drivers.

    All hardware plugins must inherit from this class and implement
    the abstract methods. The async design ensures non-blocking operation
    compatible with GLIDER's asyncio-based event loop.
    """

    def __init__(self, port: str | None = None, auto_reconnect: bool = False):
        """
        Initialize the board interface.

        Args:
            port: Connection port (e.g., COM3, /dev/ttyUSB0)
            auto_reconnect: Whether to automatically attempt reconnection on failure
        """
        self._id = str(uuid.uuid4())
        self._port = port
        self._auto_reconnect = auto_reconnect
        self._state = BoardConnectionState.DISCONNECTED
        self._callbacks: dict[int, list[Callable]] = {}
        self._error_callbacks: list[Callable] = []
        self._state_callbacks: list[Callable[[BoardConnectionState], None]] = []
        self._reconnect_task: asyncio.Task | None = None
        self._reconnect_interval = 5.0  # seconds (increased to reduce spam)
        self._i2c_lock = asyncio.Lock()  # Shared lock for I2C operations

    @property
    def i2c_lock(self) -> asyncio.Lock:
        """Shared lock for I2C operations on this board."""
        return self._i2c_lock

    @property
    def id(self) -> str:
        """Unique identifier for this board instance."""
        return self._id

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the board type."""
        ...

    @property
    @abstractmethod
    def board_type(self) -> str:
        """Driver/board type identifier (e.g., 'telemetrix', 'pigpio')."""
        ...

    @property
    def port(self) -> str | None:
        """Connection port for the board."""
        return self._port

    @property
    def state(self) -> BoardConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether the board is currently connected."""
        return self._state == BoardConnectionState.CONNECTED

    @property
    @abstractmethod
    def capabilities(self) -> BoardCapabilities:
        """
        Returns the capabilities map for this board.

        Used by the GUI to filter available pins in dropdown menus,
        preventing invalid configurations.
        """
        ...

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish the physical connection to the board.

        Returns:
            True if connection successful, False otherwise
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly shut down the connection."""
        ...

    @abstractmethod
    async def set_pin_mode(
        self, pin: int, mode: PinMode, pin_type: PinType = PinType.DIGITAL
    ) -> None:
        """
        Configure a pin's mode.

        Args:
            pin: Pin number
            mode: Input or Output mode
            pin_type: Type of pin operation
        """
        ...

    @abstractmethod
    async def write_digital(self, pin: int, value: bool) -> None:
        """
        Write a digital value to a pin.

        Args:
            pin: Pin number
            value: True for HIGH, False for LOW
        """
        ...

    @abstractmethod
    async def read_digital(self, pin: int) -> bool:
        """
        Read a digital value from a pin.

        Args:
            pin: Pin number

        Returns:
            True for HIGH, False for LOW
        """
        ...

    @abstractmethod
    async def write_analog(self, pin: int, value: int) -> None:
        """
        Write an analog (PWM) value to a pin.

        Args:
            pin: Pin number
            value: PWM value (0 to max based on resolution)
        """
        ...

    @abstractmethod
    async def read_analog(self, pin: int) -> int:
        """
        Read an analog value from a pin.

        Args:
            pin: Pin number

        Returns:
            Analog value (0 to max based on resolution)
        """
        ...

    async def write_pin(self, pin: int, pin_type: PinType, value: Any) -> None:
        """
        Generic write method that dispatches to specific implementations.

        Args:
            pin: Pin number
            pin_type: Type of write operation
            value: Value to write
        """
        if pin_type == PinType.DIGITAL:
            await self.write_digital(pin, bool(value))
        elif pin_type in (PinType.ANALOG, PinType.PWM):
            await self.write_analog(pin, int(value))
        elif pin_type == PinType.SERVO:
            await self.write_servo(pin, int(value))
        else:
            raise ValueError(f"Unsupported pin type for write: {pin_type}")

    async def read_pin(self, pin: int, pin_type: PinType) -> Any:
        """
        Generic read method that dispatches to specific implementations.

        Args:
            pin: Pin number
            pin_type: Type of read operation

        Returns:
            Value read from pin
        """
        if pin_type == PinType.DIGITAL:
            return await self.read_digital(pin)
        elif pin_type == PinType.ANALOG:
            return await self.read_analog(pin)
        else:
            raise ValueError(f"Unsupported pin type for read: {pin_type}")

    async def write_servo(self, pin: int, angle: int) -> None:
        """
        Write a servo angle. Override in subclass if supported.

        Args:
            pin: Pin number
            angle: Servo angle (0-180)
        """
        raise NotImplementedError("Servo not supported on this board")

    def register_callback(self, pin: int, callback: Callable[[int, Any], None]) -> None:
        """
        Register a callback for pin value changes.

        Used by Telemetrix-style boards that push data on changes.

        Args:
            pin: Pin number to watch
            callback: Function to call with (pin, value) when data arrives
        """
        if pin not in self._callbacks:
            self._callbacks[pin] = []
        self._callbacks[pin].append(callback)

    def unregister_callback(self, pin: int, callback: Callable[[int, Any], None]) -> None:
        """Remove a registered callback."""
        if pin in self._callbacks and callback in self._callbacks[pin]:
            self._callbacks[pin].remove(callback)

    def register_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Register a callback for error events."""
        self._error_callbacks.append(callback)

    def register_state_callback(self, callback: Callable[[BoardConnectionState], None]) -> None:
        """Register a callback for state change events."""
        self._state_callbacks.append(callback)

    def _set_state(self, new_state: BoardConnectionState) -> None:
        """Set the connection state and notify callbacks."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            logger.debug(f"Board {self._id} state: {old_state.name} -> {new_state.name}")
            self._notify_state_change(new_state)

    def _notify_state_change(self, state: BoardConnectionState) -> None:
        """Notify all registered state change callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception:
                pass  # Don't let callback errors propagate

    def _notify_callbacks(self, pin: int, value: Any) -> None:
        """Notify all registered callbacks for a pin."""
        if pin in self._callbacks:
            for callback in self._callbacks[pin]:
                try:
                    callback(pin, value)
                except Exception:
                    pass  # Don't let callback errors propagate

    def _notify_error(self, error: Exception) -> None:
        """Notify all registered error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception:
                pass

    async def _attempt_reconnect(self) -> None:
        """Background task for automatic reconnection."""
        while self._auto_reconnect and self._state == BoardConnectionState.RECONNECTING:
            await asyncio.sleep(self._reconnect_interval)
            try:
                if await self.connect():
                    break
            except Exception:
                pass  # Continue trying

    def start_reconnect(self) -> None:
        """Start the automatic reconnection process."""
        if self._auto_reconnect and self._reconnect_task is None:
            self._set_state(BoardConnectionState.RECONNECTING)
            self._reconnect_task = asyncio.create_task(self._attempt_reconnect())

    def stop_reconnect(self) -> None:
        """Stop the automatic reconnection process."""
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            self._reconnect_task = None

    async def emergency_stop(self) -> None:
        """
        Emergency stop - set all outputs to safe state.

        Override in subclass for board-specific behavior.
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize board configuration to dictionary."""
        return {
            "id": self._id,
            "name": self.name,
            "port": self._port,
            "auto_reconnect": self._auto_reconnect,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseBoard":
        """Create board instance from dictionary configuration."""
        instance = cls(port=data.get("port"), auto_reconnect=data.get("auto_reconnect", False))
        instance._id = data.get("id", instance._id)
        return instance
