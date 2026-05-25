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
        # Output-change callbacks. Concrete subclasses must call
        # `_notify_output_change(pin, pin_type, value)` after a successful
        # write_digital / write_analog / write_pwm / write_servo so listeners
        # (e.g. DeviceEventLogger) can record output events on the same event
        # stream as input edges.
        self._output_callbacks: list[Callable[[int, PinType, Any], None]] = []
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

    def set_port(self, port: str | None) -> None:
        """Set the connection port for the board.

        Args:
            port: New port string (e.g., 'COM3', '/dev/ttyUSB0'), or None.
                  Takes effect on the next connect() call.
        """
        self._port = port

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

    def register_output_callback(self, callback: Callable[[int, "PinType", Any], None]) -> None:
        """
        Register a callback fired after a successful output write.

        Concrete board subclasses invoke ``_notify_output_change`` from
        their ``write_digital`` / ``write_analog`` / ``write_pwm`` /
        ``write_servo`` implementations, so listeners can record output
        events without instrumenting every Device class.

        Args:
            callback: Function called with (pin, pin_type, value) after each
                successful output write.
        """
        if callback not in self._output_callbacks:
            self._output_callbacks.append(callback)

    def unregister_output_callback(self, callback: Callable[[int, "PinType", Any], None]) -> None:
        """Remove a previously registered output callback."""
        if callback in self._output_callbacks:
            self._output_callbacks.remove(callback)

    def _notify_output_change(self, pin: int, pin_type: "PinType", value: Any) -> None:
        """Notify all registered output callbacks. Errors are swallowed and logged."""
        for callback in self._output_callbacks:
            try:
                callback(pin, pin_type, value)
            except Exception:
                logger.exception("Error in output callback")

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
                logger.exception("Error in state change callback")

    def _notify_callbacks(self, pin: int, value: Any) -> None:
        """Notify all registered callbacks for a pin."""
        if pin in self._callbacks:
            for callback in self._callbacks[pin]:
                try:
                    callback(pin, value)
                except Exception:
                    logger.exception("Error in pin callback")

    def _notify_error(self, error: Exception) -> None:
        """Notify all registered error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception:
                logger.exception("Error in error callback")

    # Auto-reconnect tuning. Exponential backoff capped at 60s; give up
    # after MAX_RECONNECT_ATTEMPTS and transition to ERROR so the operator
    # can intervene rather than seeing infinite silent retries.
    MAX_RECONNECT_ATTEMPTS: int = 12

    async def _attempt_reconnect(self) -> None:
        """Background task for automatic reconnection.

        Previously this loop terminated after the *first* failed
        ``connect()`` attempt because ``connect()`` transitions the state
        to ``ERROR`` on failure, which breaks the loop's
        ``_state == RECONNECTING`` predicate. The follow-up
        ``start_reconnect()`` re-fired by ``connect()`` then short-circuited
        because ``_reconnect_task`` was non-None (the just-finished task).
        Net effect: ``auto_reconnect=True`` made exactly one retry attempt
        ever, with no UI surface.

        Fixed:
          * Loop predicate is ``self._auto_reconnect and not self.is_connected``
            (no longer depends on the state being RECONNECTING).
          * State is restored to RECONNECTING after every failed connect
            so the UI shows the actual situation.
          * Exponential backoff (5 → 10 → 20 → 40 → 60s capped) replaces
            fixed-interval polling.
          * Each failure fires error callbacks for UI visibility.
          * Bounded by ``MAX_RECONNECT_ATTEMPTS``; gives up and sets
            ERROR after the cap.
          * Task handle is cleared in ``finally`` so re-entry is allowed.
        """
        attempt = 0
        try:
            while self._auto_reconnect and not self.is_connected:
                if attempt >= self.MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "Auto-reconnect for board %s gave up after %d attempts",
                        self._id,
                        attempt,
                    )
                    self._set_state(BoardConnectionState.ERROR)
                    self._notify_error(
                        RuntimeError(f"Auto-reconnect failed after {attempt} attempts")
                    )
                    return

                self._set_state(BoardConnectionState.RECONNECTING)
                backoff = min(
                    self._reconnect_interval * (2 ** min(attempt, 4)),
                    60.0,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return

                attempt += 1
                logger.info("Auto-reconnect attempt %d for board %s", attempt, self._id)
                try:
                    if await self.connect():
                        logger.info(
                            "Auto-reconnect succeeded for board %s on attempt %d",
                            self._id,
                            attempt,
                        )
                        return
                except Exception as e:
                    logger.warning(
                        "Auto-reconnect attempt %d for board %s failed: %s",
                        attempt,
                        self._id,
                        e,
                    )
                    self._notify_error(e)
        finally:
            # Allow start_reconnect() to fire a new task on the next event.
            self._reconnect_task = None

    def start_reconnect(self) -> None:
        """Start the automatic reconnection process."""
        if self._auto_reconnect and self._reconnect_task is None:
            self._set_state(BoardConnectionState.RECONNECTING)
            from glider.core.async_utils import log_task_exception

            self._reconnect_task = asyncio.create_task(self._attempt_reconnect())
            self._reconnect_task.add_done_callback(log_task_exception)

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
