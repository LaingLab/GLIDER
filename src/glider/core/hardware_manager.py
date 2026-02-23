"""
Hardware Manager - Manages the lifecycle of hardware connections.

Maintains the registry of active Board instances and handles
connection/disconnection, device initialization, and error recovery.
"""

import logging
from typing import TYPE_CHECKING, Callable, Optional

from glider.hal.base_board import BaseBoard, BoardConnectionState
from glider.hal.base_device import BaseDevice, create_device_from_dict
from glider.hal.pin_manager import PinConflictError, PinManager

if TYPE_CHECKING:
    from glider.core.experiment_session import BoardConfig, DeviceConfig

logger = logging.getLogger(__name__)


class HardwareError(Exception):
    """Base exception for hardware errors."""

    pass


class BoardNotFoundError(HardwareError):
    """Raised when a referenced board is not found."""

    pass


class DeviceNotFoundError(HardwareError):
    """Raised when a referenced device is not found."""

    pass


class HardwareManager:
    """
    Manages the lifecycle of hardware connections.

    This sub-controller handles:
    - Board registration and driver loading
    - Connection management
    - Device initialization
    - Pin allocation tracking
    - Error recovery and reconnection
    """

    # Registry of available board drivers (intentionally class-level: shared across all
    # HardwareManager instances so that driver registration is global and persistent)
    _driver_registry: dict[str, type[BaseBoard]] = {}

    def __init__(self):
        """Initialize the hardware manager."""
        self._boards: dict[str, BaseBoard] = {}
        self._devices: dict[str, BaseDevice] = {}
        self._pin_managers: dict[str, PinManager] = {}
        self._error_callbacks: list[Callable[[str, Exception], None]] = []
        self._connection_callbacks: list[Callable[[str, BoardConnectionState], None]] = []
        self._initialized = False

    @classmethod
    def register_driver(cls, name: str, driver_class: type[BaseBoard]) -> None:
        """
        Register a board driver.

        Args:
            name: Driver identifier (e.g., "arduino", "raspberry_pi")
            driver_class: Board class implementing BaseBoard
        """
        cls._driver_registry[name] = driver_class
        logger.debug(f"Registered driver: {name}")

    @classmethod
    def get_available_drivers(cls) -> list[str]:
        """Get list of available driver names."""
        return list(cls._driver_registry.keys())

    @classmethod
    def get_driver_class(cls, name: str) -> Optional[type[BaseBoard]]:
        """Get a driver class by name."""
        return cls._driver_registry.get(name)

    @property
    def boards(self) -> dict[str, BaseBoard]:
        """Dictionary of active board instances."""
        return self._boards.copy()

    @property
    def devices(self) -> dict[str, BaseDevice]:
        """Dictionary of active device instances."""
        return self._devices.copy()

    def get_board(self, board_id: str) -> Optional[BaseBoard]:
        """Get a board by ID."""
        return self._boards.get(board_id)

    def get_device(self, device_id: str) -> Optional[BaseDevice]:
        """Get a device by ID."""
        return self._devices.get(device_id)

    def get_pin_manager(self, board_id: str) -> Optional[PinManager]:
        """Get the pin manager for a board."""
        return self._pin_managers.get(board_id)

    def on_error(self, callback: Callable[[str, Exception], None]) -> None:
        """Register callback for hardware errors."""
        self._error_callbacks.append(callback)

    def on_connection_change(self, callback: Callable[[str, BoardConnectionState], None]) -> None:
        """Register callback for connection state changes."""
        self._connection_callbacks.append(callback)

    def _notify_error(self, source: str, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(source, error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def _notify_connection_change(self, board_id: str, state: BoardConnectionState) -> None:
        """Notify connection state change callbacks."""
        for callback in self._connection_callbacks:
            try:
                callback(board_id, state)
            except Exception as e:
                logger.error(f"Connection callback failed: {e}")

    async def create_board(self, config: "BoardConfig") -> BaseBoard:
        """
        Create a board instance from configuration.

        Args:
            config: Board configuration

        Returns:
            Created board instance
        """
        driver_class = self._driver_registry.get(config.driver_type)
        if driver_class is None:
            raise HardwareError(f"Unknown driver type: {config.driver_type}")

        # Create board instance
        board = driver_class(
            port=config.port,
            auto_reconnect=config.auto_reconnect,
        )
        board._id = config.id

        # Apply board type if applicable
        if hasattr(board, "_board_type") and config.board_type:
            board._board_type = config.board_type
            if hasattr(board, "BOARD_CONFIGS"):
                board._board_config = board.BOARD_CONFIGS.get(
                    config.board_type, board._board_config
                )

        # Register error callback
        board.register_error_callback(lambda e: self._notify_error(config.id, e))

        # Register state change callback to propagate to listeners
        board.register_state_callback(
            lambda state, bid=config.id: self._notify_connection_change(bid, state)
        )

        # Store board and create pin manager
        self._boards[config.id] = board
        self._pin_managers[config.id] = PinManager(board)

        logger.info(f"Created board: {board.name} (ID: {config.id})")
        return board

    async def connect_board(self, board_id: str) -> bool:
        """
        Connect to a board.

        Args:
            board_id: Board ID

        Returns:
            True if connected successfully
        """
        board = self._boards.get(board_id)
        if board is None:
            raise BoardNotFoundError(f"Board not found: {board_id}")

        logger.info(f"Connecting to board: {board_id}")
        success = await board.connect()

        self._notify_connection_change(board_id, board.state)
        return success

    async def disconnect_board(self, board_id: str) -> None:
        """
        Disconnect from a board.

        Args:
            board_id: Board ID
        """
        board = self._boards.get(board_id)
        if board is None:
            return

        logger.info(f"Disconnecting board: {board_id}")

        # Shutdown all devices on this board first
        devices_to_shutdown = [d for d in self._devices.values() if d.board.id == board_id]
        for device in devices_to_shutdown:
            try:
                await device.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down device {device.id}: {e}")

        await board.disconnect()
        self._notify_connection_change(board_id, board.state)

    async def remove_board(self, board_id: str) -> None:
        """
        Remove a board and its devices.

        Args:
            board_id: Board ID
        """
        await self.disconnect_board(board_id)

        # Remove all devices on this board
        device_ids_to_remove = [d.id for d in self._devices.values() if d.board.id == board_id]
        for device_id in device_ids_to_remove:
            del self._devices[device_id]

        # Remove pin manager and board
        self._pin_managers.pop(board_id, None)
        self._boards.pop(board_id, None)

        logger.info(f"Removed board: {board_id}")

    async def create_device(self, config: "DeviceConfig") -> BaseDevice:
        """
        Create a device instance from configuration.

        Args:
            config: Device configuration

        Returns:
            Created device instance
        """
        board = self._boards.get(config.board_id)
        if board is None:
            raise BoardNotFoundError(f"Board not found: {config.board_id}")

        pin_manager = self._pin_managers.get(config.board_id)

        # Create device data for factory
        device_data = {
            "id": config.id,
            "device_type": config.device_type,
            "name": config.name,
            "board_id": config.board_id,
            "config": {
                "pins": config.pins,
                "settings": config.settings,
            },
        }

        # Create device instance
        device = create_device_from_dict(device_data, board)

        # Validate configuration
        errors = device.validate_config()
        if errors:
            raise HardwareError(f"Device configuration errors: {errors}")

        # Allocate pins
        if pin_manager:
            try:
                pin_manager.allocate_device_pins(device)
            except PinConflictError as e:
                raise HardwareError(str(e)) from e

        self._devices[config.id] = device
        logger.info(f"Created device: {device.name} (ID: {config.id})")
        return device

    async def initialize_device(self, device_id: str) -> None:
        """
        Initialize a device.

        Args:
            device_id: Device ID
        """
        device = self._devices.get(device_id)
        if device is None:
            raise DeviceNotFoundError(f"Device not found: {device_id}")

        if not device.board.is_connected:
            raise HardwareError(f"Board not connected: {device.board.id}")

        logger.info(f"Initializing device: {device_id}")
        await device.initialize()

    async def shutdown_device(self, device_id: str) -> None:
        """
        Shutdown a device safely.

        Args:
            device_id: Device ID
        """
        device = self._devices.get(device_id)
        if device is None:
            return

        logger.info(f"Shutting down device: {device_id}")
        try:
            await device.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down device {device_id}: {e}")

    async def remove_device(self, device_id: str) -> None:
        """
        Remove a device.

        Args:
            device_id: Device ID
        """
        device = self._devices.get(device_id)
        if device is None:
            return

        # Shutdown device
        await self.shutdown_device(device_id)

        # Release pins
        pin_manager = self._pin_managers.get(device.board.id)
        if pin_manager:
            pin_manager.release_device_pins(device_id)

        del self._devices[device_id]
        logger.info(f"Removed device: {device_id}")

    def add_board(
        self, board_id: str, driver_type: str, port: Optional[str] = None, **kwargs
    ) -> None:
        """
        Add a board to the manager (simplified API).

        Args:
            board_id: Unique board identifier
            driver_type: Driver type (e.g., "telemetrix", "pigpio")
            port: Serial port (for Arduino)
            **kwargs: Additional board settings
        """
        driver_class = self._driver_registry.get(driver_type)
        if driver_class is None:
            # Try alternate names
            alt_names = {"telemetrix": "arduino", "pigpio": "raspberry_pi"}
            driver_class = self._driver_registry.get(alt_names.get(driver_type, ""))

        if driver_class is None:
            raise HardwareError(
                f"Unknown driver type: {driver_type}. Available: {list(self._driver_registry.keys())}"
            )

        # Create board instance
        board = driver_class(port=port, **kwargs)
        board._id = board_id

        # Register error callback
        board.register_error_callback(lambda e: self._notify_error(board_id, e))

        # Store board and create pin manager
        self._boards[board_id] = board
        self._pin_managers[board_id] = PinManager(board)

        logger.info(f"Added board: {board_id} (type: {driver_type})")

    def add_device(
        self,
        device_id: str,
        device_type: str,
        board_id: str,
        pin: int,
        name: Optional[str] = None,
        pin_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Add a device to the manager (simplified API).

        Args:
            device_id: Unique device identifier
            device_type: Device type (e.g., "DigitalOutput", "AnalogInput")
            board_id: ID of the board this device is on
            pin: Pin number
            name: Optional display name
            pin_name: Name for the pin (e.g., "output", "input", "signal")
            **kwargs: Additional device settings
        """
        board = self._boards.get(board_id)
        if board is None:
            raise BoardNotFoundError(f"Board not found: {board_id}")

        # Determine pin name based on device type if not provided
        if pin_name is None:
            pin_name_map = {
                "DigitalOutput": "output",
                "DigitalInput": "input",
                "AnalogInput": "input",
                "PWMOutput": "output",
                "Servo": "signal",
            }
            pin_name = pin_name_map.get(device_type, "pin")

        # Create device data for factory
        device_data = {
            "id": device_id,
            "device_type": device_type,
            "name": name or device_id,
            "board_id": board_id,
            "config": {
                "pins": {pin_name: pin},
                "settings": kwargs,
            },
        }

        # Create device instance
        device = create_device_from_dict(device_data, board)

        # Store pins list for the tree view
        device._pins = [pin]

        # Allocate pins
        pin_manager = self._pin_managers.get(board_id)
        if pin_manager:
            try:
                pin_manager.allocate_device_pins(device)
            except PinConflictError as e:
                raise HardwareError(str(e)) from e

        self._devices[device_id] = device
        logger.info(f"Added device: {device_id} (type: {device_type}, pin: {pin})")

    def add_device_multi_pin(
        self,
        device_id: str,
        device_type: str,
        board_id: str,
        pins: dict[str, int],
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Add a device with multiple pins to the manager.

        Args:
            device_id: Unique device identifier
            device_type: Device type (e.g., "MotorGovernor")
            board_id: ID of the board this device is on
            pins: Dictionary mapping pin names to pin numbers
            name: Optional display name
            **kwargs: Additional device settings
        """
        board = self._boards.get(board_id)
        if board is None:
            raise BoardNotFoundError(f"Board not found: {board_id}")

        # Create device data for factory
        device_data = {
            "id": device_id,
            "device_type": device_type,
            "name": name or device_id,
            "board_id": board_id,
            "config": {
                "pins": pins,
                "settings": kwargs,
            },
        }

        # Create device instance
        device = create_device_from_dict(device_data, board)

        # Store pins list for the tree view
        device._pins = list(pins.values())

        # Allocate pins
        pin_manager = self._pin_managers.get(board_id)
        if pin_manager:
            try:
                pin_manager.allocate_device_pins(device)
            except PinConflictError as e:
                raise HardwareError(str(e)) from e

        self._devices[device_id] = device
        pins_str = ", ".join(f"{k}={v}" for k, v in pins.items())
        logger.info(f"Added device: {device_id} (type: {device_type}, pins: {pins_str})")

    def clear(self) -> None:
        """Clear all boards and devices."""
        self._devices.clear()
        self._boards.clear()
        self._pin_managers.clear()
        logger.info("Cleared all hardware")

    async def connect_all(self) -> dict[str, bool]:
        """
        Connect to all boards.

        Returns:
            Dictionary of board_id -> success
        """
        results = {}
        for board_id in self._boards:
            try:
                results[board_id] = await self.connect_board(board_id)
            except Exception as e:
                logger.error(f"Failed to connect board {board_id}: {e}")
                results[board_id] = False
        return results

    async def initialize_all_devices(self) -> dict[str, bool]:
        """
        Initialize all devices.

        Returns:
            Dictionary of device_id -> success
        """
        results = {}
        for device_id in self._devices:
            try:
                await self.initialize_device(device_id)
                results[device_id] = True
            except Exception as e:
                logger.error(f"Failed to initialize device {device_id}: {e}")
                results[device_id] = False
        return results

    async def emergency_stop(self) -> None:
        """
        Trigger emergency stop on all hardware.

        Sets all outputs to safe state.
        """
        logger.warning("EMERGENCY STOP triggered!")

        # Shutdown all devices
        for device in self._devices.values():
            try:
                await device.shutdown()
            except Exception as e:
                logger.error(f"Emergency shutdown error for device {device.id}: {e}")

        # Emergency stop all boards
        for board in self._boards.values():
            try:
                await board.emergency_stop()
            except Exception as e:
                logger.error(f"Emergency stop error for board {board.id}: {e}")

    async def disconnect_all(self) -> None:
        """Disconnect from all boards."""
        for board_id in list(self._boards.keys()):
            await self.disconnect_board(board_id)

    async def shutdown(self) -> None:
        """Shutdown the hardware manager."""
        logger.info("Shutting down hardware manager")
        await self.emergency_stop()
        await self.disconnect_all()
        self._boards.clear()
        self._devices.clear()
        self._pin_managers.clear()


# Register built-in drivers
def _register_builtin_drivers():
    """Register the built-in board drivers."""
    try:
        from glider.hal.boards.telemetrix_board import TelemetrixBoard

        HardwareManager.register_driver("arduino", TelemetrixBoard)
    except ImportError:
        logger.warning("TelemetrixBoard driver not available")

    try:
        from glider.hal.boards.pi_gpio_board import PiGPIOBoard

        HardwareManager.register_driver("raspberry_pi", PiGPIOBoard)
    except ImportError:
        logger.warning("PiGPIOBoard driver not available")


_register_builtin_drivers()
