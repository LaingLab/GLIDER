"""
Abstract Base Class for hardware devices.

Devices represent higher-level components attached to boards
(e.g., "Stepper Motor", "Temperature Sensor"). They wrap BaseBoard
methods into semantic actions.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard


@dataclass
class DeviceConfig:
    """Configuration for a device including pin assignments."""

    pins: dict[str, int] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)


class BaseDevice(ABC):
    """
    Abstract Base Class for hardware devices.

    Devices are logical entities that abstract pin numbers into functionality.
    They wrap the BaseBoard methods into semantic actions.
    """

    def __init__(
        self,
        board: "BaseBoard",
        config: DeviceConfig,
        name: Optional[str] = None,
    ):
        """
        Initialize the device.

        Args:
            board: The board this device is connected to
            config: Device configuration including pin assignments
            name: Optional human-readable name for this device instance
        """
        self._id = str(uuid.uuid4())
        self._board = board
        self._config = config
        self._name = name or self.device_type
        self._initialized = False
        self._enabled = True

    @property
    def id(self) -> str:
        """Unique identifier for this device instance."""
        return self._id

    @property
    @abstractmethod
    def device_type(self) -> str:
        """Type name for this device (e.g., 'LED', 'DHT22')."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for this device instance."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def board(self) -> "BaseBoard":
        """The board this device is connected to."""
        return self._board

    @property
    def config(self) -> DeviceConfig:
        """Device configuration including pin assignments."""
        return self._config

    @property
    def pins(self) -> dict[str, int]:
        """Pin assignments for this device."""
        return self._config.pins

    @property
    def is_initialized(self) -> bool:
        """Whether the device has been initialized."""
        return self._initialized

    @property
    def is_enabled(self) -> bool:
        """Whether the device is enabled."""
        return self._enabled

    @property
    @abstractmethod
    def actions(self) -> dict[str, Callable]:
        """
        Dictionary mapping command strings to methods.

        This structure allows the Flow Engine to trigger device actions
        generically without needing to know the specific class type.

        Example:
            {'activate': self.turn_on, 'deactivate': self.turn_off}
        """
        ...

    @property
    def required_pins(self) -> list[str]:
        """
        List of required pin names for this device.

        Override in subclass to specify which pins must be configured.
        """
        return []

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the device hardware.

        This should configure pin modes and set initial states.
        Called after the board connection is established.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the device safely.

        This should set the device to a safe state (e.g., motors stopped,
        heaters off). Called during emergency stop or normal shutdown.
        """
        ...

    async def enable(self) -> None:
        """Enable the device for operation."""
        self._enabled = True

    async def disable(self) -> None:
        """Disable the device (stops responding to commands)."""
        self._enabled = False

    def validate_config(self) -> list[str]:
        """
        Validate the device configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        for pin_name in self.required_pins:
            if pin_name not in self._config.pins:
                errors.append(f"Missing required pin: {pin_name}")
        return errors

    async def execute_action(self, action_name: str, *args, **kwargs) -> Any:
        """
        Execute a named action on this device.

        Args:
            action_name: Name of the action to execute
            *args: Positional arguments for the action
            **kwargs: Keyword arguments for the action

        Returns:
            Result of the action
        """
        if not self._enabled:
            raise RuntimeError(f"Device {self._name} is disabled")

        if action_name not in self.actions:
            raise ValueError(f"Unknown action: {action_name}")

        action = self.actions[action_name]
        return await action(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize device configuration to dictionary."""
        return {
            "id": self._id,
            "device_type": self.device_type,
            "name": self._name,
            "board_id": self._board.id,
            "config": {
                "pins": self._config.pins,
                "settings": self._config.settings,
            },
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "BaseDevice":
        """Create device instance from dictionary configuration."""
        ...


class DigitalOutputDevice(BaseDevice):
    """Simple digital output device (e.g., LED, Relay)."""

    @property
    def device_type(self) -> str:
        return "DigitalOutput"

    @property
    def required_pins(self) -> list[str]:
        return ["output"]

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "on": self.turn_on,
            "off": self.turn_off,
            "toggle": self.toggle,
            "set": self.set_state,
        }

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: Optional[str] = None):
        super().__init__(board, config, name)
        self._state = False

    @property
    def state(self) -> bool:
        """Current output state."""
        return self._state

    async def initialize(self) -> None:
        from glider.hal.base_board import PinMode, PinType

        pin = self._config.pins["output"]
        await self._board.set_pin_mode(pin, PinMode.OUTPUT, PinType.DIGITAL)
        await self._board.write_digital(pin, False)
        self._state = False
        self._initialized = True

    async def shutdown(self) -> None:
        if self._initialized:
            pin = self._config.pins["output"]
            await self._board.write_digital(pin, False)
            self._state = False

    async def turn_on(self) -> None:
        """Turn the output on."""
        pin = self._config.pins["output"]
        await self._board.write_digital(pin, True)
        self._state = True

    async def turn_off(self) -> None:
        """Turn the output off."""
        pin = self._config.pins["output"]
        await self._board.write_digital(pin, False)
        self._state = False

    async def toggle(self) -> None:
        """Toggle the output state."""
        if self._state:
            await self.turn_off()
        else:
            await self.turn_on()

    async def set_state(self, state: bool) -> None:
        """Set the output to a specific state."""
        if state:
            await self.turn_on()
        else:
            await self.turn_off()

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "DigitalOutputDevice":
        config = DeviceConfig(
            pins=data["config"]["pins"],
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


class DigitalInputDevice(BaseDevice):
    """Simple digital input device (e.g., Button, Beam Break Sensor)."""

    @property
    def device_type(self) -> str:
        return "DigitalInput"

    @property
    def required_pins(self) -> list[str]:
        return ["input"]

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "read": self.read,
        }

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: Optional[str] = None):
        super().__init__(board, config, name)
        self._last_value: Optional[bool] = None
        self._on_change_callbacks: list[Callable[[bool], None]] = []

    @property
    def last_value(self) -> Optional[bool]:
        """Last read value."""
        return self._last_value

    async def initialize(self) -> None:
        from glider.hal.base_board import PinMode, PinType

        pin = self._config.pins["input"]
        pullup = self._config.settings.get("pullup", False)
        mode = PinMode.INPUT_PULLUP if pullup else PinMode.INPUT
        await self._board.set_pin_mode(pin, mode, PinType.DIGITAL)
        self._initialized = True

    async def shutdown(self) -> None:
        pass  # No shutdown needed for input

    async def read(self) -> bool:
        """Read the current input state."""
        pin = self._config.pins["input"]
        value = await self._board.read_digital(pin)
        if value != self._last_value:
            self._last_value = value
            for callback in self._on_change_callbacks:
                try:
                    callback(value)
                except Exception:
                    pass
        return value

    def on_change(self, callback: Callable[[bool], None]) -> None:
        """Register a callback for value changes."""
        self._on_change_callbacks.append(callback)

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "DigitalInputDevice":
        config = DeviceConfig(
            pins=data["config"]["pins"],
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


class AnalogInputDevice(BaseDevice):
    """Analog input device (e.g., Potentiometer, Light Sensor)."""

    @property
    def device_type(self) -> str:
        return "AnalogInput"

    @property
    def required_pins(self) -> list[str]:
        return ["input"]

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "read": self.read,
            "read_voltage": self.read_voltage,
        }

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: Optional[str] = None):
        super().__init__(board, config, name)
        self._last_value: Optional[int] = None
        self._reference_voltage = config.settings.get("reference_voltage", 5.0)

    async def initialize(self) -> None:
        from glider.hal.base_board import PinMode, PinType

        pin = self._config.pins["input"]
        await self._board.set_pin_mode(pin, PinMode.INPUT, PinType.ANALOG)
        self._initialized = True

    async def shutdown(self) -> None:
        pass

    async def read(self) -> int:
        """Read the raw analog value."""
        pin = self._config.pins["input"]
        value = await self._board.read_analog(pin)

        # Validate the value is within expected range (0-1023 for 10-bit ADC)
        if not isinstance(value, (int, float)):
            logger.warning(f"Invalid analog value type: {type(value)}, value: {value}")
            value = 0
        elif value < 0 or value > 1023:
            logger.warning(f"Analog value out of range: {value}, clamping to 0-1023")
            value = max(0, min(1023, int(value)))
        else:
            value = int(value)

        self._last_value = value
        return value

    async def read_voltage(self) -> float:
        """Read the analog value as voltage."""
        raw = await self.read()
        max_value = 2**self._board.capabilities.analog_resolution - 1
        return (raw / max_value) * self._reference_voltage

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "AnalogInputDevice":
        config = DeviceConfig(
            pins=data["config"]["pins"],
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


class PWMOutputDevice(BaseDevice):
    """PWM output device (e.g., LED with brightness, Motor speed)."""

    @property
    def device_type(self) -> str:
        return "PWMOutput"

    @property
    def required_pins(self) -> list[str]:
        return ["output"]

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "set": self.set_value,
            "set_percent": self.set_percent,
            "off": self.off,
        }

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: Optional[str] = None):
        super().__init__(board, config, name)
        self._value = 0

    @property
    def value(self) -> int:
        """Current PWM value."""
        return self._value

    async def initialize(self) -> None:
        from glider.hal.base_board import PinMode, PinType

        pin = self._config.pins["output"]
        await self._board.set_pin_mode(pin, PinMode.OUTPUT, PinType.PWM)
        await self._board.write_analog(pin, 0)
        self._value = 0
        self._initialized = True

    async def shutdown(self) -> None:
        if self._initialized:
            await self.off()

    async def set_value(self, value: int) -> None:
        """Set the raw PWM value."""
        max_value = 2**self._board.capabilities.pwm_resolution - 1
        value = max(0, min(value, max_value))
        pin = self._config.pins["output"]
        await self._board.write_analog(pin, value)
        self._value = value

    async def set_percent(self, percent: float) -> None:
        """Set the PWM value as a percentage (0-100)."""
        max_value = 2**self._board.capabilities.pwm_resolution - 1
        value = int((percent / 100.0) * max_value)
        await self.set_value(value)

    async def off(self) -> None:
        """Turn off the PWM output."""
        await self.set_value(0)

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "PWMOutputDevice":
        config = DeviceConfig(
            pins=data["config"]["pins"],
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


class ServoDevice(BaseDevice):
    """Servo motor device."""

    @property
    def device_type(self) -> str:
        return "Servo"

    @property
    def required_pins(self) -> list[str]:
        return ["signal"]

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "set_angle": self.set_angle,
            "center": self.center,
        }

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: Optional[str] = None):
        super().__init__(board, config, name)
        self._angle = 90
        self._min_angle = config.settings.get("min_angle", 0)
        self._max_angle = config.settings.get("max_angle", 180)

    @property
    def angle(self) -> int:
        """Current servo angle."""
        return self._angle

    async def initialize(self) -> None:
        from glider.hal.base_board import PinMode, PinType

        pin = self._config.pins["signal"]
        await self._board.set_pin_mode(pin, PinMode.OUTPUT, PinType.SERVO)
        await self.center()
        self._initialized = True

    async def shutdown(self) -> None:
        pass  # Servos typically hold position

    async def set_angle(self, angle: int) -> None:
        """Set the servo angle."""
        angle = max(self._min_angle, min(angle, self._max_angle))
        pin = self._config.pins["signal"]
        await self._board.write_servo(pin, angle)
        self._angle = angle

    async def center(self) -> None:
        """Center the servo."""
        center = (self._min_angle + self._max_angle) // 2
        await self.set_angle(center)

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "ServoDevice":
        config = DeviceConfig(
            pins=data["config"]["pins"],
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


class ADS1115Device(BaseDevice):
    """
    ADS1115 16-bit ADC device for analog input via I2C.

    The ADS1115 is a 16-bit ADC with 4 single-ended channels (A0-A3)
    or 2 differential channels. Connected to Raspberry Pi via I2C
    (SDA on GPIO2, SCL on GPIO3).

    Settings:
    - i2c_address: I2C address (default 0x48, can be 0x49, 0x4A, 0x4B)
    - gain: Programmable gain amplifier setting (default 1)
        - 2/3: +/- 6.144V (not for single-ended use)
        - 1: +/- 4.096V
        - 2: +/- 2.048V
        - 4: +/- 1.024V
        - 8: +/- 0.512V
        - 16: +/- 0.256V
    - data_rate: Samples per second (8, 16, 32, 64, 128, 250, 475, 860)
    """

    # Gain settings mapping to voltage ranges
    GAIN_RANGES = {
        2 / 3: 6.144,
        1: 4.096,
        2: 2.048,
        4: 1.024,
        8: 0.512,
        16: 0.256,
    }

    @property
    def device_type(self) -> str:
        return "ADS1115"

    @property
    def required_pins(self) -> list[str]:
        # I2C doesn't use traditional pin assignments in the same way
        # The SDA/SCL are fixed on the Pi (GPIO2/GPIO3)
        return []

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "read": self.read,
            "read_channel": self.read_channel,
            "read_voltage": self.read_voltage,
            "read_all": self.read_all,
        }

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: Optional[str] = None):
        super().__init__(board, config, name)
        self._i2c_address = config.settings.get("i2c_address", 0x48)
        self._gain = config.settings.get("gain", 1)
        self._data_rate = config.settings.get("data_rate", 128)
        self._ads = None  # Will hold the ADS1115 object
        self._channels: dict[int, Any] = {}  # Cache for AnalogIn objects
        self._last_values: dict[int, int] = {}  # Channel -> raw value
        self._lock = asyncio.Lock()  # Async lock to serialize hardware access

    @property
    def i2c_address(self) -> int:
        """I2C address of the ADS1115."""
        return self._i2c_address

    @property
    def gain(self) -> float:
        """Current gain setting."""
        return self._gain

    @property
    def voltage_range(self) -> float:
        """Maximum voltage for current gain setting."""
        return self.GAIN_RANGES.get(self._gain, 4.096)

    async def initialize(self) -> None:
        """Initialize the ADS1115 via I2C."""
        import asyncio

        def _init_ads():
            try:
                import adafruit_ads1x15.ads1115 as ADS
                import board
                import busio
                from adafruit_ads1x15.analog_in import AnalogIn

                # Create I2C bus
                i2c = busio.I2C(board.SCL, board.SDA)

                # Create ADS1115 object
                ads = ADS.ADS1115(i2c, address=self._i2c_address)
                ads.gain = self._gain
                ads.data_rate = self._data_rate

                # Pre-create AnalogIn objects for all 4 channels
                channels = {}
                for i in range(4):
                    channels[i] = AnalogIn(ads, i)

                return ads, channels
            except ImportError as e:
                raise RuntimeError(
                    "ADS1115 libraries not installed. Run: "
                    "pip install adafruit-circuitpython-ads1x15"
                ) from e
            except Exception as e:
                logger.error(f"Failed to initialize ADS1115: {e}")
                raise

        self._ads, self._channels = await asyncio.to_thread(_init_ads)
        self._initialized = True
        logger.info(f"ADS1115 initialized at address 0x{self._i2c_address:02X}")

    async def shutdown(self) -> None:
        """Shutdown the ADS1115."""
        self._ads = None
        self._channels = {}
        self._initialized = False

    async def read(self, channel: int = 0) -> int:
        """
        Read raw ADC value from a channel.

        Args:
            channel: Channel number (0-3)

        Returns:
            Raw 16-bit ADC value (-32768 to 32767 for differential,
            0 to 32767 for single-ended)
        """
        return await self.read_channel(channel)

    async def read_channel(self, channel: int = 0) -> int:
        """
        Read raw ADC value from a specific channel.

        Args:
            channel: Channel number (0-3)

        Returns:
            Raw ADC value
        """
        import asyncio

        if not self._initialized or self._ads is None:
            raise RuntimeError("ADS1115 not initialized")

        if channel < 0 or channel > 3:
            raise ValueError(f"Invalid channel {channel}. Must be 0-3.")

        chan = self._channels.get(channel)
        if chan is None:
            raise RuntimeError(f"Channel {channel} not initialized")

        # Use board-level I2C lock if available, otherwise device-level lock
        lock = getattr(self._board, "i2c_lock", self._lock)
        async with lock:
            value = await asyncio.to_thread(lambda: chan.value)
            self._last_values[channel] = value
            return value

    async def read_voltage(self, channel: int = 0) -> float:
        """
        Read voltage from a channel.

        Args:
            channel: Channel number (0-3)

        Returns:
            Voltage reading
        """
        import asyncio

        if not self._initialized or self._ads is None:
            raise RuntimeError("ADS1115 not initialized")

        if channel < 0 or channel > 3:
            raise ValueError(f"Invalid channel {channel}. Must be 0-3.")

        chan = self._channels.get(channel)
        if chan is None:
            raise RuntimeError(f"Channel {channel} not initialized")

        lock = getattr(self._board, "i2c_lock", self._lock)
        async with lock:
            return await asyncio.to_thread(lambda: chan.voltage)

    async def read_all(self) -> dict[int, int]:
        """
        Read raw values from all 4 channels.

        Returns:
            Dictionary mapping channel number to raw value
        """
        results = {}
        for channel in range(4):
            results[channel] = await self.read_channel(channel)
        return results

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "ADS1115Device":
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


class MotorGovernorDevice(BaseDevice):
    """
    Motor Governor device for controlling motorized positioning.

    Uses three pins:
    - up: Digital output to move the governor up
    - down: Digital output to move the governor down
    - signal: Analog input to read position feedback
    """

    @property
    def device_type(self) -> str:
        return "MotorGovernor"

    @property
    def required_pins(self) -> list[str]:
        return ["up", "down", "signal"]

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "up": self.move_up,
            "down": self.move_down,
            "stop": self.stop,
            "read_position": self.read_position,
        }

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: Optional[str] = None):
        super().__init__(board, config, name)
        self._position: Optional[int] = None

    @property
    def position(self) -> Optional[int]:
        """Last read position value from signal pin."""
        return self._position

    async def initialize(self) -> None:
        from glider.hal.base_board import PinMode, PinType

        # Configure up pin as digital output
        up_pin = self._config.pins["up"]
        await self._board.set_pin_mode(up_pin, PinMode.OUTPUT, PinType.DIGITAL)
        await self._board.write_digital(up_pin, False)

        # Configure down pin as digital output
        down_pin = self._config.pins["down"]
        await self._board.set_pin_mode(down_pin, PinMode.OUTPUT, PinType.DIGITAL)
        await self._board.write_digital(down_pin, False)

        # Configure signal pin as analog input
        signal_pin = self._config.pins["signal"]
        await self._board.set_pin_mode(signal_pin, PinMode.INPUT, PinType.ANALOG)

        self._initialized = True

    async def shutdown(self) -> None:
        """Stop all movement on shutdown."""
        if self._initialized:
            await self.stop()

    async def move_up(self) -> None:
        """
        Move the governor up by one increment.

        Logic: Set DOWN high, then toggle UP high->low.
        """
        import asyncio

        up_pin = self._config.pins["up"]
        down_pin = self._config.pins["down"]

        # Set DOWN high
        await self._board.write_digital(down_pin, True)

        # Toggle UP: high then low
        await self._board.write_digital(up_pin, True)
        await asyncio.sleep(0.05)  # Brief pulse
        await self._board.write_digital(up_pin, False)

    async def move_down(self) -> None:
        """
        Move the governor down by one increment.

        Logic: Set UP high, then toggle DOWN high->low.
        """
        import asyncio

        up_pin = self._config.pins["up"]
        down_pin = self._config.pins["down"]

        # Set UP high
        await self._board.write_digital(up_pin, True)

        # Toggle DOWN: high then low
        await self._board.write_digital(down_pin, True)
        await asyncio.sleep(0.05)  # Brief pulse
        await self._board.write_digital(down_pin, False)

    async def stop(self) -> None:
        """Set both pins low (idle state)."""
        up_pin = self._config.pins["up"]
        down_pin = self._config.pins["down"]

        await self._board.write_digital(up_pin, False)
        await self._board.write_digital(down_pin, False)

    async def read_position(self) -> int:
        """Read the current position from the signal pin."""
        signal_pin = self._config.pins["signal"]
        self._position = await self._board.read_analog(signal_pin)
        return self._position

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "MotorGovernorDevice":
        config = DeviceConfig(
            pins=data["config"]["pins"],
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


# Registry of built-in device types
DEVICE_REGISTRY: dict[str, type] = {
    "DigitalOutput": DigitalOutputDevice,
    "DigitalInput": DigitalInputDevice,
    "AnalogInput": AnalogInputDevice,
    "PWMOutput": PWMOutputDevice,
    "Servo": ServoDevice,
    "MotorGovernor": MotorGovernorDevice,
    "ADS1115": ADS1115Device,
}


def create_device_from_dict(data: dict[str, Any], board: "BaseBoard") -> BaseDevice:
    """Factory function to create a device from dictionary configuration."""
    device_type = data.get("device_type")
    if device_type not in DEVICE_REGISTRY:
        raise ValueError(f"Unknown device type: {device_type}")
    return DEVICE_REGISTRY[device_type].from_dict(data, board)
