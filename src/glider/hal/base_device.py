"""
Abstract Base Class for hardware devices.

Devices represent higher-level components attached to boards
(e.g., "Stepper Motor", "Temperature Sensor"). They wrap BaseBoard
methods into semantic actions.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from glider.hal.value_spec import KIND_SWITCH, KIND_WHOLE, ActionValueSpec, clamp_to_spec

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard
    from glider.hal.input_behavior import InputBehavior


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
        name: str | None = None,
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
        # Serializes commands to THIS device across all sources (experiment
        # flow, functions, manual controls) so their hardware steps never
        # interleave. Held across the whole action body in execute_action.
        # Emergency stop is intentionally exempt and does not acquire it.
        self._command_lock = asyncio.Lock()
        # Fired when an action mutates config.settings at runtime (e.g. an
        # HX711 tare writing its offset). HardwareManager wires this so the
        # session can adopt the change and mark itself dirty; without it the
        # value lives only on the device, the close prompt never appears, and
        # the calibration is lost on exit.
        self._settings_changed_cb: Callable[[BaseDevice], None] | None = None

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

    def state_columns(self) -> list[str] | None:
        """
        Sub-column names this device contributes to the data CSV.

        Return ``None`` (the default) for single-column behaviour: the
        recorder emits one column named ``{device_id}:{device_type}`` and
        writes this device's recorded state as a scalar.

        Return a list of names to contribute several columns. The recorder
        then emits ``{device_id}:{name}`` per entry and expects this
        device's recorded state to be a dict keyed by those names. Missing
        keys are written as empty cells.

        Names must be non-empty, unique within the device, and must not
        contain ``:``. The recorder builds headers as
        ``{device_id}:{name}`` and recovers the pair by splitting on the
        *first* colon, so a colon inside the name would in fact survive
        the round trip — the restriction is for the humans and downstream
        tools that read these headers, which have no way to know which
        colon was the separator. (A colon in the *device id* genuinely
        does break the split, but that is not something a device
        controls.) A device with a single column returns ``None``, never
        ``[]``.
        """
        return None

    def recording_warnings(self) -> list[str]:
        """
        Ways this device's columns will say less than they look like they do.

        Each string becomes a ``# WARNING`` row in the CSV metadata block,
        beside the device id. Return short sentences describing a *known*
        limitation of what will be recorded — a column that is wired to
        something that never changes, a sensor configured out of range —
        not transient errors, which belong in the log.

        The CSV is what outlives the session. A device that only logs its
        limitation announces it to nobody: an unattended overnight run
        writes the log to a file nobody opens, and the person reading the
        recording six months later has no way to learn that a column was
        never going to move. Default: none.
        """
        return []

    @property
    def input_behaviors(self) -> list["InputBehavior"]:
        """Wait/input behaviors this device offers to WaitForInput (default none).

        Override to advertise named things the node can wait on. See
        glider.hal.input_behavior.InputBehavior.
        """
        return []

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

    def set_settings_changed_callback(
        self, callback: Callable[["BaseDevice"], None] | None
    ) -> None:
        """Register the listener notified when an action mutates settings."""
        self._settings_changed_cb = callback

    def _notify_settings_changed(self) -> None:
        """
        Announce that this device mutated its own ``config.settings``.

        Call from any action that writes a setting at runtime. Errors in the
        listener are logged and swallowed — a bookkeeping failure must not
        break the hardware action that triggered it.
        """
        if self._settings_changed_cb is None:
            return
        try:
            self._settings_changed_cb(self)
        except Exception:
            logger.exception("Error in settings-changed callback for device %s", self._name)

    def apply_settings(self, settings: dict[str, Any]) -> None:
        """
        Adopt edited settings on a live (already constructed) device.

        The edit-device dialog calls this instead of mutating
        ``config.settings`` directly. Devices that cache setting values at
        construction time must override it to refresh those caches, or an edit
        will update the saved file while the running device keeps using its
        old values. Overrides should validate before committing so a rejected
        edit leaves the device unchanged rather than half-applied.

        Args:
            settings: The changed keys. Merged over the existing settings, so
                a partial dict is fine.
        """
        self._config.settings.update(settings)

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

        if not self._initialized:
            raise RuntimeError(
                f"Device {self._name} is not initialized "
                "(it may have been shut down; re-initialize before use)"
            )

        if action_name not in self.actions:
            raise ValueError(f"Unknown action: {action_name}")

        action = self.actions[action_name]
        # Hold the per-device lock across the WHOLE action body (including any
        # nested to_thread hops) so a multi-step board write cannot interleave
        # with a concurrent command to the same device. The checks above do not
        # await, so lock-acquisition order equals call order (FIFO).
        async with self._command_lock:
            if asyncio.iscoroutinefunction(action):
                return await action(*args, **kwargs)
            else:
                return await asyncio.to_thread(action, *args, **kwargs)

    def value_spec(self, action_name: str) -> ActionValueSpec | None:
        """Single authority for an action's value semantics (range/unit/label).

        Returns ``None`` when the action carries no value. Node property editors,
        the generated runner controls, and write-time clamping all read range and
        unit through this one call, so no layer keeps a private hardcoded range.
        A declared spec (from a custom definition or a built-in) takes precedence;
        otherwise the op-inferred fallback preserves today's behavior.
        """
        declared = self._declared_value_spec(action_name)
        return declared if declared is not None else self._default_value_spec(action_name)

    def _declared_value_spec(self, action_name: str) -> ActionValueSpec | None:
        """An explicitly declared spec for ``action_name``, or ``None`` to fall back.

        Overridden by built-in device classes (e.g. the servo's angle range) and
        by declarative devices that carry a ``value`` block. The base device
        declares nothing.
        """
        return None

    def _default_value_spec(self, action_name: str) -> ActionValueSpec | None:
        """The fallback spec used when nothing is declared (preserves today's ranges).

        The base class knows no ops, so it returns ``None``; subclasses that write
        values infer a default from the action's op and the board's capabilities.
        """
        return None

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

    def _default_value_spec(self, action_name: str) -> ActionValueSpec | None:
        # "set" takes an on/off value; on/off/toggle take none.
        return ActionValueSpec(KIND_SWITCH, 0, 1) if action_name == "set" else None

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
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
        try:
            if self._initialized:
                pin = self._config.pins["output"]
                await self._board.write_digital(pin, False)
                self._state = False
        finally:
            # Always clear the flag, even if the safe-state write raised —
            # the caller may immediately re-initialize against a different
            # board, and ``is_initialized`` lying about state corrupts that
            # path. ``_set_all_devices_low`` in GliderCore records the
            # failure separately.
            self._initialized = False

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

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        self._last_value: bool | None = None
        self._on_change_callbacks: list[Callable[[bool], None]] = []

    @property
    def last_value(self) -> bool | None:
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
        # Inputs have no safe-state write, but the flag must still clear
        # so a subsequent ``initialize()`` re-runs ``set_pin_mode``.
        self._initialized = False

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

    def _default_value_spec(self, action_name: str) -> ActionValueSpec | None:
        if action_name == "read":
            bits = self._board.capabilities.analog_resolution
            return ActionValueSpec(KIND_WHOLE, 0, (1 << bits) - 1)
        return None

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        self._last_value: int | None = None
        self._reference_voltage = config.settings.get("reference_voltage", 5.0)

    async def initialize(self) -> None:
        from glider.hal.base_board import PinMode, PinType

        pin = self._config.pins["input"]
        await self._board.set_pin_mode(pin, PinMode.INPUT, PinType.ANALOG)
        self._initialized = True

    async def shutdown(self) -> None:
        # No safe-state write for analog inputs; clear the flag so a
        # subsequent initialize re-runs set_pin_mode against the (possibly
        # reconnected, possibly different) board.
        self._initialized = False

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

    def _default_value_spec(self, action_name: str) -> ActionValueSpec | None:
        if action_name == "set":
            bits = self._board.capabilities.pwm_resolution
            return ActionValueSpec(KIND_WHOLE, 0, (1 << bits) - 1)
        if action_name == "set_percent":
            return ActionValueSpec(KIND_WHOLE, 0, 100, unit="%")
        return None

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
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
        try:
            if self._initialized:
                await self.off()
        finally:
            self._initialized = False

    async def set_value(self, value: int) -> None:
        """Set the raw PWM value.

        Clamps to the device's declared range via ``clamp_to_spec`` so a
        non-finite/non-numeric value is rejected (ValueError) rather than
        silently becoming 0 — matching the declarative-device write path.
        """
        spec = self.value_spec("set")
        if spec is not None:
            value, _ = clamp_to_spec(value, spec)
        else:
            max_value = 2**self._board.capabilities.pwm_resolution - 1
            value = max(0, min(int(value), max_value))
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

    def _declared_value_spec(self, action_name: str) -> ActionValueSpec | None:
        # The servo's configurable angle bounds are per-instance, so they feed a
        # declared (not merely default) spec. "center" takes no value.
        if action_name == "set_angle":
            return ActionValueSpec(KIND_WHOLE, self._min_angle, self._max_angle, unit="deg")
        return None

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        self._angle = 90
        self._min_angle = config.settings.get("min_angle", 0)
        self._max_angle = config.settings.get("max_angle", 180)
        # Inverted/degenerate bounds would collapse both the angle clamp (always
        # min) and the generated slider (QSlider.setRange(hi, lo) fixes the
        # value), silently ignoring every commanded angle. Normalize to a usable
        # range and warn rather than misbehave.
        if self._min_angle >= self._max_angle:
            logger.warning(
                "Servo %s has invalid angle bounds (min=%s, max=%s); using 0..180",
                self._name,
                self._min_angle,
                self._max_angle,
            )
            self._min_angle, self._max_angle = 0, 180

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
        # Servos hold their last commanded position — there is no
        # "safe" angle in general. Clear the flag so reinit works
        # correctly after a disconnect/reconnect cycle.
        self._initialized = False

    async def set_angle(self, angle: int) -> None:
        """Set the servo angle.

        Clamps to the declared angle range via ``clamp_to_spec`` so a
        non-finite/non-numeric angle is rejected rather than silently collapsing
        to ``min_angle``.
        """
        spec = self.value_spec("set_angle")
        if spec is not None:
            angle, _ = clamp_to_spec(angle, spec)
        else:
            angle = max(self._min_angle, min(int(angle), self._max_angle))
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

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        self._i2c_address = config.settings.get("i2c_address", 0x48)
        self._gain = config.settings.get("gain", 1)
        self._data_rate = config.settings.get("data_rate", 128)
        self._channel = config.settings.get("channel", 0)
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

    async def read(self, channel: int | None = None) -> int:
        """
        Read raw ADC value from a channel.

        Args:
            channel: Channel number (0-3). If None, uses the channel
                configured on this device instance.

        Returns:
            Raw 16-bit ADC value (-32768 to 32767 for differential,
            0 to 32767 for single-ended)
        """
        if channel is None:
            channel = self._channel
        return await self.read_channel(channel)

    async def read_channel(self, channel: int = 0) -> int:
        """
        Read raw ADC value from a specific channel.

        Args:
            channel: Channel number (0-3)

        Returns:
            Raw ADC value
        """
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

    async def read_voltage(self, channel: int | None = None) -> float:
        """
        Read voltage from a channel.

        Args:
            channel: Channel number (0-3). If None, uses the channel
                configured on this device instance.

        Returns:
            Voltage reading
        """
        if not self._initialized or self._ads is None:
            raise RuntimeError("ADS1115 not initialized")

        if channel is None:
            channel = self._channel

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


class GenericI2CDevice(BaseDevice):
    """
    Generic I2C device exposing the full SMBus surface via ``smbus2``.

    Talks to any I2C peripheral on a Raspberry Pi by address, without needing a
    dedicated device class per chip. One device instance binds one bus + one
    7-bit address. Every transfer runs in a worker thread under the board's
    shared ``i2c_lock`` so it coexists safely with other I2C devices (e.g. an
    ADS1115) on the same bus.

    Settings:
    - i2c_bus: I2C bus number (default 1 — the Pi's primary bus on GPIO2/GPIO3)
    - i2c_address: 7-bit address, 0x03-0x77 (default 0x40)
    - register: optional default register for the no-arg ``read`` action. When
        None (default), ``read`` performs a raw byte receive instead.
    - read_word: when True (and ``register`` is set), the no-arg ``read`` reads
        two bytes big-endian (MSB first) and returns the combined 16-bit value,
        instead of a single byte. This is what most 16-bit sensors need — e.g.
        the AS5600 magnetic encoder, whose 12-bit angle lives in ANGLE_H/ANGLE_L
        at 0x0E/0x0F. Default False. (SMBus-native little-endian word reads are
        still available explicitly via the ``read_word_data`` action.)

    The transfer methods take ``None``-sentinel defaults for their required
    arguments and validate presence explicitly. This is deliberate: the
    ``DeviceActionNode`` only forwards a node input when it is non-``None``, so
    an unconnected port arrives as a *missing* positional — the sentinels turn
    that into a clear ``ValueError`` rather than a raw ``TypeError``.
    """

    ADDR_MIN = 0x03
    ADDR_MAX = 0x77
    BLOCK_MAX = 32

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        self._bus_num = int(config.settings.get("i2c_bus", 1))
        self._address = self._validate_address(config.settings.get("i2c_address", 0x40))
        register = config.settings.get("register", None)
        self._register = self._validate_register(register) if register is not None else None
        self._read_word = bool(config.settings.get("read_word", False))
        self._bus = None  # smbus2.SMBus handle, created in initialize()
        self._lock = asyncio.Lock()  # fallback when the board exposes no i2c_lock

    # --- validation helpers ---

    @classmethod
    def _validate_address(cls, address: Any) -> int:
        addr = int(address)
        if addr < cls.ADDR_MIN or addr > cls.ADDR_MAX:
            raise ValueError(
                f"I2C address 0x{addr:02X} out of range "
                f"(0x{cls.ADDR_MIN:02X}-0x{cls.ADDR_MAX:02X})"
            )
        return addr

    @staticmethod
    def _validate_register(register: Any) -> int:
        if register is None:
            raise ValueError("register is required")
        reg = int(register)
        if reg < 0x00 or reg > 0xFF:
            raise ValueError(f"register 0x{reg:X} out of range (0x00-0xFF)")
        return reg

    @staticmethod
    def _validate_byte(value: Any, label: str = "value") -> int:
        if value is None:
            raise ValueError(f"{label} is required")
        v = int(value)
        if v < 0x00 or v > 0xFF:
            raise ValueError(f"{label} {v} out of range (0x00-0xFF)")
        return v

    @staticmethod
    def _validate_word(value: Any, label: str = "value") -> int:
        if value is None:
            raise ValueError(f"{label} is required")
        v = int(value)
        if v < 0x0000 or v > 0xFFFF:
            raise ValueError(f"{label} {v} out of range (0x0000-0xFFFF)")
        return v

    @classmethod
    def _validate_length(cls, length: Any) -> int:
        n = int(length)
        if n < 1 or n > cls.BLOCK_MAX:
            raise ValueError(f"block length {n} out of range (1-{cls.BLOCK_MAX})")
        return n

    @classmethod
    def _validate_block(cls, data: Any) -> list[int]:
        if data is None:
            raise ValueError("data is required")
        block = list(data)
        if len(block) < 1 or len(block) > cls.BLOCK_MAX:
            raise ValueError(f"block data length {len(block)} out of range (1-{cls.BLOCK_MAX})")
        return [cls._validate_byte(b, "data byte") for b in block]

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "GenericI2C"

    @property
    def required_pins(self) -> list[str]:
        # I2C: SDA/SCL are fixed (GPIO2/GPIO3); no GPIO pins are allocated.
        return []

    @property
    def i2c_bus(self) -> int:
        """Configured I2C bus number."""
        return self._bus_num

    @property
    def i2c_address(self) -> int:
        """Configured 7-bit I2C address."""
        return self._address

    @property
    def register(self) -> int | None:
        """Optional default register used by the no-arg ``read`` action."""
        return self._register

    @property
    def read_word(self) -> bool:
        """Whether the no-arg ``read`` returns a big-endian 16-bit word."""
        return self._read_word

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "read": self.read,
            "read_byte": self.read_byte,
            "write_byte": self.write_byte,
            "read_byte_data": self.read_byte_data,
            "write_byte_data": self.write_byte_data,
            "read_word_data": self.read_word_data,
            "write_word_data": self.write_word_data,
            "read_word_be": self.read_word_be,
            "read_block": self.read_block,
            "write_block": self.write_block,
        }

    # --- lifecycle ---

    async def initialize(self) -> None:
        """Open the I2C bus via smbus2 (lazy-imported in a worker thread)."""

        def _open():
            try:
                import smbus2
            except ImportError as e:
                raise RuntimeError(
                    "smbus2 not installed. Run: pip install 'GLIDER[i2c]' "
                    "(or pip install smbus2)"
                ) from e
            return smbus2.SMBus(self._bus_num)

        self._bus = await asyncio.to_thread(_open)
        self._initialized = True
        logger.info("GenericI2C initialized on bus %d at 0x%02X", self._bus_num, self._address)

    async def shutdown(self) -> None:
        """Close the bus and clear state (safe to call before initialize)."""
        try:
            if self._bus is not None:
                await asyncio.to_thread(self._bus.close)
        finally:
            self._bus = None
            self._initialized = False

    # --- transfer plumbing ---

    async def _transfer(self, method_name: str, *args) -> Any:
        """Run an smbus2 call in a thread under the board's I2C lock."""
        if not self._initialized or self._bus is None:
            raise RuntimeError("GenericI2C device not initialized")
        fn = getattr(self._bus, method_name)
        lock = getattr(self._board, "i2c_lock", self._lock)
        async with lock:
            return await asyncio.to_thread(fn, *args)

    # --- actions ---

    async def read(self) -> int:
        """No-arg read.

        - no default register -> a raw byte receive
        - default register + read_word -> a big-endian 16-bit word
        - default register otherwise -> a single byte from that register
        """
        if self._register is None:
            return await self.read_byte()
        if self._read_word:
            return await self.read_word_be(self._register)
        return await self.read_byte_data(self._register)

    async def read_byte(self) -> int:
        """Receive a single byte (no register)."""
        return await self._transfer("read_byte", self._address)

    async def write_byte(self, value: Any = None) -> None:
        """Send a single byte (no register)."""
        value = self._validate_byte(value)
        await self._transfer("write_byte", self._address, value)

    async def read_byte_data(self, register: Any = None) -> int:
        """Read a byte from ``register``."""
        register = self._validate_register(register)
        return await self._transfer("read_byte_data", self._address, register)

    async def write_byte_data(self, register: Any = None, value: Any = None) -> None:
        """Write ``value`` (a byte) to ``register``."""
        register = self._validate_register(register)
        value = self._validate_byte(value)
        await self._transfer("write_byte_data", self._address, register, value)

    async def read_word_data(self, register: Any = None) -> int:
        """Read a 16-bit word from ``register`` (SMBus-native, little-endian)."""
        register = self._validate_register(register)
        return await self._transfer("read_word_data", self._address, register)

    async def read_word_be(self, register: Any = None) -> int:
        """Read two bytes starting at ``register``, MSB first (big-endian).

        Reads ``register`` (high byte) and ``register+1`` (low byte) in address
        order and returns ``(high << 8) | low``. This is the layout used by most
        16-bit I2C sensors — e.g. the AS5600's 12-bit angle at 0x0E/0x0F — and
        avoids the byte-swap you'd get from the SMBus-native ``read_word_data``.
        """
        register = self._validate_register(register)
        data = await self.read_block(register, 2)
        return (int(data[0]) << 8) | int(data[1])

    async def write_word_data(self, register: Any = None, value: Any = None) -> None:
        """Write ``value`` (a 16-bit word) to ``register``."""
        register = self._validate_register(register)
        value = self._validate_word(value)
        await self._transfer("write_word_data", self._address, register, value)

    async def read_block(self, register: Any = None, length: Any = BLOCK_MAX) -> list[int]:
        """Read ``length`` bytes (1-32) starting at ``register``."""
        register = self._validate_register(register)
        length = self._validate_length(length)
        return await self._transfer("read_i2c_block_data", self._address, register, length)

    async def write_block(self, register: Any = None, data: Any = None) -> None:
        """Write a list of bytes (1-32) starting at ``register``."""
        register = self._validate_register(register)
        data = self._validate_block(data)
        await self._transfer("write_i2c_block_data", self._address, register, data)

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "GenericI2CDevice":
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

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        self._position: int | None = None
        self._motor_lock = asyncio.Lock()

    @property
    def position(self) -> int | None:
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
        try:
            if self._initialized:
                await self.stop()
        finally:
            self._initialized = False

    async def move_up(self) -> None:
        """
        Move the governor up by one increment.

        Logic: Set DOWN high, then toggle UP high->low.
        """
        async with self._motor_lock:
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
        async with self._motor_lock:
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
        async with self._motor_lock:
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


class BLEWriteDevice(BaseDevice):
    """
    Generic BLE peripheral: writes string commands to one GATT characteristic.

    Talks to any Bluetooth LE peripheral that exposes a writable characteristic,
    without needing a dedicated device class per gadget. One device instance owns
    one ``BleakClient`` connection to one peripheral address. Commands are
    arbitrary strings (e.g. ``"on"``, ``"off"``, ``"20,10"``) encoded to bytes
    and written to the configured characteristic -- e.g. an optogenetic stimulator
    whose command characteristic accepts ``"<hz>,<seconds>"`` or ``"on"``/``"off"``.

    Settings:
    - address: peripheral BLE address (MAC on Windows/Linux, UUID on macOS).
        Required.
    - char_uuid: UUID of the writable command characteristic. Required.
    - service_uuid: optional service UUID (kept for reference / future filtering).
    - write_response: preferred write mode used only when the characteristic
        advertises BOTH modes (or its properties can't be read). The mode is
        otherwise auto-detected from the characteristic: a characteristic that
        supports only Write Request is written with response, one that supports
        only Write Without Response is written without -- so this rarely needs
        setting. True = Write Request (acknowledged), False = Write Without
        Response (default).
    - encoding: text encoding for commands (default "utf-8").

    The connection is opened lazily in ``initialize()`` and reused. A dropped
    link is transparently re-established on the next ``write``.
    """

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        s = config.settings
        self._address = str(s.get("address", "")).strip()
        self._char_uuid = str(s.get("char_uuid", "")).strip()
        self._service_uuid = str(s.get("service_uuid", "")).strip() or None
        self._write_response = bool(s.get("write_response", False))
        self._encoding = s.get("encoding", "utf-8")
        self._client = None  # BleakClient, created in initialize()
        self._lock = asyncio.Lock()  # serialize writes / (re)connects

    @property
    def device_type(self) -> str:
        return "BLEWrite"

    @property
    def required_pins(self) -> list[str]:
        # BLE has no GPIO pins.
        return []

    @property
    def address(self) -> str:
        """Configured peripheral BLE address."""
        return self._address

    @property
    def char_uuid(self) -> str:
        """Configured writable characteristic UUID."""
        return self._char_uuid

    @property
    def actions(self) -> dict[str, Callable]:
        return {"write": self.write}

    async def initialize(self) -> None:
        """Validate settings and open the BLE connection."""
        if not self._address:
            raise ValueError("BLEWrite: 'address' setting is required")
        if not self._char_uuid:
            raise ValueError("BLEWrite: 'char_uuid' setting is required")
        await self._ensure_connected()
        self._initialized = True
        logger.info("BLEWrite initialized: %s char %s", self._address, self._char_uuid)

    async def shutdown(self) -> None:
        """Disconnect the peripheral (safe to call before initialize).

        Clears ``_initialized`` FIRST so a ``write()`` queued behind the lock
        refuses to run instead of reconnecting a device that was just shut
        down (e.g. by an emergency stop), then serializes on the same lock
        ``write()`` uses so an in-flight write finishes before the disconnect.
        """
        self._initialized = False
        async with self._lock:
            client = self._client
            self._client = None
            if client is not None:
                try:
                    await client.disconnect()
                except Exception as e:  # pragma: no cover - disconnect best-effort
                    logger.warning("BLEWrite: error during disconnect: %s", e)

    async def _ensure_connected(self) -> None:
        """Open (or reopen) the BleakClient connection if not already up."""
        if self._client is not None and self._client.is_connected:
            return
        try:
            from bleak import BleakClient
        except ImportError as e:
            raise RuntimeError(
                "bleak not installed. Run: pip install bleak (or reinstall GLIDER)."
            ) from e
        client = BleakClient(self._address)
        await client.connect()
        self._client = client
        logger.info("BLEWrite: connected to %s", self._address)

    @staticmethod
    def _format_arg(value: Any) -> str:
        """Render one command argument as text.

        Whole-number floats become bare ints ("20" not "20.0") since the
        DeviceAction node's Number Input ports deliver floats -- so two of them
        wired to ``write`` produce e.g. "20,10".
        """
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _char_properties(self) -> set | None:
        """The connected characteristic's GATT properties, or None if unknown."""
        client = self._client
        if client is None:
            return None
        try:
            char = client.services.get_characteristic(self._char_uuid)
            if char is None:
                return None
            return set(char.properties)
        except Exception:
            return None

    def _effective_response(self) -> bool:
        """Pick the write mode, auto-detecting from the characteristic.

        A characteristic that supports only one write mode is written that way;
        when it supports both (or its properties can't be read) the configured
        ``write_response`` preference is honoured.
        """
        props = self._char_properties()
        if props is None:
            return self._write_response
        has_with = "write" in props
        has_without = "write-without-response" in props
        if has_with and not has_without:
            return True
        if has_without and not has_with:
            return False
        return self._write_response

    async def write(self, *args: Any) -> None:
        """Write a command to the characteristic.

        Accepts one or more arguments and joins them with commas, so a single
        string command (``write("on")`` / ``write("500,10")``) and a pair of
        Number Inputs (``write(500, 10)`` -> ``"500,10"``) both work. The command
        is encoded with ``encoding``. If the link dropped, one reconnect is
        attempted before the write is retried. No args is an explicit error
        (an unconnected DeviceAction port arrives as a missing positional).
        """
        if not args or all(a is None for a in args):
            raise ValueError("BLEWrite.write: command is required")
        command = ",".join(self._format_arg(a) for a in args if a is not None)
        data = command.encode(self._encoding)
        async with self._lock:
            if not self._initialized:
                raise RuntimeError(f"BLEWrite device {self._name} is not initialized")
            try:
                await self._ensure_connected()
                await self._client.write_gatt_char(
                    self._char_uuid, data, response=self._effective_response()
                )
            except Exception:
                # Link may have dropped; reconnect once and retry — unless a
                # shutdown ran meanwhile (e.g. e-stop), in which case
                # reconnecting would re-arm a device that was just stopped.
                if not self._initialized:
                    raise
                self._client = None
                await self._ensure_connected()
                await self._client.write_gatt_char(
                    self._char_uuid, data, response=self._effective_response()
                )
        logger.info("BLEWrite: wrote %r to %s", command, self._char_uuid)

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "BLEWriteDevice":
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
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
    "GenericI2C": GenericI2CDevice,
    "BLEWrite": BLEWriteDevice,
}


def create_device_from_dict(data: dict[str, Any], board: "BaseBoard") -> BaseDevice:
    """Factory function to create a device from dictionary configuration."""
    device_type = data.get("device_type")
    if device_type not in DEVICE_REGISTRY:
        raise ValueError(f"Unknown device type: {device_type}")
    return DEVICE_REGISTRY[device_type].from_dict(data, board)
