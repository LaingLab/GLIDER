"""
Declarative ("no-code") custom devices.

A declarative device is described by data, not Python: a ``.gdevice`` definition
names a transport (``i2c`` or ``gpio``), a settings schema, and a set of named
actions that each map to a primitive operation. ``DeclarativeDevice`` interprets
that definition at runtime against the chosen transport. ``build_device_class``
turns one definition into a concrete ``BaseDevice`` subclass so it can be
registered in ``DEVICE_REGISTRY`` and used exactly like a built-in device.

Because the definition is pure data (no executable code), loading user-authored
devices is safe by default -- unlike directory plugins, which exec Python.

Definition shape::

    {
      "schema_version": "1.0",
      "name": "MySensor",
      "description": "...",
      "transport": "i2c",            # "i2c" | "gpio"
      "settings": [ {key,label,type,default,...}, ... ],
      "actions": [
        {"name": "read_temp", "op": "read_word", "params": {"register": 0}, "primary": true},
        {"name": "set_cfg",  "op": "write_byte", "params": {"register": 1},
         "runtime_args": ["value"]},
      ]
    }

Supported ops:
  i2c:  read_byte, read_word (big-endian), write_byte, write_word  (each needs a
        ``register`` param; write ops take their ``value`` as a runtime arg)
  gpio: set_high, set_low, read_digital, read_analog, write_pwm    (act on the
        device's ``pin`` setting; write_pwm takes ``value`` as a runtime arg)
"""

import asyncio
import logging
import time

from glider.hal.base_board import PinMode, PinType
from glider.hal.base_device import BaseDevice, DeviceConfig
from glider.hal.value_spec import KIND_WHOLE, ActionValueSpec, clamp_to_spec

logger = logging.getLogger(__name__)

# Log at most one clamp warning per action per this interval, so a fast wire
# feeding an out-of-range value can't flood the log on an unattended run.
_CLAMP_WARN_INTERVAL_S = 5.0

# Default whole-number ceilings inferred from a value op's wire width when no
# explicit declaration is present (preserves today's ranges).
_OP_DEFAULT_MAX = {"write_byte": 0xFF, "write_word": 0xFFFF}

# Standard settings injected per transport (the builder pre-fills these).
I2C_SETTINGS = [
    {"key": "i2c_bus", "label": "I2C Bus", "type": "int", "default": 1, "min": 0, "max": 1},
    {
        "key": "i2c_address",
        "label": "I2C Address",
        "type": "hex",
        "default": 0x48,
        "min": 0x03,
        "max": 0x77,
    },
]
GPIO_SETTINGS = [
    {"key": "pin", "label": "Pin", "type": "int", "default": 0, "min": 0, "max": 53},
]

_I2C_OPS = {"read_byte", "read_word", "write_byte", "write_word"}
_GPIO_OPS = {"set_high", "set_low", "read_digital", "read_analog", "write_pwm"}
# Ops whose value is supplied at runtime (from a Device Action arg).
WRITE_VALUE_OPS = {"write_byte", "write_word", "write_pwm"}
# I2C ops that read the cumulative revolution accumulator (no register needed).
REVOLUTION_OPS = {"read_revolutions", "read_angle", "read_total_counts", "reset_revolutions"}
# Extra settings added to an I2C device when revolution tracking is enabled.
REVOLUTION_SETTINGS = [
    {
        "key": "angle_register",
        "label": "Angle Register",
        "type": "hex",
        "default": 0x0E,
        "min": 0x00,
        "max": 0xFF,
    },
    {
        "key": "counts_per_turn",
        "label": "Counts/Turn",
        "type": "int",
        "default": 4096,
        "min": 2,
        "max": 65535,
    },
    {
        "key": "decimals",
        "label": "Rounding (decimals)",
        "type": "int",
        "default": 2,
        "min": 0,
        "max": 6,
    },
]
_REV_POLL_INTERVAL = 0.02  # seconds between angle samples


class DeclarativeDevice(BaseDevice):
    """Runtime interpreter for a ``.gdevice`` definition.

    Concrete per-definition subclasses are produced by ``build_device_class`` and
    carry the definition in ``_definition`` and its settings in ``SETTINGS_SCHEMA``.
    """

    _definition: dict = {}
    SETTINGS_SCHEMA: list = []

    def __init__(self, board, config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        defn = type(self)._definition
        self._transport = defn.get("transport", "i2c")
        self._actions_def = defn.get("actions", [])
        self._clamp_warn_at: dict[str, float] = {}
        s = config.settings
        # Resolve declared settings (schema default -> saved value).
        self._settings = {
            field["key"]: s.get(field["key"], field.get("default"))
            for field in type(self).SETTINGS_SCHEMA
        }
        self._bus = None  # smbus2 handle for i2c
        self._lock = asyncio.Lock()  # fallback when the board exposes no i2c_lock

        # Revolution tracking (I2C only): a background loop unwraps the angle
        # register into a cumulative signed count.
        self._track_revolutions = bool(defn.get("track_revolutions"))
        self._angle_register = int(self._settings.get("angle_register", 0x0E) or 0x0E)
        self._counts_per_turn = int(self._settings.get("counts_per_turn", 4096) or 4096)
        self._decimals = int(self._settings.get("decimals", 2) or 0)
        self._total_counts = 0.0
        self._last_raw: int | None = None
        self._poll_task: asyncio.Task | None = None

    # --- identity ---

    @property
    def device_type(self) -> str:
        return type(self)._definition.get("name", "CustomDevice")

    @property
    def required_pins(self) -> list[str]:
        return []

    @property
    def actions(self):
        return {a["name"]: self._make_action(a) for a in self._actions_def}

    def _make_action(self, action_def: dict):
        async def _run(*args):
            return await self._execute(action_def, args)

        return _run

    # --- lifecycle ---

    async def initialize(self) -> None:
        if self._transport == "i2c":
            await self._open_i2c()
            if self._track_revolutions:
                self._last_raw = await self._read_angle_raw()
                self._poll_task = asyncio.create_task(self._rev_poll_loop())
        elif self._transport == "gpio":
            await self._configure_gpio()
        self._initialized = True
        logger.info("DeclarativeDevice '%s' initialized (%s)", self.device_type, self._transport)

    async def shutdown(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._bus is not None:
            await asyncio.to_thread(self._bus.close)
            self._bus = None
        self._initialized = False

    async def _open_i2c(self) -> None:
        def _open():
            try:
                import smbus2
            except ImportError as e:
                raise RuntimeError(
                    "smbus2 not installed. Run: pip install 'GLIDER[i2c]' (or pip install smbus2)"
                ) from e
            return smbus2.SMBus(int(self._settings.get("i2c_bus", 1)))

        self._bus = await asyncio.to_thread(_open)

    async def _configure_gpio(self) -> None:
        """Infer pin direction from the device's ops and set the pin mode."""
        ops = {a.get("op") for a in self._actions_def}
        if ops & {"set_high", "set_low"}:
            mode, ptype = PinMode.OUTPUT, PinType.DIGITAL
        elif "write_pwm" in ops:
            mode, ptype = PinMode.OUTPUT, PinType.PWM
        elif "read_analog" in ops:
            mode, ptype = PinMode.INPUT, PinType.ANALOG
        else:  # read_digital (or nothing)
            mode, ptype = PinMode.INPUT, PinType.DIGITAL
        if self._board is not None:
            await self._board.set_pin_mode(self._pin(), mode, ptype)

    # --- dispatch ---

    # --- value semantics ---

    def _action_def(self, action_name: str) -> dict | None:
        return next((a for a in self._actions_def if a.get("name") == action_name), None)

    def _declared_value_spec(self, action_name: str) -> ActionValueSpec | None:
        """Build the spec from an action's optional ``value`` block, if present."""
        a = self._action_def(action_name)
        v = a.get("value") if a else None
        if not isinstance(v, dict):
            return None
        try:
            return ActionValueSpec(
                kind=str(v.get("kind", KIND_WHOLE)),
                min=int(v["min"]),
                max=int(v["max"]),
                step=int(v.get("step", 1)),
                unit=str(v.get("unit", "")),
                label=str(v.get("label", "")),
            )
        except (KeyError, TypeError, ValueError):
            # Malformed blocks are rejected at load by validate_definition; be
            # defensive at runtime and fall back rather than crash.
            return None

    def _default_value_spec(self, action_name: str) -> ActionValueSpec | None:
        """Op-inferred fallback = today's ranges when nothing is declared (D8)."""
        a = self._action_def(action_name)
        op = a.get("op") if a else None
        if op == "write_pwm":
            bits = getattr(getattr(self._board, "capabilities", None), "pwm_resolution", 8)
            return ActionValueSpec(KIND_WHOLE, 0, (1 << bits) - 1)
        if op in _OP_DEFAULT_MAX:
            return ActionValueSpec(KIND_WHOLE, 0, _OP_DEFAULT_MAX[op])
        return None

    def _warn_clamp(self, action_name: str, spec: ActionValueSpec) -> None:
        now = time.monotonic()
        if now - self._clamp_warn_at.get(action_name, 0.0) >= _CLAMP_WARN_INTERVAL_S:
            self._clamp_warn_at[action_name] = now
            logger.warning(
                "Clamped out-of-range value for %s.%s to [%d, %d]",
                self._name,
                action_name,
                spec.min,
                spec.max,
            )

    async def _execute(self, action_def: dict, runtime_args: tuple):
        op = action_def.get("op")
        kwargs = dict(action_def.get("params", {}))
        for i, arg_name in enumerate(action_def.get("runtime_args", [])):
            if i < len(runtime_args) and runtime_args[i] is not None:
                kwargs[arg_name] = runtime_args[i]
        # Clamp a value-carrying op to its declared range BEFORE dispatch, so an
        # over-range value can never wrap (the old byte/word masks) or pass raw
        # (pwm). clamp_to_spec raises on NaN/inf/non-numeric, rejecting the write.
        if op in WRITE_VALUE_OPS and "value" in kwargs:
            spec = self.value_spec(action_def.get("name", ""))
            if spec is not None:
                clamped, did_clamp = clamp_to_spec(kwargs["value"], spec)
                kwargs["value"] = clamped
                if did_clamp:
                    self._warn_clamp(action_def.get("name", ""), spec)
        handler = getattr(self, f"_op_{op}", None)
        if handler is None:
            raise ValueError(f"Unknown op '{op}' for device '{self.device_type}'")
        return await handler(**kwargs)

    async def read(self):
        """Run the action flagged ``primary`` (or the first action)."""
        if not self._actions_def:
            return None
        primary = next((a for a in self._actions_def if a.get("primary")), self._actions_def[0])
        return await self._execute(primary, ())

    # --- i2c ops ---

    async def _i2c_call(self, method: str, *args):
        if self._bus is None:
            raise RuntimeError(f"Device '{self.device_type}' not initialized")
        addr = int(self._settings.get("i2c_address", 0x48))
        fn = getattr(self._bus, method)
        lock = getattr(self._board, "i2c_lock", None) or self._lock
        async with lock:
            return await asyncio.to_thread(fn, addr, *args)

    async def _op_read_byte(self, register):
        return await self._i2c_call("read_byte_data", int(register))

    async def _op_read_word(self, register):
        data = await self._i2c_call("read_i2c_block_data", int(register), 2)
        return (data[0] << 8) | data[1]  # big-endian

    async def _op_write_byte(self, register, value):
        # Clamp (never mask) to the wire width — a value in range for a wider
        # declaration than a byte is capped, not aliased to an unrelated number.
        return await self._i2c_call("write_byte_data", int(register), max(0, min(0xFF, int(value))))

    async def _op_write_word(self, register, value):
        v = max(0, min(0xFFFF, int(value)))
        return await self._i2c_call(
            "write_i2c_block_data", int(register), [(v >> 8) & 0xFF, v & 0xFF]
        )

    # --- gpio ops ---

    def _pin(self) -> int:
        return int(self._settings.get("pin", 0))

    async def _op_set_high(self):
        await self._board.write_digital(self._pin(), True)

    async def _op_set_low(self):
        await self._board.write_digital(self._pin(), False)

    async def _op_read_digital(self):
        return await self._board.read_digital(self._pin())

    async def _op_read_analog(self):
        return await self._board.read_analog(self._pin())

    async def _op_write_pwm(self, value):
        # Already clamped to the declared range in _execute; the board driver
        # applies the final hardware cap.
        await self._board.write_analog(self._pin(), int(value))

    # --- revolution tracking ---

    async def _read_angle_raw(self) -> int:
        data = await self._i2c_call("read_i2c_block_data", self._angle_register, 2)
        return ((data[0] << 8) | data[1]) & 0x0FFF  # AS5600-style 12-bit, big-endian

    async def _rev_poll_loop(self) -> None:
        errors = 0
        while True:
            await asyncio.sleep(_REV_POLL_INTERVAL)
            try:
                self._accumulate(await self._read_angle_raw())
                errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # transient I2C hiccup
                errors += 1
                if errors <= 3:
                    logger.warning("RotaryEncoder/declarative poll error: %s", e)

    def _accumulate(self, raw: int) -> None:
        if self._last_raw is not None:
            delta = raw - self._last_raw
            half = self._counts_per_turn / 2
            if delta > half:
                delta -= self._counts_per_turn
            elif delta < -half:
                delta += self._counts_per_turn
            self._total_counts += delta
        self._last_raw = raw

    async def _op_read_revolutions(self):
        return round(self._total_counts / self._counts_per_turn, self._decimals)

    async def _op_read_angle(self):
        return int(self._last_raw or 0)

    async def _op_read_total_counts(self):
        return int(self._total_counts)

    async def _op_reset_revolutions(self):
        self._total_counts = 0.0
        return 0.0

    @classmethod
    def from_dict(cls, data: dict, board) -> "DeclarativeDevice":
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


def standard_settings(transport: str) -> list[dict]:
    """Transport-standard settings the builder injects into a definition."""
    if transport == "gpio":
        return [dict(f) for f in GPIO_SETTINGS]
    return [dict(f) for f in I2C_SETTINGS]


def revolution_settings() -> list[dict]:
    """Extra settings for an I2C device with revolution tracking enabled."""
    return [dict(f) for f in REVOLUTION_SETTINGS]


def validate_definition(definition: dict) -> list[str]:
    """Return a list of problems with a definition (empty if valid)."""
    errors = []
    name = definition.get("name", "")
    if not name or not str(name).strip():
        errors.append("Device needs a name")
    transport = definition.get("transport")
    if transport not in ("i2c", "gpio"):
        errors.append(f"Unknown transport: {transport!r}")
    actions = definition.get("actions") or []
    if not actions:
        errors.append("Device needs at least one action")
    valid_ops = _I2C_OPS if transport == "i2c" else _GPIO_OPS
    if transport == "i2c" and definition.get("track_revolutions"):
        valid_ops = valid_ops | REVOLUTION_OPS
    seen = set()
    for a in actions:
        an = a.get("name", "")
        if not an:
            errors.append("Every action needs a name")
        elif an in seen:
            errors.append(f"Duplicate action name: {an}")
        seen.add(an)
        if a.get("op") not in valid_ops:
            errors.append(f"Action '{an}': op {a.get('op')!r} not valid for {transport}")
        errors.extend(f"Action '{an}': {e}" for e in _value_block_errors(a))
    return errors


def _value_block_errors(action: dict) -> list[str]:
    """Validate an action's optional ``value`` declaration (empty if absent/valid)."""
    value = action.get("value")
    if value is None:
        return []
    if not isinstance(value, dict):
        return ["value must be an object with kind/min/max"]
    # A value range only makes sense on a single-argument value-writing op —
    # not on a no-value op (set_high) or a multi-argument action.
    if action.get("op") not in WRITE_VALUE_OPS:
        return ["a value range can only be set on a value-writing op"]
    if len(action.get("runtime_args", [])) > 1:
        return ["value ranges are not supported on multi-argument actions"]
    for key in ("min", "max"):
        if key not in value:
            return [f"value missing '{key}'"]
    try:
        spec = ActionValueSpec(
            kind=str(value.get("kind", KIND_WHOLE)),
            min=int(value["min"]),
            max=int(value["max"]),
            step=int(value.get("step", 1)),
            unit=str(value.get("unit", "")),
            label=str(value.get("label", "")),
        )
    except (TypeError, ValueError):
        return ["value min/max/step must be whole numbers"]
    return spec.validate()


def build_device_class(definition: dict) -> type:
    """Create a concrete BaseDevice subclass from a declarative definition."""
    name = definition["name"]
    attrs = {
        "_definition": definition,
        "SETTINGS_SCHEMA": definition.get("settings", []),
        "__doc__": definition.get("description", "") or f"Custom device '{name}'",
    }
    return type(name, (DeclarativeDevice,), attrs)
