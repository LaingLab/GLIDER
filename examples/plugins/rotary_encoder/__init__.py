"""
Rotary Encoder device plugin for GLIDER.

A self-contained example of a GLIDER **device plugin**: drop this folder into
``~/.glider/plugins/`` (and enable directory plugins in your config) and a new
``RotaryEncoder`` device type appears, fully usable from the hardware panel,
DeviceRead nodes, and the data recorder -- no changes to GLIDER core.

The device drives an AS5600 magnetic rotary encoder over I2C. It runs a small
background poll loop that unwraps the 12-bit raw angle (0..4095, which sawtooths
as the shaft turns) into a continuous, signed count, so it can report:

  - ``revolutions``  -- cumulative turns (signed), divided by ``gear_ratio``
  - ``angle``        -- current raw angle, 0..counts_per_turn-1
  - ``degrees``      -- current angle in degrees, 0..360
  - ``total_counts`` -- raw cumulative counts

plus a ``reset`` action to zero the count. Conversions (degrees, gear ratio)
and rounding (``decimals``) happen inside the device, so downstream nodes read
a ready-to-use number.

----------------------------------------------------------------------------
PLUGIN CONTRACT (what GLIDER looks for in this module):
  - DEVICE_TYPES: dict[str, type]  -- device types to register (required here)
  - SETTINGS_SCHEMA on the device class -- lets the hardware panel render a
    settings form automatically (see RotaryEncoderDevice.SETTINGS_SCHEMA)
  - setup() / teardown() -- optional lifecycle hooks
Other plugins may also expose BOARD_DRIVERS or NODE_TYPES the same way.
----------------------------------------------------------------------------
"""

import asyncio
import logging

from glider.hal.base_device import BaseDevice, DeviceConfig

logger = logging.getLogger(__name__)

# AS5600 raw-angle resolution (12-bit).
_RAW_MASK = 0x0FFF


class RotaryEncoderDevice(BaseDevice):
    """AS5600 rotary encoder: tracks cumulative revolutions over I2C."""

    # Declarative settings schema. The GLIDER hardware panel renders an Add
    # Device form from this automatically. Supported types: int, float, hex,
    # bool, str. Optional keys: min, max, decimals (for float), help.
    SETTINGS_SCHEMA = [
        {"key": "i2c_bus", "label": "I2C Bus", "type": "int", "default": 1, "min": 0, "max": 1},
        {
            "key": "i2c_address",
            "label": "I2C Address",
            "type": "hex",
            "default": 0x36,
            "min": 0x03,
            "max": 0x77,
            "help": "AS5600 default is 0x36",
        },
        {
            "key": "angle_register",
            "label": "Angle Register",
            "type": "hex",
            "default": 0x0E,
            "min": 0x00,
            "max": 0xFF,
            "help": "AS5600 RAW ANGLE high byte (0x0E)",
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
            "key": "gear_ratio",
            "label": "Gear Ratio",
            "type": "float",
            "default": 1.0,
            "min": 0.0001,
            "max": 100000.0,
            "decimals": 4,
            "help": "Output-shaft revs = encoder revs / gear_ratio",
        },
        {
            "key": "decimals",
            "label": "Rounding (decimals)",
            "type": "int",
            "default": 2,
            "min": 0,
            "max": 6,
        },
        {
            "key": "poll_interval",
            "label": "Poll Interval (s)",
            "type": "float",
            "default": 0.02,
            "min": 0.005,
            "max": 1.0,
            "decimals": 3,
        },
    ]

    def __init__(self, board, config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        s = config.settings
        self._bus_num = int(s.get("i2c_bus", 1))
        self._address = int(s.get("i2c_address", 0x36))
        self._angle_register = int(s.get("angle_register", 0x0E))
        self._counts_per_turn = int(s.get("counts_per_turn", 4096))
        self._gear_ratio = float(s.get("gear_ratio", 1.0)) or 1.0
        self._decimals = int(s.get("decimals", 2))
        self._poll_interval = float(s.get("poll_interval", 0.02))

        self._bus = None  # smbus2.SMBus handle, opened in initialize()
        self._lock = asyncio.Lock()  # fallback when the board exposes no i2c_lock
        self._poll_task: asyncio.Task | None = None
        self._last_raw: int | None = None  # previous raw angle (for unwrap)
        self._total_counts: float = 0.0  # signed cumulative counts

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "RotaryEncoder"

    @property
    def required_pins(self) -> list[str]:
        return []  # I2C: SDA/SCL are fixed; no GPIO pins allocated

    @property
    def actions(self):
        return {
            "angle": self.read_angle,
            "degrees": self.read_degrees,
            "revolutions": self.read_revolutions,
            "total_counts": self.read_total_counts,
            "reset": self.reset,
        }

    # --- lifecycle ---

    async def initialize(self) -> None:
        """Open the I2C bus, seed the angle, and start the poll loop."""

        def _open():
            try:
                import smbus2
            except ImportError as e:
                raise RuntimeError(
                    "smbus2 not installed. Run: pip install 'GLIDER[i2c]' (or pip install smbus2)"
                ) from e
            return smbus2.SMBus(self._bus_num)

        self._bus = await asyncio.to_thread(_open)
        self._last_raw = await self._read_raw()  # seed so the first delta is 0
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._initialized = True
        logger.info(
            "RotaryEncoder initialized on bus %d at 0x%02X (poll %.3fs)",
            self._bus_num,
            self._address,
            self._poll_interval,
        )

    async def shutdown(self) -> None:
        """Stop the poll loop and close the bus (safe before initialize)."""
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

    # --- polling / accumulation ---

    async def _poll_loop(self) -> None:
        """Continuously unwrap the raw angle into a cumulative count."""
        errors = 0
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                raw = await self._read_raw()
                self._accumulate(raw)
                errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # transient I2C hiccup -> log occasionally
                errors += 1
                if errors <= 3:
                    logger.warning("RotaryEncoder poll error: %s", e)

    async def _read_raw(self) -> int:
        """Read the current 12-bit raw angle under the board's I2C lock."""
        if self._bus is None:
            raise RuntimeError("RotaryEncoder not initialized")
        lock = getattr(self._board, "i2c_lock", None) or self._lock
        async with lock:
            data = await asyncio.to_thread(
                self._bus.read_i2c_block_data, self._address, self._angle_register, 2
            )
        return ((data[0] << 8) | data[1]) & _RAW_MASK

    def _accumulate(self, raw: int) -> None:
        """Add the shortest-path delta from the last reading (handles wrap)."""
        if self._last_raw is not None:
            delta = raw - self._last_raw
            half = self._counts_per_turn / 2
            if delta > half:
                delta -= self._counts_per_turn
            elif delta < -half:
                delta += self._counts_per_turn
            self._total_counts += delta
        self._last_raw = raw

    # --- reads / actions ---

    async def read(self) -> float:
        """Primary read: cumulative revolutions (the tracked number)."""
        return self.read_revolutions()

    def read_angle(self) -> int:
        """Current raw angle (0..counts_per_turn-1)."""
        return int(self._last_raw or 0)

    def read_degrees(self) -> float:
        """Current angle in degrees (0..360)."""
        angle = self._last_raw or 0
        return round(angle / self._counts_per_turn * 360.0, self._decimals)

    def read_revolutions(self) -> float:
        """Cumulative (signed) revolutions of the output shaft."""
        revs = self._total_counts / self._counts_per_turn / self._gear_ratio
        return round(revs, self._decimals)

    def read_total_counts(self) -> int:
        """Raw signed cumulative counts."""
        return int(self._total_counts)

    def reset(self) -> float:
        """Zero the cumulative count at the current position."""
        self._total_counts = 0.0
        return 0.0

    @classmethod
    def from_dict(cls, data, board) -> "RotaryEncoderDevice":
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


# --- plugin contract ---

DEVICE_TYPES = {"RotaryEncoder": RotaryEncoderDevice}


def setup() -> None:
    """Optional load hook. Registration happens via DEVICE_TYPES above."""
    logger.info("rotary_encoder plugin loaded")
