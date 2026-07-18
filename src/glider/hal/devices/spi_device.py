"""Generic SPI device.

Talks to any SPI peripheral on a Raspberry Pi by bus + chip-select, without a
dedicated class per chip -- the SPI counterpart to ``GenericI2CDevice``. One
device instance binds one ``spidev.SpiDev`` handle (one bus + one CE line).
Every transfer runs in a worker thread under a per-device lock so it never
blocks the event loop and this device's transfers never interleave.

Covers real hardware-SPI parts: MCP3008/MCP3208 ADCs, ADS1256, SPI DACs,
digital pots, SPI motor drivers, etc. Not the HX711 (that is a bit-banged
custom two-wire protocol, not real SPI -- it has its own driver).

Pi/Linux only (``spidev`` imports Linux ``ioctl``); on a Mac/Windows host SPI
needs a USB-SPI bridge, which is out of scope here. ``spidev`` is lazy-imported
inside ``initialize()`` so importing this module is safe on any OS.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from glider.hal.base_device import BaseDevice, DeviceConfig

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard

logger = logging.getLogger(__name__)

# Words per transfer are one byte each; a value outside this range is a
# programming/config error, not a wire value to be silently masked.
_BYTE_MIN, _BYTE_MAX = 0x00, 0xFF
_MAX_TRANSFER = 4096  # generous cap; guards against a runaway readbytes(n)


class GenericSPIDevice(BaseDevice):
    """A hardware-SPI peripheral on one bus + chip-select.

    Settings:
    - spi_bus: SPI bus number (default 0 -> /dev/spidev0.*).
    - spi_device: chip-select / CE line (default 0 -> /dev/spidev0.0).
    - max_speed_hz: SCLK clock in Hz (default 500000).
    - spi_mode: clock polarity/phase 0-3 (default 0).
    """

    SETTINGS_SCHEMA = [
        {"key": "spi_bus", "label": "SPI Bus", "type": "int", "default": 0, "min": 0, "max": 8},
        {"key": "spi_device", "label": "Chip Select (CE)", "type": "int", "default": 0,
         "min": 0, "max": 8},
        {"key": "max_speed_hz", "label": "Clock (Hz)", "type": "int", "default": 500000,
         "min": 1, "max": 100_000_000},
        {"key": "spi_mode", "label": "SPI Mode", "type": "enum", "default": 0,
         "choices": [[0, "Mode 0"], [1, "Mode 1"], [2, "Mode 2"], [3, "Mode 3"]]},
    ]

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        parsed = self._parse_settings(config.settings)
        self._bus = parsed["spi_bus"]
        self._cs = parsed["spi_device"]
        self._max_speed_hz = parsed["max_speed_hz"]
        self._mode = parsed["spi_mode"]
        self._spi: Any = None  # spidev.SpiDev handle, opened in initialize()
        self._lock = asyncio.Lock()  # serializes this device's transfers

    # --- settings ---

    @staticmethod
    def _parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
        bus = int(settings.get("spi_bus", 0))
        if bus < 0:
            raise ValueError(f"spi_bus must be >= 0, got {bus}")
        cs = int(settings.get("spi_device", 0))
        if cs < 0:
            raise ValueError(f"spi_device must be >= 0, got {cs}")
        speed = int(settings.get("max_speed_hz", 500000))
        if speed <= 0:
            raise ValueError(f"max_speed_hz must be positive, got {speed}")
        mode = int(settings.get("spi_mode", 0))
        if mode not in (0, 1, 2, 3):
            raise ValueError(f"spi_mode must be 0-3, got {mode}")
        return {"spi_bus": bus, "spi_device": cs, "max_speed_hz": speed, "spi_mode": mode}

    def apply_settings(self, settings: dict[str, Any]) -> None:
        """Adopt edited settings (validated first). Connection params take effect
        on the next initialize()."""
        parsed = self._parse_settings({**self._config.settings, **settings})
        self._config.settings.update(settings)
        self._bus = parsed["spi_bus"]
        self._cs = parsed["spi_device"]
        self._max_speed_hz = parsed["max_speed_hz"]
        self._mode = parsed["spi_mode"]

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "GenericSPI"

    @property
    def required_pins(self) -> list[str]:
        # SPI: SCLK/MOSI/MISO/CE are fixed bus pins; none are allocated here.
        return []

    @property
    def spi_bus(self) -> int:
        """Configured SPI bus number."""
        return self._bus

    @property
    def spi_device(self) -> int:
        """Configured chip-select (CE) line."""
        return self._cs

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "transfer": self.transfer,
            "write": self.write,
            "read": self.read,
            "read_register": self.read_register,
        }

    # --- validation helpers ---

    @staticmethod
    def _to_byte_list(data: Any, label: str = "data") -> list[int]:
        """Coerce ``data`` to a list of 0-255 ints.

        Accepts a list/tuple of ints, a single int, or a string of
        comma/space-separated tokens (decimal or ``0x``-prefixed hex) -- so a
        Device Action node can pass ``"0x01,0x80"`` and a wired list both work.
        """
        if data is None:
            raise ValueError(f"{label} is required")
        if isinstance(data, str):
            tokens = [t for t in data.replace(",", " ").split() if t]
            items = [int(t, 0) for t in tokens]
        elif isinstance(data, (list, tuple)):
            items = [int(x) for x in data]
        else:
            items = [int(data)]
        if not items:
            raise ValueError(f"{label} is empty")
        if len(items) > _MAX_TRANSFER:
            raise ValueError(f"{label} too long ({len(items)} > {_MAX_TRANSFER})")
        for b in items:
            if b < _BYTE_MIN or b > _BYTE_MAX:
                raise ValueError(f"{label} byte {b} out of range (0x00-0xFF)")
        return items

    @classmethod
    def _validate_length(cls, n: Any) -> int:
        count = int(n)
        if count < 1 or count > _MAX_TRANSFER:
            raise ValueError(f"length {count} out of range (1-{_MAX_TRANSFER})")
        return count

    # --- lifecycle ---

    async def initialize(self) -> None:
        """Open the SPI handle via spidev (lazy-imported in a worker thread)."""

        def _open():
            try:
                import spidev
            except ImportError as e:
                raise RuntimeError(
                    "spidev not installed. Run: pip install 'GLIDER[spi]' "
                    "(or pip install spidev). SPI is Raspberry Pi / Linux only."
                ) from e
            spi = spidev.SpiDev()
            spi.open(self._bus, self._cs)
            spi.max_speed_hz = self._max_speed_hz
            spi.mode = self._mode
            return spi

        self._spi = await asyncio.to_thread(_open)
        self._initialized = True
        logger.info(
            "GenericSPI initialized on bus %d.%d (%d Hz, mode %d)",
            self._bus,
            self._cs,
            self._max_speed_hz,
            self._mode,
        )

    async def shutdown(self) -> None:
        """Close the SPI handle (safe before initialize).

        Clears ``_initialized`` FIRST, then closes inside try/finally so a
        failing ``close()`` still clears state rather than leaving the device
        looking usable with a dead handle.
        """
        self._initialized = False
        spi, self._spi = self._spi, None
        try:
            if spi is not None:
                try:
                    await asyncio.to_thread(spi.close)
                except Exception as e:  # close is best-effort
                    logger.warning("GenericSPI %s: error during close: %s", self._name, e)
        finally:
            self._initialized = False

    # --- transfer plumbing ---

    async def _call(self, method_name: str, *args) -> Any:
        if not self._initialized or self._spi is None:
            raise RuntimeError(f"GenericSPI {self._name} not initialized")
        fn = getattr(self._spi, method_name)
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    # --- actions ---

    async def transfer(self, data: Any = None) -> list[int]:
        """Full-duplex ``xfer2``: clock ``data`` out while reading the same
        number of bytes back. Returns the received bytes as a list of ints."""
        payload = self._to_byte_list(data)
        result = await self._call("xfer2", list(payload))
        return list(result)

    async def write(self, data: Any = None) -> None:
        """Write bytes with no read (``writebytes``)."""
        payload = self._to_byte_list(data)
        await self._call("writebytes", list(payload))

    async def read(self, length: Any = 1) -> list[int]:
        """Read ``length`` bytes with no write (``readbytes``)."""
        n = self._validate_length(length)
        result = await self._call("readbytes", n)
        return list(result)

    async def read_register(self, register: Any = None, length: Any = 1) -> list[int]:
        """Common register read: clock out ``register`` then ``length`` zero
        bytes, returning just the ``length`` bytes read after the command byte."""
        reg = self._to_byte_list(register, "register")
        n = self._validate_length(length)
        frame = list(reg) + [0x00] * n
        result = await self._call("xfer2", frame)
        return list(result[len(reg):])

    # --- serialization ---

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "GenericSPIDevice":
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance
