"""
BLE (Bluetooth Low Energy) board driver.

Represents the host machine's Bluetooth adapter as a GLIDER "board". Unlike
pin-based boards (Arduino, Pi GPIO), a BLE adapter has no GPIO -- it is a
transport over which one or more BLE peripheral *devices* (e.g. BLEWriteDevice)
each open their own connection. This board therefore:

- verifies the ``bleak`` stack is importable on connect()
- offers ``scan()`` to discover nearby peripherals (used by the hardware
  panel's "Scan" button)
- raises a clear error for any pin operation, since BLE has no pins

Cross-platform via bleak: Windows (WinRT), macOS (CoreBluetooth), Linux (BlueZ).
"""

import logging

from glider.hal.base_board import (
    BaseBoard,
    BoardCapabilities,
    BoardConnectionState,
    PinMode,
    PinType,
)

logger = logging.getLogger(__name__)


class BLEBoard(BaseBoard):
    """Host Bluetooth LE adapter. Peripherals connect per-device via bleak."""

    def __init__(self, port: str | None = None, auto_reconnect: bool = False):
        # `port` is unused for BLE (kept for the BaseBoard / driver-factory
        # signature); the adapter is implicit.
        super().__init__(port, auto_reconnect)

    @property
    def board_type(self) -> str:
        return "bluetooth"

    @property
    def name(self) -> str:
        return "Bluetooth LE"

    @property
    def capabilities(self) -> BoardCapabilities:
        # No GPIO pins on a BLE transport.
        return BoardCapabilities(name="Bluetooth LE", pins={})

    async def connect(self) -> bool:
        """Verify the bleak stack is importable and mark the adapter ready."""
        try:
            import bleak  # noqa: F401
        except ImportError as e:
            self._set_state(BoardConnectionState.ERROR)
            raise RuntimeError(
                "bleak not installed. Run: pip install bleak (or reinstall GLIDER)."
            ) from e
        self._set_state(BoardConnectionState.CONNECTED)
        logger.info("BLEBoard: adapter ready")
        return True

    async def disconnect(self) -> None:
        # Peripheral connections live on the individual devices; nothing to do
        # at the adapter level beyond updating state.
        self._set_state(BoardConnectionState.DISCONNECTED)
        logger.info("BLEBoard: adapter released")

    async def scan(self, timeout: float = 5.0) -> list[tuple[str, str]]:
        """Discover nearby BLE peripherals.

        Returns a list of ``(name, address)`` tuples; ``name`` falls back to
        ``"(unknown)"`` when the peripheral advertises none.
        """
        from bleak import BleakScanner

        devices = await BleakScanner.discover(timeout=timeout)
        results = [(getattr(d, "name", None) or "(unknown)", d.address) for d in devices]
        logger.info("BLEBoard: scan found %d peripheral(s)", len(results))
        return results

    # --- pin operations are not applicable to BLE ---

    @staticmethod
    def _no_pins(op: str) -> None:
        raise NotImplementedError(
            f"BLEBoard has no GPIO pins ({op} is not supported). "
            "Use a BLE device (e.g. BLEWrite) to talk to a peripheral."
        )

    async def set_pin_mode(
        self, pin: int, mode: PinMode, pin_type: PinType = PinType.DIGITAL
    ) -> None:
        self._no_pins("set_pin_mode")

    async def write_digital(self, pin: int, value: bool) -> None:
        self._no_pins("write_digital")

    async def read_digital(self, pin: int) -> bool:
        self._no_pins("read_digital")

    async def write_analog(self, pin: int, value: int) -> None:
        self._no_pins("write_analog")

    async def read_analog(self, pin: int) -> int:
        self._no_pins("read_analog")
