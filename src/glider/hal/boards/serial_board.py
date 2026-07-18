"""
Serial (UART) board driver.

Represents the host machine's serial/UART transport as a GLIDER "board". Like
:class:`~glider.hal.boards.ble_board.BLEBoard`, a serial adapter has no GPIO --
it is a transport over which one or more serial *devices* (e.g.
``GenericSerialDevice``) each open their own port. This board therefore:

- verifies the ``pyserial`` stack is importable on connect()
- offers ``scan()`` to enumerate the host's serial ports (used by the hardware
  panel's "Scan" button)
- raises a clear error for any pin operation, since a serial transport has no
  pins

Cross-platform via pyserial: Windows (COMx), macOS (/dev/cu.*), Linux
(/dev/ttyUSB*, /dev/ttyACM*). This is the one bus that is natively available on
every desktop OS, and once an OS-level Bluetooth-Classic (SPP/RFCOMM) pairing
exists the peripheral also shows up here as an ordinary serial port.
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


class SerialBoard(BaseBoard):
    """Host serial/UART transport. Peripherals connect per-device via pyserial."""

    def __init__(self, port: str | None = None, auto_reconnect: bool = False):
        # `port` is unused at the board level (kept for the BaseBoard /
        # driver-factory signature); each serial device owns its own port.
        super().__init__(port, auto_reconnect)

    @property
    def board_type(self) -> str:
        return "serial"

    @property
    def name(self) -> str:
        return "Serial (UART)"

    @property
    def capabilities(self) -> BoardCapabilities:
        # No GPIO pins on a serial transport.
        return BoardCapabilities(name="Serial (UART)", pins={})

    async def connect(self) -> bool:
        """Verify the pyserial stack is importable and mark the adapter ready."""
        try:
            import serial  # noqa: F401  (pyserial)
        except ImportError as e:
            self._set_state(BoardConnectionState.ERROR)
            raise RuntimeError(
                "pyserial not installed. Run: pip install pyserial (or reinstall GLIDER)."
            ) from e
        self._set_state(BoardConnectionState.CONNECTED)
        logger.info("SerialBoard: transport ready")
        return True

    async def disconnect(self) -> None:
        # Port connections live on the individual devices; nothing to do at the
        # transport level beyond updating state.
        self._set_state(BoardConnectionState.DISCONNECTED)
        logger.info("SerialBoard: transport released")

    @staticmethod
    async def scan() -> list[tuple[str, str]]:
        """Enumerate the host's serial ports.

        Returns a list of ``(description, device)`` tuples, where ``device`` is
        the port path/name to put in a device's ``port`` setting (e.g.
        ``"/dev/ttyUSB0"`` or ``"COM3"``) and ``description`` is a human-readable
        label (often the USB product string). Static so callers can enumerate
        the host without needing a board instance, mirroring ``BLEBoard.scan``.
        """
        import asyncio

        def _list() -> list[tuple[str, str]]:
            from serial.tools import list_ports

            results = []
            for p in list_ports.comports():
                label = (p.description or "").strip()
                results.append((label or p.device, p.device))
            return results

        # comports() does blocking I/O on some platforms; keep it off the loop.
        results = await asyncio.to_thread(_list)
        logger.info("SerialBoard: scan found %d serial port(s)", len(results))
        return results

    # --- pin operations are not applicable to a serial transport ---

    @staticmethod
    def _no_pins(op: str) -> None:
        raise NotImplementedError(
            f"SerialBoard has no GPIO pins ({op} is not supported). "
            "Use a serial device (e.g. GenericSerial) to talk to a peripheral."
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
