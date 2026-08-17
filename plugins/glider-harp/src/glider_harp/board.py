"""Harp transport board.

Represents the host's connection to Harp hardware as a GLIDER "board". Like
:class:`~glider.hal.boards.serial_board.SerialBoard` and
:class:`~glider.hal.boards.ble_board.BLEBoard`, this is a *transport*, not a
pin-based board: a Harp device is addressed by register over its own serial
port, and each ``HarpDevice`` opens that port itself. The board exists so the
hardware panel has something to attach devices to. It therefore:

- verifies the ``pyserial`` **and** ``harp.protocol`` stacks are usable on
  connect()
- offers ``scan()`` to enumerate the host's serial ports (used by the hardware
  panel's "Scan" button)
- raises a clear error for any pin operation, since Harp has no GPIO

Why connect() checks two stacks rather than one, which is the only place this
differs materially from ``SerialBoard``: ``harp``'s ``Requires-Dist:
harp-protocol`` carries no lower bound, so a naive resolve installs an
incompatible ``harp-protocol`` 0.4.0 alongside a 0.5.x ``harp`` and reports
success. The two releases share no API. Left unchecked, that mis-resolve
surfaces much later as an ``AttributeError`` or a missing name deep inside a
register build, at which point nothing points at the real cause. So the check
imports a *name* rather than the module: ``harp.protocol`` itself imports fine
under either version, and only the name tells the two apart.

Nothing here opens a port, owns a device, or knows about registers -- that is
all ``HarpDevice``'s.
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


class HarpBoard(BaseBoard):
    """Host transport for Harp hardware. Each Harp device opens its own port."""

    def __init__(self, port: str | None = None, auto_reconnect: bool = False):
        # `port` is unused at the board level (kept for the BaseBoard /
        # driver-factory signature); each Harp device owns its own port.
        super().__init__(port, auto_reconnect)

    @property
    def board_type(self) -> str:
        return "harp"

    @property
    def name(self) -> str:
        return "Harp"

    @property
    def capabilities(self) -> BoardCapabilities:
        # No GPIO pins on a Harp transport.
        return BoardCapabilities(name="Harp", pins={})

    async def connect(self) -> bool:
        """Verify the Harp stack is usable and mark the transport ready."""
        try:
            import serial  # noqa: F401  (pyserial)
        except ImportError as e:
            self._set_state(BoardConnectionState.ERROR)
            raise RuntimeError(
                "pyserial not installed. Run: pip install pyserial (or reinstall GLIDER)."
            ) from e

        try:
            # A name, not the module: see the module docstring. `harp.protocol`
            # imports cleanly under the incompatible 0.4.0 that a naive resolve
            # pulls in, and only a name it does not define reveals which one is
            # actually installed.
            from harp.protocol import HarpMessage  # noqa: F401
        except ImportError as e:
            self._set_state(BoardConnectionState.ERROR)
            raise RuntimeError(
                f"The Harp protocol stack is not usable: {e}. "
                "Install a matched pair: pip install 'harp>=0.5.0rc1' "
                "'harp-protocol>=0.5.0rc1'. (harp's own requirement on "
                "harp-protocol has no lower bound, so a plain install can "
                "resolve to an incompatible 0.4.0 and report success.)"
            ) from e

        self._set_state(BoardConnectionState.CONNECTED)
        logger.info("HarpBoard: transport ready")
        return True

    async def disconnect(self) -> None:
        # Port connections live on the individual devices; nothing to do at the
        # transport level beyond updating state.
        self._set_state(BoardConnectionState.DISCONNECTED)
        logger.info("HarpBoard: transport released")

    @staticmethod
    async def scan() -> list[tuple[str, str]]:
        """Enumerate the host's serial ports.

        Returns a list of ``(description, device)`` tuples, where ``device`` is
        the port path/name to put in a device's ``port`` setting (e.g.
        ``"/dev/ttyUSB0"`` or ``"COM3"``) and ``description`` is a human-readable
        label (often the USB product string), falling back to the port itself
        when the OS reports none. Static so callers can enumerate the host
        without needing a board instance, mirroring ``SerialBoard.scan``.

        Label first, identifier second, matching ``SerialBoard`` and
        ``BLEBoard``: the hardware panel unpacks these as ``for nm, addr in
        results``, showing the first element and storing the second as the
        port. Both halves are strings, so the reversed order raises nothing --
        it displays the port as a label and writes the USB product string into
        the device's ``port`` setting, which then fails to open with a message
        about a port that is not a port.

        Every port is returned, not just the ones that look like Harp hardware:
        Harp boards enumerate as ordinary FTDI/CDC adapters with descriptions
        that are frequently generic, so filtering on the label would hide real
        devices with no way to get them back.
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
        logger.info("HarpBoard: scan found %d serial port(s)", len(results))
        return results

    # --- pin operations are not applicable to a Harp transport ---

    @staticmethod
    def _no_pins(op: str) -> None:
        raise NotImplementedError(
            f"HarpBoard has no GPIO pins ({op} is not supported). "
            "Harp hardware is addressed by register, not by pin -- use a Harp "
            "device to talk to it."
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
