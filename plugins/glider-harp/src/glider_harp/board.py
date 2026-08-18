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
success. The two releases share no API. The check imports a *name* rather than
the module, because ``harp.protocol`` itself imports fine under either version
and only the name tells them apart.

**This guard is defence-in-depth, not the first line, and in a real
mis-resolve it will not be what fires.** ``frames`` does
``from harp.protocol import HarpMessage, HarpParseError`` at module level and
the package ``__init__`` imports ``frames``, so importing *anything* from
``glider_harp`` -- including this module, since a submodule import runs the
package ``__init__`` first -- already raises ``ImportError`` at ``frames.py``
before ``HarpBoard`` is so much as a name. The install message below is
therefore unlikely ever to reach an operator; the failure they will actually
see is the one from ``frames``. The guard stays because it is a cheap
assertion that costs nothing and stays correct if the import graph is ever
rearranged, but do not mistake it for the thing standing between a bad install
and a confusing error. Pinning the dependency at packaging time is what does
that.

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

    def report_transport_failure(self, error: Exception) -> None:
        """Record that a device's link to its hardware has broken.

        Harp devices own their own ports, so this board never touches the
        thing that fails and cannot notice a pulled cable itself. But the
        board's state is what the hardware panel shows, and its error
        callbacks are what ``HardwareManager`` wires its own listeners to, so
        the board is nonetheless where a broken link has to surface -- and
        without this a device whose reader thread died has nowhere at all to
        say so.

        ``ERROR`` rather than ``DISCONNECTED``: the transport was not shut
        down, it broke, and those want different responses from whoever is
        watching. Deliberately does *not* stop the recording -- a session that
        loses one device should come back with everything the others recorded,
        annotated, rather than not at all.
        """
        self._set_state(BoardConnectionState.ERROR)
        self._notify_error(error)
        logger.error("HarpBoard: transport failure reported: %s", error)

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
