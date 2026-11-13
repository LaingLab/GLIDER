"""VapePod: the vape box side pod's Bluetooth controller.

The side pod carries a second pump and atomizer on the far side of the glass
box from the PiBox. Its own XIAO nRF52840 (``firmware/vapepod/vapepod.ino``)
switches them, so only a mains cord has to cross over. The XIAO is a BLE
peripheral with two characteristics:

- **command** (write): one UTF-8 command per write::

      off                     atomizer and pump off
      pump,<percent>          pump at 0-100 %; stays until changed
      puff,<ms>,<percent>     atomizer for ms milliseconds at percent, then off
      puff,0                  stop a puff early

- **status** (read, notify): ``atom=<percent> pump=<percent>``, pushed on every
  change and once a second.

Like :class:`glider_maimu.device.MaimuDevice` this subclasses
:class:`~glider.hal.devices.ble_device.BLEDevice` for its connection plumbing
(name-to-address resolution, write-mode detection, the supervised reconnect,
the shutdown race guard) and adds the protocol on top.

**The firmware enforces the safety rules, not this class.** Both outputs turn
off the moment the link drops, and the atomizer only ever runs for a stated
time of at most ``MAX_PUFF_MS``. A host can crash; the pod cannot be left
heating because of it.

**A dropped link** is handled per output. The pump is what the experiment asked
to be running, so the device remembers it and puts it back on reconnect, the way
the Maimu puts back its train. A puff is a dose, so it is never resumed or
deferred: one cut short by a drop is logged, and one asked for while the link is
down raises, so the flow does not carry on as though the animal had been dosed.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from glider.hal.base_board import ConnectionState
from glider.hal.base_device import DeviceConfig
from glider.hal.devices.ble_device import BLEDevice

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard

logger = logging.getLogger(__name__)

# The pod's GATT layout and advertised name, the same as in the firmware.
# Exposed as settings defaults so a second pod (renamed in its firmware) or a
# firmware revision needs a config edit, not a code change.
DEFAULT_NAME = "VapePod"
DEFAULT_SERVICE_UUID = "70c4957d-11e9-41a8-83fc-8783938e1e90"
DEFAULT_COMMAND_CHAR_UUID = "70c4957d-11e9-41a8-83fc-8783938e1e91"
DEFAULT_STATUS_CHAR_UUID = "70c4957d-11e9-41a8-83fc-8783938e1e92"

# The longest puff the firmware runs (MAX_PUFF_MS in vapepod.ino). It refuses a
# longer one, and a refusal is silent over BLE, so it is checked here as well:
# a legible node error beats a flow that thinks it dosed the animal.
MAX_PUFF_MS = 10_000

# How long the courtesy "off" on shutdown may take. HardwareManager bounds each
# device.shutdown() at DEVICE_IO_TIMEOUT_S (2.0s), so this has to leave room for
# the disconnect that follows it.
OFF_ON_SHUTDOWN_S = 0.75


# What the control panels put in front of the puff and pump buttons. The
# bounds are the ones whole_number enforces at call time; the puff's upper
# bound is the firmware's. Shared by both transports.
ACTION_ARGS_SCHEMA = {
    "puff": [
        {
            "key": "duration_ms",
            "label": "Duration (ms)",
            "type": "int",
            "default": 3000,
            "min": 1,
            "max": MAX_PUFF_MS,
            "help": (
                f"How long the atomizer runs, at most {MAX_PUFF_MS} ms. The pod "
                "times it and turns the atomizer off itself."
            ),
        },
        {
            "key": "power",
            "label": "Power (%)",
            "type": "int",
            "default": 100,
            "min": 1,
            "max": 100,
            "help": "Atomizer duty cycle. 100 is the full 12 V.",
        },
    ],
    "pump": [
        {
            "key": "percent",
            "label": "Speed (%)",
            "type": "int",
            "default": 100,
            "min": 0,
            "max": 100,
            "help": "0 is off. Below 100 the pump is switched at 1 kHz and whines.",
        },
    ],
}


def whole_number(value: Any, field: str, low: int, high: int) -> int:
    """Coerce an argument to a whole number in ``[low, high]``.

    The firmware refuses anything else, silently over BLE, so a bad value is
    turned into a legible node error here instead.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"VapePod: {field} must be a number, got {value!r}") from e
    if not number.is_integer():
        raise ValueError(f"VapePod: {field} must be a whole number, got {value!r}")
    number = int(number)
    if not low <= number <= high:
        raise ValueError(f"VapePod: {field} must be {low}-{high}, got {value!r}")
    return number


class VapePodDevice(BLEDevice):
    """The side pod: timed puffs from its atomizer, and its pump on or off.

    Settings:
    - address: peripheral address (MAC on Windows/Linux, UUID on macOS). Leave
        blank to find the pod by its advertised name.
    - name: advertised name; "VapePod" unless the firmware was changed.
    - write_char_uuid / read_char_uuid / service_uuid: pre-filled for the pod.
    """

    SETTINGS_SCHEMA = [
        {
            "key": "address",
            "label": "Address / UUID",
            # An editable combo with a Scan button: as a plugin this is the
            # only route to Scan (the panel special-cases the built-in BLE
            # devices by name).
            "type": "ble_address",
            "default": "",
            "help": "MAC (Windows/Linux) or UUID (macOS). Leave blank to find the pod by name.",
        },
        {
            "key": "name",
            "label": "Advertised name",
            "type": "str",
            "default": DEFAULT_NAME,
            "help": "DEVICE_NAME in the pod's firmware. Give each pod its own if you build two.",
        },
        {
            "key": "write_char_uuid",
            "label": "Command characteristic",
            "type": "str",
            "default": DEFAULT_COMMAND_CHAR_UUID,
            "help": "Advanced: pre-filled for the VapePod firmware.",
        },
        {
            "key": "read_char_uuid",
            "label": "Status characteristic",
            "type": "str",
            "default": DEFAULT_STATUS_CHAR_UUID,
            "help": "Advanced: pre-filled for the VapePod firmware.",
        },
        {
            "key": "service_uuid",
            "label": "Service UUID",
            "type": "str",
            "default": DEFAULT_SERVICE_UUID,
            "help": "Advanced: pre-filled for the VapePod firmware.",
        },
    ]

    ACTION_ARGS_SCHEMA = ACTION_ARGS_SCHEMA

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        # Filled in before BLEDevice.__init__ reads config.settings into its
        # caches, so a device added (or a file saved) without them still
        # connects. An explicit value always wins.
        config.settings.setdefault("name", DEFAULT_NAME)
        config.settings.setdefault("write_char_uuid", DEFAULT_COMMAND_CHAR_UUID)
        config.settings.setdefault("read_char_uuid", DEFAULT_STATUS_CHAR_UUID)
        config.settings.setdefault("service_uuid", DEFAULT_SERVICE_UUID)
        # Subscribed, so read() and the recorder see the pod's own account of
        # its outputs rather than what was last sent to it.
        config.settings.setdefault("notify", True)
        super().__init__(board, config, name)
        # The pump speed the experiment last asked for, put back on reconnect.
        self._pump_percent = 0
        # When the puff last sent ends, on the perf_counter clock; None when
        # none is running. Only used to report a puff a drop cut short.
        self._puff_ends_at: float | None = None
        # Serializes "remember, then send" against the reconnect hook, so the
        # hook can never send an older pump speed over a newer one.
        self._state_lock = asyncio.Lock()

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "VapePod"

    @property
    def actions(self) -> dict[str, Callable]:
        """The pod's command set. ``read`` returns its status text."""
        return {
            "puff": self.puff,
            "pump_on": self.pump_on,
            "pump_off": self.pump_off,
            "pump": self.pump,
            "off": self.off,
            "read": self.read,
            "write": self.write,
        }

    # --- actions ---

    async def puff(self, duration_ms: Any, power: Any = 100) -> None:
        """Run the atomizer for ``duration_ms`` at ``power`` %, then stop.

        The pod times the puff itself, so this returns as soon as the command
        lands. A puff sent while one is running replaces it from now.

        Raises if the link is down rather than deferring the puff: a dose that
        arrives seconds late, or not at all, must show up as an error.
        """
        ms = self._whole_number(duration_ms, "duration_ms", 1, MAX_PUFF_MS)
        percent = self._whole_number(power, "power", 1, 100)
        async with self._state_lock:
            if self._link_is_down():
                # Normally already running; this covers a drop no callback reported.
                self._start_reconnect()
                raise ConnectionError(f"VapePod {self._name}: link is down; puff not delivered")
            try:
                await self.write("puff", ms, percent, retry=False)
            except Exception:
                if self._link_went_down():
                    self._start_reconnect()
                raise
            self._puff_ends_at = time.perf_counter() + ms / 1000

    async def pump(self, percent: Any) -> None:
        """Run the pump at ``percent`` % (0 is off) until told otherwise."""
        await self._set_pump(self._whole_number(percent, "percent", 0, 100))

    async def pump_on(self) -> None:
        """Run the pump at full speed."""
        await self._set_pump(100)

    async def pump_off(self) -> None:
        """Stop the pump."""
        await self._set_pump(0)

    async def off(self) -> None:
        """Turn the atomizer and the pump off."""
        async with self._state_lock:
            self._pump_percent = 0
            self._puff_ends_at = None
            await self._send_or_defer(("off",))

    # --- remembered pump state ---

    async def _set_pump(self, percent: int) -> None:
        async with self._state_lock:
            self._pump_percent = percent
            await self._send_or_defer(("pump", percent))

    def _link_is_down(self) -> bool:
        return self._initialized and self.link_state in (
            ConnectionState.DISCONNECTED,
            ConnectionState.RECONNECTING,
        )

    def _link_went_down(self) -> bool:
        """After a failed write: did the link go with it?

        Asked of the client, not ``link_state``: the write can be the first to
        find out, before bleak's disconnect callback (if it fires at all).
        """
        client = self._client
        return self._initialized and (client is None or not client.is_connected)

    async def _send_or_defer(self, wire: tuple) -> None:
        """Send a pump or off command, or leave it to the reconnect.

        Called with the state already remembered. With the link down a write
        would hold the flow up behind a reconnect of several seconds; the pod
        switched itself off when the link went, and the reconnect sends the
        remembered pump speed, so returning at once loses nothing.
        """
        command = ",".join(str(a) for a in wire)
        if self._link_is_down():
            logger.warning(
                "VapePod %s: link is down; %s will be applied when it reconnects",
                self._name,
                command,
            )
            self._start_reconnect()
            return
        try:
            await self.write(*wire, retry=False)
        except Exception:
            if not self._link_went_down():
                raise  # the link is up (or we are shutting down): a real failure
            logger.warning(
                "VapePod %s: link dropped while sending %s; it will be applied when it "
                "reconnects",
                self._name,
                command,
            )
            self._start_reconnect()

    # Kept as a method: tests and subclasses call it through the class.
    _whole_number = staticmethod(whole_number)

    # --- lifecycle ---

    def _set_link(self, state: ConnectionState) -> None:
        """Report a puff that a lost link cut short, then move the link state.

        Every route to a lost link (bleak's callback, the poll backstop, the
        reconnect loop) passes through here, which is why the check lives here
        rather than in ``_on_disconnected``. Shutdown passes through too, but
        with ``_initialized`` already cleared: stopping a puff is its job.
        """
        if state is not ConnectionState.CONNECTED and self._puff_ends_at is not None:
            if self._initialized and time.perf_counter() < self._puff_ends_at:
                logger.warning(
                    "VapePod %s: link lost mid-puff; the pod stopped the atomizer early",
                    self._name,
                )
            self._puff_ends_at = None
        super()._set_link(state)

    async def _on_reconnected(self) -> None:
        """Put the pump back as the experiment last left it.

        The pod turned everything off when the link dropped. The pump goes back
        on at its remembered speed; otherwise ``off`` is written, so the pod is
        always left in a known state. A puff is never resumed.

        Held under the state lock so a command the flow sends at the same
        moment is never overwritten by an older one from here. Sent once, with
        no reconnect inside the write: if the link dies again first, the
        reconnect loop calls this again.
        """
        async with self._state_lock:
            percent = self._pump_percent
            if percent:
                logger.warning(
                    "VapePod %s: link restored; pump back on at %d%%", self._name, percent
                )
                await self.write("pump", percent, retry=False)
            else:
                await self.write("off", retry=False)

    async def shutdown(self) -> None:
        """Turn the pod off, then disconnect.

        The firmware turns everything off when the link drops anyway, so the
        ``off`` here is the belt to that braces: it makes emergency stop, End
        Experiment and app quit stop the pod at once, rather than after the
        link's supervision timeout. Best-effort and bounded, and written before
        ``super().shutdown()`` because that clears ``_initialized``, which
        ``write()`` needs.
        """
        # Forget the state first, so a reconnect landing during the shutdown
        # can only ever put the pod back to off.
        self._pump_percent = 0
        self._puff_ends_at = None
        try:
            if self._initialized and self._client is not None:
                try:
                    await asyncio.wait_for(self.write("off"), timeout=OFF_ON_SHUTDOWN_S)
                except Exception as e:  # noqa: BLE001 - teardown must not raise
                    logger.warning(
                        "VapePod %s: could not send 'off' before disconnect (%s); "
                        "it turns off when the link drops",
                        self._name,
                        e,
                    )
        finally:
            await super().shutdown()
