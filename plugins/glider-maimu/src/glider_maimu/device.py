"""Maimu BLE stimulator.

The Maimu is a Nordic/Zephyr Bluetooth LE peripheral with exactly one writable
GATT characteristic that accepts three UTF-8 commands::

    on                              turn on and stay on
    off                             turn off
    <period_ms>,<duration_s>        toggle every period_ms ms, for duration_s seconds
    p<on_ms>,<off_ms>,<duration_s>  on_ms on / off_ms off, for duration_s seconds
                                    (firmware 0.1.0+; older firmware ignores it)

Running one through the generic ``BLEWrite`` / ``BLE`` device works, but means
pasting both UUIDs into the Add Device dialog and then remembering that the
action is called ``write`` and that a pulse is spelled ``"500,10"``. This device
bakes the protocol in: the UUIDs are defaults and the commands are named actions
(``on`` / ``off`` / ``pulse`` / ``train``).

It subclasses :class:`~glider.hal.devices.ble_device.BLEDevice` rather than
``BaseDevice`` so it inherits that class's connection plumbing unchanged --
advertised-name-to-address resolution (so a ``.glider`` file opens on both macOS
and Windows), write-mode auto-detection from the characteristic's GATT
properties, reconnect-once-and-retry on a dropped link, and the shutdown race
guard that stops a retry from re-arming a device an emergency stop just stopped.

**Stopping matters here.** The firmware runs a pulse *autonomously*: dropping the
BLE link mid-pulse leaves the device stimulating with nothing connected to stop
it. So unlike the generic BLE devices, :meth:`MaimuDevice.shutdown` writes
``off`` before it disconnects -- which puts emergency stop, End Experiment and
app quit all on the same path.

A *dropped* link is different: the firmware turns the light off the moment the
link goes, but the experiment still wants whatever it last asked for. So the
device remembers its last command, and :meth:`MaimuDevice._on_reconnected`
puts it back the moment the link returns -- a train for whatever is left of its
duration, ``off`` if nothing was asked or the train has run out. Commands that
arrive while the link is down update that memory and return at once instead of
holding the flow up behind a reconnect; the newest one is what gets sent.
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

# The stimulator's GATT layout. Shared with the lab's optogenetic stimulator --
# same firmware family. Exposed as settings defaults (not constants baked into
# the writes) so a firmware revision that moves them needs a config edit, not a
# code change.
DEFAULT_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
DEFAULT_WRITE_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"

# How long the courtesy "off" on shutdown may take. HardwareManager bounds each
# device.shutdown() at DEVICE_IO_TIMEOUT_S (2.0s), so this has to leave room for
# the disconnect that follows it -- a wedged write must not starve the teardown.
OFF_ON_SHUTDOWN_S = 0.75


class MaimuDevice(BLEDevice):
    """A Maimu BLE stimulator: ``on``, ``off``, or a timed pulse.

    Settings:
    - address: peripheral address (MAC on Windows/Linux, UUID on macOS). Leave
        blank and set ``name`` for a file that opens on any host.
    - name: advertised local name; used to resolve ``address`` when it is blank.
    - write_char_uuid / service_uuid: pre-filled with the stimulator's UUIDs.
    """

    SETTINGS_SCHEMA = [
        {
            "key": "address",
            "label": "Address / UUID",
            # Renders as an editable combo with a Scan button. As a plugin this
            # is the only route to Scan: the hardware panel special-cases the
            # built-in BLE devices by name, which no plugin can be.
            "type": "ble_address",
            "default": "",
            "help": "MAC (Windows/Linux) or UUID (macOS). Leave blank and set Name for portability.",
        },
        {
            "key": "name",
            "label": "Advertised name",
            "type": "str",
            "default": "",
            "help": "Resolve the address by scanning for this name when Address is blank.",
        },
        {
            "key": "write_char_uuid",
            "label": "Command characteristic",
            "type": "str",
            "default": DEFAULT_WRITE_CHAR_UUID,
            "help": "Advanced: the writable command characteristic. Pre-filled for the Maimu.",
        },
        {
            "key": "service_uuid",
            "label": "Service UUID",
            "type": "str",
            "default": DEFAULT_SERVICE_UUID,
            "help": "Advanced: reference only. Pre-filled for the Maimu.",
        },
    ]

    # What the control panels put in front of a pulse button. The bounds are
    # the same contract _whole_number enforces at call time -- whole numbers,
    # at least 1 -- because the firmware atoi()s both fields, so a spin box
    # that offered 0 or a fraction would only produce a legible error later.
    # The defaults match MaimuNode's, so the number a researcher sees does not
    # change when they move between the graph and the panel.
    ACTION_ARGS_SCHEMA = {
        "pulse": [
            {
                "key": "period_ms",
                "label": "Period (ms)",
                "type": "int",
                "default": 500,
                "min": 1,
                "max": 3_600_000,
                "help": (
                    "On/off toggle period in milliseconds -- a period, not a "
                    "frequency. 500 ms toggles about once a second."
                ),
            },
            {
                "key": "duration_s",
                "label": "Duration (s)",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 86_400,
                "help": "How long the train runs. The firmware stops on its own.",
            },
        ],
        "train": [
            {
                "key": "on_ms",
                "label": "On (ms)",
                "type": "int",
                "default": 500,
                "min": 1,
                "max": 3_600_000,
                "help": "How long each pulse stays on. 20 Hz with 10 ms pulses: On 10, Off 40.",
            },
            {
                "key": "off_ms",
                "label": "Off (ms)",
                "type": "int",
                "default": 500,
                "min": 1,
                "max": 3_600_000,
                "help": "Gap between pulses. Unequal On/Off needs firmware 0.1.0+.",
            },
            {
                "key": "duration_s",
                "label": "Duration (s)",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 86_400,
                "help": "How long the train runs. The firmware stops on its own.",
            },
        ],
    }

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        # Fill the UUIDs in before BLEDevice.__init__ reads config.settings into
        # its caches, so a device added (or a file saved) without them still
        # connects. An explicit value always wins -- setdefault, not overwrite.
        config.settings.setdefault("write_char_uuid", DEFAULT_WRITE_CHAR_UUID)
        config.settings.setdefault("service_uuid", DEFAULT_SERVICE_UUID)
        super().__init__(board, config, name)
        # The last command the experiment gave, so a reconnect can put it back:
        # ("on",), ("off",) or ("train", on_ms, off_ms, duration_s, ends_at)
        # with ends_at on the perf_counter clock. None until the first command.
        self._desired: tuple | None = None
        # Serializes "remember a command, send it" against the reconnect hook,
        # so the hook can never send an older command over a newer one.
        self._desired_lock = asyncio.Lock()

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "Maimu"

    @property
    def actions(self) -> dict[str, Callable]:
        """The Maimu's command set.

        ``read`` is deliberately absent: the peripheral has no read or notify
        characteristic, so offering it would only let a Device Action node
        select an action that fails at runtime. ``write`` stays as an escape
        hatch for firmware commands this class does not model.
        """
        return {
            "on": self.on,
            "off": self.off,
            "pulse": self.pulse,
            "train": self.train,
            "write": self.write,
        }

    # --- actions ---

    async def on(self) -> None:
        """Turn the stimulator on and leave it on."""
        await self._command(("on",))

    async def off(self) -> None:
        """Turn the stimulator off."""
        await self._command(("off",))

    async def pulse(self, period_ms: Any, duration_s: Any) -> None:
        """Toggle every ``period_ms`` milliseconds for ``duration_s`` seconds.

        e.g. ``pulse(500, 10)`` writes ``"500,10"`` -- a 500 ms on/off period
        (~1 Hz) held for 10 seconds. The firmware runs the pattern itself and
        stops on its own, so this returns as soon as the write lands.

        Note ``period_ms`` is a *period in milliseconds*, not a frequency.
        """
        period = self._whole_number(period_ms, "period_ms")
        duration = self._whole_number(duration_s, "duration_s")
        await self._command(("train", period, period, duration, self._deadline(duration)))

    async def train(self, on_ms: Any, off_ms: Any, duration_s: Any) -> None:
        """Pulse ``on_ms`` on / ``off_ms`` off for ``duration_s`` seconds.

        e.g. ``train(10, 40, 10)`` writes ``"p10,40,10"`` -- 10 ms pulses at
        20 Hz for 10 seconds. Equal on and off times go out as the original
        ``<period_ms>,<duration_s>`` command, which every firmware understands.
        """
        on = self._whole_number(on_ms, "on_ms")
        off = self._whole_number(off_ms, "off_ms")
        duration = self._whole_number(duration_s, "duration_s")
        await self._command(("train", on, off, duration, self._deadline(duration)))

    # --- remembered command ---

    @staticmethod
    def _deadline(duration_s: int) -> float:
        return time.perf_counter() + duration_s

    @staticmethod
    def _wire(desired: tuple, *, resume: bool = False) -> tuple:
        """The write() arguments for a remembered command.

        A first send uses the train's full duration. A resend after a reconnect
        uses what is left of it, rounded down so the train never outlasts the
        one that was asked for, and turns into ``off`` once under a second is
        left.
        """
        if desired[0] != "train":
            return desired
        _, on, off, duration, ends_at = desired
        if resume:
            left = ends_at - time.perf_counter()
            if left < 1:
                return ("off",)
            duration = int(left)
        # Two or three args -> BLEDevice.write comma-joins them: "500,10".
        return (on, duration) if on == off else (f"p{on}", off, duration)

    def _link_is_down(self) -> bool:
        return self._initialized and self.link_state in (
            ConnectionState.DISCONNECTED,
            ConnectionState.RECONNECTING,
        )

    async def _command(self, desired: tuple) -> None:
        """Remember ``desired``, then send it -- unless the link is down.

        With the link down a write would block the flow behind a reconnect of
        several seconds, and every command queued up that way used to land in a
        burst once it came back, oldest first. Remembering the newest and
        returning means the reconnect sends exactly one command: the current one.
        """
        async with self._desired_lock:
            self._desired = desired
            if self._link_is_down():
                logger.warning(
                    "Maimu %s: link is down; %s will be sent as soon as it reconnects",
                    self._name,
                    ",".join(str(a) for a in self._wire(desired)),
                )
                # Normally already running; this covers a drop no callback reported.
                self._start_reconnect()
                return
            await self.write(*self._wire(desired))

    @staticmethod
    def _whole_number(value: Any, field: str) -> int:
        """Coerce a pulse or train argument to a positive whole number.

        The firmware ``atoi``s both fields, so a fractional or non-numeric value
        would be silently truncated into a command that does something other
        than what the flow asked for. Rejecting it here turns that into a
        legible node error instead.
        """
        try:
            number = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Maimu: {field} must be a number, got {value!r}") from e
        if not number.is_integer():
            raise ValueError(f"Maimu: {field} must be a whole number, got {value!r}")
        number = int(number)
        if number < 1:
            raise ValueError(f"Maimu: {field} must be at least 1, got {value!r}")
        return number

    # --- lifecycle ---

    async def _on_reconnected(self) -> None:
        """Put back the last command the experiment gave.

        The firmware turns the light off when the link drops, so whatever the
        experiment asked for has been interrupted, not continued. A train that
        still has time left is resent for that remaining time; ``on`` is resent
        as ``on``; anything else -- ``off``, a train that ran out, or no command
        yet -- writes ``off``, so the device is always left in a known state.

        Held under the command lock so a command the flow sends at the same
        moment is never overwritten by an older one from here.

        Best-effort, like :meth:`shutdown`: BLEDevice logs a failure here and
        leaves the link up, because the link genuinely did reconnect. This and
        :meth:`shutdown` can both fire for the same event, so ``off`` may be
        written twice in a row; that is safety-neutral and deliberate.
        """
        async with self._desired_lock:
            desired = self._desired
            wire = self._wire(desired, resume=True) if desired is not None else ("off",)
            if wire != ("off",):
                logger.warning(
                    "Maimu %s: link restored; resuming %s",
                    self._name,
                    ",".join(str(a) for a in wire),
                )
            await self.write(*wire)

    async def shutdown(self) -> None:
        """Stop the stimulator, then disconnect.

        ``BaseDevice.shutdown`` is the safe-state hook -- it is what
        ``HardwareManager.emergency_stop`` calls, and what runs at End
        Experiment and app quit. The generic BLE devices only disconnect, which
        is not enough here: a pulse in flight is driven by the firmware, so
        dropping the link would leave the device stimulating.

        The ``off`` is best-effort and bounded. The write happens *before*
        ``super().shutdown()`` because that is what clears ``_initialized`` --
        and ``write()`` refuses to run once it is cleared. ``finally`` makes the
        disconnect unconditional, so a peripheral that is already gone still
        tears down cleanly.

        **Both ways of not sending it are reported.** There is no link to write
        over while a reconnect attempt is in flight (``_client`` is None for the
        length of it), so quitting or e-stopping right then skips the write
        entirely rather than failing it -- and a skip that says nothing leaves
        the operator with no indication a stimulator was left in an unknown
        state. That is the same news as a failed write, so it gets the same
        warning.
        """
        # Forget the last command first, so a reconnect landing during the
        # shutdown can only ever put the device back to off.
        self._desired = None
        try:
            if self._initialized and self._client is not None:
                try:
                    await asyncio.wait_for(self.write("off"), timeout=OFF_ON_SHUTDOWN_S)
                except Exception as e:  # noqa: BLE001 - teardown must not raise
                    # Includes the timeout. A cancelled write releases the lock
                    # on its way out, so the disconnect below still gets it.
                    logger.warning(
                        "Maimu %s: could not send 'off' before disconnect (%s); "
                        "the device may still be running",
                        self._name,
                        e,
                    )
            elif self._initialized:
                # Initialized but no client: mid-reconnect, or the link went
                # away and has not been rebuilt. A device that was never
                # initialized is silent here on purpose -- nothing has ever
                # commanded it, so there is nothing to warn about.
                logger.warning(
                    "Maimu %s: no link to send 'off' over at shutdown; "
                    "the device may still be running",
                    self._name,
                )
        finally:
            await super().shutdown()
