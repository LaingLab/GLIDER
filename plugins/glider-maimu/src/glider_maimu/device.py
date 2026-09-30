"""Maimu BLE stimulator.

The Maimu is a Nordic/Zephyr Bluetooth LE peripheral with exactly one writable
GATT characteristic that accepts three UTF-8 commands::

    on                                                       turn on and stay on
    off                                                      turn off
    <period_ms>,<pulse_width_ms>,<count>,<intensity_pct>     deliver count pulses

Running one through the generic ``BLEWrite`` / ``BLE`` device works, but means
pasting both UUIDs into the Add Device dialog and then remembering that the
action is called ``write`` and that a pulse is spelled ``"50,4,4,100"``. This device
bakes the protocol in: the UUIDs are defaults and the commands are named actions
(``on`` / ``off`` / ``pulse``).

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

The same reasoning covers a *dropped* link: :meth:`MaimuDevice._on_reconnected`
writes ``off`` when the automatic reconnect succeeds, so a stimulator that came
back mid-pattern is put in a known state instead of left running one nobody
asked for.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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

    # What the control panels put in front of a pulse button. The bounds are the
    # firmware's bounds (src/train.h), so a value the spin box offers is always a
    # value the parser accepts. The defaults match MaimuNode's, so the number a
    # researcher sees does not change when they move between the graph and the
    # panel.
    ACTION_ARGS_SCHEMA = {
        "pulse": [
            {
                "key": "period_ms",
                "label": "Period (ms)",
                "type": "int",
                "default": 50,
                "min": 1,
                "max": 3_600_000,
                "help": (
                    "Full cycle period in milliseconds -- a period, not a "
                    "frequency. 50 ms is 20 Hz."
                ),
            },
            {
                "key": "pulse_width_ms",
                "label": "Pulse width (ms)",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 3_600_000,
                "help": (
                    "On-time within each cycle; cannot exceed the period. Equal "
                    "to the period means continuous light, which needs Pulses "
                    "set to 0."
                ),
            },
            {
                "key": "count",
                "label": "Pulses",
                "type": "int",
                "default": 20,
                "min": 0,
                "max": 65_535,
                "help": "How many pulses to deliver. 0 runs until stopped.",
            },
            {
                "key": "intensity_pct",
                "label": "Intensity (%)",
                "type": "int",
                "default": 100,
                "min": 0,
                "max": 100,
                "help": (
                    "Relative light output, not calibrated optical power. "
                    "Per-unit brightness is trimmed in firmware."
                ),
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
            "write": self.write,
        }

    # --- actions ---

    async def on(self) -> None:
        """Turn the stimulator on and leave it on."""
        await self.write("on")

    async def off(self) -> None:
        """Turn the stimulator off."""
        await self.write("off")

    async def pulse(
        self,
        period_ms: Any,
        pulse_width_ms: Any,
        count: Any,
        intensity_pct: Any,
    ) -> None:
        """Deliver ``count`` pulses of ``pulse_width_ms`` every ``period_ms``.

        e.g. ``pulse(50, 4, 4, 100)`` writes ``"50,4,4,100"`` -- four 4 ms pulses
        at 20 Hz, full intensity. The firmware runs the train itself and stops on
        its own, so this returns as soon as the write lands.

        ``count`` of 0 runs until stopped. ``pulse_width_ms`` equal to
        ``period_ms`` is continuous light at that intensity, and is legal only
        with ``count`` of 0 -- continuous light has exactly one spelling.
        """
        period = self._whole_number(period_ms, "period_ms", 1, 3_600_000)
        width = self._whole_number(pulse_width_ms, "pulse_width_ms", 1, 3_600_000)
        pulses = self._whole_number(count, "count", 0, 65_535)
        intensity = self._whole_number(intensity_pct, "intensity_pct", 0, 100)

        # The firmware rejects this too, but silently -- it has no way to answer
        # back. Raising here is what turns "the device did nothing" into a
        # legible node error.
        if width > period:
            raise ValueError(
                f"Maimu.pulse: pulse_width_ms ({width}) cannot exceed "
                f"period_ms ({period}) -- the pulse would not fit in its cycle"
            )

        # width == period is continuous light, which must be spelled with
        # count = 0. With a non-zero count the train has no gap to end on, so
        # the firmware would run forever while the command reads as bounded --
        # an implanted LED left lit for the rest of the session. Nothing is
        # lost: a bounded single 500 ms pulse is pulse(1000, 500, 1, 40), where
        # count = 1 ends the train at the first pulse end so the trailing gap
        # never elapses.
        if width >= period and pulses != 0:
            raise ValueError(
                f"Maimu.pulse: pulse_width_ms ({width}) equal to period_ms "
                f"({period}) is continuous light, which must be spelled with "
                f"count = 0, not count = {pulses} -- a train with no gap has "
                f"nothing to end on. For one bounded pulse, make period_ms "
                f"longer than pulse_width_ms."
            )

        await self.write(period, width, pulses, intensity)

    @staticmethod
    def _whole_number(value: Any, field: str, minimum: int, maximum: int) -> int:
        """Coerce a pulse argument to a whole number within the firmware's bounds.

        The firmware's parser is strict: a fractional or non-numeric field is
        rejected outright and the command silently does nothing. Rejecting it
        here turns that into a legible node error instead.
        """
        try:
            number = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Maimu.pulse: {field} must be a number, got {value!r}") from e
        if not number.is_integer():
            raise ValueError(f"Maimu.pulse: {field} must be a whole number, got {value!r}")
        number = int(number)
        if number < minimum or number > maximum:
            raise ValueError(
                f"Maimu.pulse: {field} must be between {minimum} and {maximum}, got {value!r}"
            )
        return number

    # --- lifecycle ---

    async def _on_reconnected(self) -> None:
        """Come back off.

        The firmware runs a pulse autonomously, so a link that dropped
        mid-train left the stimulator running with nothing attached to stop
        it. Whatever it is doing, it has been doing it unsupervised, and this
        is the first moment anyone can say otherwise -- so the device is put
        in a known state rather than resumed in an unknown one.

        Same reasoning as :meth:`shutdown`, and the same best-effort
        treatment: BLEDevice logs a failure here and leaves the link up,
        because the link genuinely did reconnect.

        This and :meth:`shutdown` are independent safety paths that can both
        fire for the same event -- a reconnect landing just as a shutdown
        begins -- so ``off`` may be written twice in a row. That duplication
        is safety-neutral and deliberate; it is not a bug to collapse.
        """
        await self.write("off")

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
