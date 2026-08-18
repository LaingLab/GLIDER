"""A Harp board as a GLIDER device: one serial port, one register cache.

Everything the other modules in this package build is assembled here.
``schema`` says what the device has, ``derivation`` says which of that the
experiment wants, ``reader`` drains the port into a ``RegisterCache``, and this
module owns the port and the order the three are driven in.

The lifecycle is not negotiable, and each step is here because leaving it out
fails silently:

1. Build the register classes (``build_registers``), which is where a schema
   fault raises instead of decoding every event of the session wrongly, and
   resolve the profile (``derive``) into columns and actions.
2. Open the port.
3. Read ``WhoAmI`` (register 0) and check it against the schema. A schema for
   the wrong board derives cleanly and records the wrong registers.
4. **Write ``OperationControl`` (register 10) to Active, and read it back** --
   but only when the profile asked for something to be recorded. A Harp device
   boots in **Standby**, where it answers commands but emits no events at all.
   Miss this and the device connects, answers, records nothing, and reports no
   error anywhere -- the recording simply comes back empty. A device with no
   profile is the one case that genuinely wants Standby: nothing would drain
   what Active makes it emit.
5. Build a ``RegisterCache`` and a fresh ``HarpReader``, and start it.

Step 1 is before step 2 rather than after it, which is the one place this
differs from the order the steps are numbered in elsewhere: everything in it
can fail (a moved ``device.yml``, a register type that will not build, a
profile naming a register the board does not have), and doing it first means
those failures cost nothing to recover from, because there is no handle open
yet. ``MockHarpDevice._open_port`` also reads the schema, so the dependency is
now real as well as convenient.

Steps 3 and 4 are register round-trips, and they are both before step 5 for the
reason ``HarpReader`` states: the reader consumes every byte the port produces
and hands only Events to the cache, so a reply arriving while it runs is
decoded, counted, and dropped. A round-trip during a run does not race -- it
simply never returns. That is also why a *read* action refuses while the reader
is alive, and why a *write* action does not wait for its echo.

``shutdown`` runs the same order backwards -- ``reader.stop()``, then Standby,
then ``close()`` -- and takes ``stop()``'s return value seriously: ``False``
means the thread is still reading the port, so neither the register write nor
the close may proceed.

What the recorder sees:

* ``state_columns()`` -- the cache's columns, or ``None`` when the device has
  no profile and records nothing.
* ``get_state()`` -- ``cache.snapshot()``. The single consuming read, for
  whoever writes the CSV.
* ``read()`` -- ``cache.peek()``. Non-consuming, because ``WaitForInput`` and
  the Input node both prefer ``read()`` and one of them polls at 50 ms.

Both of those reads are also where a link that broke mid-session is noticed.
The reader thread records why it stopped and exits, and nothing above it would
otherwise look: the cache simply stops changing, so the CSV carries the last
value with a count of zero for the rest of the run and is indistinguishable
from a subject that went quiet. See ``_check_link``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from glider.hal.base_device import BaseDevice, DeviceConfig
from glider.hal.value_spec import KIND_WHOLE, ActionValueSpec
from glider_harp.derivation import PROFILE_DIR, Derived, derive, load_profile
from glider_harp.frames import (
    MESSAGE_READ,
    MESSAGE_WRITE,
    FrameError,
    FrameSplitter,
    HarpFrame,
    decode,
    encode,
)
from glider_harp.reader import HarpReader, RegisterCache
from glider_harp.schema import build_registers, load_schema

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard

logger = logging.getLogger(__name__)

# Core register addresses, fixed by the Harp specification rather than by any
# one device's schema -- so they are reachable even on a board whose schema
# omits them, which hand-written schemas routinely do.
WHO_AM_I = 0
OPERATION_CONTROL = 10

# ...and their payload types, likewise fixed by the specification.
_WHO_AM_I_TYPE = "U16"
_OPERATION_CONTROL_TYPE = "U8"

# OperationControl's low two bits select the operating mode; the rest are
# independent flags (heartbeat, LEDs, dump-on-connect) that belong to the
# device's own configuration. Both mode changes here are read-modify-write so
# those flags survive: an operator watching the board's operation LED should
# not lose it because GLIDER connected.
_OPERATION_MODE_MASK = 0x03
_MODE_STANDBY = 0x00
_MODE_ACTIVE = 0x01
_MODE_NAMES = {_MODE_STANDBY: "Standby", _MODE_ACTIVE: "Active"}

# How long a register round-trip waits for its reply, and how long one read
# inside it may block. A Harp device answers in well under a millisecond; the
# second is only there so the loop can notice its own deadline.
ROUND_TRIP_TIMEOUT_S = 1.0
ROUND_TRIP_READ_TIMEOUT_S = 0.05

# The same wait, on the way out, and much shorter. ``HardwareManager`` bounds
# ``device.shutdown()`` at 2 s, of which the reader's join may already have
# taken most; two full-length round-trips on top of it overrun the budget, and
# the caller's ``wait_for`` then cancels shutdown part-way -- historically
# before the port was closed. A board that is powered off but still enumerated
# burns exactly this path. Returning the device to Standby is a courtesy;
# releasing the port is not, so the courtesy gets the small budget.
SHUTDOWN_ROUND_TRIP_TIMEOUT_S = 0.25

_DEFAULT_BAUDRATE = 115200

# Widest bound a declared ``ActionValueSpec`` will carry. A U32 or U64
# register's true range is meaningless to the controls that read this -- the
# runner builds a Qt slider, whose bounds are 32-bit -- so the *declared*
# range is clamped here while ``_pack`` stays the authority on what actually
# fits the register and raises on anything that does not. Clamping the
# display bound cannot admit a bad write; widening it past what Qt can hold
# would produce a control that silently misbehaves.
_MAX_SPEC_VALUE = (1 << 31) - 1
_MIN_SPEC_VALUE = -(1 << 31)


def _profile_choices() -> list[list[str]]:
    """Every profile shipped in the package, for the hardware panel's dropdown.

    Free text here is a setting whose only failure mode is a typo discovered
    at initialize() time, on a bench, with the animal already in the rig. The
    list is short, fixed at install time, and enumerable -- so enumerate it.
    The empty choice is a real one: no profile means record nothing, which is
    ``derive``'s documented default rather than an unset field.
    """
    names = sorted(path.stem for path in PROFILE_DIR.glob("*.json"))
    return [["", "None (record nothing)"], *([name, name] for name in names)]


class HarpDevice(BaseDevice):
    """One Harp board on one host serial port.

    Settings:

    - ``port``: serial port path/name (``/dev/ttyUSB0``, ``COM3``). Required.
    - ``baudrate``: bits per second (default 115200).
    - ``device_yml``: path to the board's vendor ``device.yml``. Required --
      it is what says which registers exist, how wide they are, and which of
      them can be written. Nothing here guesses it.
    - ``profile``: name of a profile shipped inside this package (see
      ``glider_harp/profiles``), naming the registers worth recording. A
      **name**, never a path: a device setting that could name any file on
      disk is a setting that can read one we never shipped. Optional --
      without it the device records nothing and still exposes every writable
      register as an action, which is ``derive``'s documented rule.
    """

    # Rendered by the hardware panel's schema form when adding/editing a device.
    SETTINGS_SCHEMA = [
        {
            "key": "port",
            "label": "Port",
            "type": "str",
            "default": "",
            "help": "Serial port, e.g. /dev/ttyUSB0 or COM3. Use Scan on the board to list ports.",
        },
        {
            "key": "baudrate",
            "label": "Baud rate",
            "type": "enum",
            "default": _DEFAULT_BAUDRATE,
            "choices": [[115200, "115200"], [1000000, "1000000"]],
        },
        {
            "key": "device_yml",
            "label": "Device schema (device.yml)",
            "type": "str",
            "default": "",
            "help": "Path to the board's Harp device.yml, which lists its registers.",
        },
        {
            "key": "profile",
            "label": "Recording profile",
            "type": "enum",
            "default": "",
            "choices": _profile_choices(),
            "help": "Which registers to record. Without one the device records nothing.",
        },
    ]

    def __init__(self, board: BaseBoard, config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        parsed = self._parse_settings(config.settings)
        self._port = parsed["port"]
        self._baudrate = parsed["baudrate"]
        self._device_yml = parsed["device_yml"]
        self._profile_name = parsed["profile"]

        self._serial: Any = None  # serial.Serial handle, opened in initialize()
        self._reader: HarpReader | None = None
        self._cache: RegisterCache | None = None
        self._schema: dict[str, Any] = {}
        self._registers: dict[str, Any] = {}
        self._derived = Derived()
        self._actions: dict[str, Callable] = {}
        self._who_am_i: int | None = None
        # Warnings about *this run* rather than about the configuration, and
        # latched: a link that broke must be reported once, not once per row.
        self._runtime_warnings: list[str] = []
        self._link_failed = False
        # Whether the schema/profile have been resolved at least once. The node
        # editor asks for ``actions`` long before any hardware exists, so the
        # derivation is loaded lazily there and strictly in initialize().
        self._derivation_loaded = False
        # Serializes everything that touches the handle -- round-trips, writes,
        # and shutdown's close -- so the port is never closed out from under an
        # in-flight action (emergency stop bypasses the command lock).
        self._port_lock = asyncio.Lock()

    # --- settings ---

    @staticmethod
    def _parse_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a settings dict; the single place these are interpreted."""
        port = str(settings.get("port", "")).strip()
        baudrate = int(settings.get("baudrate", _DEFAULT_BAUDRATE))
        if baudrate <= 0:
            raise ValueError(f"baudrate must be positive, got {baudrate}")
        device_yml = str(settings.get("device_yml", "")).strip()
        profile = str(settings.get("profile", "")).strip()
        return {
            "port": port,
            "baudrate": baudrate,
            "device_yml": device_yml,
            "profile": profile,
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        """Adopt edited settings (validated against the merged result first).

        Every setting here decides how the port was opened or what the running
        reader is filling, none of which can change under a live handle, so
        while the port is held the edit is recorded for the saved file and
        takes effect on the next initialize(). Before that the caches are
        refreshed immediately. Mirrors ``GenericSerialDevice``.

        Keyed on the port still being held, exactly as ``initialize()`` is,
        rather than on ``_initialized``. The two differ in one reachable case:
        a shutdown whose join was refused clears ``_initialized`` while the
        reader thread still owns the handle. Keyed on the flag, an edit there
        would rewrite ``_port`` to name a device this one is still reading.
        """
        parsed = self._parse_settings({**self._config.settings, **settings})
        self._config.settings.update(settings)
        if self._serial is not None or self._reader is not None:
            logger.info("Harp %s: settings saved; reconnect to apply", self._name)
            return
        self._port = parsed["port"]
        self._baudrate = parsed["baudrate"]
        self._device_yml = parsed["device_yml"]
        self._profile_name = parsed["profile"]
        # The schema or profile may have changed, so what was derived from the
        # old ones is no longer an answer to anything.
        self._derivation_loaded = False

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "Harp"

    @property
    def required_pins(self) -> list[str]:
        # Harp is addressed by register, not by pin; nothing is allocated.
        return []

    @property
    def port(self) -> str:
        """Configured serial port path/name."""
        return self._port

    @property
    def baudrate(self) -> int:
        """Configured baud rate."""
        return self._baudrate

    @property
    def who_am_i(self) -> int | None:
        """What the connected device answered for register 0, once asked."""
        return self._who_am_i

    @property
    def reader(self) -> HarpReader | None:
        """The running reader, or None. Its counters are the link's health."""
        return self._reader

    @property
    def actions(self) -> dict[str, Callable]:
        """One callable per readable/writable register, as ``derive`` decided.

        Available before the device is initialized, because the node editor
        offers these while the hardware is still in a box. A derivation that
        cannot be loaded there is logged and yields no actions rather than
        raising: an unopenable ``device.yml`` should not make a saved
        experiment unopenable too. initialize() loads it strictly.
        """
        self._ensure_derivation()
        return dict(self._actions)

    def _declared_value_spec(self, action_name: str) -> ActionValueSpec | None:
        """What an action's value means, or ``None`` for one that carries none.

        This is the hook every layer already asks -- node property editors,
        the runner's generated controls, write-time clamping -- and answering
        it is what makes a Harp register usable from the GUI at all. Without
        it ``DeviceControlsPanel`` finds no spec for any action, renders every
        one as a bare button, and invokes each with no value: every button on
        the panel becomes a *read*, which is wrong for every writable register
        and impossible for most of them.

        So a **writable** register declares a spec, and the register's own
        width supplies the range; a **read-only** register declares none,
        which is exactly how the runner learns to draw it as a button that
        reads. The two halves of the answer come from ``Derived.access``,
        which is why that had to exist.
        """
        self._ensure_derivation()
        if "Write" not in self._derived.access.get(action_name, frozenset()):
            return None
        built = self._registers.get(action_name)
        if built is None:
            return None
        name = str(built.payload_type.name)
        if name == "Float":
            # No fractional value kind exists yet (see value_spec), and a
            # float register cannot be written at all -- see _pack.
            return None
        bits = built.payload_type.numpy_dtype.itemsize * 8
        if name.startswith("S"):
            low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        else:
            low, high = 0, (1 << bits) - 1
        return ActionValueSpec(
            KIND_WHOLE,
            max(low, _MIN_SPEC_VALUE),
            min(high, _MAX_SPEC_VALUE),
            label=action_name,
        )

    def recording_warnings(self) -> list[str]:
        """Ways this device's columns will say less than they look like they do.

        Written into the CSV metadata block by ``DataRecorder``. Reported by
        ``derive``, which owns the predicate and the wording; this only passes
        them on. A second copy of the predicate here was the previous shape,
        and it got the list-of-access-modes case wrong.

        Two kinds, and the second is why the recorder reads this again at
        stop() as well as at start(): what the *configuration* cannot report
        is known before a row is written, while a link that broke mid-run is
        only known afterwards. See ``_check_link``.
        """
        self._ensure_derivation()
        return [*self._derived.warnings, *self._runtime_warnings]

    def _ensure_derivation(self) -> None:
        """Resolve the schema and profile once, tolerating failure.

        For the callers that run outside ``initialize()`` -- the node editor
        asking what this device can do, the recorder asking what it has to warn
        about. A derivation that cannot be loaded there is logged and yields
        nothing rather than raising: an unopenable ``device.yml`` should not
        make a saved experiment unopenable too. ``initialize()`` calls
        ``_load_derivation`` directly, where every one of those failures is an
        error.
        """
        if self._derivation_loaded:
            return
        try:
            self._load_derivation()
        except Exception:
            logger.exception(
                "Harp %s: could not read the schema at %s", self._name, self._device_yml
            )
            # Marked loaded so a failing file read is not repeated on every
            # repaint of the node editor.
            self._derivation_loaded = True

    # --- what the recorder reads ---

    def state_columns(self) -> list[str] | None:
        """The columns the profile asks for, or None when it asks for none.

        Answered from the profile rather than from the live cache, so a device
        whose initialize() failed still contributes the columns it was
        configured for -- empty, since ``get_state()`` has nothing to return,
        which is what a device that is not recording should write. Answered
        from the cache alone it would instead collapse to one unnamed column
        while ``recording_warnings()`` went on describing columns that were no
        longer in the header.

        ``None`` rather than ``[]``: an empty list reads to ``DataRecorder`` as
        single-column behaviour anyway, and ``BaseDevice`` documents ``None``
        as the way to ask for it.
        """
        cache = self._cache
        if cache is not None:
            return cache.columns()
        self._ensure_derivation()
        recorded = self._derived.recorded
        return _columns_for_recorded(recorded) if recorded else None

    async def get_state(self) -> dict[str, int | float | None] | None:
        """The consuming read, for whoever writes the CSV.

        ``snapshot`` clears the event counters, so this must have exactly one
        caller. Everything else wants ``read``.

        Also where a broken link is noticed, because this is the call the
        recorder makes once per row -- see ``_check_link``.
        """
        self._check_link()
        cache = self._cache
        return cache.snapshot() if cache is not None else None

    async def read(self) -> dict[str, int | float | None] | None:
        """The non-consuming read, for everybody else.

        ``WaitForInput`` polls ``read()`` every 50 ms and the Input node
        prefers it over ``get_state()``. Wired to ``peek``, dropping an Input
        node onto this device costs nothing; wired to ``snapshot`` it would eat
        counts out of the CSV twenty times a second, with no symptom anywhere.
        """
        self._check_link()
        cache = self._cache
        return cache.peek() if cache is not None else None

    def _check_link(self) -> None:
        """Notice a reader thread that has stopped, once.

        ``HarpReader._run`` records why it stopped and exits; until this,
        nothing anywhere read that. The failure it leaves behind is the worst
        shape a failure can have. A cable pulled twenty minutes into a
        four-hour unattended run kills the thread, and every row from then on
        carries the last state it saw, a count of zero, and a device time that
        never moves again -- byte for byte what an animal that stopped licking
        looks like. A plausible result is worse than a broken one, because
        nobody investigates it.

        So it is reported three ways, none of which stops the recording: the
        board goes to ERROR (which is what the hardware panel shows and what
        ``HardwareManager``'s error listeners are wired to), the log gets one
        line, and -- the one that survives the session -- a warning is added
        for the CSV. Losing one device should still leave a recording of
        everything the others saw, annotated, rather than nothing at all.

        Latched, so a four-hour run does not write the same line every 33 ms.
        """
        reader = self._reader
        if reader is None or self._link_failed or reader.is_alive():
            return
        self._link_failed = True

        failure = reader.failure
        cause = f"{type(failure).__name__}: {failure}" if failure is not None else "no error"
        message = (
            f"the reader thread stopped during recording ({cause}); every row after this "
            "point repeats the last value with a count of zero and is not a reading"
        )
        self._runtime_warnings.append(message)
        logger.error("Harp %s: %s", self._name, message)

        report = getattr(self._board, "report_transport_failure", None)
        if report is None:
            return
        try:
            report(failure if failure is not None else RuntimeError(message))
        except Exception:
            # A board that cannot record the failure must not also swallow the
            # sample the recorder came here for.
            logger.exception("Harp %s: could not report the transport failure", self._name)

    # --- lifecycle ---

    async def initialize(self) -> None:
        """Open the port, identify the device, put it in Active, start reading.

        Refuses if the port is still held. A second initialize() would open a
        second handle and leave the first reader -- a daemon thread -- running
        on the old one, consuming the frames the new one is waiting for, for
        the rest of the process. Nothing outside this device can notice that,
        so nothing else can prevent it. Call ``shutdown()`` first.

        A failure part way through closes the port and clears everything, so
        the refusal above never fires on a device that never came up.
        """
        if self._serial is not None or self._reader is not None:
            raise RuntimeError(
                f"Harp {self._name} still holds {self._port or 'its port'}; "
                "call shutdown() before initializing it again"
            )
        if not self._port:
            raise ValueError("Harp: 'port' setting is required")

        # Disk I/O and register-class construction, off the event loop. Done
        # before the port is opened so a bad schema costs nothing to recover
        # from: there is no handle to close yet.
        await asyncio.to_thread(self._load_derivation)

        self._serial = await asyncio.to_thread(self._open_port)
        try:
            self._who_am_i = await self._read_who_am_i()
            self._check_identity()
            if self._derived.recorded:
                # Active is what makes events flow, so it belongs with the
                # reader that drains them and not before it. A device with no
                # profile has nothing listening: putting it in Active would
                # stream into a port nobody reads, which is the one
                # configuration where Standby is the right answer rather than
                # the failure this whole sequence exists to prevent.
                await self._set_operation_mode(_MODE_ACTIVE)
                cache = RegisterCache(self._derived.recorded)
                # A fresh reader per connection, always. ``HarpReader`` is
                # one-shot -- start() after a stop() raises -- and a reused one
                # would also still hold the previous session's bytes in its
                # splitter.
                reader = HarpReader(self._serial, cache)
                reader.start()
                # Only plain assignments follow start(), so a rollback from
                # here can never find a thread it has to stop.
                self._cache = cache
                self._reader = reader
                # Re-arm the broken-link detector for the new reader, and drop
                # what the old one reported. Both halves matter, and the latch
                # is the one that bites: ``_check_link`` returns early while it
                # is set, so a detector left latched never looks at this reader
                # at all. A second cable pull would then go unnoticed, and the
                # recording would go back to being indistinguishable from a
                # subject that went quiet -- the whole failure this detector
                # exists to close, regressing on every run after the first.
                #
                # The warnings go with it because they describe a link that no
                # longer exists. They are *not* cleared per recording: the
                # recorder re-reads them at the start of each one, so a reader
                # that is still dead correctly annotates every recording made
                # while it stays that way. Only a new link clears them.
                self._link_failed = False
                self._runtime_warnings = []
            else:
                logger.info(
                    "Harp %s: no profile, so nothing is recorded and the device is left "
                    "in Standby; %d action(s) available",
                    self._name,
                    len(self._derived.actions),
                )
        except BaseException:
            await self._close_port()
            self._cache = None
            raise

        self._initialized = True
        logger.info(
            "Harp %s initialized on %s @ %d (WhoAmI %s, %d column(s), %d action(s))",
            self._name,
            self._port,
            self._baudrate,
            self._who_am_i,
            len(self._cache.columns()) if self._cache else 0,
            len(self._derived.actions),
        )

    async def shutdown(self) -> None:
        """Stop the reader, return the device to Standby, close the port.

        In that order, and the first step gates the other two. ``stop()``
        returning ``False`` means the thread is still inside a read on this
        handle: writing OperationControl would race it for the reply and see
        only a timeout, and closing the handle under an in-flight read is
        indistinguishable from an unplugged cable. So on a refused join
        nothing further is attempted, the reader and handle are kept (which
        also keeps ``initialize()`` refused, since the port really is still in
        use), and the error is logged. ``stop()`` keeps asking, so a later
        shutdown() can still succeed.

        Everything after the join is bounded and best-effort, because the
        caller bounds this whole method: ``HardwareManager`` gives it 2 s and
        cancels at the deadline. Standby gets a short round-trip budget and
        the close runs from a ``finally``, shielded -- releasing the port is
        the part that must happen, and a cancellation must not be what stops
        it.
        """
        self._initialized = False

        reader = self._reader
        if reader is not None:
            # Up to 2 s inside the join, off the event loop -- on the loop it
            # would stall the GUI and every other device mid-recording.
            if not await asyncio.to_thread(reader.stop):
                logger.error(
                    "Harp %s: the reader thread would not stop and still owns %s; "
                    "leaving the device Active and the port open",
                    self._name,
                    self._port,
                )
                return
            self._reader = None

        try:
            if self._serial is not None:
                await self._set_operation_mode(_MODE_STANDBY, timeout=SHUTDOWN_ROUND_TRIP_TIMEOUT_S)
        except Exception as e:
            # Best-effort: a device that will not answer is still a device
            # whose port has to be released.
            logger.warning("Harp %s: could not return the device to Standby: %s", self._name, e)
        finally:
            # In a ``finally`` and shielded, because the caller bounds this
            # whole method with ``wait_for``: a shutdown cancelled part-way
            # must still release the handle, or a device that merely took too
            # long leaves a port nothing in the process can reopen. Shielding
            # lets the close finish even though the ``await`` here is about to
            # raise ``CancelledError`` -- which is a ``BaseException``, so a
            # bare ``except Exception`` would have skipped this entirely.
            # ``initialize()`` catches ``BaseException`` for the same reason;
            # two halves of one lifecycle should not disagree about it.
            if self._serial is not None:
                await asyncio.shield(self._close_port())
            self._cache = None

    async def _close_port(self) -> None:
        """Release the handle, whatever close() thinks of it.

        The lock is held **across** the close, not merely while the reference
        is taken. Released first, an action already waiting on the lock wakes
        up and writes to a handle that is closing underneath it, which is the
        exact interleaving the lock is claimed to prevent (and which raises
        ``OSError: the port is closed`` from somewhere that looks like a
        hardware fault). Held across it, the waiter acquires afterwards, finds
        ``_serial`` is None, and refuses cleanly.
        """
        async with self._port_lock:
            handle, self._serial = self._serial, None
            if handle is None:
                return
            try:
                await asyncio.to_thread(handle.close)
            except Exception as e:  # close is best-effort
                logger.warning("Harp %s: error closing %s: %s", self._name, self._port, e)

    def _open_port(self) -> Any:
        """Open the serial handle. Overridden by ``MockHarpDevice``."""
        try:
            import serial
        except ImportError as e:
            raise RuntimeError(
                "pyserial not installed. Run: pip install pyserial (or reinstall GLIDER)."
            ) from e
        return serial.Serial(
            port=self._port, baudrate=self._baudrate, timeout=ROUND_TRIP_READ_TIMEOUT_S
        )

    # --- schema, profile and what they derive ---

    def _load_derivation(self) -> None:
        """Read the schema and profile and work out columns and actions.

        Blocking (it reads files), so callers on the event loop hand it to a
        thread. Every failure raises: this is where a mistyped path, a schema
        that cannot build, or a profile naming a register the board does not
        have has to become an error, because after this point every one of
        them turns into a CSV that is quietly wrong.
        """
        if not self._device_yml:
            raise ValueError("Harp: 'device_yml' setting is required")
        schema = load_schema(Path(self._device_yml))
        # Built for its validation as much as for its result: a register whose
        # type or mask cannot be resolved raises here, once, instead of
        # decoding every event of the session through the wrong lens.
        registers = build_registers(schema)
        profile = load_profile(self._profile_name) if self._profile_name else None
        derived = derive(schema, profile)

        self._schema = schema
        self._registers = registers
        self._derived = derived
        self._actions = {
            name: self._make_action(name, address) for name, address in derived.actions.items()
        }
        self._derivation_loaded = True

    def _check_identity(self) -> None:
        """Refuse a schema written for a different board.

        Only checkable when the schema says who it is for; a hand-written
        schema that omits ``whoAmI`` is taken at its word, exactly as
        ``derive`` treats a profile against such a schema. The failure this
        prevents is the quiet one: a schema whose register names happen to
        exist on the connected board builds, derives and records, and records
        the wrong registers.
        """
        declared = _as_who_am_i(self._schema.get("whoAmI"))
        if declared is None or self._who_am_i is None:
            return
        if declared != self._who_am_i:
            raise RuntimeError(
                f"Harp {self._name} on {self._port} reports WhoAmI {self._who_am_i}, but "
                f"{self._device_yml} describes WhoAmI {declared}; this is a schema for "
                "another board"
            )

    # --- register round-trips (before start(), after stop(), never during) ---

    async def _read_who_am_i(self) -> int | None:
        frame = await self._round_trip(WHO_AM_I, _WHO_AM_I_TYPE)
        return int.from_bytes(frame.payload, "little") if frame.payload else None

    async def _read_operation_control(self, timeout: float = ROUND_TRIP_TIMEOUT_S) -> int:
        frame = await self._round_trip(OPERATION_CONTROL, _OPERATION_CONTROL_TYPE, timeout)
        if not frame.payload:
            raise RuntimeError(
                f"Harp {self._name}: the device answered OperationControl with an empty payload"
            )
        return frame.payload[0]

    async def _set_operation_mode(self, mode: int, timeout: float = ROUND_TRIP_TIMEOUT_S) -> int:
        """Move the device between Standby and Active, and confirm it moved.

        Read-modify-write, so the flags that share the register -- heartbeat,
        LEDs -- keep whatever the device was configured with.

        The confirming read is a separate round-trip rather than the write's
        own echo, and the difference matters: the echo says the command was
        received, while the read says the register holds what was asked for. A
        device that acknowledges a mode it did not enter looks identical
        through the echo, and the symptom of that is a recording with no rows
        in it and nothing in any log.
        """
        current = await self._read_operation_control(timeout)
        wanted = (current & ~_OPERATION_MODE_MASK) | mode
        await self._send(
            encode(MESSAGE_WRITE, OPERATION_CONTROL, _OPERATION_CONTROL_TYPE, bytes([wanted]))
        )
        confirmed = await self._read_operation_control(timeout)
        if confirmed & _OPERATION_MODE_MASK != mode:
            raise RuntimeError(
                f"Harp {self._name}: asked for OperationControl 0x{wanted:02X} "
                f"({_MODE_NAMES.get(mode, mode)}) but it reads back 0x{confirmed:02X}; "
                "the device is not in the mode it was told to enter"
            )
        logger.debug(
            "Harp %s: OperationControl 0x%02X -> 0x%02X (%s)",
            self._name,
            current,
            confirmed,
            _MODE_NAMES.get(mode, mode),
        )
        return confirmed

    async def _round_trip(
        self, address: int, payload_type: str, timeout: float = ROUND_TRIP_TIMEOUT_S
    ) -> HarpFrame:
        """Read one register and wait for the reply.

        Only legal before ``reader.start()`` or after ``reader.stop()``. While
        the reader runs it consumes every byte the port produces and hands only
        Events to the cache, so the reply to this is decoded and dropped and
        this call can only ever time out.
        """
        request = encode(MESSAGE_READ, address, payload_type)
        async with self._port_lock:
            # Read inside the lock, never captured before it: a handle taken
            # first is a handle that shutdown() may have closed by the time
            # this runs, and the failure surfaces as an I/O error rather than
            # as "not connected".
            handle = self._serial
            if handle is None:
                raise RuntimeError(f"Harp {self._name} is not connected")
            return await asyncio.to_thread(_exchange, handle, request, address, timeout, self._name)

    async def _send(self, request: bytes) -> None:
        """Write one frame and do not wait for anything.

        Safe with the reader running -- pyserial permits one reader and one
        writer -- precisely because it reads nothing back. The device's echo is
        decoded by the reader thread, found not to be an Event, and dropped.
        """
        async with self._port_lock:
            handle = self._serial  # inside the lock; see _round_trip
            if handle is None:
                raise RuntimeError(f"Harp {self._name} is not connected")
            await asyncio.to_thread(_write_frame, handle, request)

    # --- actions ---

    def _make_action(self, register: str, address: int) -> Callable:
        async def action(value: Any = None) -> Any:
            return await self._register_action(register, address, value)

        action.__name__ = register
        action.__doc__ = (
            f"Write a value to the {register!r} register, or read it when called "
            "with no value. Reading is only possible while the device is not "
            "recording; see HarpDevice._round_trip."
        )
        return action

    async def _register_action(self, register: str, address: int, value: Any) -> Any:
        """Write ``value`` to a register, or read it when ``value`` is None.

        A ``DeviceAction`` node forwards an input port only when it carries a
        value, so an unconnected port arrives here as ``None`` -- which is
        exactly the request to read. ``DeviceReadNode`` and the runner's
        buttons call with no value at all, and mean the same thing.
        """
        if self._serial is None:
            raise RuntimeError(f"Harp {self._name} is not connected")

        if value is None:
            return await self._read_register(register, address)

        if "Write" not in self._derived.access.get(register, frozenset()):
            # Checked against the schema rather than discovered on the wire.
            # A device answers a write to a read-only register by ignoring it,
            # so the alternative is an action that reports success and does
            # nothing for the rest of the session.
            raise ValueError(
                f"Harp {self._name}: register {register!r} is not writable "
                f"({self._access_description(register)})"
            )
        payload_type = self._payload_type_of(register)
        await self._send(encode(MESSAGE_WRITE, address, payload_type, self._pack(register, value)))
        return None

    async def _read_register(self, register: str, address: int) -> Any:
        """Answer a read: from the cache if it is recorded, else from the wire."""
        column = self._derived.recorded.get(address)
        cache = self._cache
        if column is not None and cache is not None:
            # A recorded register is already being read, continuously, by the
            # reader thread. Answering from the cache is not a shortcut -- it
            # is the only answer available at all while the reader owns the
            # port, and it is a better one than the wire could give: no
            # round-trip, and it works for a Write+Event register that the
            # wire would refuse to read. ``peek``, so a DeviceRead node
            # dropped onto the graph cannot eat counts out of the CSV.
            return cache.peek().get(f"{column}_state")

        if "Read" not in self._derived.access.get(register, frozenset()):
            # Refused from the schema, instantly. A Read sent to a write-only
            # register gets no reply, so without this the caller waits out the
            # full round-trip timeout and is then told the device did not
            # answer -- which reads as broken hardware rather than as an
            # action that never existed.
            raise ValueError(
                f"Harp {self._name}: register {register!r} is not readable "
                f"({self._access_description(register)}); "
                "record it in the profile to read it continuously instead"
            )

        reader = self._reader
        if reader is not None and reader.is_alive():
            raise RuntimeError(
                f"Harp {self._name}: cannot read register {register!r} while recording. "
                "The reader thread consumes every reply, so this would wait for one that "
                "never arrives. Record the register instead by naming it in the profile."
            )
        frame = await self._round_trip(address, self._payload_type_of(register))
        return int.from_bytes(frame.payload, "little") if frame.payload else None

    def _access_description(self, register: str) -> str:
        """How a message should describe what a register does allow."""
        modes = sorted(self._derived.access.get(register, frozenset()))
        return f"access {', '.join(modes)}" if modes else "no access modes declared"

    def _payload_type_of(self, register: str) -> str:
        """The register's payload-type name, from the class ``schema`` built."""
        built = self._registers.get(register)
        if built is None:
            raise RuntimeError(f"Harp {self._name}: register {register!r} is not in the schema")
        return str(built.payload_type.name)

    def _pack(self, register: str, value: Any) -> bytes:
        """Render a value as the register's payload bytes.

        Integers only, matching what ``RegisterCache`` decodes on the way back
        in. Float registers raise rather than being written through a lossy
        int conversion that would look like it worked.
        """
        built = self._registers[register]
        dtype = built.payload_type.numpy_dtype
        name = str(built.payload_type.name)
        if name == "Float":
            raise ValueError(
                f"Harp {self._name}: register {register!r} is a Float register, which this "
                "device cannot write yet"
            )
        try:
            return int(value).to_bytes(dtype.itemsize, "little", signed=name.startswith("S"))
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Harp {self._name}: {value!r} does not fit register {register!r} ({name})"
            ) from exc

    # --- serialization ---

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: BaseBoard) -> HarpDevice:
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance


def _columns_for_recorded(recorded: Mapping[int, str]) -> list[str]:
    """The columns a register map produces, spelled by the one authority.

    A throwaway ``RegisterCache`` rather than a second copy of the naming
    rule. The cache is pure state -- no thread, no port, no side effect --
    so building one costs a dict and a lock, while an f-string here would be
    a second place the ``_state``/``_count``/``_last_ms`` suffixes are
    written and the first opportunity for a header to disagree with its rows.
    """
    return RegisterCache(dict(recorded)).columns()


def _as_who_am_i(raw: Any) -> int | None:
    """A schema's ``whoAmI`` as a number, however it was written.

    A schema hand-copied from a datasheet may quote it or write it in hex;
    ``"1400"`` and ``0x578`` are the same board as ``1400``. Anything that is
    not a number at all comes back as ``None``, which skips the check rather
    than failing it -- ``derive`` treats an unreadable WhoAmI the same way.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    try:
        return int(str(raw), 0)
    except ValueError:
        return None


def _write_frame(handle: Any, request: bytes) -> None:
    """Put one frame on the wire, flushing if the handle can."""
    handle.write(request)
    flush = getattr(handle, "flush", None)
    if flush is not None:
        flush()


def _exchange(
    handle: Any, request: bytes, address: int, timeout: float, device_name: str
) -> HarpFrame:
    """Send a Read request and return the reply, blocking. Runs in a thread.

    The reply is picked out of the stream by address rather than assumed to be
    the next thing that arrives, because it very often is not: at shutdown the
    device is still Active and emitting, so the reply comes back somewhere
    behind a run of events. Frames that are not the one asked for are dropped
    -- they belong to a recording that is already over.

    The handle's read timeout is borrowed and put back, for the same reason
    ``HarpReader`` borrows it: the caller chose that value for its own reads,
    and a value silently changed underneath it surfaces much later as a read
    that returned too early.
    """
    previous = getattr(handle, "timeout", None)
    handle.timeout = ROUND_TRIP_READ_TIMEOUT_S
    try:
        _write_frame(handle, request)
        splitter = FrameSplitter()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            waiting = getattr(handle, "in_waiting", 0) or 0
            chunk = handle.read(max(1, waiting))
            if not chunk:
                continue
            for raw in splitter.feed(chunk):
                try:
                    frame = decode(raw)
                except FrameError:
                    # Unreachable while the splitter validates by decoding;
                    # skipped rather than trusted, as everywhere else here.
                    continue
                if frame.message_type == MESSAGE_READ and frame.address == address:
                    return frame
        raise TimeoutError(
            f"Harp {device_name}: no reply to a read of register {address} within "
            f"{timeout:.1f}s"
        )
    finally:
        # Guarded for the reason ``HarpReader._restore_timeout`` documents: on
        # pyserial this is not an assignment, it reconfigures the open port,
        # and that raises on a device that has been unplugged -- which is
        # exactly when the read above has just failed. Unguarded, the
        # reconfigure error replaces the true failure and the operator reads
        # "cannot reconfigure a port" instead of the disconnect.
        try:
            handle.timeout = previous
        except Exception:
            logger.warning(
                "Harp %s: could not restore the port read timeout (the device may be "
                "gone); reporting the original failure instead",
                device_name,
                exc_info=True,
            )
