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
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from glider.hal.base_device import BaseDevice, DeviceConfig
from glider_harp.derivation import Derived, derive, load_profile
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

_DEFAULT_BAUDRATE = 115200


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
            "type": "str",
            "default": "",
            "help": "Name of a shipped profile naming the registers to record. Blank records none.",
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
        self._warnings: list[str] = []
        self._who_am_i: int | None = None
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
        while the device is initialized the edit is recorded for the saved file
        and takes effect on the next initialize(). Before init the caches are
        refreshed immediately. Mirrors ``GenericSerialDevice``.
        """
        parsed = self._parse_settings({**self._config.settings, **settings})
        self._config.settings.update(settings)
        if self._initialized:
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

    def recording_warnings(self) -> list[str]:
        """Ways this device's columns will say less than they look like they do.

        Written into the CSV metadata block by ``DataRecorder``. ``derive``
        already logs these, and a log line during a long unattended run reaches
        nobody -- the CSV is the artefact that outlives the session.
        """
        self._ensure_derivation()
        return list(self._warnings)

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
        """The cache's columns, or None when nothing is recorded.

        ``None`` rather than ``[]``: an empty list reads to ``DataRecorder`` as
        single-column behaviour anyway, and ``BaseDevice`` documents ``None``
        as the way to ask for it.
        """
        cache = self._cache
        return cache.columns() if cache is not None else None

    async def get_state(self) -> dict[str, int | float | None] | None:
        """The consuming read, for whoever writes the CSV.

        ``snapshot`` clears the event counters, so this must have exactly one
        caller. Everything else wants ``read``.
        """
        cache = self._cache
        return cache.snapshot() if cache is not None else None

    async def read(self) -> dict[str, int | float | None] | None:
        """The non-consuming read, for everybody else.

        ``WaitForInput`` polls ``read()`` every 50 ms and the Input node
        prefers it over ``get_state()``. Wired to ``peek``, dropping an Input
        node onto this device costs nothing; wired to ``snapshot`` it would eat
        counts out of the CSV twenty times a second, with no symptom anywhere.
        """
        cache = self._cache
        return cache.peek() if cache is not None else None

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

        if self._serial is not None:
            try:
                await self._set_operation_mode(_MODE_STANDBY)
            except Exception as e:
                # Best-effort: a device that will not answer is still a device
                # whose port has to be released.
                logger.warning("Harp %s: could not return the device to Standby: %s", self._name, e)
            await self._close_port()
        self._cache = None

    async def _close_port(self) -> None:
        """Release the handle, whatever close() thinks of it."""
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
        self._warnings = self._collect_warnings(schema, derived)
        self._derivation_loaded = True

    @staticmethod
    def _collect_warnings(schema: Mapping[str, Any], derived: Derived) -> list[str]:
        """Recorded registers that cannot report anything.

        ``derive`` already notices this and logs it. The same finding is
        rebuilt here rather than captured from the log because a warning that
        only exists in a log line is a warning nobody reads six months later
        with the CSV open -- and the CSV is what the trial leaves behind.
        """
        registers = schema.get("registers") or {}
        if not isinstance(registers, Mapping):
            return []
        by_address: dict[int, tuple[str, Any]] = {}
        for name, meta in registers.items():
            if isinstance(meta, Mapping) and isinstance(meta.get("address"), int):
                by_address[meta["address"]] = (str(name), meta)

        warnings: list[str] = []
        for address, column in derived.recorded.items():
            entry = by_address.get(address)
            if entry is None:
                continue
            name, meta = entry
            access = meta.get("access")
            modes = {access} if isinstance(access, str) else set(map(str, access or ()))
            if "Event" not in modes:
                warnings.append(
                    f"register {name} (column {column}) is not an Event register; "
                    "its columns will never change"
                )
        return warnings

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

    async def _read_operation_control(self) -> int:
        frame = await self._round_trip(OPERATION_CONTROL, _OPERATION_CONTROL_TYPE)
        if not frame.payload:
            raise RuntimeError(
                f"Harp {self._name}: the device answered OperationControl with an empty payload"
            )
        return frame.payload[0]

    async def _set_operation_mode(self, mode: int) -> int:
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
        current = await self._read_operation_control()
        wanted = (current & ~_OPERATION_MODE_MASK) | mode
        await self._send(
            encode(MESSAGE_WRITE, OPERATION_CONTROL, _OPERATION_CONTROL_TYPE, bytes([wanted]))
        )
        confirmed = await self._read_operation_control()
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

    async def _round_trip(self, address: int, payload_type: str) -> HarpFrame:
        """Read one register and wait for the reply.

        Only legal before ``reader.start()`` or after ``reader.stop()``. While
        the reader runs it consumes every byte the port produces and hands only
        Events to the cache, so the reply to this is decoded and dropped and
        this call can only ever time out.
        """
        handle = self._serial
        if handle is None:
            raise RuntimeError(f"Harp {self._name} is not connected")
        request = encode(MESSAGE_READ, address, payload_type)
        async with self._port_lock:
            return await asyncio.to_thread(
                _exchange, handle, request, address, ROUND_TRIP_TIMEOUT_S, self._name
            )

    async def _send(self, request: bytes) -> None:
        """Write one frame and do not wait for anything.

        Safe with the reader running -- pyserial permits one reader and one
        writer -- precisely because it reads nothing back. The device's echo is
        decoded by the reader thread, found not to be an Event, and dropped.
        """
        handle = self._serial
        if handle is None:
            raise RuntimeError(f"Harp {self._name} is not connected")
        async with self._port_lock:
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
        exactly the request to read.
        """
        if self._serial is None:
            raise RuntimeError(f"Harp {self._name} is not connected")
        payload_type = self._payload_type_of(register)

        if value is None:
            reader = self._reader
            if reader is not None and reader.is_alive():
                raise RuntimeError(
                    f"Harp {self._name}: cannot read register {register!r} while recording. "
                    "The reader thread consumes every reply, so this would wait for one that "
                    "never arrives. Record the register instead by naming it in the profile."
                )
            frame = await self._round_trip(address, payload_type)
            return int.from_bytes(frame.payload, "little") if frame.payload else None

        await self._send(encode(MESSAGE_WRITE, address, payload_type, self._pack(register, value)))
        return None

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
        handle.timeout = previous
