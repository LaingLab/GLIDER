"""The read side of a Harp link: what the device reported, and what a row says.

``RegisterCache`` is pure state -- no threads of its own, no I/O, no serial
port -- so the ``state``/``count``/``last_ms`` semantics can be decided and
tested with no device in the way. ``HarpReader`` is the thread that owns the
port and fills the cache; between them they are the whole read path, and the
event loop touches only the cache.
"""

import logging
import struct
import threading
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

from glider_harp.frames import FrameError, FrameSplitter, HarpFrame, decode

logger = logging.getLogger(__name__)

# Payload types this module can turn into a number, and the two facts that
# decide how. Spelled as the names a ``device.yml`` uses, which is what
# ``derivation`` carries across -- a register's declared type is the only thing
# the cache needs from a schema, so that is the whole of the coupling.
_SIGNED_TYPES = frozenset({"S8", "S16", "S32", "S64"})
_UNSIGNED_TYPES = frozenset({"U8", "U16", "U32", "U64"})
_FLOAT_TYPE = "Float"
DECODABLE_TYPES = _UNSIGNED_TYPES | _SIGNED_TYPES | {_FLOAT_TYPE}

# Harp's ``Float`` is one IEEE-754 single, little-endian.
_FLOAT = struct.Struct("<f")

# Read=1 and Write=2 are host-initiated traffic: a Read is our own request and
# a Write is our own command echoed back. Only Event=3 is the device reporting
# something it did on its own, which is the only thing an experiment record is
# asking about.
_EVENT = 3

_MS_PER_SECOND = 1000.0

# How long the reader may sit inside one blocking read. The floor keeps a port
# opened with timeout=0 from spinning a core; the ceiling bounds how long
# ``stop`` waits for the loop to notice the stop event, well inside its 2 s
# join. Everything the reader must do on a silent line happens only when a read
# returns, so ``start`` aims for half the idle window -- but the floor wins
# below a 0.1 s window, and the flush then lands within one read of the window
# rather than at it. That only matters to a test choosing a small window; the
# 0.5 s default sits well above the floor.
_MIN_READ_TIMEOUT_S = 0.05
_MAX_READ_TIMEOUT_S = 0.5


def decode_payload(payload: bytes, declared: str | None) -> int | float | None:
    """One event's payload, read as the type the schema declares for it.

    The single spelling of "what does a Harp payload mean", shared by
    ``RegisterCache`` and by ``HarpDevice``'s wire reads -- a signed register
    read over the wire and the same register read out of the record must not
    disagree about its sign.

    ``declared`` of ``None`` means "unsigned, width unchecked", which is what
    this did for every register before types reached it. It stays the default
    so a caller with no schema to hand keeps the old behaviour rather than
    getting a new failure.

    Two payloads decode to ``None``, and they mean the same thing -- *the event
    happened, this part of it is unknown*, which is exactly what the cache's
    ``state`` column is documented to report:

    * an **empty** payload, which carries no value at all;
    * a **``Float`` whose payload is not four bytes**, which is a schema that
      disagrees with the hardware. Reported unknown rather than invented from
      whatever bytes arrived, or raised: a blank column is a visible gap, and
      raising here would cost the rest of the frames in that same read.
    """
    if not payload:
        return None
    if declared == _FLOAT_TYPE:
        if len(payload) != _FLOAT.size:
            return None
        return _FLOAT.unpack(payload)[0]
    return int.from_bytes(payload, "little", signed=declared in _SIGNED_TYPES)


def _columns_for(name: str) -> tuple[str, str, str]:
    """The three columns one register contributes, in order.

    The single spelling of the column names, shared by ``columns`` and the
    reads. Sharing it is what makes a header that disagrees with its rows
    structurally impossible rather than merely tested for.

    The separator is ``_``, not ``:``, because these are sub-column names: the
    recorder builds its header as ``{device_id}:{sub_column}``, so a colon here
    would produce ``harp1:lick:state``, in which nothing downstream can tell
    which colon was the separator. ``BaseDevice.state_columns`` forbids it for
    that reason.
    """
    return (f"{name}_state", f"{name}_count", f"{name}_last_ms")


class _RegisterState:
    """Mutable state for one register. Guarded by the cache's lock."""

    __slots__ = ("count", "last_ms", "state")

    def __init__(self) -> None:
        self.state: int | None = None
        self.count = 0
        self.last_ms: float | None = None


class RegisterCache:
    """The current value of every register that becomes a CSV column.

    Each register contributes three columns, and they answer different
    questions on purpose:

    * ``state`` -- the value carried by the most recent event. Persists across
      reads, so a row always reports the level the device is at, not just the
      rows where it changed.
    * ``count`` -- events since the previous ``snapshot``, **cleared on read**.
      This is what makes the record honest: a row is written about every 33 ms
      at 30 fps while a lick lasts 20-50 ms, so a column that reported only the
      current level would drop whole events between rows, and two licks inside
      one interval would look like one. A counter cleared on read cannot lose
      one -- every event is reported in exactly one row.
    * ``last_ms`` -- device time of the most recent event, in milliseconds,
      sub-millisecond digits kept. Persists. Device time is the only clock that
      can place an event inside the poll interval it was found in; host arrival
      time cannot.

    Where a value is unknown, the column reports ``None`` rather than the
    previous value. An event whose payload is empty leaves no value to report
    and an untimestamped event leaves no device time to report; carrying the
    older reading forward would present a number that is stale by an unbounded
    interval as if it described the event just counted. ``None`` alongside a
    non-zero ``count`` says what actually happened -- the event occurred, this
    part of it is unknown.

    That unknown is sticky. ``state`` stays ``None`` until some later event
    arrives carrying a payload, so one empty-payload event blanks the column
    for the rest of the session if no other event follows. This is the intended
    trade -- a blank column is a visible gap, a stale one is not -- but it is
    reachable without a broken device: a 6-byte all-header frame that happens
    to satisfy the checksum while ``FrameSplitter`` hunts through noise decodes
    as a perfectly valid empty-payload event, roughly once in 256 tried offsets.

    Payloads are read as the type the schema declares for each register --
    unsigned and signed integers of any width, and ``Float`` -- which is why
    the cache is built with a type map beside its name map. See
    ``decode_payload``. Arrays are not decoded: ``derivation`` refuses to
    record one, because how several values should appear in one CSV cell is
    undecided rather than merely unimplemented.

    Reading comes in two forms, and which one a caller wants depends on whether
    it owns the record. ``snapshot`` consumes the counters and belongs to
    whatever writes the CSV; ``peek`` reports without consuming and is for
    everybody else. See ``peek`` for why that distinction is not optional.

    Thread-safe: the reader thread ingests while the event loop reads. Be
    precise about what that rests on, because no test here can show it. On
    CPython today the lock is unobservable -- the suite passes with it removed,
    and not by luck: the interpreter offers a thread switch only at bytecodes
    like ``RESUME`` and ``JUMP_BACKWARD``, and neither critical section here
    contains one, so no interleaving exists to find at any switch interval.
    That is an implementation detail of one interpreter, unspecified by the
    language, and it does not hold on a free-threaded build (3.13t/3.14t),
    where a read could clear a count it never reported and lose events in
    silence. Unobservable here, unspecified by the language, required there --
    so do not remove it on the strength of a green suite.

    Every read returns a fresh dict, so a caller may hold rows and batch them.
    """

    def __init__(self, registers: dict[int, str], types: Mapping[int, str] | None = None) -> None:
        """``registers`` maps a register address to the base name of its columns.

        ``types`` maps the same addresses to the payload type the schema
        declares -- ``derivation.Derived.recorded_types``. Optional, and an
        address it omits is decoded as unsigned, which is what every register
        got before types reached here; that default is what lets a caller with
        no schema (``_columns_for_recorded``, which only wants column names)
        build one at all.

        Everything is checked here rather than at the first read, because every
        way it can be wrong produces a CSV that is malformed but not obviously
        so, and by then a trial is running. A type for an address that is not
        recorded is checked for the same reason it looks harmless: the two maps
        are filled together by ``derive``, so an address in one and not the
        other means they have drifted, and the register the caller thought it
        was typing is being decoded as something else.
        """
        self._names = dict(registers)
        types = dict(types or {})
        if strays := sorted(set(types) - set(self._names)):
            raise ValueError(
                "Register types name addresses that are not recorded: "
                + ", ".join(str(address) for address in strays)
            )
        if unknown := sorted({str(t) for t in types.values()} - DECODABLE_TYPES):
            raise ValueError(f"Register types the cache cannot decode: {', '.join(unknown)}")
        self._types: dict[int, str | None] = {
            address: types.get(address) for address in self._names
        }

        if not self._names:
            # An empty columns() is falsy to DataRecorder, which then treats
            # the device as single-column: it emits one header,
            # "{device_id}:{device_type}", and per row partitions that header
            # on the colon and looks the device *type* up in the dict
            # get_state() returned. That key is never there, so every row gets
            # an empty cell. The device records nothing and nothing raises.
            raise ValueError("RegisterCache needs at least one register")

        if any(not name for name in self._names.values()):
            # Not caught by the column checks below: an unnamed register
            # yields "_state", which is unique and non-empty and tells a
            # reader of the CSV nothing at all.
            raise ValueError("Register names must be non-empty")

        # Validated on the produced columns rather than on the names, because
        # the columns are what DataRecorder validates and what a person reads.
        columns = self.columns()
        if colons := sorted(column for column in columns if ":" in column):
            raise ValueError(f"Column names must not contain ':': {', '.join(colons)}")
        if repeated := sorted(name for name, n in Counter(columns).items() if n > 1):
            raise ValueError(f"Column names must be unique; repeated: {', '.join(repeated)}")

        self._states = {address: _RegisterState() for address in registers}
        self._lock = threading.Lock()

    def columns(self) -> list[str]:
        """Every column name this cache reports, in register order."""
        return [column for name in self._names.values() for column in _columns_for(name)]

    def ingest(self, frame: HarpFrame) -> None:
        """Absorb one decoded frame.

        Frames that are not events, and events for addresses this cache was not
        built with, are ignored: a Harp device emits registers no experiment
        asked for, and counting them would inflate a column that is not theirs.
        """
        if frame.message_type != _EVENT:
            return
        register = self._states.get(frame.address)
        if register is None:
            return

        value = decode_payload(frame.payload, self._types[frame.address])
        last_ms = None if frame.timestamp is None else frame.timestamp * _MS_PER_SECOND
        with self._lock:
            register.state = value
            register.count += 1
            register.last_ms = last_ms

    def peek(self) -> dict[str, int | float | None]:
        """Read every column without clearing anything.

        Because ``snapshot`` consumes the counters, only one caller may use it
        -- and in GLIDER a second caller already exists. Both ``WaitForInput``
        and the Input node try ``device.read()`` first and fall back to
        ``get_state()``, the latter on a 50 ms loop, so an experimenter who
        drops an Input node onto a Harp device would otherwise consume counts
        twenty times a second and write a CSV that is wrong with no symptom
        anywhere. Wiring ``read()`` here makes that harmless.

        ``peek`` is the fix rather than making the ownership a rule, because it
        has no clearing behaviour to get wrong: it cannot lose a count, so a
        second poller becomes safe instead of forbidden. Enforcing ownership
        instead would trade a miscount for a ``RuntimeError`` in the middle of
        a trial, which is worse.

        The counts it reports are those accumulated since the last ``snapshot``
        -- a partial interval, which is what an unsynchronised observer should
        see. Two peeks with no snapshot between them report the same numbers.
        """
        return self._read(clear=False)

    def snapshot(self) -> dict[str, int | float | None]:
        """Read every column and clear the event counters.

        For the one caller that owns the record; everybody else wants ``peek``.

        Reading and clearing are one operation under the lock. Splitting them
        would let an event that arrived in between be counted into no row at
        all, which is the one failure the counter exists to prevent.
        """
        return self._read(clear=True)

    def _read(self, clear: bool) -> dict[str, int | float | None]:
        """Collect every column, optionally consuming the counters."""
        values: dict[str, int | float | None] = {}
        with self._lock:
            for address, name in self._names.items():
                register = self._states[address]
                state_column, count_column, last_ms_column = _columns_for(name)
                values[state_column] = register.state
                values[count_column] = register.count
                values[last_ms_column] = register.last_ms
                if clear:
                    register.count = 0
        return values


class HarpReader:
    """Drains a serial port into a ``RegisterCache`` on a background thread.

    A Harp device streams events whenever it likes, so nothing on the event
    loop can afford to wait for one. The thread owns the read loop and the
    ``FrameSplitter``; the loop only ever reads the cache, which is already
    filled by the time it looks. This is ``GenericSerialDevice``'s streaming
    pattern with a binary framer in place of the line framer.

    Ownership, because two of these bite quietly:

    * The **port** belongs to the caller, but not while the reader is running.
      The reader never opens or closes it, and it borrows the handle's read
      timeout between ``start`` and ``stop`` -- the read cadence is the
      reader's own heartbeat -- putting the caller's value back afterwards.
      Two orderings follow, and both are the caller's to keep:

      - ``stop`` before ``close``. A read in flight when the handle closes is
        indistinguishable from an unplugged cable and is recorded as a failure.
      - Every write and register round-trip goes **before ``start`` or after
        ``stop``, never during**. This thread reads every byte the port
        produces and hands only events to the cache; a Read reply arriving
        while it runs is decoded, counted in ``frame_count``, and dropped. A
        concurrent round-trip therefore cannot see its own reply -- it does not
        race, it simply never returns one.
    * ``snapshot`` belongs to whoever writes the CSV. This thread calls
      ``cache.ingest`` and nothing else -- ``snapshot`` clears the counters, so
      a second caller silently eats events out of the record.

    What the counters mean:

    * ``error_count`` -- corrupt frames, and nothing else. It is exactly
      ``FrameSplitter.checksum_errors``, which is what "bad cable" looks like.
      Framing noise (``resyncs``, ``bytes_discarded``) is reported separately
      on purpose: a device that was mid-transmission when the port opened is
      normal and costs a resync, and folding that in would make every healthy
      connect look like a failing link. Note the splitter counts *recovered*
      corruption -- a burst that never resynchronises registers one error, not
      one per frame lost inside it -- so it is a symptom, not a frame-accurate
      total.
    * ``frame_count`` -- frames decoded and handed to the cache, whatever their
      message type. The cache ignores Read and Write traffic, so this is
      larger than the events recorded, and deliberately: it is the answer to
      "is anything arriving at all?".
    * ``decode_failures`` -- frames the splitter accepted that then failed to
      decode. Structurally impossible today, since the splitter validates by
      decoding, and counted anyway because the alternative to a visible zero is
      a silent swallow if the two ever disagree.
    * ``processing_errors`` -- reads the splitter or the cache could not be
      made to swallow at all. This one means a bug in this package rather than
      anything on the wire, so it is kept out of ``error_count``; see ``_run``.
    * ``idle_flushes`` -- times the reader forced framing open because the
      device went quiet (see ``_flush_stalled_buffer``).

    Counters are plain ints written only by the reader thread and read from
    anywhere; they are advisory, so a poll may see the count from a moment ago.
    The cache, where losing an event actually matters, has its own lock.

    One reader runs once: ``start`` after a ``stop`` raises rather than
    resurrecting a thread whose splitter still holds the last session's bytes.
    Build a new reader per connection, as ``GenericSerialDevice`` builds a new
    thread and event per initialize().
    """

    def __init__(self, serial: Any, cache: RegisterCache, idle_flush_s: float = 0.5) -> None:
        """``serial`` needs pyserial's ``read`` and a settable ``timeout``.

        ``cache`` receives every event decoded. ``idle_flush_s`` is how long
        the line must be silent before framing is forced open; see
        ``_flush_stalled_buffer`` for what that trades.
        """
        if idle_flush_s <= 0:
            # Zero would flush on every read that returned nothing, which on a
            # device that pauses mid-frame means hunting for a boundary inside
            # a frame that was simply still arriving.
            raise ValueError(f"idle_flush_s must be positive, got {idle_flush_s}")
        self._serial = serial
        self._cache = cache
        self._idle_flush_s = idle_flush_s
        self._splitter = FrameSplitter()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_count = 0
        self._decode_failures = 0
        self._processing_errors = 0
        self._idle_flushes = 0
        self._last_data_at = 0.0
        # (borrowed?, the caller's read timeout) -- see ``start`` and ``stop``.
        self._borrowed_timeout: tuple[bool, Any] = (False, None)
        self._failure: Exception | None = None

    # --- lifecycle ---

    def start(self) -> None:
        """Start the reader thread. Once per instance."""
        # The stop event half matters as much as the thread half: it is
        # latched, so a reader started after any stop would end on its first
        # loop test -- alive for microseconds, then quietly gone, with an empty
        # recording and nothing anywhere saying the reader never ran.
        if self._thread is not None or self._stop_event.is_set():
            raise RuntimeError("HarpReader.start() is one-shot; build a new reader to reconnect")

        # The reader borrows the handle's read timeout while it runs. Both
        # things it must do on a silent line -- notice the stop event, notice
        # the device stopped talking -- can only happen when a read returns, so
        # a port opened with no timeout would leave the thread unstoppable and
        # the idle flush unreachable. Half the idle window so a stalled buffer
        # is released near ``idle_flush_s`` rather than a whole read later.
        #
        # Borrowed, not taken: ``stop`` puts the caller's value back. The
        # caller owns this port for writes and register round-trips too, and a
        # timeout it chose and silently lost is the kind of thing that surfaces
        # much later as a read that returns too early. Set before the thread
        # exists so a handle that rejects it fails out of ``start`` cleanly,
        # with nothing running and the reader still startable.
        wanted = max(_MIN_READ_TIMEOUT_S, min(self._idle_flush_s / 2, _MAX_READ_TIMEOUT_S))
        previous = getattr(self._serial, "timeout", None)
        if previous is not None and previous != wanted:
            logger.debug(
                "HarpReader: borrowing the port read timeout (%s -> %.2fs) until stop",
                previous,
                wanted,
            )
        self._serial.timeout = wanted
        self._borrowed_timeout = (True, previous)
        self._last_data_at = time.monotonic()

        thread = threading.Thread(target=self._run, name="harp-reader", daemon=True)
        try:
            thread.start()
        except Exception:
            # A Pi already running the GUI, vision and the recorder can refuse
            # a new thread. Nothing is running, so hand the port back and leave
            # the reader startable: recording _thread first would make the
            # caller's stop() raise "cannot join thread before it is started",
            # masking this error and stranding the borrowed timeout.
            self._restore_timeout()
            raise
        self._thread = thread

    def stop(self, timeout: float = 2.0) -> bool:
        """Ask the reader to finish, wait for it, and give the port back.

        Returns whether the reader is now stopped: **False means the thread is
        still reading the port**, and the caller must not use the handle. The
        return value exists because that outcome is otherwise invisible -- a
        caller that carried on would write a register and read the reply back
        with a live reader thread consuming it, and see only a timeout. It is
        reachable: a loaded Pi at the end of a trial may not schedule the
        thread through a read plus overhead inside ``timeout``.

        Idempotent, safe before ``start`` -- which retires the reader, see
        ``start`` -- and worth retrying, since it keeps asking. The thread
        reference is kept rather than cleared, so a thread that refused to join
        stays visible through ``is_alive`` instead of being reported as gone.

        The read timeout is handed back only once the thread has actually
        exited. A thread still running needs the short timeout it was given to
        keep noticing the stop event; handing it back a long one would strand
        it further.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        if thread.is_alive():
            logger.error(
                "HarpReader: thread did not exit within %.1fs; the port is still in use", timeout
            )
            return False
        self._restore_timeout()
        return True

    def _restore_timeout(self) -> None:
        """Give the caller's read timeout back, if we took one.

        Guarded, because on pyserial this is not an assignment: the setter
        reconfigures the open port, and that raises on a device that has been
        unplugged -- which is the single likeliest reason a reader is being
        stopped at all. Unguarded, the exception leaves ``stop`` by a path no
        caller expects, *before* the caller can release the handle, so a pulled
        cable strands the port until the process exits and re-plugging does
        not help.

        The courtesy is not worth that. The handle is about to be closed, and
        a device that is gone has no configuration left to restore. Cleared
        before the attempt so a failed restore is not retried on the next stop.
        """
        borrowed, previous = self._borrowed_timeout
        if not borrowed:
            return
        self._borrowed_timeout = (False, None)
        try:
            self._serial.timeout = previous
        except Exception:
            logger.warning(
                "HarpReader: could not restore the port read timeout (the device may be "
                "gone); continuing so the handle can still be released",
                exc_info=True,
            )

    def is_alive(self) -> bool:
        """Whether the reader thread is running."""
        return self._thread is not None and self._thread.is_alive()

    # --- counters ---

    @property
    def error_count(self) -> int:
        """Corrupt frames recovered from. Not framing noise -- see the class docstring."""
        return self._splitter.checksum_errors

    @property
    def frame_count(self) -> int:
        """Frames decoded and passed to the cache, of every message type."""
        return self._frame_count

    @property
    def resyncs(self) -> int:
        """Times the splitter lost framing and hunted for the next boundary."""
        return self._splitter.resyncs

    @property
    def bytes_discarded(self) -> int:
        """Bytes thrown away without ever forming a frame."""
        return self._splitter.bytes_discarded

    @property
    def decode_failures(self) -> int:
        """Frames the splitter accepted that then failed to decode."""
        return self._decode_failures

    @property
    def processing_errors(self) -> int:
        """Reads that could not be processed at all. A bug here, not on the wire."""
        return self._processing_errors

    @property
    def idle_flushes(self) -> int:
        """Times a silent line forced framing open."""
        return self._idle_flushes

    @property
    def failure(self) -> Exception | None:
        """The read that ended the thread, or None. See ``_run``."""
        return self._failure

    # --- the reader thread ---

    def _run(self) -> None:
        """Read, split, decode, ingest, until stopped or the port fails.

        Nothing may escape this method. There is nobody on this stack to catch
        it: an exception here ends the thread, the traceback goes to
        ``threading.excepthook`` on stderr -- invisible under a packaged GUI,
        and never seen by ``logger`` -- and the device then polls a cache that
        will never change again, reporting zeros with nothing anywhere saying
        why. That is the same silent-empty-recording failure the ``FrameError``
        handler exists to prevent, through a different door.

        So both halves are guarded, and they are guarded differently because
        they fail differently:

        * A **read** that raises is the port itself: unplugged, or closed under
          us. Nothing later will read from it either, and a loop that swallowed
          it would spin on the same error for the rest of the trial, so the
          thread stops and records ``failure``.
        * Anything raised while **processing** what was read is a bug in this
          package -- the splitter, the decoder, or the cache -- not a fault on
          the wire. It is data-dependent, so the next read may well be fine:
          the rest of that one read is skipped, ``processing_errors`` counts
          it, and the thread carries on. (Not reachable today, and payload
          decoding growing types of its own did not make it so:
          ``decode_payload`` answers a ``Float`` whose payload is the wrong
          width with ``None`` rather than a ``struct.error``, precisely
          because one misdeclared register must not cost the rest of the
          frames in the same read.)
        """
        while not self._stop_event.is_set():
            try:
                chunk = self._read_chunk()
            except Exception as exc:
                logger.exception("HarpReader: serial read failed; reader stopping")
                self._failure = exc
                self._stop_event.set()
                return
            try:
                if chunk:
                    self._last_data_at = time.monotonic()
                    self._ingest_frames(self._splitter.feed(chunk))
                elif self._is_stalled():
                    self._flush_stalled_buffer()
            except Exception:
                self._processing_errors += 1
                logger.exception("HarpReader: could not process %d bytes read", len(chunk))

    def _read_chunk(self) -> bytes:
        """One read, taking whatever the port has already buffered.

        ``in_waiting`` collapses a burst into a single read instead of one call
        per byte, which is worth having on a Pi at 115200 baud. It is reached
        through ``getattr`` so the read loop needs nothing of a handle but
        ``read``; without it the loop still works, one byte per call. (The
        handle does need a settable ``timeout`` -- but that is ``start``'s
        requirement, and it fails there, before any thread exists.)
        """
        waiting = getattr(self._serial, "in_waiting", 0) or 0
        return self._serial.read(max(1, waiting))

    def _ingest_frames(self, raw_frames: list[bytes]) -> None:
        """Decode what the splitter returned and hand each frame to the cache.

        Every failure is caught by **type**, not by message: ``FrameError`` is
        the base of the whole hierarchy, so this one clause covers truncation,
        corruption and an invalid header alike. Catching a narrower type would
        let the others end the thread, and the recording would come back empty
        with no error anywhere -- while the messages themselves cannot be used
        to tell the cases apart, since upstream validates the checksum before
        the length field and so calls most partial reads a checksum mismatch.

        Decoding goes through ``decode`` rather than ``harp.protocol`` for the
        same reason: ``HarpParseError`` is not a ``ValueError``, so a frame
        parsed directly would throw straight past this handler.
        """
        for raw in raw_frames:
            try:
                frame = decode(raw)
            except FrameError:
                # Unreachable while the splitter validates by decoding, which
                # is exactly why it is counted and logged rather than trusted.
                self._decode_failures += 1
                logger.warning("HarpReader: splitter returned an undecodable frame %r", raw)
                continue
            self._frame_count += 1
            self._cache.ingest(frame)

    # --- the idle flush ---

    def _is_stalled(self) -> bool:
        """Whether a silent line has left the splitter holding bytes."""
        if not self._splitter.pending_bytes:
            return False
        return time.monotonic() - self._last_data_at >= self._idle_flush_s

    def _flush_stalled_buffer(self) -> None:
        """Force framing open after the device has gone quiet.

        ``FrameSplitter`` has one residual stall it cannot fix alone, and says
        so: a noise byte that happens to be a valid message type followed by a
        length byte claiming a long frame parks it in "waiting for the rest",
        withholding every frame behind it. At most 255 further bytes settle it,
        so a live stream always recovers -- but if the device falls silent
        first the frames are held for good. Measured: a ``(3, 255)`` head ahead
        of 36 events emits none of them and holds 254 bytes across a hundred
        idle polls. The splitter has no clock, so only the reader can tell the
        difference between "still arriving" and "never coming".

        ``force_resync`` is that message: the head stops being believed as a
        frame boundary, so the splitter's own resync hunts forward and releases
        everything parked behind it. The policy is here, where the clock is;
        the mechanism stays in ``frames``, next to the invariant it breaks.

        What it costs when it fires on a frame that really was still arriving
        depends entirely on what that frame's payload happens to contain, and
        one of the two cases is worse than it first looks:

        * An ordinary payload survives. Resync finds nothing decodable inside a
          partial frame and, below one maximum frame, discards nothing, so the
          frame completes and is emitted when its tail lands. Swept across
          arrival gaps from 0.02x to 4x the window with zero and random
          payloads: 1/1 delivered every time.
        * A payload that **contains a decodable frame** is mis-framed, and
          deterministically -- not at "one offset in 256". Clearing ``_synced``
          is exactly what disarms the splitter's guard against hunting inside
          an in-flight head, so resync finds the embedded frame, consumes the
          outer frame's header with it, and emits it. The recording then gets a
          phantom event on the *inner* frame's register and loses the real
          event entirely, with no counter moving. Reproduced 3/3, and pinned in
          ``test_an_embedded_frame_is_mis_framed_by_a_mid_frame_flush``.

        The second case is accepted rather than prevented, on reachability: it
        needs a mid-frame stall longer than ``idle_flush_s`` (0.5 s is already
        pathological at 115200 baud, where a whole frame takes ~1.5 ms) *and*
        an embedded frame, which needs a payload of at least six bytes -- the
        digital-input and counter registers this is built for send one to four.
        12,000 realistic timestamped events produced no instance.

        Preventing it means deciding whether an in-flight head is a real frame
        before disarming the guard, and the only evidence available is the
        header's payload-type bits -- the parser knowledge ``frames`` exists to
        keep in one place, reimplemented here, and still only a heuristic: a
        noise head with plausible bits would then hold the stall open and lose
        every frame behind it at the end of a trial, which is the failure this
        flush exists to prevent and the far likelier one. Not worth the trade
        unless a device appears with long payloads.

        Waiting a whole ``idle_flush_s`` rather than flushing on the first
        empty read is what keeps the second case as rare as it is.
        """
        self._idle_flushes += 1
        logger.debug(
            "HarpReader: %.2fs idle with %d bytes held; forcing a resync",
            self._idle_flush_s,
            self._splitter.pending_bytes,
        )
        self._ingest_frames(self._splitter.force_resync())
        # Restart the window rather than flushing on every read from here on:
        # a buffer with nothing framable in it stays put, and re-hunting it
        # each time would burn a full rescan per read for the rest of the trial.
        self._last_data_at = time.monotonic()
