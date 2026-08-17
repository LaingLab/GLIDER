"""The read side of a Harp link: what the device reported, and what a row says.

``RegisterCache`` is pure state -- no threads of its own, no I/O, no serial
port -- so the ``state``/``count``/``last_ms`` semantics can be decided and
tested with no device in the way. The thread that owns the port joins it here.
"""

import threading
from collections import Counter

from glider_harp.frames import HarpFrame

# Read=1 and Write=2 are host-initiated traffic: a Read is our own request and
# a Write is our own command echoed back. Only Event=3 is the device reporting
# something it did on its own, which is the only thing an experiment record is
# asking about.
_EVENT = 3

_MS_PER_SECOND = 1000.0


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

    Payloads are read as little-endian unsigned integers, which is what the
    digital-input and counter registers this is built for send. Signed and
    floating-point payload types would need decoding per ``payload_type``; add
    that when a device needs it rather than guessing now.

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

    def __init__(self, registers: dict[int, str]) -> None:
        """``registers`` maps a register address to the base name of its columns.

        The names are checked here rather than at the first read, because every
        way they can be wrong produces a CSV that is malformed but not
        obviously so, and by then a trial is running.
        """
        self._names = dict(registers)

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

        value = int.from_bytes(frame.payload, "little") if frame.payload else None
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
