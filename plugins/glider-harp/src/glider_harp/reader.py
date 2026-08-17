"""Per-register state, the layer between decoded frames and CSV columns.

Pure state: no threads of its own, no I/O, no serial port. The reader thread
that owns the port calls ``ingest``; the event loop calls ``snapshot`` once per
recorded row. Keeping the two apart is what lets the ``state``/``count``/
``last_ms`` semantics below be decided and tested without a device in the way.
"""

import threading

from glider_harp.frames import HarpFrame

# Read=1 and Write=2 are host-initiated traffic: a Read is our own request and
# a Write is our own command echoed back. Only Event=3 is the device reporting
# something it did on its own, which is the only thing an experiment record is
# asking about.
_EVENT = 3

_MS_PER_SECOND = 1000.0


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
      snapshots, so a row always reports the level the device is at, not just
      the rows where it changed.
    * ``count`` -- events since the previous snapshot, **cleared on read**.
      This is what makes the record honest: a row is written about every 33 ms
      at 30 fps while a lick lasts 20-50 ms, so a column that reported only the
      current level would drop whole events between rows, and two licks inside
      one interval would look like one. A counter cleared on read cannot lose
      one -- every event is reported in exactly one row.
    * ``last_ms`` -- device time of the most recent event, in milliseconds.
      Persists. Device time is the only clock that can place an event inside
      the poll interval it was found in; host arrival time cannot.

    Where a value is unknown, the column reports ``None`` rather than the
    previous value. An event whose payload is empty leaves no value to report
    and an untimestamped event leaves no device time to report; carrying the
    older reading forward would present a number that is stale by an unbounded
    interval as if it described the event just counted. ``None`` alongside a
    non-zero ``count`` says what actually happened -- the event occurred, this
    part of it is unknown.

    Payloads are read as little-endian unsigned integers, which is what the
    digital-input and counter registers this is built for send. Signed and
    floating-point payload types would need decoding per ``payload_type``; add
    that when a device needs it rather than guessing now.

    Thread-safe: the reader thread ingests while the event loop snapshots.
    The lock is not decoration, and the suite passing without it is not
    evidence that it can go: under CPython's GIL these few attribute writes are
    very hard to interleave destructively, so no test here can tell locked from
    unlocked code. On a free-threaded build they can, and a snapshot that
    cleared a count it never reported would lose events silently.

    Each snapshot is a fresh dict, so a caller may hold rows and batch them.
    """

    def __init__(self, registers: dict[int, str]) -> None:
        """``registers`` maps a register address to the base name of its columns."""
        names = list(registers.values())
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            # Two addresses under one name would collide in the CSV header and
            # silently interleave two registers' events in one set of columns.
            raise ValueError(f"Register names must be unique; repeated: {', '.join(duplicates)}")

        self._names = dict(registers)
        self._states = {address: _RegisterState() for address in registers}
        self._lock = threading.Lock()

    def columns(self) -> list[str]:
        """Every column name this cache reports, in register order.

        Spelled out here and again in ``snapshot`` rather than shared, because
        sharing them would need ``getattr`` on the field names to build the
        snapshot. The suite asserts the two agree instead.
        """
        return [
            column
            for name in self._names.values()
            for column in (f"{name}:state", f"{name}:count", f"{name}:last_ms")
        ]

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

    def snapshot(self) -> dict[str, int | float | None]:
        """Read every column and clear the event counters.

        Reading and clearing are one operation under the lock. Splitting them
        would let an event that arrived in between be counted into no row at
        all, which is the one failure the counter exists to prevent.
        """
        values: dict[str, int | float | None] = {}
        with self._lock:
            for address, name in self._names.items():
                register = self._states[address]
                values[f"{name}:state"] = register.state
                values[f"{name}:count"] = register.count
                values[f"{name}:last_ms"] = register.last_ms
                register.count = 0
        return values
