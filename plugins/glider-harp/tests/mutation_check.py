"""Mutation check for every module in ``glider_harp`` that fails quietly.

``FrameSplitter``, ``RegisterCache``, ``HarpReader``, ``schema`` and
``derivation``.

Not collected by pytest -- deliberately named without a ``test_`` prefix, since
it rewrites source files on disk and shells out to pytest. Run it directly:

    python plugins/glider-harp/tests/mutation_check.py

Every mutant below must be killed by the suite, except those marked EQUIVALENT,
which provably cannot be killed and are asserted to survive so that a future
change making one observable shows up as a failure rather than as a silently
stricter suite. That has already earned its keep once: the lazy-genexp mutant
was EQUIVALENT until ``feed`` was annotated ``list[bytes]``, and this check is
what flagged that the justification had gone stale.

This exists because every one of these modules fails quietly. Several splitter
bugs found in review (a stale byte after each frame, noise stranding the frames
behind it) left the yielded frames looking correct and were invisible to
assertions on output alone; a register cache that drops an event writes a CSV
that is wrong in no visible way at all; a register built one byte too narrow
decodes every event of the session and never raises; a column rule obeyed by
halves writes a file that opens cleanly. A mutant that survives means a test
constant is doing the work.

Mutants are written against the *contract* each module documents, not against
the code as written -- including rules the implementation might have obeyed by
halves. Writing them off the implementation only measures the author's
imagination.

Three categories, tallied separately, because one number over all of them would
say less than it appears to:

* plain -- must be killed, and a kill is ordinary evidence that a test pins the
  behaviour.
* ``EQUIVALENT:`` -- provably unkillable, asserted to survive (above).
* ``RACE-WINDOW:`` -- the two lock mutants. These are killed **only because the
  mutant itself inserts a ``time.sleep`` into the window it opens**. With the
  lock simply removed and no sleep added, the concurrency tests still pass:
  15/15 green, and a stress harness at 8-16 writers x 3000 events found zero
  lost updates over 40 trials, down to a switch interval of 1e-7.

  That is structural, not luck, and not something a harsher setting will
  overturn: CPython 3.12 offers a thread switch only at bytecodes like
  ``RESUME`` and ``JUMP_BACKWARD``, and neither critical section contains one,
  so no interleaving exists to find at any switch interval. Squeezing the
  interval will never promote these two.

  So their kill shows the suite catches an artificially widened race -- not
  that the lock as written is load-bearing on this interpreter. The lock stays
  regardless, because "unobservable on today's CPython" is not "unnecessary":
  the property is an implementation detail of one interpreter, unspecified by
  the language, and it does not hold on a free-threaded build. Do not quote
  these two alongside the killable score as if they measured the same thing.

  Exit condition: if a free-threaded (3.13t/3.14t) job ever lands in CI, the
  interleaving becomes reachable and these stop being self-fulfilling. Drop the
  ``RACE-WINDOW:`` prefix then and let them count as plain killable mutants.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "plugins" / "glider-harp" / "src" / "glider_harp"
FRAMES_TARGET = SOURCE / "frames.py"
READER_TARGET = SOURCE / "reader.py"
SCHEMA_TARGET = SOURCE / "schema.py"
DERIVATION_TARGET = SOURCE / "derivation.py"
TESTS = "plugins/glider-harp/tests/"

CONSUME = "                del self._buffer[: len(candidate)]\n                self._synced = True"
FEED_BODY = (
    "        self._buffer += chunk\n"
    "        frames: list[bytes] = []\n"
    "        while True:\n"
    "            frame = self._take_frame()\n"
    "            if frame is None:\n"
    "                break\n"
    "            frames.append(frame)"
)
FRAME_AT_DECODE = (
    "        try:\n"
    "            decode(candidate)\n"
    "        except FrameError:\n"
    "            return None\n"
    "        return candidate"
)

# (name, exact source to replace, replacement)
FRAME_MUTANTS: list[tuple[str, str, str]] = [
    ("never buffers the remainder", "self._buffer += chunk", "self._buffer = bytearray(chunk)"),
    ("never resyncs", "        return self._resync()", "        return None"),
    ("returns frames in reverse order", "        return frames", "        return frames[::-1]"),
    (
        "off-by-one short on frame length",
        "size = self._buffer[offset + _LENGTH_OFFSET] + 2",
        "size = self._buffer[offset + _LENGTH_OFFSET] + 1",
    ),
    (
        "off-by-one long on frame length",
        "size = self._buffer[offset + _LENGTH_OFFSET] + 2",
        "size = self._buffer[offset + _LENGTH_OFFSET] + 3",
    ),
    (
        "never drops unframeable bytes (unbounded buffer)",
        "            dropped = len(self._buffer) - keep\n"
        "            del self._buffer[:dropped]\n"
        "            self.bytes_discarded += dropped",
        "            return",
    ),
    (
        "scans forward even while synced",
        "elif self._synced and self._head_may_still_complete():",
        "elif False and self._head_may_still_complete():",
    ),
    (
        "trusts a length byte at an unvalidated head",
        "        if self._buffer[0] not in _MESSAGE_TYPES:\n            return False\n",
        "",
    ),
    (
        "accepts a resync candidate without decoding it",
        FRAME_AT_DECODE,
        "        return candidate",
    ),
    (
        "forgets to consume the skipped bytes on resync",
        "del self._buffer[: offset + len(frame)]",
        "del self._buffer[: len(frame)]",
    ),
    (
        "off-by-one short consuming a frame on the synced path",
        CONSUME,
        CONSUME.replace("len(candidate)]", "len(candidate) - 1]"),
    ),
    (
        "over-consumes a frame on the synced path",
        CONSUME,
        CONSUME.replace("len(candidate)]", "len(candidate) + 1]"),
    ),
    (
        "feed is a true generator (buffer append deferred)",
        FEED_BODY,
        "        self._buffer += chunk\n"
        "        while True:\n"
        "            frame = self._take_frame()\n"
        "            if frame is None:\n"
        "                break\n"
        "            yield frame\n"
        "        return",
    ),
    # --- counters, which are Task 8's only view of what was swallowed ---
    (
        "counts corruption while scanning noise",
        FRAME_AT_DECODE,
        "        try:\n"
        "            decode(candidate)\n"
        "        except ChecksumError:\n"
        "            self.checksum_errors += 1\n"
        "            return None\n"
        "        except FrameError:\n"
        "            return None\n"
        "        return candidate",
    ),
    (
        "never counts corruption",
        "                if self._synced:\n                    self.checksum_errors += 1",
        "                pass",
    ),
    (
        # Escaped the first counter round: every counter test delivered its
        # whole stream in a single feed(), so nothing exercised a corrupt frame
        # sitting at the head across reads, which is the normal serial case.
        "counts corruption regardless of _synced",
        "                if self._synced:\n                    self.checksum_errors += 1",
        "                self.checksum_errors += 1",
    ),
    (
        # The degenerate fix for the mutant above.
        "counts corruption once per session instead of once per frame",
        "                if self._synced:\n                    self.checksum_errors += 1",
        "                self.checksum_errors = 1",
    ),
    ("never counts resyncs", "            self.resyncs += 1\n", ""),
    (
        "never counts discarded bytes on resync",
        "                self.bytes_discarded += offset\n",
        "",
    ),
    # --- provably unkillable; see module docstring ---
    (
        "EQUIVALENT: resync starts at the failed offset instead of past it",
        # _resync is only ever reached immediately after _candidate_at(0)/decode
        # failed on an unchanged buffer, so retrying offset 0 is deterministically
        # another miss. Costs one wasted call; changes no output.
        "for offset in range(1, len(self._buffer)):",
        "for offset in range(0, len(self._buffer)):",
    ),
    (
        # Was classified EQUIVALENT while feed returned Iterator[bytes] and the
        # tests wrapped every call in list(). Annotating the return as
        # list[bytes] and comparing against lists directly made the difference
        # observable, so it is now an ordinary killable mutant. Kept as the
        # regression guard for that decision.
        "returns a lazy genexp instead of a list",
        "        return frames",
        "        return (f for f in frames)",
    ),
    # --- the public hand-off to the reader, which is the half with a clock ---
    (
        "force_resync leaves the head believed, so nothing is released",
        '        self._synced = False\n        return self.feed(b"")',
        '        return self.feed(b"")',
    ),
    (
        "force_resync releases the frames and drops them",
        '        self._synced = False\n        return self.feed(b"")',
        '        self._synced = False\n        self.feed(b"")\n        return []',
    ),
    (
        "pending_bytes never reports anything held",
        "        return len(self._buffer)",
        "        return 0",
    ),
]

# ``ingest``'s guarded write, and the shared read loop behind ``peek`` and
# ``snapshot``. Quoted whole so the race-window mutants below can remove the
# lock and widen the window in one edit -- a lock only fails where two threads
# meet, so mutating it in place would just be dropping a statement no
# single-threaded test can see. Read the RACE-WINDOW note above before quoting
# their result as coverage.
INGEST_WRITE = (
    "        with self._lock:\n"
    "            register.state = value\n"
    "            register.count += 1\n"
    "            register.last_ms = last_ms"
)
READ_BODY = (
    "        with self._lock:\n"
    "            for address, name in self._names.items():\n"
    "                register = self._states[address]\n"
    "                state_column, count_column, last_ms_column = _columns_for(name)\n"
    "                values[state_column] = register.state\n"
    "                values[count_column] = register.count\n"
    "                values[last_ms_column] = register.last_ms\n"
    "                if clear:\n"
    "                    register.count = 0"
)
LOOKUP = (
    "        register = self._states.get(frame.address)\n"
    "        if register is None:\n"
    "            return"
)
COLUMN_NAMES = '    return (f"{name}_state", f"{name}_count", f"{name}_last_ms")'
COLUMNS = (
    "        return [column for name in self._names.values() for column in _columns_for(name)]"
)

READER_MUTANTS: list[tuple[str, str, str]] = [
    # --- "state -- latest value seen, persists across reads" ---
    (
        "state keeps the first value instead of the latest",
        "            register.state = value",
        "            register.state = register.state if register.state is not None else value",
    ),
    (
        "a read clears state as well as count",
        "                    register.count = 0",
        "                    register.count = 0\n                    register.state = None",
    ),
    (
        "payload read big-endian",
        'int.from_bytes(frame.payload, "little")',
        'int.from_bytes(frame.payload, "big")',
    ),
    (
        "only the first payload byte is read",
        'value = int.from_bytes(frame.payload, "little") if frame.payload else None',
        "value = frame.payload[0] if frame.payload else None",
    ),
    (
        # The documented rule is that an unknown value is reported as unknown,
        # not carried forward. Zero is a real lick level, so this mutant writes
        # a reading the device never sent.
        "an empty payload reports 0 instead of no value",
        'value = int.from_bytes(frame.payload, "little") if frame.payload else None',
        'value = int.from_bytes(frame.payload, "little")',
    ),
    (
        "an empty payload leaves the previous state standing",
        "            register.state = value",
        "            if value is not None:\n                register.state = value",
    ),
    # --- "count -- events since the previous snapshot, cleared on read" ---
    (
        "count never increments",
        "            register.count += 1",
        "            register.count += 0",
    ),
    (
        "count increments by two",
        "            register.count += 1",
        "            register.count += 2",
    ),
    (
        # The degenerate "did anything happen" column: right for a single event
        # in the interval, wrong for the two-licks-in-one-frame case the counter
        # exists for.
        "count is a flag rather than a count",
        "            register.count += 1",
        "            register.count = 1",
    ),
    (
        "nothing ever clears the counters",
        "                if clear:",
        "                if False:",
    ),
    (
        "count is cleared before it is reported, so every row reads zero",
        "                values[count_column] = register.count\n"
        "                values[last_ms_column] = register.last_ms\n"
        "                if clear:\n"
        "                    register.count = 0",
        "                if clear:\n"
        "                    register.count = 0\n"
        "                values[count_column] = register.count\n"
        "                values[last_ms_column] = register.last_ms",
    ),
    (
        # Dedented out of the loop, so only the last register is cleared: the
        # CSV then double-counts every register but one.
        "only one register's count is cleared per read",
        "                if clear:\n                    register.count = 0",
        "            if clear:\n                register.count = 0",
    ),
    # --- peek: the non-consuming read a second poller needs ---
    (
        # WaitForInput and the Input node poll read() every 50 ms. If that path
        # consumes, an Input node dropped onto the device silently eats counts
        # out of the CSV twenty times a second.
        "peek consumes the counters like snapshot",
        "        return self._read(clear=False)",
        "        return self._read(clear=True)",
    ),
    (
        "snapshot does not consume the counters",
        "        return self._read(clear=True)",
        "        return self._read(clear=False)",
    ),
    (
        "the clear flag is ignored and every read consumes",
        "                if clear:",
        "                if True:",
    ),
    # --- "last_ms -- device timestamp of the most recent event, in ms" ---
    (
        "last_ms reports seconds",
        "frame.timestamp * _MS_PER_SECOND",
        "frame.timestamp * 1.0",
    ),
    (
        "last_ms reports microseconds",
        "frame.timestamp * _MS_PER_SECOND",
        "frame.timestamp * _MS_PER_SECOND * 1000.0",
    ),
    (
        # Device time is Seconds(U32) + Micros(U16) at 32 us per tick, so the
        # sub-millisecond digits are resolution the device really has.
        "last_ms is rounded to whole milliseconds",
        "frame.timestamp * _MS_PER_SECOND",
        "float(int(frame.timestamp * _MS_PER_SECOND))",
    ),
    (
        "last_ms keeps the first event's time instead of the latest",
        "            register.last_ms = last_ms",
        "            if register.last_ms is None:\n                register.last_ms = last_ms",
    ),
    (
        "a read clears last_ms",
        "                    register.count = 0",
        "                    register.count = 0\n                    register.last_ms = None",
    ),
    (
        # The other half of the timestamp-is-None decision: an untimestamped
        # event would leave the previous event's device time in place, so a row
        # would read count=1 beside a last_ms that never moved.
        "an untimestamped event leaves the previous device time standing",
        "            register.last_ms = last_ms",
        "            if last_ms is not None:\n                register.last_ms = last_ms",
    ),
    # --- "only message_type == 3 (Event) frames count" ---
    (
        "every message type is counted",
        "        if frame.message_type != _EVENT:\n            return\n",
        "",
    ),
    (
        "Write echoes are counted as events",
        "        if frame.message_type != _EVENT:",
        "        if frame.message_type not in (2, _EVENT):",
    ),
    (
        "Read replies are counted as events",
        "        if frame.message_type != _EVENT:",
        "        if frame.message_type not in (1, _EVENT):",
    ),
    (
        "the wrong message type is treated as the event",
        "_EVENT = 3",
        "_EVENT = 2",
    ),
    # --- "frames for unmapped addresses are ignored" ---
    (
        # Not simply deleting the guard, which would raise and be killed by any
        # unmapped frame at all. This is the quiet version: the frame lands in
        # somebody else's column.
        "an unmapped address falls into the first register",
        LOOKUP,
        "        register = self._states.get(frame.address, next(iter(self._states.values())))",
    ),
    (
        "every register absorbs every event",
        INGEST_WRITE,
        "        with self._lock:\n"
        "            for register in self._states.values():\n"
        "                register.state = value\n"
        "                register.count += 1\n"
        "                register.last_ms = last_ms",
    ),
    (
        "the address is ignored and every event lands in one register",
        "        register = self._states.get(frame.address)",
        "        register = self._states.get(frame.address)\n"
        "        for register in self._states.values():\n"
        "            pass",
    ),
    # --- the columns themselves: this list is the CSV header ---
    (
        # A colon here yields "harp1:lick:state" once the recorder prefixes the
        # device id, and nothing downstream can tell which colon was the
        # separator. BaseDevice.state_columns forbids it.
        "columns are separated by a colon",
        COLUMN_NAMES,
        '    return (f"{name}:state", f"{name}:count", f"{name}:last_ms")',
    ),
    (
        # Mutating columns() alone, not the shared speller: the header then
        # disagrees with the rows it labels.
        "columns omits last_ms",
        COLUMNS,
        "        return [\n"
        "            column for name in self._names.values() for column in _columns_for(name)[:2]\n"
        "        ]",
    ),
    (
        "columns orders the fields differently from the rows",
        COLUMNS,
        "        return [\n"
        "            column\n"
        "            for name in self._names.values()\n"
        "            for column in reversed(_columns_for(name))\n"
        "        ]",
    ),
    (
        "columns are ordered by field rather than by register",
        COLUMNS,
        "        return [\n"
        '            f"{name}_{field}"\n'
        '            for field in ("state", "count", "last_ms")\n'
        "            for name in self._names.values()\n"
        "        ]",
    ),
    # --- construction-time validation, all of it protecting the CSV ---
    (
        "duplicate columns are accepted",
        "        if repeated := sorted(name for name, n in Counter(columns).items() if n > 1):",
        "        if False:",
    ),
    (
        "a colon in a register name is accepted",
        '        if colons := sorted(column for column in columns if ":" in column):',
        "        if False:",
    ),
    (
        "an unnamed register is accepted",
        "        if any(not name for name in self._names.values()):",
        "        if False:",
    ),
    (
        # columns() == [] reads to DataRecorder as single-column behaviour: it
        # emits "{device_id}:{device_type}" and then looks the device type up
        # in the state dict, so every row gets an empty cell.
        "an empty register map is accepted",
        "        if not self._names:",
        "        if False:",
    ),
    (
        # Every value in the shared dict is one the cache really did report, so
        # a caller batching rows sees row N rewrite itself when row N+1 is
        # taken, with nothing anywhere looking wrong.
        "a read returns one shared dict reused across calls",
        "        values: dict[str, int | float | None] = {}",
        '        values: dict[str, int | float | None] = getattr(self, "_shared", {})\n'
        "        self._shared = values",
    ),
    (
        "the caller's register map is aliased rather than copied",
        "        self._names = dict(registers)",
        "        self._names = registers",
    ),
    # --- the lock; see the RACE-WINDOW note, these are not coverage evidence ---
    (
        "RACE-WINDOW: a read reports and clears without the lock",
        READ_BODY,
        "        for address, name in self._names.items():\n"
        "            register = self._states[address]\n"
        "            state_column, count_column, last_ms_column = _columns_for(name)\n"
        "            values[state_column] = register.state\n"
        "            values[count_column] = register.count\n"
        '            __import__("time").sleep(0.0002)\n'
        "            values[last_ms_column] = register.last_ms\n"
        "            if clear:\n"
        "                register.count = 0",
    ),
    (
        "RACE-WINDOW: ingest increments without the lock",
        INGEST_WRITE,
        "        register.state = value\n"
        "        seen = register.count\n"
        '        __import__("time").sleep(0.0002)\n'
        "        register.count = seen + 1\n"
        "        register.last_ms = last_ms",
    ),
]

DECODE_CALL = (
    "            try:\n                frame = decode(raw)\n            except FrameError:"
)
FLUSH = "        self._ingest_frames(self._splitter.force_resync())"
READ_TIMEOUT = (
    "        wanted = max(_MIN_READ_TIMEOUT_S, min(self._idle_flush_s / 2, _MAX_READ_TIMEOUT_S))"
)
STALL_CHECK = (
    "        if not self._splitter.pending_bytes:\n"
    "            return False\n"
    "        return time.monotonic() - self._last_data_at >= self._idle_flush_s"
)
LOOP_BODY = (
    "            try:\n"
    "                if chunk:\n"
    "                    self._last_data_at = time.monotonic()\n"
    "                    self._ingest_frames(self._splitter.feed(chunk))\n"
    "                elif self._is_stalled():\n"
    "                    self._flush_stalled_buffer()\n"
    "            except Exception:\n"
    "                self._processing_errors += 1\n"
    '                logger.exception("HarpReader: could not process %d bytes read", len(chunk))'
)
START_THREAD = (
    "        try:\n"
    "            thread.start()\n"
    "        except Exception:\n"
    "            # A Pi already running the GUI, vision and the recorder can refuse\n"
    "            # a new thread. Nothing is running, so hand the port back and leave\n"
    "            # the reader startable: recording _thread first would make the\n"
    '            # caller\'s stop() raise "cannot join thread before it is started",\n'
    "            # masking this error and stranding the borrowed timeout.\n"
    "            self._restore_timeout()\n"
    "            raise\n"
    "        self._thread = thread"
)
JOIN_RESULT = (
    "        if thread.is_alive():\n"
    "            logger.error(\n"
    '                "HarpReader: thread did not exit within %.1fs; the port is still in use",'
    " timeout\n"
    "            )\n"
    "            return False\n"
    "        self._restore_timeout()"
)

# ``HarpReader``. Written against the rules the thread inherited from the three
# pieces it composes -- the exception hierarchy, the counter meanings, the
# splitter's residual stall, cache ownership -- rather than against the loop as
# written, since a rule obeyed by halves looks fully covered from the code.
HARP_READER_MUTANTS: list[tuple[str, str, str]] = [
    # --- branch on the exception TYPE, never the message ---
    (
        # The inherited defect this task exists to avoid: FrameError is the base
        # of all three, and anything narrower lets a sibling end the thread. The
        # recording then comes back empty with no error anywhere.
        "catches only ChecksumError, so a plain FrameError ends the thread",
        DECODE_CALL,
        "            from glider_harp.frames import ChecksumError\n\n"
        "            try:\n"
        "                frame = decode(raw)\n"
        "            except ChecksumError:",
    ),
    (
        "no decode failure is caught at all",
        DECODE_CALL,
        "            if True:\n                frame = decode(raw)\n            if False:",
    ),
    (
        # Upstream validates the checksum before the length field, so a
        # truncated frame's message reads "Checksum mismatch". Any handler that
        # reads the text is deciding on a string upstream never promised.
        "branches on the exception message instead of its type",
        "            except FrameError:\n",
        '            except FrameError as exc:\n                if "checksum" not in str(exc).lower():\n                    raise\n',
    ),
    (
        # Provably unkillable: FrameError subclasses ValueError and decode
        # raises nothing else, a pairing test_frames pins directly. Kept so that
        # decoupling the hierarchy from ValueError shows up here as a kill
        # rather than as a silently weaker handler.
        "EQUIVALENT: catches ValueError rather than FrameError",
        "            except FrameError:\n",
        "            except ValueError:\n",
    ),
    (
        # Requirement 4: HarpParseError is not a ValueError, so a frame parsed
        # directly throws straight past the handler above.
        "parses frames directly instead of through decode()",
        "                frame = decode(raw)\n",
        "                from harp.protocol import HarpMessage\n\n"
        "                frame = HarpMessage.parse(raw)\n",
    ),
    # --- error_count is corruption, and nothing else ---
    (
        "error_count folds in resyncs, so a noisy connect reads as a bad cable",
        "        return self._splitter.checksum_errors",
        "        return self._splitter.checksum_errors + self._splitter.resyncs",
    ),
    (
        "error_count reports discarded bytes",
        "        return self._splitter.checksum_errors",
        "        return self._splitter.bytes_discarded",
    ),
    (
        "error_count folds in decode failures",
        "        return self._splitter.checksum_errors",
        "        return self._splitter.checksum_errors + self._decode_failures",
    ),
    (
        "error_count folds in a dead port",
        "        return self._splitter.checksum_errors",
        "        return self._splitter.checksum_errors + (1 if self.failure else 0)",
    ),
    (
        "corruption is never surfaced at all",
        "        return self._splitter.checksum_errors",
        "        return 0",
    ),
    # --- the idle flush: the stall only a clock can resolve ---
    (
        "never flushes, so frames behind a stalled head are lost at the end of a trial",
        "                elif self._is_stalled():",
        "                elif False:",
    ),
    (
        "flushes on every read that came back empty",
        STALL_CHECK,
        "        return bool(self._splitter._buffer)",
    ),
    (
        "flushes on a silent line even with nothing held",
        STALL_CHECK,
        "        return time.monotonic() - self._last_data_at >= self._idle_flush_s",
    ),
    (
        # The window is silence on the line. Timed from anywhere else it expires
        # mid-stream and tears a frame that was merely arriving slowly.
        "the idle window is not restarted by the bytes that arrive",
        "                if chunk:\n                    self._last_data_at = time.monotonic()\n",
        "                if chunk:\n",
    ),
    (
        "a buffer with nothing framable in it is rescanned on every read",
        "        # Restart the window rather than flushing on every read from here on:",
        "        return\n        # Restart the window rather than flushing on every read from here on:",
    ),
    (
        "the flush drops the held bytes instead of reframing them",
        FLUSH,
        "        self._splitter._buffer.clear()",
    ),
    (
        # The idle flush that isn't one: feed(b"") without giving up on the head
        # re-runs the same wait that is already stuck.
        "the flush re-feeds without giving up on the head",
        FLUSH,
        '        self._ingest_frames(self._splitter.feed(b""))',
    ),
    (
        # Requirement 5: feed() returns the frames it completed, so a caller that
        # ignores the return value silently drops them.
        "the flush reframes but throws away what came back",
        FLUSH,
        "        self._splitter.force_resync()",
    ),
    (
        "the read path throws away what feed returned",
        "                    self._ingest_frames(self._splitter.feed(chunk))",
        "                    self._splitter.feed(chunk)",
    ),
    # --- exactly one caller may poll snapshot(), and it is not this thread ---
    (
        "the thread polls snapshot() each time round the loop",
        "                elif self._is_stalled():",
        "                self._cache.snapshot()\n                if self._is_stalled():",
    ),
    (
        "the flush consumes the cache it just filled",
        FLUSH,
        "        self._ingest_frames(self._splitter.force_resync())\n"
        "        self._cache.snapshot()",
    ),
    # --- nothing may escape the loop; see _run ---
    (
        # The other door into a silent empty recording: the traceback goes to
        # threading.excepthook on stderr, logger never sees it, and the device
        # polls a cache that will never change again.
        "an unexpected error while processing a read ends the thread",
        "            except Exception:\n"
        "                self._processing_errors += 1\n"
        '                logger.exception("HarpReader: could not process %d bytes read", len(chunk))',
        "            except Exception:\n                raise",
    ),
    (
        "the guard covers the read path but leaves the flush bare",
        LOOP_BODY,
        "            try:\n"
        "                if chunk:\n"
        "                    self._last_data_at = time.monotonic()\n"
        "                    self._ingest_frames(self._splitter.feed(chunk))\n"
        "            except Exception:\n"
        "                self._processing_errors += 1\n"
        "                logger.exception"
        '("HarpReader: could not process %d bytes read", len(chunk))\n'
        "            if not chunk and self._is_stalled():\n"
        "                self._flush_stalled_buffer()",
    ),
    (
        # A bug in this package is not a dead port: one malformed payload would
        # end the recording instead of costing the read it arrived in.
        "a processing error is treated as a dead port and stops the reader",
        "                self._processing_errors += 1\n"
        '                logger.exception("HarpReader: could not process %d bytes read", len(chunk))',
        "                self._stop_event.set()\n                return",
    ),
    (
        "an unexpected processing error leaves no trace",
        "                self._processing_errors += 1\n",
        "",
    ),
    (
        "processing errors are folded into error_count",
        "        return self._splitter.checksum_errors",
        "        return self._splitter.checksum_errors + self._processing_errors",
    ),
    # --- what the frames do on the way to the cache ---
    (
        "frame_count counts frames that never decoded",
        DECODE_CALL,
        "            self._frame_count += 1\n"
        "            try:\n"
        "                frame = decode(raw)\n"
        "            except FrameError:",
    ),
    (
        "frame_count never moves, so a live link is indistinguishable from a dead one",
        "            self._frame_count += 1",
        "            self._frame_count += 0",
    ),
    (
        "a swallowed decode failure leaves no trace",
        "                self._decode_failures += 1",
        "                pass",
    ),
    (
        "reads one byte per call however much the port is holding",
        '        waiting = getattr(self._serial, "in_waiting", 0) or 0\n'
        "        return self._serial.read(max(1, waiting))",
        "        return self._serial.read(1)",
    ),
    # --- lifecycle: the thread the event loop is trusting to be gone ---
    (
        "stop() asks nothing to stop",
        "        self._stop_event.set()\n        thread = self._thread",
        "        thread = self._thread",
    ),
    (
        # Killed by the 0.2 s empty read in the fixture, not by a sleep this
        # mutant inserts: without the join the thread is provably still inside
        # that read when stop() returns.
        "stop() returns without waiting for the thread",
        "        thread.join(timeout)",
        "        pass",
    ),
    (
        "stop() reports success by forgetting the thread it left running",
        "        thread.join(timeout)",
        "        self._thread = None",
    ),
    (
        "the reader thread is not a daemon",
        'name="harp-reader", daemon=True',
        'name="harp-reader", daemon=False',
    ),
    (
        "a failing port is retried forever instead of stopping the reader",
        "                self._stop_event.set()\n                return",
        "                continue",
    ),
    (
        "a failing port leaves no record of what happened",
        "                self._failure = exc\n",
        "",
    ),
    # --- start/stop, which Task 11 drives from an event loop ---
    (
        # Thread exhaustion on a Pi: recording the thread before it started
        # makes the caller's stop() raise "cannot join thread before it is
        # started", masking the real error and stranding the borrowed timeout.
        "the thread is recorded before it has actually started",
        START_THREAD,
        "        self._thread = thread\n        thread.start()",
    ),
    (
        "a failed start keeps the port's read timeout",
        START_THREAD,
        "        thread.start()\n        self._thread = thread",
    ),
    (
        # False is the only way a caller learns the thread still owns the port.
        # Reported as stopped, it writes a register and reads the reply back
        # while the reader consumes it, and sees an unexplainable timeout.
        "stop() reports success whatever the join did",
        JOIN_RESULT,
        "        self._restore_timeout()",
    ),
    (
        "stop() reports failure when there was nothing to stop",
        "        if thread is None:\n            return True",
        "        if thread is None:\n            return False",
    ),
    (
        "stop() hands the port back to a thread that is still reading it",
        JOIN_RESULT,
        "        self._restore_timeout()\n"
        "        if thread.is_alive():\n"
        "            logger.error(\n"
        '                "HarpReader: thread did not exit within %.1fs; the port is still in use",'
        " timeout\n"
        "            )\n"
        "            return False",
    ),
    (
        "start() leaves the port's read timeout as it found it",
        "        self._serial.timeout = wanted",
        "        pass",
    ),
    (
        "the read timeout ignores the idle window",
        READ_TIMEOUT,
        "        wanted = _MAX_READ_TIMEOUT_S",
    ),
    (
        "the read timeout is unbounded above, so stop() cannot join",
        READ_TIMEOUT,
        "        wanted = max(_MIN_READ_TIMEOUT_S, self._idle_flush_s / 2)",
    ),
    (
        # The caller owns this port for writes and register round-trips too, so
        # a borrowed timeout that is never handed back surfaces later as a read
        # that returned early on a port nobody remembers editing.
        "stop() keeps the read timeout it borrowed",
        "        borrowed, previous = self._borrowed_timeout\n"
        "        if borrowed:\n"
        "            self._serial.timeout = previous\n"
        "            self._borrowed_timeout = (False, None)",
        "        return",
    ),
    (
        "start() does not remember what the caller's read timeout was",
        "        self._borrowed_timeout = (True, previous)",
        "        self._borrowed_timeout = (True, None)",
    ),
    (
        # The wrong place to put it back: a reader that never ran borrowed
        # nothing, and restoring regardless clears a timeout it never took.
        "stop() restores a timeout it never borrowed",
        "        self._stop_event.set()\n        thread = self._thread",
        "        self._stop_event.set()\n"
        "        self._serial.timeout = self._borrowed_timeout[1]\n"
        "        thread = self._thread",
    ),
    (
        "start() may be called on a reader that is already running",
        "        if self._thread is not None or self._stop_event.is_set():",
        "        if False:",
    ),
    (
        # The half that is easy to leave out: the stop event is latched, so a
        # reader started after a stop runs a thread that ends on its first loop
        # test and takes the whole recording with it, silently.
        "a reader that was already stopped starts a thread that exits at once",
        "        if self._thread is not None or self._stop_event.is_set():",
        "        if self._thread is not None:",
    ),
]

TYPE_GUARD = "    if type_name not in _REGISTER_TYPES:"
BUILD_SCALAR = "    register = array(address, length=length) if length > 1 else scalar(address)"
MASK_MEMBER_VALUE = '        raw = value.get("value") if isinstance(value, Mapping) else value'

# ``schema``. Written against what each schema key is *documented to mean* --
# a type is a width and a signedness, a length is an element count, a maskType
# is a decoding -- rather than against the branches as written. Every one of
# these builds a register class successfully; the damage is that it decodes
# every event of the session wrongly and looks identical from the outside.
SCHEMA_MUTANTS: list[tuple[str, str, str]] = [
    # --- the type table: one wrong row is one register decoded wrongly forever ---
    (
        "U16 registers are built one byte wide",
        '    "U16": (RegisterU16, RegisterU16Array),',
        '    "U16": (RegisterU8, RegisterU8Array),',
    ),
    (
        "S16 registers are built unsigned, so negatives read as large positives",
        '    "S16": (RegisterS16, RegisterS16Array),',
        '    "S16": (RegisterU16, RegisterU16Array),',
    ),
    (
        "Float registers are built as integers of the same width",
        '    "Float": (RegisterFloat, RegisterFloatArray),',
        '    "Float": (RegisterU32, RegisterU32Array),',
    ),
    (
        # The quiet version of an unknown type: something is built, and it is
        # the commonest width, so it works for exactly the registers that would
        # have worked anyway.
        "an unknown type silently defaults to U8",
        TYPE_GUARD + "\n        raise SchemaError(",
        '    if type_name not in _REGISTER_TYPES:\n        type_name = "U8"\n    if False:\n        raise SchemaError(',
    ),
    # --- length: an element count, not a flag ---
    (
        "length is ignored and every register is a scalar",
        BUILD_SCALAR,
        "    register = scalar(address)",
    ),
    (
        "an array register is built one element short",
        BUILD_SCALAR,
        "    register = array(address, length=length - 1) if length > 1 else scalar(address)",
    ),
    (
        "a scalar register is built as a one-element array",
        BUILD_SCALAR,
        "    register = array(address, length=length) if length >= 1 else scalar(address)",
    ),
    (
        "a length below one is accepted",
        "    if not isinstance(length, int) or isinstance(length, bool) or length < 1:",
        "    if False:",
    ),
    # --- maskType: a decoding, and the two kinds decode differently ---
    (
        "maskType is ignored, so a flag register decodes to a bare number",
        "    if mask_type is not None:",
        "    if False:",
    ),
    (
        # Bits combine. An enum of alternatives rejects Channel0|Channel1,
        # which is the ordinary case of two licks at once.
        "a bit mask is built as an enum of alternatives",
        "        return enum.IntFlag(name, dict(members))",
        "        return enum.IntEnum(name, dict(members))",
    ),
    (
        "a group mask is built as a set of combinable bits",
        "        return enum.IntEnum(name, dict(members))",
        "        return enum.IntFlag(name, dict(members))",
    ),
    (
        # BitMask defaults its mask to the whole base element; pinning it to one
        # byte truncates a 16-bit flag word to its low half.
        "a bit mask reads only the low byte of a wide register",
        "        return BitMask(enum=flags, offset=offset)",
        "        return BitMask(enum=flags, mask=0xFF, offset=offset)",
    ),
    (
        "an unknown maskType is silently ignored",
        '    raise SchemaError(\n        f"Register {name!r} names mask {mask_type!r}, which is in neither "\n        "bitMasks nor groupMasks"\n    )',
        '    return BitMask(enum=enum.IntFlag(mask_type, {"Bit0": 1}), offset=offset)',
    ),
    (
        "a masked array register masks its first element and drops the rest",
        "        if length > 1:",
        "        if False:",
    ),
    (
        # The payloadSpec branch returns before maskType is read, so declaring
        # both dropped the mask in silence -- including a mask name present in
        # neither section, which is how it was found.
        "payloadSpec and maskType on one register silently drops the mask",
        "    if payload_spec is not None and mask_type is not None:",
        "    if False:",
    ),
    (
        # "Bits" in "Bits" is substring membership, so a section written as a
        # bare string passes the maskType lookup and dies on the subscript.
        "a mask section that is not a mapping is indexed anyway",
        "    if not isinstance(section, Mapping):",
        "    if False:",
    ),
    (
        "a register with an empty name builds a class called ''",
        "        if not str(name):",
        "        if False:",
    ),
    (
        "a mask member written with a description is read as the description",
        MASK_MEMBER_VALUE,
        "        raw = value",
    ),
    (
        "a 0x-prefixed mask value is read as decimal",
        "        return int(str(raw), 0)",
        "        return int(str(raw))",
    ),
    # --- payloadSpec: named members at element offsets ---
    (
        "payloadSpec is ignored and the register decodes as a bare scalar",
        "    if payload_spec is not None:",
        "    if False:",
    ),
    (
        "payloadSpec offsets are ignored and every member reads the first element",
        '        offset = member.get("offset", 0)',
        "        offset = 0",
    ),
    # --- addresses: the key every frame is dispatched by ---
    (
        "the address is off by one",
        "    return address",
        "    return address + 1",
    ),
    (
        "an address no frame header could carry is accepted",
        "    if not 0 <= address <= _MAX_ADDRESS:",
        "    if False:",
    ),
    (
        "a non-integer address is accepted",
        "    if not isinstance(address, int) or isinstance(address, bool):",
        "    if False:",
    ),
    (
        # An address-keyed map keeps one and loses the other, so every event of
        # the loser is attributed to the winner for the whole session.
        "two registers at one address are accepted",
        "        if register.address in seen:",
        "        if False:",
    ),
    # --- the surrounding shape ---
    (
        "registers come back sorted rather than in schema order",
        "    return built",
        "    return dict(sorted(built.items()))",
    ),
    (
        "load_schema returns whatever the file happened to contain",
        "    if not isinstance(loaded, dict):",
        "    if False:",
    ),
]

CORE_SKIP = "        if address in CORE_REGISTERS:\n            continue"
CORE_PROFILE_GUARD = (
    "        if address in CORE_REGISTERS:\n"
    "            # Silently dropping it would leave a profile that looks like it\n"
    "            # asked for a column and produced none.\n"
    "            raise ValueError("
)
NO_PROFILE = "    if not profile:\n        return result"
ACCESS_LIST = (
    "    if isinstance(access, (list, tuple, set, frozenset)):\n"
    "        return frozenset(str(item) for item in access)"
)

# ``derivation``. Written against the three rules the module docstring states
# and the four validation rules it owns, not against the branches -- a rule
# obeyed for the one register a test happens to name looks fully covered from
# the code. Every mutant here produces a CSV that opens cleanly and is wrong.
DERIVATION_MUTANTS: list[tuple[str, str, str]] = [
    # --- "core registers are never columns" ---
    (
        "one core register is missing from the set, so it leaks into the columns",
        "CORE_REGISTERS = frozenset({0, 1, 2, 6, 7, 8, 10, 14})",
        "CORE_REGISTERS = frozenset({0, 1, 2, 6, 7, 8, 10})",
    ),
    (
        "no register is core, so identity and lifecycle become actions",
        "CORE_REGISTERS = frozenset({0, 1, 2, 6, 7, 8, 10, 14})",
        "CORE_REGISTERS = frozenset()",
    ),
    (
        "core registers become actions",
        CORE_SKIP,
        "        if False:\n            continue",
    ),
    (
        # The leak the profile side owns: a profile naming WhoAmI would write
        # the same number in every row of every trial.
        "a profile may record a core register",
        CORE_PROFILE_GUARD,
        "        if False:\n            raise ValueError(",
    ),
    # --- "without a profile, record nothing" ---
    (
        # The tempting default, and the reason the rule exists: a Behavior board
        # would drop ~30 columns into the CSV that nobody asked for.
        "a device with no profile records every event register",
        NO_PROFILE,
        "    if not profile:\n"
        "        for name, meta in registers.items():\n"
        "            address = _address_of(str(name), meta)\n"
        '            if address not in CORE_REGISTERS and "Event" in _access_of(meta):\n'
        "                result.recorded[address] = str(name).lower()\n"
        "        return result",
    ),
    (
        "a profile is required, so an unrecognised device has no actions either",
        NO_PROFILE,
        "    if not profile:\n        return Derived()",
    ),
    # --- "Write and Read registers become actions" ---
    (
        "Write registers do not become actions",
        '_ACTION_ACCESS = frozenset({"Read", "Write"})',
        '_ACTION_ACCESS = frozenset({"Read"})',
    ),
    (
        "Read registers do not become actions",
        '_ACTION_ACCESS = frozenset({"Read", "Write"})',
        '_ACTION_ACCESS = frozenset({"Write"})',
    ),
    (
        # An Event register is the device talking to us; there is nothing to
        # call, and offering it in the node editor is an action that times out.
        "Event registers become actions",
        '_ACTION_ACCESS = frozenset({"Read", "Write"})',
        '_ACTION_ACCESS = frozenset({"Read", "Write", "Event"})',
    ),
    (
        "nothing becomes an action",
        '_ACTION_ACCESS = frozenset({"Read", "Write"})',
        "_ACTION_ACCESS = frozenset()",
    ),
    (
        # ``access: [Write, Event]`` is ordinary. Read as one opaque value it
        # matches nothing, and a writable register vanishes from the graph.
        "a list of access modes is read as a single mode",
        ACCESS_LIST,
        "    if isinstance(access, (list, tuple, set, frozenset)):\n"
        "        return frozenset({str(access)})",
    ),
    # --- what a profile selects, and under what name ---
    (
        "the profile's column name is ignored and the register name is used",
        "        result.recorded[address] = column",
        "        result.recorded[address] = register",
    ),
    (
        # RegisterCache is built from address -> name; keyed by name it would
        # match no frame, and every column would stay at its initial value.
        "recorded is keyed by register name rather than address",
        "        result.recorded[address] = column",
        "        result.recorded[register] = column",
    ),
    (
        "a profile naming a register the device does not have is ignored",
        '            raise ValueError(f"Profile entry names unknown register {register!r}")',
        "            continue",
    ),
    (
        "a profile records the same register twice, keeping whichever came last",
        "        if address in result.recorded:",
        "        if False:",
    ),
    (
        "a profile written for another device is accepted",
        "    _check_who_am_i(schema, profile)",
        "    pass",
    ),
    (
        # The other half: a schema that does not say who it is must not be
        # rejected, or every hand-written schema fails against every profile.
        "a schema that declares no WhoAmI is rejected by every profile",
        "    if declared is None or expected is None:\n        return",
        "    if False:\n        return",
    ),
    # --- the column-name invariant this module owns ---
    (
        "an empty column name is accepted",
        "        if not isinstance(column, str) or not column:",
        "        if False:",
    ),
    (
        "a column name that is not a string is accepted",
        "        if not isinstance(column, str) or not column:",
        "        if not column:",
    ),
    (
        # "harp1:lick:state" -- nothing downstream can tell which colon was the
        # separator. BaseDevice.state_columns forbids it.
        "a colon in a column name is accepted",
        '        if ":" in column:',
        "        if False:",
    ),
    (
        "two registers may share a column name",
        "        if column in claimed:",
        "        if False:",
    ),
    (
        # The degenerate version: the collision check tracks register names, so
        # it never fires for two different registers -- which is the only case
        # that can collide.
        "the collision check tracks register names instead of column names",
        "        claimed[column] = register",
        "        claimed[register] = register",
    ),
    (
        # The hole the action-side sweep already closed, on the record side:
        # pinned by address 0 alone, this survived the whole suite, leaving a
        # profile recording OperationControl or ClockConfig unconstrained.
        "the profile-side core guard covers WhoAmI only",
        CORE_PROFILE_GUARD,
        "        if address in {0}:\n            raise ValueError(",
    ),
    # --- malformed JSON a hand-editing user can actually produce ---
    (
        # A JSON list or object is unhashable, so the membership test below
        # raises TypeError: the one malformed entry that did not come back as
        # the ValueError every other one does.
        "a non-string register reaches the membership test",
        "        if not isinstance(register, str):",
        "        if False:",
    ),
    (
        "an unknown key in a record entry is silently ignored",
        "        if unknown := sorted(str(key) for key in entry if key not in _RECORD_KEYS):",
        "        if False:",
    ),
    (
        # "record": {"LickState": "lick"} iterates its keys, so every register
        # name comes back as a malformed entry -- naming a register that does
        # exist, and pointing at the wrong half of the file.
        "a record block that is not a list is iterated anyway",
        "    if not isinstance(records, (list, tuple)):",
        "    if False:",
    ),
    (
        # ``mode`` is reserved, carried by the shipped profile, and read by
        # nothing. Rejecting it would make the shipped profile unloadable.
        "the reserved mode key is rejected",
        '_RECORD_KEYS = frozenset({"register", "as", "mode"})',
        '_RECORD_KEYS = frozenset({"register", "as"})',
    ),
    # --- the version gate that makes strict record keys survivable ---
    (
        # Without it, a 1.1 profile adding a key fails with "unknown keys:
        # scale" -- naming the key rather than the version, and reading like a
        # typo in a file that is perfectly correct for a newer GLIDER.
        "no version gate, so a future profile fails as a typo",
        '    declared = profile.get("schema_version")\n    if declared is None:\n        return',
        '    declared = profile.get("schema_version")\n    if True:\n        return',
    ),
    (
        "the gate rejects a minor bump as well as a major one",
        '    major = str(declared).split(".")[0].strip()',
        '    major = str(declared).replace(".", "")',
    ),
    (
        # load_profile is only the shipped path; Task 11 reads user files of
        # its own, so gating in one place leaves the other open.
        "derive does not gate the version, only load_profile",
        "    _check_schema_version(profile)\n    _check_who_am_i(schema, profile)",
        "    _check_who_am_i(schema, profile)",
    ),
    (
        "load_profile does not gate the version, only derive",
        "    _check_schema_version(loaded)\n    return loaded",
        "    return loaded",
    ),
    # --- WhoAmI, which is the only thing that does not overlap across boards ---
    (
        "a WhoAmI written as a string or in hex never matches",
        "        return int(str(raw), 0)",
        "        return raw",
    ),
    (
        # Compared raw, a quoted WhoAmI produced "is for WhoAmI 1400, but this
        # schema declares 1400" -- a mismatch naming two identical numbers.
        "the mismatch message hides how each value was written",
        "            f\"Profile {profile.get('name', '?')!r} is for WhoAmI {expected!r}, \"\n"
        '            f"but this schema declares {declared!r}"',
        "            f\"Profile {profile.get('name', '?')!r} is for WhoAmI {expected}, \"\n"
        '            f"but this schema declares {declared}"',
    ),
    # --- a recorded register that cannot report anything ---
    (
        "recording a register that emits no events passes without a word",
        "            logger.warning(\n"
        '                "Harp profile records register %r, which is not an Event register; "\n'
        '                "its columns will never change",\n'
        "                register,\n"
        "            )",
        "            pass",
    ),
    (
        "the warning fires for event registers instead",
        '        if "Event" not in _access_of(by_name[register]):',
        '        if "Event" in _access_of(by_name[register]):',
    ),
    # --- load_profile: a name from a device setting, resolved inside the package ---
    (
        "a profile name is used as a path",
        "    if not _PROFILE_NAME.match(name):",
        "    if False:",
    ),
    (
        "a missing profile yields an empty one instead of raising",
        "        raise FileNotFoundError(",
        "        return {}\n        raise FileNotFoundError(",
    ),
]

# (target file, mutants). Each file is restored before the next is touched.
SUITES: list[tuple[Path, list[tuple[str, str, str]]]] = [
    (FRAMES_TARGET, FRAME_MUTANTS),
    (READER_TARGET, READER_MUTANTS + HARP_READER_MUTANTS),
    (SCHEMA_TARGET, SCHEMA_MUTANTS),
    (DERIVATION_TARGET, DERIVATION_MUTANTS),
]

EXPECTED_SURVIVORS = {
    name for _, mutants in SUITES for name, _, _ in mutants if name.startswith("EQUIVALENT:")
}

RACE_WINDOW = {
    name for _, mutants in SUITES for name, _, _ in mutants if name.startswith("RACE-WINDOW:")
}


def run_tests() -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH="src;plugins/glider-harp/src")
    return subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "-x", "--no-header"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def run_suite(target: Path, mutants: list[tuple[str, str, str]], unexpected: list[str]) -> None:
    """Apply every mutant to ``target`` in turn, restoring it afterwards."""
    original = target.read_text(encoding="utf-8")
    print(f"--- {target.name} ---")
    try:
        for name, old, new in mutants:
            count = original.count(old)
            if count != 1:
                print(f"[NOT APPLIED] {name}: pattern matched {count} times, expected 1")
                unexpected.append(name)
                continue

            mutated = original.replace(old, new)
            if mutated == original:
                print(f"[NO-OP] {name}: replacement identical to original")
                unexpected.append(name)
                continue

            target.write_text(mutated, encoding="utf-8")
            if target.read_text(encoding="utf-8") != mutated:
                print(f"[NOT ON DISK] {name}")
                unexpected.append(name)
                continue

            result = run_tests()
            killed = result.returncode != 0
            should_survive = name in EXPECTED_SURVIVORS
            if killed and should_survive:
                print(f"[KILLED, EXPECTED SURVIVAL] {name} -- equivalence claim is now wrong")
                unexpected.append(name)
            elif killed:
                first = next(
                    (ln for ln in result.stdout.splitlines() if ln.startswith("FAILED")), ""
                )
                print(f"[KILLED]   {name}\n             <- {first}")
            elif should_survive:
                print(f"[SURVIVED] {name}  (equivalent, expected)")
            else:
                print(f"[SURVIVED] {name}  <-- NOT COVERED")
                unexpected.append(name)
    finally:
        target.write_text(original, encoding="utf-8")
    print()


def main() -> int:
    baseline = run_tests()
    if baseline.returncode != 0:
        print("BASELINE FAILS -- fix the suite before mutating")
        print(baseline.stdout[-3000:])
        return 1
    print("baseline: PASS\n")

    unexpected: list[str] = []
    for target, mutants in SUITES:
        run_suite(target, mutants, unexpected)

    restored = run_tests()
    print(f"restored baseline: {'PASS' if restored.returncode == 0 else 'FAIL'}")
    # Tallied over the killable names only. Subtracting every problem from the
    # killable total misattributes a killed expected-survivor, which is not a
    # killable mutant at all, to the killable set.
    all_mutants = [mutant for _, mutants in SUITES for mutant in mutants]
    excluded = EXPECTED_SURVIVORS | RACE_WINDOW
    killable = [name for name, _, _ in all_mutants if name not in excluded]
    killed_count = sum(name not in unexpected for name in killable)
    print(f"{killed_count}/{len(killable)} killable mutants killed")

    # Reported apart from the score on purpose. Folding these in would inflate
    # the headline number with kills the mutants arrange for themselves.
    race_killed = sum(name not in unexpected for name in RACE_WINDOW)
    print(f"{race_killed}/{len(RACE_WINDOW)} race-window mutants killed -- NOT coverage evidence")

    if unexpected:
        print("PROBLEMS: " + ", ".join(unexpected))
    # A score is only ever evidence about the mutants someone thought to write.
    print(
        f"({len(all_mutants)} mutants total, {len(EXPECTED_SURVIVORS)} equivalent, "
        f"{len(RACE_WINDOW)} race-window; a clean sweep is not proof of coverage)"
    )
    return 1 if unexpected or restored.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
