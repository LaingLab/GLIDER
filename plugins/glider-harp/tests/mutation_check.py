"""Mutation check for every module in ``glider_harp`` that fails quietly.

``FrameSplitter``, ``RegisterCache``, ``HarpReader``, ``schema``,
``derivation``, ``board`` and ``device``.

``board`` is the odd one out and is included with lower expectations. It is a
transport shim with no state machine of its own, and the eighteen mutants break
down as: **thirteen trivial**, killed by a single direct assertion and proving
nothing beyond that the assertion exists (board_type, capabilities.pins, the
state transitions); **three of modest value** (only one pin method forgets to
refuse; scan's field order; scan swallowing an OSError into "no devices
found"); and **two worth the run** -- the pin operation that returns instead of
raising, which is a flow graph running green while no hardware moves, and the
harp stack check that tests the *module* rather than a *name*, which is exactly
what a mis-resolved harp-protocol 0.4.0 passes.

Quote that breakdown, not "18/18". Two caveats sharpen it further: the field
order is killed by a literal assertion before the sibling-agreement test is
reached, so that test is drift insurance rather than coverage; and the canary
mutant is only *reachable* because the test stubs ``sys.modules`` after import
-- in a real mis-resolve ``frames.py`` raises at import time and neither the
guard nor its message is ever reached. Do not read this section's clean sweep
as evidence of the same kind the splitter's is.

Not collected by pytest -- deliberately named without a ``test_`` prefix, since
it rewrites source files on disk and shells out to pytest. Run it directly:

    python plugins/glider-harp/tests/mutation_check.py

**Check that the tree is clean before you start and after you finish.** This
tool edits the files it is measuring, and restores each one from a ``finally``
-- which does not run if the process is killed. A run interrupted by a timeout
or a Ctrl-C therefore leaves a mutant on disk, in a source file, looking like
an ordinary edit. That has already happened once: an interrupted run left
``start()``'s thread-start guard removed, and the only thing that caught it was
an unrelated test failing afterwards. It is the one failure mode of this
technique that damages the artifact rather than merely wasting a run, and it is
invisible in the tool's own output, because the tool is no longer running.

A full sweep takes tens of minutes. Run it somewhere it will not be killed --
and if you interrupt it, ``git status`` and ``git diff`` the source directory
before doing anything else.

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
BOARD_TARGET = SOURCE / "board.py"
DEVICE_TARGET = SOURCE / "device.py"
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
        '    return int.from_bytes(payload, "little", signed=declared in _SIGNED_TYPES)',
        '    return int.from_bytes(payload, "big", signed=declared in _SIGNED_TYPES)',
    ),
    (
        "only the first payload byte is read",
        '    return int.from_bytes(payload, "little", signed=declared in _SIGNED_TYPES)',
        "    return payload[0]",
    ),
    (
        # The documented rule is that an unknown value is reported as unknown,
        # not carried forward. Zero is a real lick level, so this mutant writes
        # a reading the device never sent.
        "an empty payload reports 0 instead of no value",
        "    if not payload:\n        return None",
        "    if False:\n        return None",
    ),
    # --- decoding by the register's declared type ---
    (
        # The asymmetry the recordable gate existed to prevent: HarpDevice
        # packs -1 into an S16 as ff ff, and reading it back unsigned gives
        # 65535 -- a different number, inside one program, in a CSV that opens
        # cleanly.
        "signed registers are decoded as unsigned",
        "signed=declared in _SIGNED_TYPES",
        "signed=False",
    ),
    (
        # The other direction: 255 in a U8 is 255, not -1.
        "every register is decoded as signed",
        "signed=declared in _SIGNED_TYPES",
        "signed=True",
    ),
    (
        "only one signed width is recognised, so the others decode unsigned",
        '_SIGNED_TYPES = frozenset({"S8", "S16", "S32", "S64"})',
        '_SIGNED_TYPES = frozenset({"S8"})',
    ),
    (
        # A Float of 1.5 read as an integer is 1069547520.
        "a Float register is decoded as an integer",
        "    if declared == _FLOAT_TYPE:",
        "    if False:",
    ),
    (
        # A Float whose payload is not four bytes is a schema that disagrees
        # with the hardware. Unpacked anyway it raises struct.error, out of a
        # thread, and takes the rest of that read's frames with it.
        "a Float payload of the wrong width is unpacked anyway",
        "        if len(payload) != _FLOAT.size:\n            return None",
        "        if False:\n            return None",
    ),
    (
        # The map is the only thing that makes any of the above reachable:
        # dropped, every register decodes as it did before types existed.
        "the type map is discarded, so every register decodes unsigned",
        "            address: types.get(address) for address in self._names",
        "            address: None for address in self._names",
    ),
    (
        "a type the cache cannot decode is accepted and silently read unsigned",
        "        if unknown := sorted({str(t) for t in types.values()} - DECODABLE_TYPES):",
        "        if False:",
    ),
    (
        # ``recorded`` and ``recorded_types`` are filled together by derive, so
        # an address in one and not the other means they have drifted -- and
        # the register the caller meant to type is decoded as something else.
        "a type for an address that is not recorded is accepted",
        "        if strays := sorted(set(types) - set(self._names)):",
        "        if False:",
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
RESTORE_BODY = (
    "        borrowed, previous = self._borrowed_timeout\n"
    "        if not borrowed:\n"
    "            return\n"
    "        self._borrowed_timeout = (False, None)\n"
    "        try:\n"
    "            self._serial.timeout = previous\n"
    "        except Exception:\n"
    "            logger.warning(\n"
    '                "HarpReader: could not restore the port read timeout (the device may be "\n'
    '                "gone); continuing so the handle can still be released",\n'
    "                exc_info=True,\n"
    "            )"
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
        RESTORE_BODY,
        "        return",
    ),
    (
        # pyserial's timeout setter reconfigures the open port, so it raises
        # once the device is gone -- which is the likeliest reason a reader is
        # being stopped at all. Unguarded, it leaves stop() before the caller
        # can release the handle, and a pulled cable strands the port until
        # the process exits. Re-plugging does not help.
        "restoring the read timeout is unguarded, so a vanished device strands the port",
        RESTORE_BODY,
        "        borrowed, previous = self._borrowed_timeout\n"
        "        if not borrowed:\n"
        "            return\n"
        "        self._borrowed_timeout = (False, None)\n"
        "        self._serial.timeout = previous",
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
NON_EVENT_WARNING = (
    "            warning = (\n"
    '                f"register {register} (column {column}) is not an Event register; "\n'
    '                "its columns will never change"\n'
    "            )\n"
    '            logger.warning("Harp profile records a register that cannot report: %s", warning)\n'
    "            result.warnings.append(warning)"
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
        "    try:\n"
        "        _check_schema_version(loaded)\n"
        "    except ValueError as exc:\n"
        '        raise ValueError(f"{exc} (in {path})") from None\n'
        "    return loaded",
        "    return loaded",
    ),
    (
        # A user profile is a file the reader of the message can open and fix,
        # which they cannot do if the message does not say which file it is.
        "a malformed profile does not say which file is malformed",
        '        raise ValueError(f"Profile file {path} is not valid JSON: {exc}") from exc',
        '        raise ValueError("A profile is not valid JSON") from exc',
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
        NON_EVENT_WARNING,
        "            pass",
    ),
    (
        # The half that only surfaces an experiment later. Logged and not
        # carried, the finding exists solely in a file nobody opens during an
        # unattended run, and the CSV it is about says nothing at all.
        "the warning is logged but never carried, so it dies where nobody looks",
        "            result.warnings.append(warning)",
        "            pass",
    ),
    (
        # Without access modes on the result, a caller holding only addresses
        # cannot tell a read from a write and has to find out by sending a
        # Read to a write-only register and waiting out the timeout -- and a
        # GUI cannot tell which control to draw at all.
        "Derived carries no access modes, so a caller cannot tell a read from a write",
        "        result.access[str(name)] = modes",
        "        result.access[str(name)] = frozenset()",
    ),
    (
        "the warning fires for event registers instead",
        '        if "Event" not in _access_of(by_name[register]):',
        '        if "Event" in _access_of(by_name[register]):',
    ),
    # --- what the record can actually decode ---
    (
        # RegisterCache reads every payload as one unsigned little-endian
        # integer: an S16 of -1 records as 65535, a Float of 1.5 as
        # 1069547520. The CSV opens cleanly and is wrong, and nothing
        # downstream can tell the difference from a real reading.
        "a register the cache cannot decode may be recorded",
        "        _check_recordable(register, by_name[register])\n",
        "",
    ),
    (
        "only the type is gated, so an array register records as one number",
        '    length = meta.get("length", 1)',
        "        length = 1",
    ),
    (
        # The other half: narrowing the gate past what the cache can decode
        # would refuse registers that record perfectly well.
        "the gate is narrower than what the cache can decode",
        '_RECORDABLE_TYPES = frozenset({"U8", "U16", "U32", "U64", "S8", "S16", "S32",'
        ' "S64", "Float"})',
        '_RECORDABLE_TYPES = frozenset({"U8"})',
    ),
    (
        # Only the recorded side is gated: writes already go out with the
        # correct width and signedness, so a signed register stays usable as
        # an action.
        "the gate is applied to actions as well, taking away a working write",
        "        modes = _access_of(meta)\n        result.access[str(name)] = modes",
        "        _check_recordable(str(name), meta)\n"
        "        modes = _access_of(meta)\n"
        "        result.access[str(name)] = modes",
    ),
    # --- load_profile: a name from a device setting, resolved inside the package ---
    (
        "a profile name is used as a path",
        "    if not _PROFILE_NAME.match(name):",
        "    if False:",
    ),
    (
        "a missing profile yields a path that was never there instead of raising",
        "    raise FileNotFoundError(",
        "    return shipped\n    raise FileNotFoundError(",
    ),
    # --- profiles a lab wrote itself, beside the ones we ship ---
    (
        # The whole feature: without it a second Harp device needs a file
        # edited inside an installed package, which the next upgrade wipes.
        "the user profile directory is not read at all",
        "    if user.is_file():",
        "    if False:",
    ),
    (
        # Precedence the other way round. A lab that copied a shipped profile
        # to correct it for its own firmware would silently keep getting the
        # shipped one, and nothing would say so.
        "a shipped profile wins over the user's own copy",
        "        return user\n    if shipped.is_file():",
        "        return shipped\n    if shipped.is_file():",
    ),
    (
        "available_profiles ranks the shipped directory last, contradicting load_profile",
        "    for directory in (PROFILE_DIR, user_profile_dir()):",
        "    for directory in (user_profile_dir(), PROFILE_DIR):",
    ),
    (
        # Overriding is the feature; overriding *silently* is the failure it
        # buys. A stale local copy outliving a shipped fix has to be visible
        # somewhere other than a dropdown somebody read months ago.
        "shadowing a shipped profile happens without a word",
        "        if shipped.is_file():\n            # Not a warning about a mistake",
        "        if False:\n            # Not a warning about a mistake",
    ),
    (
        # Reached from HarpDevice's class body, so an OSError escaping does not
        # misconfigure one device -- it stops the plugin importing at all.
        "an unreadable profile directory takes the whole plugin down",
        "    except OSError:\n"
        '        logger.warning("Harp profiles: could not read %s", directory, exc_info=True)\n'
        "        return []",
        "    except ValueError:\n        return []",
    ),
    (
        # Without it RegisterCache has no type for anything and decodes every
        # payload unsigned, which is exactly the behaviour the widened gate
        # stopped being safe.
        "the declared type of a recorded register is never carried",
        '        result.recorded_types[address] = str(by_name[register].get("type"))',
        "        pass",
    ),
]

NO_PINS_BODY = (
    "        raise NotImplementedError(\n"
    '            f"HarpBoard has no GPIO pins ({op} is not supported). "\n'
    '            "Harp hardware is addressed by register, not by pin -- use a Harp "\n'
    '            "device to talk to it."\n'
    "        )"
)
HARP_CHECK_FAILURE = (
    "            self._set_state(BoardConnectionState.ERROR)\n"
    "            raise RuntimeError(\n"
    '                f"The Harp protocol stack is not usable: {e}. "'
)
SCAN_APPEND = "                results.append((label or p.device, p.device))"
# connect()'s two failure paths set ERROR too, so the line alone is not unique
# to report_transport_failure; the pair that follows it is.
TRANSPORT_FAILURE = (
    "        self._set_state(BoardConnectionState.ERROR)\n        self._notify_error(error)"
)

# ``board``. A transport shim, so the set is short and derived from the four
# things the contract actually states: no GPIO, both halves of the stack
# verified at connect, the state transitions, and what scan() reports. See the
# caveat in the module docstring before quoting this section's score.
BOARD_MUTANTS: list[tuple[str, str, str]] = [
    # --- "every pin operation raises" ---
    (
        # The whole reason the board defines these at all. A board that accepts
        # write_digital and returns is a flow graph that runs green while no
        # hardware moves, and nothing anywhere reports a problem.
        "a pin operation succeeds silently instead of raising",
        NO_PINS_BODY,
        "        return None",
    ),
    (
        # The partial version: one method forgotten. Only a test that covers
        # all five sees it.
        "write_digital alone forgets to refuse",
        "    async def write_digital(self, pin: int, value: bool) -> None:\n"
        '        self._no_pins("write_digital")',
        "    async def write_digital(self, pin: int, value: bool) -> None:\n        return None",
    ),
    (
        "the refusal does not say which operation was attempted",
        '            f"HarpBoard has no GPIO pins ({op} is not supported). "',
        '            "HarpBoard has no GPIO pins. "',
    ),
    # --- connect(): the stack check, which is the only real logic here ---
    (
        # The mutant this file was worth writing for. `import harp.protocol`
        # succeeds under the incompatible harp-protocol 0.4.0 that harp's
        # unbounded requirement resolves to, so a module-level check reports a
        # healthy stack and the mismatch resurfaces as a missing name deep
        # inside a register build, pointing at nothing.
        "connect() checks that harp.protocol imports rather than that it has the right names",
        "            from harp.protocol import HarpMessage  # noqa: F401",
        "            import harp.protocol  # noqa: F401",
    ),
    (
        "connect() never checks the harp stack at all",
        "            from harp.protocol import HarpMessage  # noqa: F401",
        "            pass",
    ),
    (
        "connect() never checks pyserial",
        "            import serial  # noqa: F401  (pyserial)",
        "            pass",
    ),
    (
        "connect() reports success when the harp stack is unusable",
        HARP_CHECK_FAILURE,
        # The trailing raise is dead code, kept only so the string continuation
        # lines that follow the anchor still complete a valid statement.
        "            return True\n"
        "            raise RuntimeError(\n"
        '                f"The Harp protocol stack is not usable: {e}. "',
    ),
    (
        "a failed connect leaves the board looking merely disconnected",
        HARP_CHECK_FAILURE,
        "            raise RuntimeError(\n"
        '                f"The Harp protocol stack is not usable: {e}. "',
    ),
    (
        "the failure message does not name the mis-resolved package",
        "                \"Install a matched pair: pip install 'harp>=0.5.0rc1' \"\n"
        "                \"'harp-protocol>=0.5.0rc1'. (harp's own requirement on \"\n"
        '                "harp-protocol has no lower bound, so a plain install can "\n'
        '                "resolve to an incompatible 0.4.0 and report success.)"',
        '                "Reinstall GLIDER."',
    ),
    (
        "connect() returns True without ever reaching CONNECTED",
        "        self._set_state(BoardConnectionState.CONNECTED)\n"
        '        logger.info("HarpBoard: transport ready")',
        '        logger.info("HarpBoard: transport ready")',
    ),
    (
        "disconnect() leaves the board connected",
        "        self._set_state(BoardConnectionState.DISCONNECTED)\n"
        '        logger.info("HarpBoard: transport released")',
        '        logger.info("HarpBoard: transport released")',
    ),
    # --- scan(): (description, port), every port, failures surfaced ---
    (
        # "No devices found" for a machine whose serial subsystem is broken
        # sends the operator hunting for a cable that is fine.
        "scan() swallows an enumeration failure and reports no ports",
        "        results = await asyncio.to_thread(_list)",
        "        try:\n"
        "            results = await asyncio.to_thread(_list)\n"
        "        except Exception:\n"
        "            results = []",
    ),
    (
        # The order this task was originally briefed with, and the reason the
        # brief was corrected. Silently wrong: both halves are strings, so the
        # panel unpacking them as (label, port) shows the COM port as the label
        # and writes the USB product string into the device's port setting,
        # which then fails to open with a message about a port that is not one.
        "scan() returns (port, description), disagreeing with SerialBoard and BLEBoard",
        SCAN_APPEND,
        "                results.append((p.device, label or p.device))",
    ),
    (
        "scan() reports an empty description rather than falling back to the port",
        SCAN_APPEND,
        "                results.append((label, p.device))",
    ),
    (
        "scan() discards the description the OS gave it",
        '                label = (p.description or "").strip()',
        '                label = ""',
    ),
    (
        # Harp boards enumerate as generic FTDI/CDC adapters, so this hides real
        # hardware and leaves no way to select it.
        "scan() hides ports whose description does not mention Harp",
        SCAN_APPEND,
        '                if "harp" not in label.lower():\n'
        "                    continue\n"
        "                results.append((label or p.device, p.device))",
    ),
    # --- identity, which the hardware map is keyed by ---
    (
        "the board reports the plain serial transport's type",
        '        return "harp"',
        '        return "serial"',
    ),
    (
        # capabilities.pins is what the GUI filters pin dropdowns off, so a
        # single entry offers the operator a pin that does not exist.
        "capabilities advertise a pin",
        '        return BoardCapabilities(name="Harp", pins={})',
        "        from glider.hal.base_board import PinCapability\n\n"
        '        return BoardCapabilities(name="Harp", pins={0: PinCapability(0)})',
    ),
    # --- a device's broken link, which the board never sees for itself ---
    (
        # The board's state is what the hardware panel shows. Left healthy, an
        # operator walking past a rig whose cable came out sees nothing wrong.
        "a reported transport failure leaves the board looking healthy",
        TRANSPORT_FAILURE,
        "        self._notify_error(error)",
    ),
    (
        # DISCONNECTED is an orderly shutdown; a broken link is not, and the
        # two want different responses from whoever is watching.
        "a broken link is reported as an orderly disconnect",
        TRANSPORT_FAILURE,
        "        self._set_state(BoardConnectionState.DISCONNECTED)\n"
        "        self._notify_error(error)",
    ),
    (
        # HardwareManager wires its own error listeners to these, so this is
        # the only path from a dead reader thread to anything watching.
        "a reported transport failure notifies nobody",
        "        self._notify_error(error)",
        "        pass",
    ),
]

REFUSE_SECOND_INIT = (
    "        if self._serial is not None or self._reader is not None:\n"
    "            raise RuntimeError("
)
FRESH_READER = (
    "                reader = HarpReader(self._serial, cache)\n                reader.start()"
)
STOP_RESULT = "            if not await asyncio.to_thread(reader.stop):"
READ_ACTION_GUARD = (
    "        reader = self._reader\n"
    "        if reader is not None and reader.is_alive():\n"
    "            raise RuntimeError("
)
CONFIRM_READBACK = (
    "        confirmed = await self._read_operation_control(timeout)\n"
    "        if confirmed & _OPERATION_MODE_MASK != mode:"
)
STANDBY_CALL = (
    "await self._set_operation_mode(_MODE_STANDBY, timeout=SHUTDOWN_ROUND_TRIP_TIMEOUT_S)"
)
STANDBY_BUDGET = "_MODE_STANDBY, timeout=SHUTDOWN_ROUND_TRIP_TIMEOUT_S"
REARM = "                self._link_failed = False\n                self._runtime_warnings = []"
READ_ACCESS_GUARD = '        if "Read" not in self._derived.access.get(register, frozenset()):'
WRITE_ACCESS_GUARD = '        if "Write" not in self._derived.access.get(register, frozenset()):'
CACHE_READ = (
    "        if column is not None and cache is not None:\n"
    "            # A recorded register is already being read, continuously, by the"
)
CLOSE_BODY = (
    "        async with self._port_lock:\n"
    "            handle, self._serial = self._serial, None\n"
    "            if handle is None:\n"
    "                return\n"
    "            try:\n"
    "                await asyncio.to_thread(handle.close)\n"
    "            except Exception as e:  # close is best-effort\n"
    '                logger.warning("Harp %s: error closing %s: %s", self._name, self._port, e)'
)
SPEC_RANGE = (
    "        bits = built.payload_type.numpy_dtype.itemsize * 8\n"
    '        if name.startswith("S"):'
)

# ``HarpDevice``. This module is composition, so the mutants are written
# against the rules it **inherited** from the four pieces it drives -- the
# reader's port ownership and one-shot lifecycle, the cache's snapshot/peek
# split, derivation's silence without a profile, and the Harp specification's
# own Standby default. Written off the code instead, every one of these would
# read as already covered: the composition is short, and each of these lines
# looks optional right up until the recording comes back empty.
#
# Not covered here, and deliberately: the ``# WARNING`` row itself lives in
# ``glider.core.data_recorder``, outside this package and outside ``TESTS``, so
# a mutant of it would survive a plugin-only run for the wrong reason. It is
# pinned by ``tests/unit/core/test_data_recorder_multicolumn.py``. What this
# section covers is the half the device owns: whether the warning exists at all.
DEVICE_MUTANTS: list[tuple[str, str, str]] = [
    # --- the Standby default: the failure this whole task exists to avoid ---
    (
        # A Harp device boots in Standby, where it answers every command and
        # emits no events. Skip this and the device connects, identifies,
        # reports no error, and records nothing for the whole session.
        "never takes the device out of Standby",
        "                await self._set_operation_mode(_MODE_ACTIVE)",
        "                pass",
    ),
    (
        # The other side of the same line. Active exists to make events flow,
        # so a device with nothing recorded wants Standby: unconditionally
        # Active streams into a port with no reader draining it.
        "a device with nothing recorded is put into streaming mode anyway",
        "            if self._derived.recorded:\n"
        "                # Active is what makes events flow, so it belongs with the",
        "            await self._set_operation_mode(_MODE_ACTIVE)\n"
        "            if self._derived.recorded:\n"
        "                # Active is what makes events flow, so it belongs with the",
    ),
    (
        # The write is not the guarantee; the readback is. A device that
        # acknowledges a mode it did not enter is indistinguishable from one
        # that entered it, until the CSV comes back empty.
        "writes Active and trusts the write, never reading it back",
        CONFIRM_READBACK,
        "        confirmed = wanted\n        if False:",
    ),
    (
        "confirms the readback against the wrong thing, so any value passes",
        "        if confirmed & _OPERATION_MODE_MASK != mode:",
        "        if False:",
    ),
    (
        "shutdown leaves the device Active, streaming into a closed port",
        STANDBY_CALL,
        "pass",
    ),
    (
        # OperationControl also carries the heartbeat and LED flags. Clobbering
        # them switches off the operation LED an operator watches, and the
        # reverse direction then never clears the mode bits at all.
        "the mode change clobbers the flags that share the register",
        "        wanted = (current & ~_OPERATION_MODE_MASK) | mode",
        "        wanted = current | mode",
    ),
    (
        "the mode change writes the mode and nothing else",
        "        wanted = (current & ~_OPERATION_MODE_MASK) | mode",
        "        wanted = mode",
    ),
    # --- round-trips only before start() or after stop(), never during ---
    (
        # The reader consumes every byte and hands only Events to the cache, so
        # a reply arriving while it runs is decoded and dropped. This does not
        # race -- it simply never returns.
        "a register read is attempted while the reader owns the port",
        READ_ACTION_GUARD,
        "            reader = self._reader\n            if False:\n                raise RuntimeError(",
    ),
    (
        # Same rule, the other end of the lifecycle: the reader is started
        # before the mode round-trip, so the reply to it can never arrive.
        "the reader is started before the device is identified and made Active",
        "            self._who_am_i = await self._read_who_am_i()",
        "            if self._derived.recorded:\n"
        "                HarpReader(self._serial, RegisterCache(self._derived.recorded)).start()\n"
        "            self._who_am_i = await self._read_who_am_i()",
    ),
    (
        # A write needs no reply, which is exactly why it stays legal during a
        # recording. Waiting for the echo makes every action time out.
        "a register write waits for its echo",
        "        await self._send(encode(MESSAGE_WRITE, address, payload_type,"
        " self._pack(register, value)))",
        "        await self._round_trip(address, payload_type)",
    ),
    # --- stop() returns False, and False means the port is still in use ---
    (
        "shutdown ignores a refused join and writes the register anyway",
        STOP_RESULT,
        "            await asyncio.to_thread(reader.stop)\n            if False:",
    ),
    (
        # The other half: a thread still inside a read on this handle, and the
        # handle closes. Indistinguishable from an unplugged cable, and
        # recorded as a read failure.
        "shutdown closes the port under a reader that would not stop",
        "                return\n            self._reader = None",
        "                pass\n            self._reader = None",
    ),
    (
        # Up to 2 s inside the join. On the loop that is a two-second freeze of
        # the GUI, the recorder and every other device, mid-recording.
        "the join runs on the event loop",
        STOP_RESULT,
        "            if not reader.stop():",
    ),
    # --- one reader per connection, and one connection at a time ---
    (
        # A second initialize() opens a second handle and leaves the first
        # reader -- a daemon thread -- on the old one, eating the frames the
        # new one is waiting for, for the rest of the process.
        "initialize() may be called twice, orphaning a reader on the old handle",
        REFUSE_SECOND_INIT,
        "        if False:\n            raise RuntimeError(",
    ),
    (
        # The plausible wrong key. ``_initialized`` is already False after a
        # shutdown whose join was refused, so the reader that refused to stop
        # would be joined by a second one on the same port.
        "the refusal is keyed on _initialized rather than on the port still being held",
        REFUSE_SECOND_INIT,
        "        if self._initialized:\n            raise RuntimeError(",
    ),
    (
        # HarpReader is one-shot: start() after a stop() raises rather than
        # resurrecting a thread whose splitter still holds the last session's
        # bytes. A cached reader makes every reconnect fail.
        "the reader is reused across initialize() instead of rebuilt",
        FRESH_READER,
        '                reader = getattr(self, "_retained_reader", None)\n'
        "                if reader is None:\n"
        "                    reader = HarpReader(self._serial, cache)\n"
        "                    self._retained_reader = reader\n"
        "                reader.start()",
    ),
    (
        # Nothing outside the device can see a handle it opened, so a failed
        # initialize that keeps one leaves the port held for the process's life
        # and every retry refused.
        "a failed initialize keeps the port it opened",
        "        except BaseException:\n            await self._close_port()",
        "        except BaseException:\n            pass",
    ),
    # --- get_state() consumes, read() must not ---
    (
        # WaitForInput and the Input node both prefer read(), and one polls at
        # 50 ms. Wired to snapshot, an Input node dropped onto this device eats
        # counts out of the CSV twenty times a second, with no symptom anywhere.
        "read() consumes the counters like get_state()",
        "        return cache.peek() if cache is not None else None",
        "        return cache.snapshot() if cache is not None else None",
    ),
    (
        # The other direction: every event would then be reported in every row
        # from the one it arrived in onward.
        "get_state() does not consume, so counts accumulate across rows",
        "        return cache.snapshot() if cache is not None else None",
        "        return cache.peek() if cache is not None else None",
    ),
    (
        # None and [] are not the same answer. DataRecorder reads [] as
        # single-column behaviour and then looks the device *type* up in a dict
        # keyed by columns, so every row gets an empty cell.
        "state_columns() returns [] rather than None when nothing is recorded",
        "        return _columns_for_recorded(recorded) if recorded else None",
        "        return _columns_for_recorded(recorded) if recorded else []",
    ),
    (
        # A device whose initialize() failed has no cache and still has a
        # profile. Answered from the cache alone it collapses to one unnamed
        # column while recording_warnings() goes on describing columns that
        # are no longer in the header.
        "state_columns() forgets the profile as soon as there is no cache",
        "        return _columns_for_recorded(recorded) if recorded else None",
        "        return None",
    ),
    (
        # state and last_ms are None while count > 0 for an untimestamped or
        # empty-payload event -- the normal representation, not an error. The
        # CSV writes None as a blank cell; dropping the key writes nothing and
        # silently shortens the row.
        "columns whose value is unknown are dropped from the row",
        "        return cache.snapshot() if cache is not None else None",
        "        return (\n"
        "            {k: v for k, v in cache.snapshot().items() if v is not None}\n"
        "            if cache is not None\n"
        "            else None\n"
        "        )",
    ),
    # --- a schema for the wrong board derives cleanly and records the wrong thing ---
    (
        "WhoAmI is read and never compared against the schema",
        "            self._check_identity()",
        "            pass",
    ),
    (
        "any WhoAmI matches",
        "        if declared != self._who_am_i:",
        "        if False:",
    ),
    (
        # The other half: a hand-written schema that omits whoAmI must not be
        # rejected, or nothing without a vendor file can be used at all.
        "a schema that declares no WhoAmI is rejected by every board",
        "        if declared is None or self._who_am_i is None:\n            return",
        "        if False:\n            return",
    ),
    # --- a link that breaks mid-recording, which nothing above can see ---
    (
        # HarpReader records why it stopped and exits. Until this, nothing
        # anywhere read that: the cache simply stopped changing, and every row
        # for the rest of a four-hour run carried the last state, a count of
        # zero and a frozen device time -- byte for byte what an animal that
        # stopped licking looks like.
        "a reader thread that died is never noticed",
        "        if reader is None or self._link_failed or reader.is_alive():",
        "        if True:",
    ),
    (
        # The recorder calls get_state() once per row; a four-hour run at 30
        # fps is 430,000 of them.
        "the link failure is reported once per row instead of once",
        "        self._link_failed = True",
        "        self._link_failed = False",
    ),
    (
        # The log is the half nobody reads during an unattended run, so this
        # is the half that has to survive: without it the CSV says nothing.
        "the link failure never reaches the recording",
        "        self._runtime_warnings.append(message)",
        "        pass",
    ),
    (
        # And the half the operator sees while the rig is still running. The
        # board's state is what the hardware panel shows and what
        # HardwareManager's error listeners are wired to.
        "the link failure never reaches the board",
        "            report(failure if failure is not None else RuntimeError(message))",
        "            pass",
    ),
    (
        # The half-fix, and the reason its test asserts on the *second*
        # detection rather than on an empty warning list. ``_check_link``
        # returns early while the latch is set, so a device reconnected after
        # a cable is replaced gets a fresh reader that nothing is watching:
        # the second pull is never noticed, and C2 regresses on every run
        # after the first.
        "the link detector is not re-armed on reconnect, so only the first pull is seen",
        REARM,
        "                self._runtime_warnings = []",
    ),
    (
        # And the other half: a new link must not inherit the dead one's
        # warning, or every recording after a reconnect is annotated with a
        # failure that is already over.
        "a reconnected device carries the dead link's warning into the new one",
        REARM,
        "                self._link_failed = False",
    ),
    (
        "a healthy reader is reported as a broken link",
        "        if reader is None or self._link_failed or reader.is_alive():",
        "        if reader is None or self._link_failed:",
    ),
    (
        "runtime warnings are collected and never reported",
        "        return [*self._derived.warnings, *self._runtime_warnings]",
        "        return list(self._derived.warnings)",
    ),
    # --- shutdown inside the caller's budget, cancellation included ---
    (
        # HardwareManager bounds shutdown() at 2 s. Two full-length round-trips
        # on top of the reader's join overrun it, wait_for cancels part-way,
        # and the port leaks.
        "the Standby courtesy is given the full round-trip budget",
        STANDBY_BUDGET,
        "_MODE_STANDBY, timeout=ROUND_TRIP_TIMEOUT_S",
    ),
    (
        # CancelledError is a BaseException, so a bare except with no finally
        # skips the close entirely and leaves a handle nothing can reopen.
        "a cancelled shutdown never releases the port",
        "        finally:",
        "        else:",
    ),
    # --- the warning that has to outlive the log ---
    (
        # The predicate and the wording live in ``derive`` (see the
        # derivation section, which mutates both); this device only has to
        # pass them on rather than drop them on the floor.
        "the device swallows the warnings derive reported",
        "        return [*self._derived.warnings, *self._runtime_warnings]",
        "        return list(self._runtime_warnings)",
    ),
    # --- which half of an action a caller gets: read or write ---
    (
        # ``derive`` puts Read *or* Write registers into actions, so an action
        # name alone says nothing about which. A Read sent to a write-only
        # register gets no reply at all, so without this the caller waits out
        # the whole round-trip timeout and is told the device did not answer
        # -- which reads as broken hardware rather than as an action that was
        # never readable. DeviceReadNode calls with no args, so it is reachable
        # from a stock node.
        "a read of a write-only register goes to the wire and times out",
        READ_ACCESS_GUARD,
        "        if False:",
    ),
    (
        # A device answers a write to a read-only register by ignoring it, so
        # the mutant is an action that reports success and does nothing for
        # the rest of the session.
        "a write to a read-only register is sent anyway",
        WRITE_ACCESS_GUARD,
        "        if False:",
    ),
    (
        # The register the reader already owns is the one a round-trip can
        # never reach. Reading it from the wire is the failure that makes
        # DeviceReadNode unusable against a recording device.
        "a recorded register is read from the wire instead of from the cache",
        CACHE_READ,
        "        if False:\n"
        "            # A recorded register is already being read, continuously, by the",
    ),
    (
        # Same rule as read(): a node polling a recorded register must not
        # consume the counters the CSV is owed.
        "the cache-backed read consumes the counters",
        '            return cache.peek().get(f"{column}_state")',
        '            return cache.snapshot().get(f"{column}_state")',
    ),
    # --- value_spec: how every GUI layer tells a command from a measurement ---
    (
        # DeviceControlsPanel classifies each control by value_spec. None for
        # everything makes every Harp action a bare button invoked with no
        # value -- so every button on the runner panel becomes a read, which
        # is wrong for every writable register and impossible for most.
        "no action declares a value, so every runner control becomes a read",
        '        if "Write" not in self._derived.access.get(action_name, frozenset()):\n'
        "            return None",
        "        if True:\n            return None",
    ),
    (
        # The other direction: a read-only register offered as a slider is a
        # control that writes to something that cannot be written.
        "a read-only register declares a value, so the runner writes to it",
        '        if "Write" not in self._derived.access.get(action_name, frozenset()):\n'
        "            return None",
        "        if False:\n            return None",
    ),
    (
        # The range is the register's own width. A U16 offered as 0..255 caps
        # the control at a quarter of a percent of what the register takes.
        "the declared range ignores the register's width",
        SPEC_RANGE,
        "        bits = 8\n" '        if name.startswith("S"):',
    ),
    # --- the port lock, which is only a lock if it spans the close ---
    (
        # Released before the close, an action already waiting on it wakes up
        # and writes into a handle that is closing underneath it -- the exact
        # interleaving the lock is documented to prevent, surfacing as an
        # OSError that looks like a hardware fault.
        "the port lock is released before the port is closed",
        CLOSE_BODY,
        "        async with self._port_lock:\n"
        "            handle, self._serial = self._serial, None\n"
        "        if handle is None:\n"
        "            return\n"
        "        try:\n"
        "            await asyncio.to_thread(handle.close)\n"
        "        except Exception as e:\n"
        '            logger.warning("Harp %s: error closing %s: %s", self._name, self._port, e)',
    ),
    (
        # _initialized is already False after a shutdown whose join was
        # refused, while the reader thread still owns the handle. Keyed on the
        # flag, an edit there rewrites _port to name a device this one is
        # still reading.
        "settings may be edited while a refusing reader still holds the port",
        "        if self._serial is not None or self._reader is not None:\n"
        '            logger.info("Harp %s: settings saved; reconnect to apply", self._name)',
        "        if self._initialized:\n"
        '            logger.info("Harp %s: settings saved; reconnect to apply", self._name)',
    ),
    # --- what a device with no profile is ---
    (
        # derive()'s third rule, at the device level: without a profile nothing
        # is recorded, and the whole control surface is still reachable.
        "a device with no profile loses its actions too",
        "        self._ensure_derivation()\n        return dict(self._actions)",
        "        self._ensure_derivation()\n"
        "        return dict(self._actions) if self._derived.recorded else {}",
    ),
    (
        # The node editor asks for actions while the hardware is still in a
        # box; a hook that only answers after initialize() offers an empty
        # dropdown and no way to tell why.
        "actions are only listed once the device is initialized",
        "        self._ensure_derivation()\n        return dict(self._actions)",
        "        if not self._initialized:\n            return {}\n        return dict(self._actions)",
    ),
    (
        # The register's declared width, not a byte. A U16 truncated to its low
        # byte writes a threshold nobody asked for and raises nothing.
        "every register is written one byte wide",
        '            return int(value).to_bytes(dtype.itemsize, "little",'
        ' signed=name.startswith("S"))',
        '            return int(value).to_bytes(1, "little", signed=name.startswith("S"))',
    ),
    (
        # The dropdown is the only screen on which anybody sees that a profile
        # exists. Enumerating only the package makes a profile a lab wrote
        # unreachable without hand-editing the saved experiment.
        "the dropdown offers only the profiles shipped in the package",
        "    for name, path in available_profiles().items():",
        '    for name, path in {p.stem: p for p in PROFILE_DIR.glob("*.json")}.items():',
    ),
    (
        # A register nothing records is answered by a round trip rather than by
        # the cache. Decoding the two differently makes one register report -1
        # or 65535 depending on whether a profile happens to name it.
        "a wire read ignores the register's declared signedness",
        "        return decode_payload(frame.payload, payload_type)",
        '        return int.from_bytes(frame.payload, "little") if frame.payload else None',
    ),
]

# (target file, mutants). Each file is restored before the next is touched.
SUITES: list[tuple[Path, list[tuple[str, str, str]]]] = [
    (FRAMES_TARGET, FRAME_MUTANTS),
    (READER_TARGET, READER_MUTANTS + HARP_READER_MUTANTS),
    (SCHEMA_TARGET, SCHEMA_MUTANTS),
    (DERIVATION_TARGET, DERIVATION_MUTANTS),
    (BOARD_TARGET, BOARD_MUTANTS),
    (DEVICE_TARGET, DEVICE_MUTANTS),
]

EXPECTED_SURVIVORS = {
    name for _, mutants in SUITES for name, _, _ in mutants if name.startswith("EQUIVALENT:")
}

RACE_WINDOW = {
    name for _, mutants in SUITES for name, _, _ in mutants if name.startswith("RACE-WINDOW:")
}


def check_anchors() -> list[str]:
    """Every mutant's anchor must match its target exactly once, and change it.

    Run before the baseline, and fatal, because a stale anchor is the one
    failure of this technique that produces *false confidence* rather than a
    wasted run. ``run_suite`` does report an unapplied mutant as
    ``[NOT APPLIED]`` and tallies it -- but a mutant that never reached disk
    looks exactly like one that was killed to anyone reading the score, and
    the score is what gets quoted. Seven went stale in a single refactor of
    the sources, across four sections, and the run still printed a number for
    all four.

    The checks are the same two ``run_suite`` makes per mutant; the point is
    the timing. Finding a stale anchor after twenty-five minutes of mutating
    is finding it too late to do anything but run again.
    """
    problems: list[str] = []
    for target, mutants in SUITES:
        source = target.read_text(encoding="utf-8")
        for name, old, new in mutants:
            count = source.count(old)
            if count != 1:
                problems.append(f"{target.name}: {name!r} matched {count} times, expected 1")
            elif source.replace(old, new) == source:
                problems.append(f"{target.name}: {name!r} replacement is identical to the source")
    return problems


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
    stale = check_anchors()
    if stale:
        # Before the baseline, and fatal. See check_anchors: an anchor that no
        # longer matches is a mutant that silently never runs, and the score
        # printed at the end counts it as though it had.
        print("STALE ANCHORS -- these mutants would never reach disk:")
        for problem in stale:
            print(f"  {problem}")
        return 1

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
