"""Mutation check for ``FrameSplitter`` and ``RegisterCache``.

Not collected by pytest -- deliberately named without a ``test_`` prefix, since
it rewrites source files on disk and shells out to pytest. Run it directly:

    python plugins/glider-harp/tests/mutation_check.py

Every mutant below must be killed by the suite, except those marked EQUIVALENT,
which provably cannot be killed and are asserted to survive so that a future
change making one observable shows up as a failure rather than as a silently
stricter suite. That has already earned its keep once: the lazy-genexp mutant
was EQUIVALENT until ``feed`` was annotated ``list[bytes]``, and this check is
what flagged that the justification had gone stale.

This exists because both modules fail quietly. Several splitter bugs found in
review (a stale byte after each frame, noise stranding the frames behind it)
left the yielded frames looking correct and were invisible to assertions on
output alone; a register cache that drops an event writes a CSV that is wrong
in no visible way at all. A mutant that survives means a test constant is doing
the work.

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
  mutant itself inserts a ``time.sleep`` into the window it opens**. Under
  CPython's GIL the concurrency test also passes with the lock removed and no
  sleep added: 15/15 green, and a stress harness at 16 writers x 3000 events
  found zero lost updates over 40 trials. So their kill shows the suite catches
  an artificially widened race -- not that the lock as written is load-bearing
  on this interpreter. The lock stays regardless: it is correct, costs three
  attribute writes, and is genuinely required on free-threaded builds where the
  GIL is not quietly doing the work. Do not quote these two alongside the
  killable score as if they measured the same thing.
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
]

# ``ingest``'s guarded write, and ``snapshot``'s read-and-clear. Quoted whole so
# the race-window mutants below can remove the lock and widen the window in one
# edit -- a lock only fails where two threads meet, so mutating it in place would
# just be dropping a statement no single-threaded test can see. Read the
# RACE_WINDOW note above before quoting their result as coverage.
INGEST_WRITE = (
    "        with self._lock:\n"
    "            register.state = value\n"
    "            register.count += 1\n"
    "            register.last_ms = last_ms"
)
SNAPSHOT_BODY = (
    "        with self._lock:\n"
    "            for address, name in self._names.items():\n"
    "                register = self._states[address]\n"
    '                values[f"{name}:state"] = register.state\n'
    '                values[f"{name}:count"] = register.count\n'
    '                values[f"{name}:last_ms"] = register.last_ms\n'
    "                register.count = 0"
)
LOOKUP = (
    "        register = self._states.get(frame.address)\n"
    "        if register is None:\n"
    "            return"
)
COLUMNS = '            for column in (f"{name}:state", f"{name}:count", f"{name}:last_ms")'

READER_MUTANTS: list[tuple[str, str, str]] = [
    # --- "state -- latest value seen, persists across snapshots" ---
    (
        "state keeps the first value instead of the latest",
        "            register.state = value",
        "            register.state = register.state if register.state is not None else value",
    ),
    (
        "snapshot clears state as well as count",
        "                register.count = 0",
        "                register.count = 0\n                register.state = None",
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
        "count is not cleared on read",
        "                register.count = 0",
        "                pass",
    ),
    (
        "count is cleared before it is reported, so every row reads zero",
        '                values[f"{name}:count"] = register.count\n'
        '                values[f"{name}:last_ms"] = register.last_ms\n'
        "                register.count = 0",
        "                register.count = 0\n"
        '                values[f"{name}:count"] = register.count\n'
        '                values[f"{name}:last_ms"] = register.last_ms',
    ),
    (
        # Dedented, so only the last register in the loop is cleared: the CSV
        # then double-counts every register but one.
        "only one register's count is cleared per snapshot",
        "                register.count = 0\n        return values",
        "            register.count = 0\n        return values",
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
        "last_ms keeps the first event's time instead of the latest",
        "            register.last_ms = last_ms",
        "            if register.last_ms is None:\n                register.last_ms = last_ms",
    ),
    (
        "snapshot clears last_ms",
        "                register.count = 0",
        "                register.count = 0\n                register.last_ms = None",
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
    # --- columns() is the CSV header; it must be exactly what snapshot fills ---
    (
        "columns omits last_ms",
        COLUMNS,
        '            for column in (f"{name}:state", f"{name}:count")',
    ),
    (
        "columns uses a different separator from snapshot",
        COLUMNS,
        '            for column in (f"{name}_state", f"{name}_count", f"{name}_last_ms")',
    ),
    (
        "columns orders the fields differently from snapshot",
        COLUMNS,
        '            for column in (f"{name}:count", f"{name}:state", f"{name}:last_ms")',
    ),
    (
        # Every value in the shared dict is one the cache really did report, so
        # a caller batching rows sees row N rewrite itself when row N+1 is
        # taken, with nothing anywhere looking wrong.
        "snapshot returns one shared dict reused across calls",
        "        values: dict[str, int | float | None] = {}",
        '        values: dict[str, int | float | None] = getattr(self, "_shared", {})\n'
        "        self._shared = values",
    ),
    (
        # Device time is Seconds(U32) + Micros(U16) at 32 us per tick, so the
        # sub-millisecond digits are resolution the device really has.
        "last_ms is rounded to whole milliseconds",
        "frame.timestamp * _MS_PER_SECOND",
        "float(int(frame.timestamp * _MS_PER_SECOND))",
    ),
    (
        "the caller's register map is aliased rather than copied",
        "        self._names = dict(registers)",
        "        self._names = registers",
    ),
    (
        "duplicate register names are accepted",
        "        if duplicates:",
        "        if False:",
    ),
    # --- the lock; see the RACE_WINDOW note, these are not coverage evidence ---
    (
        "RACE-WINDOW: snapshot reads and clears without the lock",
        SNAPSHOT_BODY,
        "        for address, name in self._names.items():\n"
        "            register = self._states[address]\n"
        '            values[f"{name}:state"] = register.state\n'
        '            values[f"{name}:count"] = register.count\n'
        '            __import__("time").sleep(0.0002)\n'
        '            values[f"{name}:last_ms"] = register.last_ms\n'
        "            register.count = 0",
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

# (target file, mutants). Each file is restored before the next is touched.
SUITES: list[tuple[Path, list[tuple[str, str, str]]]] = [
    (FRAMES_TARGET, FRAME_MUTANTS),
    (READER_TARGET, READER_MUTANTS),
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
