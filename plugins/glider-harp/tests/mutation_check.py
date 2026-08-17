"""Mutation check for ``FrameSplitter``.

Not collected by pytest -- deliberately named without a ``test_`` prefix, since
it rewrites ``frames.py`` on disk and shells out to pytest. Run it directly:

    python plugins/glider-harp/tests/mutation_check.py

Every mutant below must be killed by ``test_frames.py``, except those marked
EQUIVALENT, which provably cannot be killed and are asserted to survive so that
a future change making one observable shows up as a failure rather than as a
silently stricter suite. That has already earned its keep once: the lazy-genexp
mutant was EQUIVALENT until ``feed`` was annotated ``list[bytes]``, and this
check is what flagged that the justification had gone stale.

This exists because the splitter's failure modes are quiet: several bugs found
in review (a stale byte after each frame, noise stranding the frames behind it)
left the yielded frames looking correct and were invisible to assertions on
output alone. A mutant that survives means a test constant is doing the work.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "plugins" / "glider-harp" / "src" / "glider_harp" / "frames.py"
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
MUTANTS: list[tuple[str, str, str]] = [
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
        "                self.checksum_errors += 1",
        "                pass",
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

EXPECTED_SURVIVORS = {name for name, _, _ in MUTANTS if name.startswith("EQUIVALENT:")}


def run_tests() -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH="src;plugins/glider-harp/src")
    return subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "-x", "--no-header"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")

    baseline = run_tests()
    if baseline.returncode != 0:
        print("BASELINE FAILS -- fix the suite before mutating")
        print(baseline.stdout[-3000:])
        return 1
    print("baseline: PASS\n")

    unexpected: list[str] = []
    try:
        for name, old, new in MUTANTS:
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

            TARGET.write_text(mutated, encoding="utf-8")
            if TARGET.read_text(encoding="utf-8") != mutated:
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
        TARGET.write_text(original, encoding="utf-8")

    restored = run_tests()
    print(f"\nrestored baseline: {'PASS' if restored.returncode == 0 else 'FAIL'}")
    killable = len(MUTANTS) - len(EXPECTED_SURVIVORS)
    print(f"{killable - len(unexpected)}/{killable} killable mutants killed")
    if unexpected:
        print("PROBLEMS: " + ", ".join(unexpected))
    return 1 if unexpected or restored.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
