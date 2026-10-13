"""The flow clock: one timebase for a running flow, and the thread that keeps it.

Three things live here.

**The clock.** :func:`now` is ``time.perf_counter`` -- QueryPerformanceCounter
on Windows, 0.1 us resolution. ``time.monotonic`` is not usable for this on
Python 3.12/Windows: it is GetTickCount64 and only advances every 15.6 ms.

**The timing thread.** Every timed wait in a flow (Delay, Loop, Timer) registers
its deadline with one daemon thread instead of sleeping on the qasync loop. The
thread waits on a condition variable until ~3 ms out, sleeps on Python's
high-resolution waitable timer until ~1 ms out, then spins on the clock. While
it holds any deadline it also holds Windows' 1 ms timer tick (see
:mod:`glider.core.timer_resolution`), which a condition-variable wait needs to
resolve finer than 15.6 ms. On macOS it waits on a kqueue instead: when macOS
rates the process as background (a hidden window under App Nap, a CI runner) it
holds back condition-variable waits, ``time.sleep`` and ``select`` by 60-100 ms,
but not a kqueue timeout. Measured on a lab desktop with the GUI thread busy,
the thread's own wake-up is typically within 0.02 ms of the deadline, worst case
~2 ms. The waiting coroutine then resumes on its event loop; that hop is outside
this module's control and is what moving the flow off the GUI thread addresses.

**Scheduled time.** Each exec hand-off carries the time it was *scheduled* for,
in a context variable that asyncio copies into every task it creates. A Delay
waits until scheduled time + duration, not until "duration after I happened to
start", so I/O latency upstream -- a ~130 ms BLE write -- is absorbed instead of
added to every iteration. Nodes that wait on the world (WaitForInput, a Button)
fire at the moment they observed it. A node's ``_fire_exec_output`` returns the
latest scheduled time its downstream chain reached, so a Loop knows when its
body ended *on the schedule*, not just in wall time.

Timed nodes also report scheduled vs. actual times to the active run's timing
log (:class:`TimingLog`), so every run measures its own precision.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import heapq
import itertools
import logging
import os
import select
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from glider.core.timer_resolution import high_resolution_timers

logger = logging.getLogger(__name__)

now: Callable[[], float] = time.perf_counter

# Wait on the condition variable until this far out; it can be woken early by a
# new, sooner deadline. Its worst observed overshoot with the 1 ms tick held is
# ~1.5 ms, so 3 ms keeps it clear of the deadline.
COARSE_MARGIN_S = 0.003
# Then sleep on the waitable timer until this far out (it overshoots ~0.5 ms),
# and spin the rest. Spinning holds the GIL, so it is kept to about a
# millisecond per deadline.
SPIN_MARGIN_S = 0.001
# The GIL switch interval for the length of a run. The timing thread and the
# loop that resumes after it both need the GIL; at the default 5 ms a busy GUI
# thread can hold it that long. Measured: worst-case resume 14 ms -> 4 ms.
RUN_SWITCH_INTERVAL_S = 0.001


# --- scheduled time ---------------------------------------------------------

_scheduled: ContextVar[float | None] = ContextVar("glider_flow_scheduled", default=None)
_chain_end: ContextVar[float | None] = ContextVar("glider_flow_chain_end", default=None)


def inherited_time() -> float | None:
    """The scheduled time handed to the running node, or None outside a flow."""
    return _scheduled.get()


def scheduled_or_now() -> float:
    """The scheduled time handed to the running node, else the current time."""
    scheduled = _scheduled.get()
    return now() if scheduled is None else scheduled


@contextlib.contextmanager
def scheduled_at(t: float) -> Iterator[None]:
    """Hand ``t`` to every task created inside the block.

    Exec callbacks create the downstream node's task synchronously, and
    asyncio copies the current context into a new task, so the downstream
    node reads ``t`` from :func:`inherited_time`.
    """
    token = _scheduled.set(t)
    try:
        yield
    finally:
        _scheduled.reset(token)


def begin_chain() -> None:
    """Start a fresh chain-end record for the current task.

    A new task inherits its parent's context, chain end included; the engine
    clears it before running a node so the node reports only its own chain.
    """
    _chain_end.set(None)


def note_chain_end(t: float) -> None:
    """Record that the current task's chain reached scheduled time ``t``."""
    current = _chain_end.get()
    if current is None or t > current:
        _chain_end.set(t)


def chain_end() -> float | None:
    """The latest scheduled time the current task's chain reached, if any."""
    return _chain_end.get()


# --- the timing thread -------------------------------------------------------


class _Entry:
    __slots__ = ("callback", "cancelled")

    def __init__(self, callback: Callable[[float], None]):
        self.callback = callback
        self.cancelled = False


class _TimingThread:
    """Fires callbacks at perf_counter deadlines, on a dedicated thread.

    Callbacks run on this thread and must be quick: :class:`ClockWait` only
    hands the fire time to its event loop.

    A deadline registered during another deadline's final ~3 ms is serviced
    right after that one, so it can be up to ~3 ms late. Waits that close
    together are rare in a flow and the timing log shows it if it happens.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, _Entry]] = []
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        # macOS: wait on a kqueue, woken through a pipe rather than notify().
        # Measured with the process clamped to utility QoS, a 20 ms wait
        # overshot ~70 ms on the condition variable, time.sleep and select
        # alike, and 1 ms on a kqueue. Raising the thread's own QoS did not
        # help. The second, empty kqueue is a sleep nothing can cut short.
        self._kq = self._sleep_kq = None
        if sys.platform == "darwin":
            self._kq, self._sleep_kq = select.kqueue(), select.kqueue()
            self._wake_r, self._wake_w = os.pipe()
            os.set_blocking(self._wake_w, False)
            self._kq.control([select.kevent(self._wake_r)], 0)

    def call_at(self, deadline: float, callback: Callable[[float], None]) -> _Entry:
        entry = _Entry(callback)
        with self._cv:
            heapq.heappush(self._heap, (deadline, next(self._seq), entry))
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="glider-flow-clock", daemon=True
                )
                self._thread.start()
            self._notify()
        return entry

    def _notify(self) -> None:
        """Wake the thread to look at a new deadline. Call with the lock held."""
        if self._kq is None:
            self._cv.notify()
            return
        with contextlib.suppress(BlockingIOError):  # the pipe is full: a wake is pending
            os.write(self._wake_w, b"\0")

    def _wait(self, timeout: float | None) -> None:
        """Wait for :meth:`_notify` or ``timeout``. Call with the lock held."""
        if self._kq is None:
            self._cv.wait(timeout)
            return
        self._cv.release()
        try:
            if self._kq.control(None, 1, timeout):
                os.read(self._wake_r, 4096)
        finally:
            self._cv.acquire()

    def _sleep(self, seconds: float) -> None:
        if self._sleep_kq is None:
            time.sleep(seconds)
        else:
            self._sleep_kq.control(None, 1, seconds)

    def _run(self) -> None:
        _raise_thread_priority()
        tick = contextlib.ExitStack()
        holding = False
        while True:
            with self._cv:
                while True:
                    while self._heap and self._heap[0][2].cancelled:
                        heapq.heappop(self._heap)
                    if not self._heap:
                        if holding:
                            tick.close()
                            holding = False
                        self._wait(None)
                        continue
                    if not holding:
                        tick.enter_context(high_resolution_timers())
                        holding = True
                    remaining = self._heap[0][0] - now()
                    if remaining <= COARSE_MARGIN_S:
                        break
                    self._wait(remaining - COARSE_MARGIN_S)
                deadline = self._heap[0][0]

            # Outside the lock, so registering a deadline never waits on this.
            remaining = deadline - now()
            if remaining > SPIN_MARGIN_S:
                self._sleep(remaining - SPIN_MARGIN_S)
            while now() < deadline:
                pass
            fired = now()

            due: list[_Entry] = []
            with self._cv:
                while self._heap and self._heap[0][0] <= fired:
                    entry = heapq.heappop(self._heap)[2]
                    if not entry.cancelled:
                        due.append(entry)
            for entry in due:
                try:
                    entry.callback(fired)
                except Exception:  # noqa: BLE001 - one bad callback must not stop the clock
                    logger.exception("Flow clock callback failed")


def _raise_thread_priority() -> None:
    """Ask the OS to run the timing thread ahead of ordinary threads.

    Windows only: on Linux raising a thread's priority needs privileges, and on
    macOS raising its QoS does not stop a background process's waits being held
    back -- the kqueue wait in :class:`_TimingThread` does that. HIGHEST,
    not TIME_CRITICAL: the thread spins for ~1 ms per deadline, and a bug that
    kept it spinning should not be able to starve the rest of the machine.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")
        thread_priority_highest = 2
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), thread_priority_highest)
    except Exception:  # noqa: BLE001 - a timing improvement must never break a run
        logger.debug("Could not raise the flow clock thread's priority", exc_info=True)


_thread = _TimingThread()


class ClockWait:
    """A wait on the flow clock. Await it for the time it fired.

    Awaiting returns the clock time at which the timing thread fired, or None
    if :meth:`cancel` ended the wait first. Cancelling the awaiting task also
    withdraws the deadline. Create, await and cancel it on the same event loop.
    """

    def __init__(self, deadline: float):
        self.deadline = deadline
        self._loop = asyncio.get_running_loop()
        self._future: asyncio.Future[float | None] = self._loop.create_future()
        if deadline <= now():
            self._entry: _Entry | None = None
            self._future.set_result(now())
        else:
            self._entry = _thread.call_at(deadline, self._fire)

    def _fire(self, fired: float) -> None:
        # Timing thread.
        try:
            self._loop.call_soon_threadsafe(self._resolve, fired)
        except RuntimeError:
            pass  # loop closed: nobody is waiting any more

    def _resolve(self, fired: float | None) -> None:
        if not self._future.done():
            self._future.set_result(fired)

    def cancel(self) -> None:
        """End the wait now; the awaiter gets None."""
        if self._entry is not None:
            self._entry.cancelled = True
        self._resolve(None)

    async def _wait(self) -> float | None:
        try:
            return await self._future
        except asyncio.CancelledError:
            self.cancel()
            raise

    def __await__(self):
        return self._wait().__await__()


def wait_until(deadline: float) -> ClockWait:
    """Register a wait for clock time ``deadline``; await the result."""
    return ClockWait(deadline)


# --- the run's timing log ----------------------------------------------------

TIMING_COLUMNS = [
    "node_id",
    "node_type",
    "kind",
    "scheduled_ms",
    "clock_ms",
    "started_ms",
    "done_ms",
    "late_ms",
    "note",
]


class TimingLog:
    """Scheduled vs. actual times for one run, as a CSV beside the event log.

    Times are milliseconds from the run's flow start on the flow clock:

    - ``scheduled_ms``: when the event was due.
    - ``clock_ms``: when the timing thread fired (waits only).
    - ``started_ms``: when the node's own code ran.
    - ``done_ms``: when a hardware call returned (hardware rows only).
    - ``late_ms``: started - scheduled. For a hardware row this is how long
      the command waited to be sent; done - started is how long it took.
    """

    def __init__(self, t0: float, path: Path | None = None, experiment: str = ""):
        self.t0 = t0
        self.path = path
        self._lock = threading.Lock()
        self._late: dict[str, list[float]] = {}
        self._io: list[float] = []
        self._overruns = 0
        self._rows = 0
        self._file = None
        self._writer = None
        if path is not None:
            self._file = open(path, "w", newline="", encoding="utf-8")  # noqa: SIM115
            self._file.write("# GLIDER Flow Timing\n")
            self._file.write(f"# Experiment,{experiment}\n")
            self._file.write(f"# Start Time,{datetime.now().isoformat()}\n")
            self._file.write("# Times are ms from flow start on the flow clock (perf_counter).\n\n")
            self._writer = csv.writer(self._file)
            self._writer.writerow(TIMING_COLUMNS)

    def _ms(self, t: float | None) -> str:
        return "" if t is None else f"{(t - self.t0) * 1000:.3f}"

    def record(
        self,
        node_id: str,
        node_type: str,
        kind: str,
        scheduled: float,
        *,
        clock: float | None = None,
        started: float | None = None,
        done: float | None = None,
        note: str = "",
    ) -> None:
        late = None if started is None else (started - scheduled) * 1000
        with self._lock:
            self._rows += 1
            if late is not None:
                self._late.setdefault(kind, []).append(late)
            if done is not None and started is not None:
                self._io.append((done - started) * 1000)
            if note.startswith("overrun"):
                self._overruns += 1
            if self._writer is not None:
                self._writer.writerow(
                    [
                        node_id,
                        node_type,
                        kind,
                        self._ms(scheduled),
                        self._ms(clock),
                        self._ms(started),
                        self._ms(done),
                        "" if late is None else f"{late:.3f}",
                        note,
                    ]
                )

    def summary(self) -> list[str]:
        """One line per kind: count and lateness percentiles."""
        lines = []
        with self._lock:
            for kind, late in sorted(self._late.items()):
                ordered = sorted(late)
                p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
                lines.append(
                    f"{kind}: {len(late)} events, late median {statistics.median(late):.2f} ms, "
                    f"p95 {p95:.2f} ms, max {ordered[-1]:.2f} ms"
                )
            if self._io:
                lines.append(
                    f"hardware call duration: median {statistics.median(self._io):.1f} ms, "
                    f"max {max(self._io):.1f} ms"
                )
            if self._overruns:
                lines.append(f"overruns: {self._overruns}")
        return lines

    def close(self) -> None:
        lines = self.summary()
        with self._lock:
            if self._file is not None:
                self._file.write("\n")
                for line in lines:
                    self._file.write(f"# Summary,{line}\n")
                self._file.close()
                self._file = None
                self._writer = None


class _Run:
    def __init__(self, log: TimingLog, previous_switch_interval: float):
        self.log = log
        self.previous_switch_interval = previous_switch_interval
        self.tick = contextlib.ExitStack()


_run: _Run | None = None


def begin_run(path: Path | None = None, experiment: str = "") -> float:
    """Start a run: zero its timing log and hold the timing settings.

    Returns the run's start time on the flow clock. Holds the 1 ms timer tick
    and a 1 ms GIL switch interval until :func:`end_run`. A second call while
    a run is active keeps the first run's log and returns its start time.
    """
    global _run
    if _run is not None:
        return _run.log.t0
    t0 = now()
    try:
        log = TimingLog(t0, path, experiment)
    except OSError as e:
        logger.error("Could not open the flow timing log %s: %s", path, e)
        log = TimingLog(t0, None, experiment)
    _run = _Run(log, sys.getswitchinterval())
    _run.tick.enter_context(high_resolution_timers())
    sys.setswitchinterval(RUN_SWITCH_INTERVAL_S)
    return t0


def end_run() -> list[str]:
    """Close the run's timing log, restore the settings, return its summary."""
    global _run
    run, _run = _run, None
    if run is None:
        return []
    sys.setswitchinterval(run.previous_switch_interval)
    run.tick.close()
    lines = run.log.summary()
    run.log.close()
    for line in lines:
        logger.info("Flow timing: %s", line)
    return lines


def run_start() -> float | None:
    """The active run's start time on the flow clock, or None."""
    return None if _run is None else _run.log.t0


def record(node: Any, kind: str, scheduled: float, **times: Any) -> None:
    """Add a row to the active run's timing log. A no-op outside a run."""
    run = _run
    if run is None:
        return
    node_id = getattr(node, "_glider_id", None) or ""
    node_type = getattr(getattr(node, "definition", None), "name", type(node).__name__)
    run.log.record(node_id, node_type, kind, scheduled, **times)
