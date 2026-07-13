"""
Regression test: DelayNode timing precision must be robust to event-loop
pressure.

Reproduces the bug from the field: a user ran the same
``StartExperiment → Delay(10s) → EndExperiment`` flow three times and
saw durations of 10.27, 10.01, 10.08 — 263ms of jitter. Trace timestamps
showed ``asyncio.sleep(10)`` itself waking up 0-260ms late. Root cause:
under qasync the asyncio loop is multiplexed with Qt's event loop, and
several timers tick during the sleep:

  - ``data_recorder._sampling_loop`` at 100 ms (CSV write + flush)
  - ``RunControlPanel._elapsed_timer`` at 50 ms (setText → Qt repaint)
  - ``_stall_timer`` at 50 ms
  - ``_device_refresh_timer`` at 250 ms

Qt repaints in particular block the loop long enough that
``asyncio.sleep``'s wakeup callback is delayed by up to hundreds of ms.

The fix moves the sleep onto a thread via
``asyncio.to_thread(threading.Event.wait, timeout)``. ``Event.wait`` is
backed by an OS condition variable (precision ~1 ms on macOS/Linux),
runs on a worker thread that isn't competing with Qt for the event
loop, and remains cleanly cancellable via ``Event.set()`` from
``DelayNode.stop()``.

This test simulates the loop pressure with a tight family of asyncio
tasks ticking every 1 ms and asserts the measured delay stays within
30 ms of the requested 0.5 s — comfortably tighter than the 260 ms
field bug, loose enough to absorb CI scheduling jitter.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from glider.nodes.logic.flow_nodes import DelayNode

# These assert sub-100ms / sub-200ms timing on a delay whose result is
# delivered through the asyncio loop. Shared CI runners have too much
# scheduling jitter for that to be stable, so they are excluded from CI
# (`pytest -m "not slow"`) and run locally via `pytest -m slow`.
pytestmark = pytest.mark.slow


async def _loop_pressure(stop_event: asyncio.Event, burst_ms: float, period_ms: float) -> None:
    """Periodically burn a short burst of CPU on the event loop to
    simulate the kind of pressure that comes from Qt repaints + ticking
    timers under qasync.

    Real GLIDER during an idle Delay has a data_recorder.sample tick at
    100ms (write + flush), a runner-panel elapsed_timer at 50ms (setText
    → Qt repaint), and a stall-check timer at 50ms. None of these burn
    much CPU individually, but Qt repaints can block the loop for tens
    of ms occasionally. ``burst_ms`` and ``period_ms`` let the test
    model both the "realistic" case (1ms every 50ms) and the
    "pathological" case the field bug was hit in.
    """
    while not stop_event.is_set():
        start = time.monotonic()
        x = 0
        while time.monotonic() - start < burst_ms / 1000:
            x += 1  # noqa: F841 — burn CPU
        # Sleep until the next period boundary so the CPU-burn rate is
        # bounded; otherwise this task starves the loop.
        spent = (time.monotonic() - start) * 1000
        remaining_ms = max(0.0, period_ms - spent)
        await asyncio.sleep(remaining_ms / 1000)


@pytest.mark.asyncio
async def test_delay_node_is_accurate_under_realistic_loop_pressure():
    """Run DelayNode under simulated realistic GLIDER loop pressure.

    Models the load profile during a real ``Delay`` between
    StartExperiment and EndExperiment:
      - data_recorder tick at 100ms with ~1ms CPU per write
      - elapsed_timer + stall_timer at 50ms with ~0.5ms CPU each
      - device_refresh_timer at 250ms with ~1ms CPU

    Total: ~3% CPU on the event loop. Under this load, an Event.wait-on-
    thread sleep should return within 25 ms of the requested time.
    """
    requested = 0.5

    stop_pressure = asyncio.Event()
    # Three realistic-cadence pressure tasks (data sampler, UI timer, stall check).
    pressure_tasks = [
        asyncio.create_task(_loop_pressure(stop_pressure, burst_ms=1.0, period_ms=100)),
        asyncio.create_task(_loop_pressure(stop_pressure, burst_ms=0.5, period_ms=50)),
        asyncio.create_task(_loop_pressure(stop_pressure, burst_ms=0.5, period_ms=50)),
        asyncio.create_task(_loop_pressure(stop_pressure, burst_ms=1.0, period_ms=250)),
    ]

    try:
        node = DelayNode()
        node._state["duration"] = requested
        node._state["unit"] = "seconds"

        # We don't have a real flow engine; intercept _fire_exec_output so
        # the call doesn't fail on missing wiring. The timing measurement
        # is what we care about.
        async def _fake_fire(*args, **kwargs):
            pass

        node._fire_exec_output = _fake_fire  # type: ignore[assignment]

        start = time.monotonic()
        await node.execute()
        elapsed = time.monotonic() - start
    finally:
        stop_pressure.set()
        await asyncio.gather(*pressure_tasks, return_exceptions=True)

    # Must be at least the requested duration (sleep cannot wake early).
    assert elapsed >= requested - 0.005, (
        f"DelayNode returned in {elapsed:.4f}s, earlier than requested {requested}s. "
        "Did the sleep mechanism skip waiting entirely?"
    )

    # Must not overshoot by more than 25ms under realistic loop pressure.
    # The prior asyncio.sleep impl routinely overshot 50-260ms in the field;
    # Event.wait on a worker thread should be within a few ms.
    overshoot_ms = (elapsed - requested) * 1000
    assert elapsed < requested + 0.025, (
        f"DelayNode took {elapsed:.4f}s for a {requested}s delay — "
        f"{overshoot_ms:.1f}ms overshoot. The loop-pressure jitter bug "
        f"is back. The sleep should run on a worker thread, not on the "
        f"asyncio event loop."
    )


@pytest.mark.asyncio
async def test_delay_node_can_be_stopped_mid_sleep():
    """stop() must cancel the sleep promptly, even though it runs on a thread.

    threading.Event.wait responds to .set() within microseconds, so a
    stop() call during a 60s sleep should return within ~10ms.
    """
    node = DelayNode()
    node._state["duration"] = 60.0
    node._state["unit"] = "seconds"

    async def _fake_fire(*args, **kwargs):
        pass

    node._fire_exec_output = _fake_fire  # type: ignore[assignment]

    # Kick off the long delay, then stop it after 100ms.
    delay_task = asyncio.create_task(node.execute())
    await asyncio.sleep(0.1)

    cancel_start = time.monotonic()
    await node.stop()

    # Wait for execute() to wind down. Should be near-instant after stop().
    try:
        await asyncio.wait_for(delay_task, timeout=1.0)
    except TimeoutError:
        delay_task.cancel()
        pytest.fail(
            "DelayNode.stop() did not cancel the in-flight sleep within 1s; "
            "the worker-thread sleep is not respecting the cancel signal."
        )

    cancel_elapsed = time.monotonic() - cancel_start
    assert cancel_elapsed < 0.2, (
        f"DelayNode.stop() took {cancel_elapsed*1000:.1f}ms to cancel a "
        "60s sleep. Expected near-instant (< 200ms). Event.set() may not "
        "be wired into the wait path."
    )
