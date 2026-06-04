"""
Regression test: the experiment duration reported by GliderCore must be
anchored to the *flow's* logical start/end, not to wall-clock around the
session-state transitions.

Bug being prevented: a ``StartExperiment → Delay(N) → EndExperiment`` flow
should report exactly ``N`` seconds, but previously the runner timer and
the tracking CSV's ``# Duration (s)`` footer both measured
``state→RUNNING`` through ``state→READY``. ``state→READY`` only fires
*after* ``_handle_flow_complete()`` runs the entire teardown sequence
(stop flow engine + stop recorders + drive devices low + atomic-rename
file outputs), which has 100-400ms of variable I/O latency on a Pi. A
"10s delay" displayed as 10.11s / 10.43s run-to-run.

The fix introduces three contracts on ``GliderCore``:
  1. ``_flow_start_monotonic`` is set the moment the flow becomes live
     (i.e., right after ``await self._flow_engine.start()`` returns from
     ``start_experiment``).
  2. ``_flow_end_monotonic`` is set at the *top* of ``_handle_flow_complete``
     and ``_stop_experiment_locked`` — BEFORE any teardown I/O runs.
  3. ``last_flow_duration_s`` is a property that returns
     ``_flow_end_monotonic - _flow_start_monotonic``, or ``None`` if not yet
     both set.

These tests verify each contract directly without spinning up a real
flow / camera / recorder — the contract is the unit of correctness.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from glider.core.glider_core import GliderCore


@pytest.mark.asyncio
async def test_handle_flow_complete_captures_end_time_before_teardown():
    """Simulate a 0.5s flow: set flow-start, wait, fire _on_flow_complete.

    The reported duration must reflect the 0.5s wait — NOT 0.5s plus
    whatever ``_handle_flow_complete``'s teardown took. We test by
    making the teardown artificially slow (sleep 0.3s in a patched
    ``_set_all_devices_low``) and asserting the duration is still
    ~0.5s and not ~0.8s.
    """
    core = GliderCore()
    await core.initialize()

    try:
        # Patch _set_all_devices_low to add 300ms of synthetic teardown
        # latency — simulates the variable-I/O cost on a real Pi.
        original = core._set_all_devices_low

        async def slow_safe_state():
            await asyncio.sleep(0.3)
            await original()

        core._set_all_devices_low = slow_safe_state

        # Simulate a 0.5s flow. Capture flow-start, sleep, fire complete.
        core._flow_start_monotonic = time.monotonic()
        await asyncio.sleep(0.5)
        core._on_flow_complete()

        # _on_flow_complete schedules _handle_flow_complete as a task;
        # wait for that task (including the 0.3s patched teardown) to
        # finish. We poll the duration attribute since the test has no
        # direct handle to the task.
        deadline = time.monotonic() + 5.0
        while core.last_flow_duration_s is None and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

        duration = core.last_flow_duration_s
        assert (
            duration is not None
        ), "last_flow_duration_s was never set after _on_flow_complete fired"

        # The contract: duration reflects the flow's wall-clock work
        # (0.5s), NOT the teardown (0.3s) that follows.
        assert 0.48 < duration < 0.60, (
            f"Flow duration {duration:.3f}s is wrong. Expected ~0.5s "
            f"(the flow's logical duration). If it's near 0.8s the bug "
            f"is back — the timer is capturing teardown latency too."
        )
    finally:
        await core.shutdown()


@pytest.mark.asyncio
async def test_stop_experiment_captures_end_time_before_teardown():
    """User-initiated STOP must capture _flow_end_monotonic immediately,
    before any teardown I/O. Same shape as the test above, exercising the
    ``_stop_experiment_locked`` path instead of ``_handle_flow_complete``.
    """
    core = GliderCore()
    await core.initialize()

    try:
        original = core._set_all_devices_low

        async def slow_safe_state():
            await asyncio.sleep(0.3)
            await original()

        core._set_all_devices_low = slow_safe_state

        # The session needs to exist for _stop_experiment_locked to run.
        # initialize() creates it; just sanity-check.
        assert core.session is not None

        # Simulate 0.5s of running flow, then user clicks STOP.
        core._flow_start_monotonic = time.monotonic()
        await asyncio.sleep(0.5)
        await core.stop_experiment()

        duration = core.last_flow_duration_s
        assert duration is not None, "last_flow_duration_s was never set after stop_experiment"
        assert 0.48 < duration < 0.60, (
            f"Stop-experiment duration {duration:.3f}s is wrong. "
            f"Expected ~0.5s. If it's near 0.8s the teardown latency "
            f"is leaking into the reported duration."
        )
    finally:
        await core.shutdown()


@pytest.mark.asyncio
async def test_last_flow_duration_is_none_before_first_run():
    """Fresh core: no flow has run yet, so the duration should be None
    (not 0, not a stale value)."""
    core = GliderCore()
    await core.initialize()
    try:
        assert core.last_flow_duration_s is None
    finally:
        await core.shutdown()
