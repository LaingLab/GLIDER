"""
GliderCore fire-and-forget tasks must be strongly referenced until done.

asyncio.create_task returns a task the event loop only weakly references;
without an instance-level strong reference, a task like
_handle_flow_complete (which stops recorders and drives all output devices
to a safe LOW state) can be garbage-collected before it runs. FlowEngine
already retains its tasks in _running_tasks; GliderCore must do the same
for its background tasks.

shutdown() must also DRAIN those tasks before tearing down the flow
engine / hardware manager, or a flow-complete teardown scheduled just
before shutdown races the shutdown teardown over the same state.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from glider.core.glider_core import GliderCore


@pytest.mark.asyncio
async def test_flow_complete_task_is_retained_until_done():
    core = GliderCore()
    await core.initialize()
    try:
        # Simulate a running flow so _on_flow_complete schedules teardown.
        core._flow_start_monotonic = time.monotonic()
        core._on_flow_complete()

        # The teardown task must be held in the instance-level set while
        # pending (strong reference), not just in a local variable.
        assert len(core._background_tasks) == 1
        task = next(iter(core._background_tasks))
        assert not task.done()

        await task

        # The done-callback must discard the reference once complete.
        assert task not in core._background_tasks
    finally:
        await core.shutdown()


@pytest.mark.asyncio
async def test_shutdown_drains_pending_background_tasks():
    """shutdown() must let a pending background task (e.g. the flow-complete
    teardown that drives devices to safe LOW) finish BEFORE tearing down the
    flow engine / hardware manager, instead of racing or abandoning it."""
    core = GliderCore()
    await core.initialize()

    finished = {"done": False}

    async def slow_teardown():
        # Longer than everything shutdown() itself awaits (no real hardware),
        # so without an explicit drain the flag is still False on return.
        await asyncio.sleep(0.2)
        finished["done"] = True

    core._create_background_task(slow_teardown())

    await core.shutdown()

    assert finished["done"] is True, "shutdown() returned before the background task completed"
    assert core._background_tasks == set()
