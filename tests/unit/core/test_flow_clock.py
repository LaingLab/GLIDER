"""The flow clock: its timing thread, scheduled-time context and timing log."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from glider.core import flow_clock


@pytest.fixture(autouse=True)
def _no_run_left_behind():
    yield
    flow_clock.end_run()


async def test_wait_fires_at_the_deadline_and_never_early():
    for offset in (0.01, 0.02, 0.05):
        deadline = flow_clock.now() + offset
        fired = await flow_clock.wait_until(deadline)
        resumed = flow_clock.now()
        assert fired >= deadline
        assert resumed >= deadline
        # Generous: this is the thread's wake-up on a shared CI runner. The
        # slow-marked test below holds it to the real target.
        assert fired - deadline < 0.015


async def test_a_deadline_already_past_returns_at_once():
    start = flow_clock.now()
    fired = await flow_clock.wait_until(start - 1.0)
    assert fired is not None
    assert flow_clock.now() - start < 0.01


async def test_cancel_ends_the_wait_with_none():
    wait = flow_clock.wait_until(flow_clock.now() + 10.0)
    asyncio.get_running_loop().call_later(0.02, wait.cancel)
    start = flow_clock.now()
    assert await wait is None
    assert flow_clock.now() - start < 0.5


async def test_cancelling_the_awaiting_task_withdraws_the_deadline():
    wait = flow_clock.wait_until(flow_clock.now() + 10.0)
    task = asyncio.ensure_future(wait._wait())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wait._entry is not None and wait._entry.cancelled


async def test_waits_fire_in_deadline_order():
    base = flow_clock.now() + 0.05
    order: list[int] = []

    async def wait(i: int, deadline: float) -> None:
        await flow_clock.wait_until(deadline)
        order.append(i)

    # Registered out of order, 10 ms apart.
    await asyncio.gather(*(wait(i, base + i * 0.01) for i in (3, 0, 4, 1, 2)))
    assert order == [0, 1, 2, 3, 4]


async def test_scheduled_time_reaches_tasks_created_inside_the_block():
    async def read() -> float | None:
        return flow_clock.inherited_time()

    with flow_clock.scheduled_at(42.0):
        inside = asyncio.ensure_future(read())
    outside = asyncio.ensure_future(read())

    assert await inside == 42.0
    assert await outside is None
    assert flow_clock.scheduled_or_now() != 42.0


async def test_chain_end_keeps_the_latest_and_starts_fresh_per_chain():
    flow_clock.begin_chain()
    assert flow_clock.chain_end() is None
    flow_clock.note_chain_end(5.0)
    flow_clock.note_chain_end(3.0)
    assert flow_clock.chain_end() == 5.0
    flow_clock.begin_chain()
    assert flow_clock.chain_end() is None


def test_record_outside_a_run_does_nothing():
    flow_clock.record(SimpleNamespace(_glider_id="d"), "delay", 1.0, started=1.0)


def test_a_run_writes_scheduled_and_actual_times(tmp_path):
    path = tmp_path / "x_timing.csv"
    t0 = flow_clock.begin_run(path, experiment="Maimu")
    node = SimpleNamespace(_glider_id="delay_1", definition=SimpleNamespace(name="Delay"))
    flow_clock.record(node, "delay", t0 + 1.0, clock=t0 + 1.0001, started=t0 + 1.002)
    flow_clock.record(node, "hardware", t0 + 2.0, started=t0 + 2.001, done=t0 + 2.131)
    summary = flow_clock.end_run()

    text = path.read_text(encoding="utf-8")
    assert "# Experiment,Maimu" in text
    assert "delay_1,Delay,delay,1000.000,1000.100,1002.000,,2.000," in text
    assert "delay_1,Delay,hardware,2000.000,,2001.000,2131.000,1.000," in text
    assert any(line.startswith("hardware call duration: median 130.0 ms") for line in summary)
    assert "# Summary,delay: 1 events" in text


def test_a_run_holds_a_fine_switch_interval_and_restores_it():
    before = sys.getswitchinterval()
    flow_clock.begin_run()
    assert sys.getswitchinterval() == flow_clock.RUN_SWITCH_INTERVAL_S
    flow_clock.end_run()
    assert sys.getswitchinterval() == before


def test_a_second_begin_keeps_the_running_log():
    t0 = flow_clock.begin_run()
    assert flow_clock.begin_run() == t0
    assert flow_clock.run_start() == t0
    flow_clock.end_run()
    assert flow_clock.run_start() is None


@pytest.mark.slow
async def test_the_timing_thread_wakes_within_a_couple_of_ms():
    late = []
    for _ in range(50):
        deadline = flow_clock.now() + 0.02
        fired = await flow_clock.wait_until(deadline)
        late.append((fired - deadline) * 1000)
    late.sort()
    assert late[int(0.95 * len(late)) - 1] < 2.0, late
