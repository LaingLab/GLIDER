"""
LoopNode keeps its iterations on a fixed schedule.

Iteration k is due at start + k * delay. The field case: a Loop driving a BLE
stimulator, whose write took ~130 ms to land, ran 10 iterations 1.34 s long.
Timing each iteration from its own start lets every late wakeup push all later
iterations back; aiming each wait at the absolute deadline absorbs it instead.
"""

from __future__ import annotations

import asyncio
import time

from glider.nodes.control_nodes import LoopNode


def _loop(count: int, delay: float, body_s=0.0) -> tuple[LoopNode, list[float]]:
    """A LoopNode whose body records its onset and then takes ``body_s``.

    ``body_s`` may be a callable of the iteration index, for bodies that
    vary in length.
    """
    node = LoopNode()
    node.set_state({"count": count, "delay": delay})
    onsets: list[float] = []

    async def body(i: int) -> None:
        onsets.append(time.perf_counter())
        await asyncio.sleep(body_s(i) if callable(body_s) else body_s)

    def on_output(name, _value):
        if name == "body":
            return asyncio.ensure_future(body(len(onsets)))
        return None

    node._update_callbacks.append(on_output)
    return node, onsets


async def test_late_wakeups_do_not_accumulate(monkeypatch):
    # Every wait overshoots by 50 ms, as a busy qasync loop can. Timed from
    # each iteration's own start, the fifth gap would end 5 x 50 ms late;
    # aimed at absolute deadlines, every onset is late by the same 50 ms.
    async def oversleep(self, seconds):
        await asyncio.sleep(seconds + 0.05)

    monkeypatch.setattr(LoopNode, "_wait", oversleep)
    node, onsets = _loop(count=6, delay=0.2, body_s=0.02)

    await node.execute()

    span = onsets[-1] - onsets[0]
    assert span < 5 * 0.2 + 0.05 + 0.04, f"onsets drifted: span {span:.3f}s"


async def test_body_time_counts_toward_the_period():
    # A 130 ms body inside a 300 ms period -- the Maimu-write case. Measured
    # against the grid onsets[0] + k * 0.3, not gap by gap: a loaded macOS CI
    # runner woke every wait ~100 ms late, which all lands in the first gap.
    # What must not happen is lateness growing by the body's 130 ms each time.
    node, onsets = _loop(count=5, delay=0.3, body_s=0.13)

    await node.execute()

    late = [t - (onsets[0] + k * 0.3) for k, t in enumerate(onsets)]
    # Never early. onsets[0] itself lands a little after the Loop's own start,
    # by however long the first body took to be scheduled, hence the slack.
    assert min(late) > -0.02, late
    assert late[-1] - late[1] < 0.1, late  # adding each body would make this 0.39


async def test_an_overrun_does_not_trigger_catch_up_iterations():
    # Iteration 0 takes 3.5 periods. The next one starts as soon as it ends,
    # and the ones after run on a fresh 0.1 s grid from there rather than
    # firing back to back to make up the missed deadlines.
    node, onsets = _loop(count=4, delay=0.1, body_s=lambda i: 0.35 if i == 0 else 0.0)

    await node.execute()

    # 20 ms of slack on the body: its asyncio.sleep runs on time.monotonic,
    # which ticks every 15.6 ms on Windows, and asyncio ends a sleep up to one
    # tick early -- a 0.35 s body measured 0.344 s on windows-2022.
    assert onsets[1] - onsets[0] >= 0.35 - 0.02, onsets
    # Checked against the grid, not gap by gap: a wakeup that lands late is
    # taken out of the next wait, so the gap after it is short by design.
    # Catching up would put these 0.1 and 0.2 s early.
    late = [t - (onsets[1] + k * 0.1) for k, t in enumerate(onsets[1:])]
    assert min(late) > -0.02, late


async def test_stop_ends_the_wait_between_iterations():
    node, onsets = _loop(count=0, delay=10.0)
    task = asyncio.ensure_future(node.execute())
    while not onsets:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)  # now inside the 10 s wait

    start = time.monotonic()
    await node.stop()
    await asyncio.wait_for(task, timeout=1.0)

    assert time.monotonic() - start < 0.5
    assert len(onsets) == 1
