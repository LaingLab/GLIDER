"""FlowFunctionRunner concurrency fixes (D-7): single-in-flight, timeout honesty,
shared runner, and cancel-on-clear.

These target three code-verified pre-existing bugs: a per-caller runner clobbering
the shared completion callback (C1), the timeout discarding the completion signal
(C2), and New/Open leaking in-flight tasks (C6).
"""

from __future__ import annotations

import asyncio

from glider.core.flow_engine import FlowEngine
from glider.nodes.flow_function_nodes import EndFunctionNode, FlowFunctionRunner


class _FakeEngine:
    def __init__(self):
        self.nodes: dict = {}
        self._connections: list = []

    def get_node(self, nid):
        return self.nodes.get(nid)


class _Start:
    """Fake StartFunction: on execute, (optionally wait on a gate then) reach End."""

    def __init__(self, engine, end_id, gate=None, active=None):
        self._engine = engine
        self._end_id = end_id
        self._gate = gate
        self._active = active

    async def execute(self):
        if self._active is not None:
            self._active.append(1)  # entered the function body
        if self._gate is not None:
            await self._gate.wait()
        # Reaching EndFunction fires the runner's completion callback.
        await self._engine.get_node(self._end_id).execute()
        if self._active is not None:
            self._active.pop()


def _wire(gate=None, active=None):
    eng = _FakeEngine()
    end = EndFunctionNode()
    eng.nodes["end"] = end
    eng.nodes["start"] = _Start(eng, "end", gate=gate, active=active)
    eng._connections = [{"from_node": "start", "to_node": "end"}]
    return eng


async def test_single_invocation_completes():
    runner = FlowFunctionRunner("start", _wire())
    assert await runner.execute() is True
    assert runner.running is False


async def test_concurrent_calls_serialize_never_overlap():
    gate = asyncio.Event()
    active: list = []
    runner = FlowFunctionRunner("start", _wire(gate=gate, active=active))

    t1 = asyncio.create_task(runner.execute())
    t2 = asyncio.create_task(runner.execute())
    await asyncio.sleep(0)  # let the first enter and block on the gate
    assert len(active) <= 1  # only one invocation inside the body at a time
    gate.set()
    await asyncio.gather(t1, t2)
    assert len(active) <= 1  # never overlapped throughout


async def test_timeout_notifies_and_cancels_so_caller_recovers():
    # A body whose gate never opens models a hung chain (a node raised and
    # stopped propagation, or hardware went away). The runner must notify, then
    # cancel and return False — never leave the caller awaiting forever.
    gate = asyncio.Event()  # deliberately never set
    runner = FlowFunctionRunner("start", _wire(gate=gate))
    notified: list = []

    result = await runner.execute(timeout=0.02, on_timeout=lambda: notified.append(True))

    assert notified == [True]  # reported unresponsive
    assert result is False  # did not complete — it was cancelled
    assert runner.running is False  # lock released; a later run can proceed


async def test_recovers_and_can_run_again_after_a_timeout():
    # After a hung run is cancelled, the same shared runner must accept a fresh
    # invocation (the wedge bug left it permanently busy).
    gate = asyncio.Event()
    runner = FlowFunctionRunner("start", _wire(gate=gate))

    assert await runner.execute(timeout=0.02) is False  # first run hangs → cancelled
    gate.set()  # a later run's chain completes normally
    assert await runner.execute(timeout=0.02) is True
    assert runner.running is False


async def test_missing_start_node_returns_false():
    eng = _FakeEngine()  # no "start" node
    runner = FlowFunctionRunner("start", eng)
    assert await runner.execute() is False


def test_engine_shares_one_runner_per_function():
    fe = FlowEngine()
    r1 = fe.get_function_runner("s1")
    r2 = fe.get_function_runner("s1")
    r3 = fe.get_function_runner("s2")
    assert r1 is r2 and r1 is not r3


async def test_clear_cancels_in_flight_tasks():
    fe = FlowEngine()
    task = asyncio.create_task(asyncio.sleep(10))
    fe._running_tasks.add(task)
    fe.clear()
    await asyncio.sleep(0)
    assert task.cancelled() or task.cancelling()
    assert fe._function_runners == {}
