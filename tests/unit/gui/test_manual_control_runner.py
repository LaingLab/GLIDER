# tests/unit/gui/test_manual_control_runner.py
# (asyncio_mode = "auto" — async tests need no decorator; no `import pytest` needed.)
import asyncio

from glider.gui.runner.manual_control_runner import (
    ManualControlRunner,
    RunOutcome,
)


class _FakeEngine:
    def __init__(self, nodes=None):
        self._nodes = nodes or {}
        self.setup_called = 0

    @property
    def nodes(self):
        return self._nodes

    def get_node(self, node_id):
        return self._nodes.get(node_id)


class _FakeHW:
    def __init__(self, connected):
        self._connected = connected

    def is_any_board_connected(self):
        return self._connected


class _FakeCore:
    def __init__(self, *, connected=True, nodes=None):
        self.flow_engine = _FakeEngine(nodes)
        self.hardware_manager = _FakeHW(connected)

    def setup_flow(self):
        self.flow_engine.setup_called += 1


async def test_no_hardware_returns_not_ready():
    runner = ManualControlRunner(_FakeCore(connected=False))
    result = await runner.run("s1")
    assert result.outcome is RunOutcome.NO_HARDWARE


async def test_unknown_node_returns_not_found():
    runner = ManualControlRunner(_FakeCore(connected=True, nodes={"other": object()}))
    result = await runner.run("s1")
    assert result.outcome is RunOutcome.NOT_FOUND


async def test_busy_guard_rejects_overlapping_runs():
    release = asyncio.Event()

    class _SlowRunner:
        def __init__(self, start_node_id, engine):
            pass

        async def execute(self):
            await release.wait()

    core = _FakeCore(connected=True, nodes={"s1": object()})
    runner = ManualControlRunner(core, function_runner_factory=_SlowRunner)

    first = asyncio.create_task(runner.run("s1"))
    await asyncio.sleep(0)
    assert runner.is_busy is True
    second = await runner.run("s1")
    assert second.outcome is RunOutcome.BUSY

    release.set()
    assert (await first).outcome is RunOutcome.SUCCESS
    assert runner.is_busy is False
    assert core.flow_engine.setup_called == 0


async def test_lazy_setup_flow_when_engine_empty():
    executed = {}

    class _OkRunner:
        def __init__(self, start_node_id, engine):
            executed["id"] = start_node_id

        async def execute(self):
            executed["ran"] = True

    core = _FakeCore(connected=True, nodes={})

    def _setup():
        core.flow_engine._nodes["s1"] = object()
        core.flow_engine.setup_called += 1

    core.setup_flow = _setup
    runner = ManualControlRunner(core, function_runner_factory=_OkRunner)

    result = await runner.run("s1")
    assert core.flow_engine.setup_called == 1
    assert result.outcome is RunOutcome.SUCCESS
    assert executed == {"id": "s1", "ran": True}


async def test_execution_error_returns_error_outcome():
    class _BoomRunner:
        def __init__(self, start_node_id, engine):
            pass

        async def execute(self):
            raise RuntimeError("boom")

    core = _FakeCore(connected=True, nodes={"s1": object()})
    runner = ManualControlRunner(core, function_runner_factory=_BoomRunner)
    result = await runner.run("s1")
    assert result.outcome is RunOutcome.ERROR
    assert "boom" in (result.error or "")
    assert runner.is_busy is False
