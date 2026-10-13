"""Whole flows keep their schedule on the flow clock.

The field case: ``Start -> Loop(10, delay 0) -> Maimu -> Delay 10 s`` ran
101.34 s, because the Delay counted from when the ~130 ms BLE write returned.
Here a fake hardware node stands in for the write with a set latency.
"""

from __future__ import annotations

import asyncio

from glider.core import flow_clock
from glider.core.flow_engine import FlowEngine
from glider.nodes.base_node import (
    HardwareNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)
from glider.nodes.control_nodes import register_control_nodes
from glider.nodes.experiment_nodes import register_experiment_nodes
from glider.nodes.logic.flow_nodes import register_logic_nodes


class _SlowWrite(HardwareNode):
    """Sends a command that takes ``latency_s`` to land, like a BLE write."""

    definition = NodeDefinition(
        name="SlowWrite",
        category=NodeCategory.HARDWARE,
        description="Test command with a set latency",
        inputs=[PortDefinition(name="exec", port_type=PortType.EXEC)],
        outputs=[PortDefinition(name="exec", port_type=PortType.EXEC)],
    )

    def __init__(self):
        super().__init__()
        self.onsets: list[float] = []
        self._device = object()

    async def hardware_operation(self) -> None:
        self.onsets.append(flow_clock.now())
        await asyncio.sleep(self._state.get("latency_s", 0.0))
        await self._fire_exec_output("exec")


async def _run(nodes: list[tuple], wires: list[tuple], timeout: float = 10.0):
    """Run a flow to its End node. Returns (seconds taken, engine)."""
    register_experiment_nodes(FlowEngine)
    register_control_nodes(FlowEngine)
    register_logic_nodes(FlowEngine)
    FlowEngine.register_node("SlowWrite", _SlowWrite)
    engine = FlowEngine()
    engine.initialize()
    for node_id, node_type, state in nodes:
        engine.create_node(node_id, node_type, state=state)
    for i, (src, out, dst) in enumerate(wires):
        engine.create_connection(f"c{i}", src, out, dst, 0, "exec")
    done = asyncio.Event()
    engine.on_flow_complete(done.set)

    start = flow_clock.now()
    await engine.start()
    await asyncio.wait_for(done.wait(), timeout)
    took = flow_clock.now() - start
    await engine.stop()
    return took, engine


def _gaps(onsets: list[float]) -> list[float]:
    return [b - a for a, b in zip(onsets, onsets[1:], strict=False)]


def _loop_write_delay(count_from: str = "schedule"):
    nodes = [
        ("start", "StartExperiment", {}),
        ("loop", "Loop", {"count": 5, "delay": 0.0}),
        ("write", "SlowWrite", {"latency_s": 0.1}),
        ("delay", "Delay", {"duration": 0.2, "count_from": count_from}),
        ("end", "EndExperiment", {}),
    ]
    wires = [("start", 0, "loop"), ("loop", 0, "write"), ("write", 0, "delay"), ("loop", 1, "end")]
    return nodes, wires


async def test_loop_stimulus_delay_runs_on_time():
    took, engine = await _run(*_loop_write_delay())

    # 5 x 0.2 s. Counting from each write's return made it 5 x 0.3 s.
    assert abs(took - 1.0) < 0.06, took
    gaps = _gaps(engine.get_node("write").onsets)
    assert all(abs(g - 0.2) < 0.02 for g in gaps), gaps


async def test_count_from_previous_step_keeps_the_old_behaviour():
    took, engine = await _run(*_loop_write_delay(count_from="previous"))

    gaps = _gaps(engine.get_node("write").onsets)
    assert all(abs(g - 0.3) < 0.03 for g in gaps), gaps
    assert took > 1.4


async def test_loop_period_absorbs_a_slow_write():
    nodes = [
        ("start", "StartExperiment", {}),
        ("loop", "Loop", {"count": 5, "delay": 0.2}),
        ("write", "SlowWrite", {"latency_s": 0.1}),
        ("end", "EndExperiment", {}),
    ]
    wires = [("start", 0, "loop"), ("loop", 0, "write"), ("loop", 1, "end")]
    took, engine = await _run(nodes, wires)

    assert abs(took - 1.0) < 0.06, took
    gaps = _gaps(engine.get_node("write").onsets)
    assert all(abs(g - 0.2) < 0.02 for g in gaps), gaps


async def _write_delay_write(latency_s: float, delay_s: float) -> float:
    nodes = [
        ("start", "StartExperiment", {}),
        ("a", "SlowWrite", {"latency_s": latency_s}),
        ("delay", "Delay", {"duration": delay_s}),
        ("b", "SlowWrite", {}),
        ("end", "EndExperiment", {}),
    ]
    wires = [("start", 0, "a"), ("a", 0, "delay"), ("delay", 0, "b"), ("b", 0, "end")]
    _, engine = await _run(nodes, wires)
    return engine.get_node("b").onsets[0] - engine.get_node("a").onsets[0]


async def test_a_delay_absorbs_latency_shorter_than_itself():
    gap = await _write_delay_write(latency_s=0.1, delay_s=0.3)
    assert abs(gap - 0.3) < 0.02, gap


async def test_a_delay_shorter_than_the_latency_waits_its_full_duration():
    # The write overran the whole 0.1 s delay. Firing at once would put the
    # two commands back to back; the delay counts from now instead.
    gap = await _write_delay_write(latency_s=0.3, delay_s=0.1)
    assert gap >= 0.39, gap


async def test_timer_ticks_stay_on_their_grid():
    nodes = [
        ("timer", "Timer", {"interval": 0.1}),
        ("write", "SlowWrite", {"latency_s": 0.03}),
    ]
    register_logic_nodes(FlowEngine)
    FlowEngine.register_node("SlowWrite", _SlowWrite)
    engine = FlowEngine()
    engine.initialize()
    for node_id, node_type, state in nodes:
        engine.create_node(node_id, node_type, state=state)
    engine.create_connection("c0", "timer", 0, "write", 0, "exec")
    await engine.start()
    await asyncio.sleep(0.75)
    await engine.stop()

    gaps = _gaps(engine.get_node("write").onsets)
    assert len(gaps) >= 5
    assert all(abs(g - 0.1) < 0.02 for g in gaps), gaps


async def test_a_run_logs_every_timed_step(tmp_path):
    path = tmp_path / "run_timing.csv"
    flow_clock.begin_run(path)
    try:
        await _run(*_loop_write_delay())
    finally:
        summary = flow_clock.end_run()

    rows = [
        line.split(",")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ][1:]
    kinds = [r[2] for r in rows]
    assert kinds.count("loop") == 5
    assert kinds.count("hardware") == 5
    assert kinds.count("delay") == 5
    # Each write's duration is its done - started: the fake's 100 ms.
    io = [float(r[6]) - float(r[5]) for r in rows if r[2] == "hardware"]
    assert all(95 < ms < 140 for ms in io), io
    assert any(line.startswith("delay: 5 events") for line in summary)
