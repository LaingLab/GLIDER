"""A6 (flow load reports what it skipped) and A8 (exec fired while paused is replayed)."""

import asyncio

from glider.core.experiment_session import ConnectionConfig, ExperimentSession, NodeConfig
from glider.core.flow_engine import FlowEngine
from glider.core.hardware_manager import HardwareManager
from glider.nodes.base_node import (
    GliderNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

EXECUTED: list[str] = []


class _Step(GliderNode):
    definition = NodeDefinition(
        name="step",
        category=NodeCategory.LOGIC,
        inputs=[PortDefinition(name="in", port_type=PortType.EXEC)],
        outputs=[PortDefinition(name="out", port_type=PortType.EXEC)],
    )

    def update_event(self) -> None:  # pragma: no cover - trivial
        pass

    async def execute(self) -> None:
        EXECUTED.append(self._glider_id)

    def bind_device(self, device) -> None:  # pragma: no cover - never bound here
        self.device = device


FlowEngine.register_node("step", _Step)


def test_load_keeps_a_node_whose_device_is_missing_and_reports_failures():
    session = ExperimentSession()
    session.add_node(NodeConfig(id="kept", node_type="step", device_id="gone"))
    session.add_node(NodeConfig(id="bad", node_type="NoSuchPluginNode"))
    session.add_connection(
        ConnectionConfig(id="c1", from_node="kept", from_output=0, to_node="bad", to_input=0)
    )
    engine = FlowEngine(HardwareManager())

    engine.load_from_session(session)

    assert "kept" in engine.nodes  # H1: never dropped for a missing device
    failures = engine.load_failures
    assert any("'bad'" in f for f in failures)
    assert any("Connection kept -> bad" in f for f in failures)
    warnings = engine.consume_load_warnings()
    assert any("'gone'" in w and "missing" in w for w in warnings)
    assert all(f in warnings for f in failures)


async def test_exec_fired_while_paused_runs_on_resume():
    EXECUTED.clear()
    engine = FlowEngine()
    engine.create_node("a", "step")
    engine.create_node("b", "step")
    engine.create_connection("c", "a", 0, "b", 0, connection_type="exec")
    await engine.start()
    await engine.pause()

    await engine._nodes["a"]._fire_exec_output("out")
    await asyncio.sleep(0)
    assert EXECUTED == []  # held, not dropped

    await engine.resume()
    for _ in range(5):
        await asyncio.sleep(0)
    assert EXECUTED == ["b"]
    await engine.stop()
