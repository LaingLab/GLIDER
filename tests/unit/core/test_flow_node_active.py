"""The engine reports which node the running flow is in (canvas highlight)."""

from __future__ import annotations

import asyncio

from glider.core.flow_engine import FlowEngine
from glider.nodes.control_nodes import register_control_nodes
from glider.nodes.experiment_nodes import register_experiment_nodes
from glider.nodes.logic.flow_nodes import register_logic_nodes


async def test_a_node_is_active_from_entry_until_it_fires_on():
    register_experiment_nodes(FlowEngine)
    register_control_nodes(FlowEngine)
    register_logic_nodes(FlowEngine)
    engine = FlowEngine()
    engine.initialize()
    engine.create_node("start", "StartExperiment", state={})
    engine.create_node("delay", "Delay", state={"duration": 0.05})
    engine.create_node("end", "EndExperiment", state={})
    engine.create_connection("c0", "start", 0, "delay", 0, "exec")
    engine.create_connection("c1", "delay", 0, "end", 0, "exec")

    events: list[tuple[str, bool]] = []
    engine.on_node_active(lambda node_id, active: events.append((node_id, active)))
    done = asyncio.Event()
    engine.on_flow_complete(done.set)

    await engine.start()
    await asyncio.wait_for(done.wait(), 5)
    await engine.stop()

    # The Delay goes dark before End lights: one node active at a time.
    assert events.index(("delay", True)) < events.index(("delay", False))
    assert events.index(("delay", False)) < events.index(("end", True))
    # Nothing is left lit once the flow stops.
    last = dict(events)
    assert not any(last.values())


async def test_a_loop_stays_active_through_its_body_and_waits():
    register_experiment_nodes(FlowEngine)
    register_control_nodes(FlowEngine)
    register_logic_nodes(FlowEngine)
    engine = FlowEngine()
    engine.initialize()
    engine.create_node("start", "StartExperiment", state={})
    engine.create_node("loop", "Loop", state={"count": 2, "delay": 0.1})
    engine.create_node("delay", "Delay", state={"duration": 0.02})
    engine.create_node("end", "EndExperiment", state={})
    engine.create_connection("c0", "start", 0, "loop", 0, "exec")
    engine.create_connection("c1", "loop", 0, "delay", 0, "exec")
    engine.create_connection("c2", "loop", 1, "end", 0, "exec")

    events: list[tuple[str, bool]] = []
    engine.on_node_active(lambda node_id, active: events.append((node_id, active)))
    done = asyncio.Event()
    engine.on_flow_complete(done.set)

    await engine.start()
    await asyncio.wait_for(done.wait(), 5)
    await engine.stop()

    # Lit once on entry, dark once when done fires: firing body doesn't clear it.
    assert [e for e in events if e[0] == "loop"][0] == ("loop", True)
    last_body = max(i for i, e in enumerate(events) if e == ("delay", False))
    assert events.index(("loop", False)) > last_body
