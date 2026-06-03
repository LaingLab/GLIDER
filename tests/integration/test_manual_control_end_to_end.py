# tests/integration/test_manual_control_end_to_end.py
"""End-to-end: run a graph StartFunction chain from a manual-control button at idle.

Uses a real FlowEngine + real flow-function nodes + real HardwareManager/MockBoard.
A minimal core-shim provides only the surface ManualControlRunner uses.
"""

from glider.core.experiment_session import ExperimentSession
from glider.core.flow_engine import FlowEngine
from glider.core.hardware_manager import HardwareManager
from glider.gui.runner.manual_control_runner import ManualControlRunner, RunOutcome
from glider.hal.mock_board import MockBoard
from glider.nodes.flow_function_nodes import register_flow_function_nodes


class _CoreShim:
    """Minimal GliderCore-like surface used by ManualControlRunner."""

    def __init__(self, flow_engine, hardware_manager, session):
        self.flow_engine = flow_engine
        self.hardware_manager = hardware_manager
        self.session = session

    def setup_flow(self):
        # The real lazy-load path: instantiate the graph into the live engine.
        self.flow_engine.load_from_session(self.session)


def _start_end_session():
    """An ExperimentSession whose flow is a minimal StartFunction -> EndFunction chain."""
    session = ExperimentSession()
    data = session.to_dict()
    data["flow"] = {
        "nodes": [
            {
                "id": "s1",
                "node_type": "StartFunction",
                "position": [0, 0],
                "state": {"function_name": "Purge"},
                "device_id": None,
            },
            {
                "id": "e1",
                "node_type": "EndFunction",
                "position": [200, 0],
                "state": {},
                "device_id": None,
            },
        ],
        "connections": [
            {
                "id": "c1",
                "from_node": "s1",
                "from_output": 0,
                "to_node": "e1",
                "to_input": 0,
                "connection_type": "exec",
            },
        ],
    }
    return ExperimentSession.from_dict(data)


async def test_manual_run_executes_chain_at_idle():
    engine = FlowEngine()
    register_flow_function_nodes(engine)  # registers StartFunction/EndFunction/FunctionCall

    session = _start_end_session()
    session.set_manual_controls([{"slot": 0, "start_node_id": "s1", "label": "Purge"}])

    hw = HardwareManager()
    hw._boards["b1"] = MockBoard()  # MockBoard.__init__ connects immediately
    assert hw.is_any_board_connected() is True

    core = _CoreShim(engine, hw, session)
    assert not engine.nodes  # idle: graph not instantiated yet

    runner = ManualControlRunner(core)
    result = await runner.run("s1")

    assert result.outcome is RunOutcome.SUCCESS
    assert engine.get_node("s1") is not None  # lazy setup_flow ran
    assert engine.is_running is False  # never entered RUNNING flow state
