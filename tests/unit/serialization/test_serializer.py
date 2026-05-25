"""
Round-trip regression test for the serializer's save → load path.

Before the 1.0 release-prep pass, the serializer's apply path called APIs
that did not exist: ``add_board(board_type=…)`` (real kwarg is
``driver_type``), ``create_node(node_class, …)`` (first arg is
``node_id: str``), ``flow_engine.connect(…)`` (no such method —
``create_connection`` and ``connect_nodes`` are the real ones), and the
save path iterated ``flow_engine.connections.items()`` (no such property
— the real attribute is ``_connections: list[dict]``, accessed via
``get_connections()``). The code was undetected because the schema tests
exercised dataclass validation but no end-to-end save/load test existed.

These tests do not require real hardware; they spin up a
``HardwareManager`` with a registered mock driver and put the system
through a real save → fresh-load → save cycle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glider.core.experiment_session import ExperimentSession
from glider.core.flow_engine import FlowEngine
from glider.core.hardware_manager import HardwareManager
from glider.hal.mock_board import MockBoard
from glider.nodes.control_nodes import register_control_nodes
from glider.nodes.experiment_nodes import register_experiment_nodes
from glider.nodes.flow_function_nodes import register_flow_function_nodes
from glider.nodes.hardware import register_hardware_nodes
from glider.nodes.interface import register_interface_nodes
from glider.nodes.interface.audio_nodes import register_audio_nodes
from glider.nodes.interface.video_nodes import register_video_nodes
from glider.nodes.logic import (
    register_comparison_nodes,
    register_logic_control_nodes,
    register_math_nodes,
)
from glider.nodes.logic.flow_nodes import register_logic_nodes
from glider.nodes.vision.zone_nodes import register_zone_nodes
from glider.serialization.serializer import ExperimentSerializer


def _make_engine_with_all_nodes() -> FlowEngine:
    engine = FlowEngine()
    register_experiment_nodes(engine)
    register_control_nodes(engine)
    register_logic_nodes(engine)
    register_flow_function_nodes(engine)
    register_zone_nodes(engine)
    register_audio_nodes(engine)
    register_video_nodes(engine)
    register_interface_nodes(engine)
    register_hardware_nodes(engine)
    register_math_nodes(engine)
    register_comparison_nodes(engine)
    register_logic_control_nodes(engine)
    return engine


@pytest.fixture
def hardware_manager_with_mock():
    """A HardwareManager wired with a registered mock driver."""
    hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    return hm


@pytest.fixture
def flow_engine_with_nodes():
    return _make_engine_with_all_nodes()


@pytest.fixture
def serializer():
    return ExperimentSerializer()


def test_save_with_no_state_does_not_raise(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """Empty-session save should not crash."""
    session = ExperimentSession()
    out = tmp_path / "empty.glider"
    serializer.save(
        out,
        session=session,
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_with_board_and_node_does_not_raise(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """Save with one board + one node + one connection — the path that
    historically crashed at every step.
    """
    hardware_manager_with_mock.add_board(board_id="board1", driver_type="mock", port=None)

    # Two nodes so we have a connection to serialize.
    flow_engine_with_nodes.create_node(node_id="start1", node_type="StartExperiment")
    flow_engine_with_nodes.create_node(node_id="end1", node_type="EndExperiment")
    flow_engine_with_nodes.create_connection(
        connection_id="c1",
        from_node_id="start1",
        from_output=0,
        to_node_id="end1",
        to_input=0,
        connection_type="exec",
    )

    session = ExperimentSession()
    out = tmp_path / "small.glider"
    serializer.save(
        out,
        session=session,
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )
    assert out.exists()


def test_round_trip_preserves_structure(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """Build a session, save, reload into a fresh engine, assert the
    structure (boards / nodes / connections) round-trips.
    """
    # Build initial state
    hardware_manager_with_mock.add_board(board_id="board1", driver_type="mock", port=None)
    flow_engine_with_nodes.create_node(node_id="start1", node_type="StartExperiment")
    flow_engine_with_nodes.create_node(node_id="end1", node_type="EndExperiment")
    flow_engine_with_nodes.create_connection(
        connection_id="c1",
        from_node_id="start1",
        from_output=0,
        to_node_id="end1",
        to_input=0,
        connection_type="exec",
    )

    session = ExperimentSession()
    session.name = "round-trip-test"
    out = tmp_path / "rt.glider"
    serializer.save(
        out,
        session=session,
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )

    # Now load into fresh containers
    fresh_session = ExperimentSession()
    fresh_engine = _make_engine_with_all_nodes()
    fresh_hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)

    schema = serializer.load(out)
    serializer.apply_to_session(
        schema,
        session=fresh_session,
        flow_engine=fresh_engine,
        hardware_manager=fresh_hm,
    )

    # Boards round-tripped
    assert (
        "board1" in fresh_hm.boards
    ), f"Board 'board1' did not round-trip. Loaded boards: {list(fresh_hm.boards)}"

    # Nodes round-tripped
    loaded_node_ids = set(fresh_engine.nodes.keys())
    assert {"start1", "end1"}.issubset(
        loaded_node_ids
    ), f"Expected nodes 'start1' and 'end1', got {loaded_node_ids}"

    # Connection round-tripped
    loaded_conns = fresh_engine.get_connections()
    assert any(
        c["from_node"] == "start1" and c["to_node"] == "end1" for c in loaded_conns
    ), f"start1 -> end1 connection lost on round-trip. Got: {loaded_conns}"


def test_node_state_round_trip(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """Per-node state (set via the node's own _state dict) must survive
    save → load. The pre-fix code dropped every node-local parameter
    silently because ``_extract_node_properties`` iterated a
    ``property_names`` attribute that no node class defined.
    """
    # DelayNode stores its duration in self._state — a canonical test
    flow_engine_with_nodes.create_node(node_id="start1", node_type="StartExperiment")
    flow_engine_with_nodes.create_node(
        node_id="delay1",
        node_type="Delay",
        state={"duration_seconds": 4.2, "use_input": False},
    )

    session = ExperimentSession()
    out = tmp_path / "state.glider"
    serializer.save(
        out,
        session=session,
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )

    fresh_engine = _make_engine_with_all_nodes()
    fresh_hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    fresh_session = ExperimentSession()

    schema = serializer.load(out)
    serializer.apply_to_session(
        schema,
        session=fresh_session,
        flow_engine=fresh_engine,
        hardware_manager=fresh_hm,
    )

    loaded_delay = fresh_engine.nodes.get("delay1")
    assert loaded_delay is not None, "delay1 node was not loaded back"
    state = loaded_delay.get_state()
    assert (
        state.get("duration_seconds") == 4.2
    ), f"DelayNode duration_seconds lost on round-trip. State: {state}"
    assert (
        state.get("use_input") is False
    ), f"DelayNode use_input lost on round-trip. State: {state}"
