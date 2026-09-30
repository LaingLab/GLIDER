"""Undo/redo of node delete/create keeps the real type, state and edges (F4)."""

import pytest

from glider.core.experiment_session import ExperimentSession
from glider.gui.commands import UndoStack
from glider.gui.node_graph.graph_view import NodeGraphView
from glider.gui.panels.node_editor_controller import NodeEditorController

pytestmark = pytest.mark.usefixtures("qtbot")


def _controller(qtbot):
    view = NodeGraphView()
    qtbot.addWidget(view)
    session = ExperimentSession()
    ctrl = NodeEditorController(view, lambda: session, None, UndoStack(), core=None)
    ctrl.connect_graph_signals()
    return ctrl, view, session


def test_undo_delete_restores_zone_node_and_its_connection(qtbot):
    ctrl, view, session = _controller(qtbot)
    ctrl._on_node_created("ZoneInput:z1", 0, 0)
    ctrl._on_node_created("Output", 200, 0)
    zone_id, out_id = (n.id for n in session.flow.nodes)
    ctrl._on_connection_created(zone_id, 2, out_id, 0, "exec")
    view.add_connection(session.flow.connections[0].id, zone_id, 2, out_id, 0)

    view.node_deleted.emit(zone_id)
    view.remove_node(zone_id)
    assert session.get_node(zone_id) is None and not session.flow.connections

    ctrl._undo_stack.undo()

    restored = session.get_node(zone_id)
    assert restored.node_type == "ZoneInput"
    assert restored.state["zone_id"] == "z1"
    assert view.nodes[zone_id]._actual_node_type == "ZoneInput"
    assert len(session.flow.connections) == 1
    assert len(view.connections) == 1


def test_redo_create_keeps_state_and_runner_visibility(qtbot):
    ctrl, view, session = _controller(qtbot)
    ctrl._on_node_created("ZoneInput:z1", 0, 0)
    node_id = session.flow.nodes[0].id

    ctrl._undo_stack.undo()
    assert session.get_node(node_id) is None
    ctrl._undo_stack.redo()

    assert session.get_node(node_id).state["zone_id"] == "z1"
    assert view.nodes[node_id]._actual_node_type == "ZoneInput"
