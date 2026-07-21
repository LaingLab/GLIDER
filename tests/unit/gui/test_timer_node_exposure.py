from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QMenu

from glider.gui.node_graph.graph_view import NodeGraphView
from glider.gui.node_graph.port_item import PortType
from glider.gui.panels import node_editor_controller
from glider.gui.panels.node_editor_controller import NodeEditorController
from glider.gui.panels.node_library_panel import DraggableNodeButton, NodeLibraryPanel


def test_node_library_offers_timer_node(qtbot):
    panel = NodeLibraryPanel(lambda: None, SimpleNamespace())
    qtbot.addWidget(panel)

    timer_buttons = [
        button for button in panel.findChildren(DraggableNodeButton) if button.text() == "Timer"
    ]

    assert len(timer_buttons) == 1
    assert timer_buttons[0]._node_type == "Timer"
    assert timer_buttons[0].property("nodeCategory") == "Flow"


def test_timer_node_uses_runtime_port_contract():
    node_item = SimpleNamespace(inputs=[], outputs=[])
    node_item.add_input_port = lambda name, port_type: node_item.inputs.append((name, port_type))
    node_item.add_output_port = lambda name, port_type: node_item.outputs.append((name, port_type))

    NodeEditorController.setup_node_ports(None, node_item, "Timer")

    assert node_item.inputs == [
        ("Interval", PortType.DATA),
        ("Enabled", PortType.DATA),
    ]
    assert node_item.outputs == [
        ("Tick", PortType.EXEC),
        ("Count", PortType.DATA),
    ]


def test_graph_context_menu_offers_timer_node(qtbot, monkeypatch):
    captured = {}
    monkeypatch.setattr(QMenu, "exec", lambda menu, _position: captured.setdefault("menu", menu))
    graph_view = NodeGraphView()
    qtbot.addWidget(graph_view)
    event = SimpleNamespace(pos=lambda: QPoint(), globalPos=lambda: QPoint())

    graph_view.contextMenuEvent(event)

    add_node_menu = next(
        action.menu() for action in captured["menu"].actions() if action.text() == "Add Node"
    )
    flow_menu = next(action.menu() for action in add_node_menu.actions() if action.text() == "Flow")
    assert "Timer" in [action.text() for action in flow_menu.actions()]


@pytest.mark.parametrize(
    ("node_type", "expected_category"),
    [
        ("Timer", "logic"),
        ("StartExperiment", "logic"),
        ("Loop", "interface"),
        ("Output", "hardware"),
        ("FunctionCall", "logic"),
        ("ZoneInput", "interface"),
        ("UnknownNode", "default"),
    ],
)
def test_node_category_is_consistent_for_new_and_loaded_nodes(node_type, expected_category):
    assert node_editor_controller.node_category_for_type(node_type) == expected_category
