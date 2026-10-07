"""The graph's delete shortcut, including macOS's Backspace-labelled "delete" key (#51)."""

import pytest
from PyQt6.QtCore import Qt

from glider.gui.node_graph.graph_view import NodeGraphView


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_delete_keys_remove_the_selected_node(qtbot, key):
    view = NodeGraphView()
    qtbot.addWidget(view)
    deleted = []
    view.node_deleted.connect(deleted.append)
    view.add_node("n1", "Delay", 0, 0)
    view._selected_nodes.append("n1")

    qtbot.keyClick(view, key)

    assert deleted == ["n1"]
