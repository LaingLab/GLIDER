from types import SimpleNamespace

from PyQt6.QtCore import Qt

from glider.gui.node_graph.graph_view import NodeGraphView
from glider.gui.panels.node_library_panel import DraggableNodeButton, NodeLibraryPanel


def _panel(qtbot, node_type="Delay", **kw):
    graph = NodeGraphView()
    qtbot.addWidget(graph)
    panel = NodeLibraryPanel(session_fn=lambda: None, graph_view=graph, **kw)
    qtbot.addWidget(panel)
    return graph, panel, _button(panel, node_type)


def _button(panel, node_type):
    return next(b for b in panel.findChildren(DraggableNodeButton) if b._node_type == node_type)


def test_tap_adds_the_node_at_the_visible_centre(qtbot):
    graph, _panel_, button = _panel(qtbot)
    with qtbot.waitSignal(graph.node_created) as sig:
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    centre = graph.mapToScene(graph.viewport().rect().center())
    assert sig.args == ["Delay", centre.x(), centre.y()]


def test_click_still_adds_the_node_on_desktop(qtbot):
    # Clicking a library button already dropped the node mid-graph before the
    # Pi layout existed; desktop never locks, so that must stay that way.
    graph, _p, button = _panel(qtbot)
    with qtbot.waitSignal(graph.node_created) as sig:
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert sig.args[0] == "Delay"


def test_tap_is_refused_while_the_graph_is_locked(qtbot):
    graph, _p, button = _panel(qtbot)
    graph.set_locked(True)
    with qtbot.assertNotEmitted(graph.node_created):
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def test_zone_button_taps_its_own_zone_type(qtbot):
    graph, panel, _b = _panel(qtbot)
    panel.refresh_zones(SimpleNamespace(zones=[SimpleNamespace(id="z1", name="Nest")]))
    with qtbot.waitSignal(graph.node_created) as sig:
        qtbot.mouseClick(_button(panel, "ZoneInput:z1"), Qt.MouseButton.LeftButton)
    assert sig.args[0] == "ZoneInput:z1"
