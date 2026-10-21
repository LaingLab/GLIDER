import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from glider.gui.node_graph.graph_view import NodeGraphView
from glider.gui.runner.pi_dashboard import PiDashboard

_TABS = [
    ("nodes", "Nodes"),
    ("properties", "Properties"),
    ("hardware", "Hardware"),
    ("camera", "Camera"),
    ("control", "Control"),
]


# Fixtures, not a helper that returns the widgets: qtbot.addWidget keeps only a
# weak reference, so a helper's return value dropped at the end of the test body
# destroys a shown NodeGraphView before pytest-qt's final processEvents, which
# then paints the deleted scene and segfaults. A fixture holds them to teardown.
@pytest.fixture
def dash_env(qtbot):
    graph = NodeGraphView()
    dash = PiDashboard(graph)
    qtbot.addWidget(dash)
    for key, label in _TABS:
        dash.add_tab(key, label, QLabel(label))
    dash.resize(480, 700)
    dash.show()
    return graph, dash


@pytest.fixture
def nodes_only_env(qtbot):
    graph = NodeGraphView()
    dash = PiDashboard(graph)
    qtbot.addWidget(dash)
    dash.add_tab("nodes", "Nodes", QLabel("Nodes"))
    return graph, dash


def test_graph_on_top_tabs_under_it_split_evenly(dash_env):
    graph, dash = dash_env
    assert dash.splitter().orientation() == Qt.Orientation.Vertical
    assert dash.splitter().widget(0) is graph
    assert dash.keys() == [key for key, _ in _TABS]
    top, bottom = dash.splitter().sizes()
    assert abs(top - bottom) <= 0.15 * (top + bottom)


def test_selecting_a_node_shows_properties(dash_env):
    graph, dash = dash_env
    graph.node_selected.emit("n1")
    assert dash.current() == "properties"


def test_set_current_switches_tab_and_ignores_unknown_keys(dash_env):
    _, dash = dash_env
    dash.set_current("camera")
    assert dash.current() == "camera"
    dash.set_current("nope")
    assert dash.current() == "camera"


def test_node_selected_without_a_properties_tab_leaves_the_tab_alone(nodes_only_env):
    graph, dash = nodes_only_env
    graph.node_selected.emit("n1")
    assert dash.current() == "nodes"


def test_splitter_children_cannot_collapse_and_tabs_fill_the_bar(dash_env):
    _, dash = dash_env
    splitter = dash.splitter()
    assert not splitter.childrenCollapsible()
    assert splitter.objectName() == "piDashboardSplitter"
    assert dash.tabs().objectName() == "piDashboardTabs"
    assert dash.tabs().documentMode()
    assert dash.tabs().tabBar().expanding()
    assert dash.objectName() == "piDashboard"
