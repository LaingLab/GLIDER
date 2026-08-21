"""A plugin can contribute a node that core has never heard of.

Registration already worked -- PluginManager has a ``glider.node`` entry-point
group and a ``NODE_TYPES`` table -- but a registered node was unusable: the
library palette and the canvas context menu are both hardcoded, so it could not
be placed, and the properties panel is a chain of ``node_type == "..."``
branches, so it could not be configured. No plugin had ever shipped a node, so
neither gap had been noticed.

These drive the two extension points with a node type that exists nowhere in
core, which is the only honest way to test "core does not need to know about
it".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit

from glider.core.flow_engine import FlowEngine
from glider.nodes.base_node import ExecNode, NodeCategory, NodeDefinition
from glider.plugins import plugin_manager as pm

pytestmark = pytest.mark.usefixtures("qtbot")

PLUGIN_NODE_TYPE = "AcmeShaker"


class _PluginNode(ExecNode):
    """Stands in for a node shipped by a third-party plugin."""

    definition = NodeDefinition(
        name="Acme Shaker",
        category=NodeCategory.HARDWARE,
        description="Shake the thing",
    )

    PROPERTIES_SCHEMA = [
        {"key": "amplitude", "label": "Amplitude", "type": "float", "default": 1.5},
        {
            "key": "waveform",
            "label": "Waveform",
            "type": "enum",
            "default": "sine",
            "choices": [["sine", "Sine"], ["square", "Square"]],
        },
        {"key": "inverted", "label": "Inverted", "type": "bool", "default": False},
        {"key": "note", "label": "Note", "type": "str", "default": ""},
    ]

    async def execute(self) -> None:
        pass


@pytest.fixture
def registered_plugin_node(monkeypatch):
    """Register the node the way a plugin would, and clean up after."""
    monkeypatch.setitem(FlowEngine._node_registry, PLUGIN_NODE_TYPE, _PluginNode)
    monkeypatch.setitem(pm._PLUGIN_COMPONENTS, ("node", PLUGIN_NODE_TYPE), "glider-acme")
    return _PluginNode


# --- provenance ---------------------------------------------------------------


def test_an_unregistered_node_is_not_reported():
    """Asserted about this node, not about the registry being empty: a machine
    with glider-maimu or glider-harp installed has real plugin components in
    there, and CI installs both."""
    assert PLUGIN_NODE_TYPE not in pm.plugin_components("node")


def test_a_registered_node_is_attributed_to_its_plugin(registered_plugin_node):
    assert pm.plugin_components("node").get(PLUGIN_NODE_TYPE) == "glider-acme"


def test_provenance_is_per_kind(registered_plugin_node, monkeypatch):
    """Asking for nodes must not return devices."""
    monkeypatch.setitem(pm._PLUGIN_COMPONENTS, ("device", "AcmeSensor"), "glider-acme")

    assert PLUGIN_NODE_TYPE in pm.plugin_components("node")
    assert "AcmeSensor" not in pm.plugin_components("node")
    assert "AcmeSensor" in pm.plugin_components("device")


# --- it can be placed ---------------------------------------------------------


def _library(qtbot):
    from glider.gui.panels.node_library_panel import NodeLibraryPanel

    panel = NodeLibraryPanel(lambda: None, SimpleNamespace())
    qtbot.addWidget(panel)
    return panel


def _buttons(panel):
    from glider.gui.panels.node_library_panel import DraggableNodeButton

    return {b._node_type: b for b in panel.findChildren(DraggableNodeButton)}


def test_the_library_offers_a_plugin_node(qtbot, registered_plugin_node):
    panel = _library(qtbot)

    button = _buttons(panel).get(PLUGIN_NODE_TYPE)
    assert button is not None, "a registered plugin node never reached the palette"
    assert button.text() == "Acme Shaker", "the palette should show the node's display name"


def test_the_button_says_where_the_node_came_from(qtbot, registered_plugin_node):
    """Two plugins can register similar nodes; the row has to say whose it is."""
    panel = _library(qtbot)

    assert "glider-acme" in _buttons(panel)[PLUGIN_NODE_TYPE].toolTip()


def test_the_section_refreshes_when_a_plugin_loads(qtbot, monkeypatch):
    """A plugin installed from the Plugins window loads without a restart, so
    the palette has to be able to pick it up without one either."""
    panel = _library(qtbot)
    assert PLUGIN_NODE_TYPE not in _buttons(panel)

    monkeypatch.setitem(FlowEngine._node_registry, PLUGIN_NODE_TYPE, _PluginNode)
    monkeypatch.setitem(pm._PLUGIN_COMPONENTS, ("node", PLUGIN_NODE_TYPE), "glider-acme")
    panel.refresh_plugin_nodes()

    assert PLUGIN_NODE_TYPE in _buttons(panel)


def test_an_empty_section_says_so(qtbot, monkeypatch):
    """With no plugin nodes the section explains itself rather than sitting
    blank. Forced empty, because a developer machine (and CI) has real plugins
    installed."""
    from PyQt6.QtWidgets import QLabel

    monkeypatch.setattr(pm, "_PLUGIN_COMPONENTS", {})
    panel = _library(qtbot)

    labels = [lb.text() for lb in panel.findChildren(QLabel)]
    assert any("No plugin nodes installed" in t for t in labels)


# --- it can be configured -----------------------------------------------------


def _controller(state):
    node_config = SimpleNamespace(device_id=None, state=state)
    from glider.gui.panels.node_editor_controller import NodeEditorController

    ctrl = NodeEditorController.__new__(NodeEditorController)
    ctrl._graph_view = SimpleNamespace(
        nodes={"n1": SimpleNamespace(node_type=PLUGIN_NODE_TYPE, _actual_node_type=None)}
    )
    saved: list = []
    ctrl._session_fn = lambda: SimpleNamespace(
        get_node=lambda nid: node_config,
        update_node_state=lambda nid, patch: saved.append((nid, patch)),
    )
    ctrl._hardware_manager = SimpleNamespace(devices={}, get_device=lambda i: None)
    ctrl._core = SimpleNamespace()
    ctrl._zone_config = None
    captured: dict = {}
    ctrl._properties_dock = SimpleNamespace(setWidget=lambda w: captured.__setitem__("w", w))
    return ctrl, captured, saved


def test_a_declared_schema_becomes_a_properties_form(qtbot, registered_plugin_node):
    ctrl, captured, _ = _controller({})

    ctrl._update_properties_panel("n1")

    panel = captured["w"]
    assert panel.findChildren(QDoubleSpinBox), "no widget for the float field"
    assert panel.findChildren(QComboBox), "no widget for the enum field"
    assert panel.findChildren(QCheckBox), "no widget for the bool field"
    assert panel.findChildren(QLineEdit), "no widget for the str field"


def test_saved_values_are_shown(qtbot, registered_plugin_node):
    ctrl, captured, _ = _controller({"amplitude": 4.25, "waveform": "square", "inverted": True})

    ctrl._update_properties_panel("n1")

    panel = captured["w"]
    assert panel.findChildren(QDoubleSpinBox)[0].value() == pytest.approx(4.25)
    assert panel.findChildren(QComboBox)[0].currentData() == "square"
    assert panel.findChildren(QCheckBox)[0].isChecked() is True


def test_defaults_are_used_when_nothing_is_saved(qtbot, registered_plugin_node):
    ctrl, captured, _ = _controller({})

    ctrl._update_properties_panel("n1")

    assert captured["w"].findChildren(QDoubleSpinBox)[0].value() == pytest.approx(1.5)


def test_editing_a_field_persists_into_node_state(qtbot, registered_plugin_node):
    ctrl, captured, saved = _controller({})
    ctrl._update_properties_panel("n1")

    captured["w"].findChildren(QDoubleSpinBox)[0].setValue(9.0)

    assert ("n1", {"amplitude": 9.0}) in saved


def test_toggling_a_bool_persists(qtbot, registered_plugin_node):
    ctrl, captured, saved = _controller({})
    ctrl._update_properties_panel("n1")

    captured["w"].findChildren(QCheckBox)[0].setChecked(True)

    assert ("n1", {"inverted": True}) in saved


def test_a_node_with_no_schema_renders_nothing_extra(qtbot, monkeypatch):
    """Most nodes have no properties; an empty PROPERTIES header would be noise."""

    class _Bare(ExecNode):
        definition = NodeDefinition(name="Bare", category=NodeCategory.LOGIC)

        async def execute(self) -> None:
            pass

    monkeypatch.setitem(FlowEngine._node_registry, "Bare", _Bare)
    ctrl, captured, _ = _controller({})
    ctrl._graph_view = SimpleNamespace(
        nodes={"n1": SimpleNamespace(node_type="Bare", _actual_node_type=None)}
    )

    ctrl._update_properties_panel("n1")

    from PyQt6.QtWidgets import QLabel

    headers = [lb.text() for lb in captured["w"].findChildren(QLabel)]
    assert not any("PROPERTIES" in t for t in headers)
