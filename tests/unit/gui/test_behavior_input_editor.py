"""The Behavior Input node has to be reachable from the app, not just the engine.

Before this, the node was registered with the flow engine and had no GUI
presence at all: no palette entry, no port configuration, no properties. The
port fallback in ``setup_node_ports`` gave it one generic input and one generic
output, so its real shape -- no inputs, four outputs -- was wrong on the canvas
and ``On Enter`` at index 2 could never be wired to anything.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QComboBox, QSpinBox

from glider.core.live_signals import LiveSignalBus
from glider.gui.node_graph.port_item import PortType
from glider.gui.panels.node_editor_controller import (
    NodeEditorController,
    node_category_for_type,
)

pytestmark = pytest.mark.usefixtures("qtbot")


def _controller(state, behaviors=None):
    node_config = SimpleNamespace(device_id=None, state=state)
    ctrl = NodeEditorController.__new__(NodeEditorController)  # skip heavy __init__
    ctrl._graph_view = SimpleNamespace(
        nodes={"b1": SimpleNamespace(node_type="BehaviorInput", _actual_node_type=None)}
    )
    saved: list = []
    ctrl._session_fn = lambda: SimpleNamespace(
        get_node=lambda nid: node_config,
        update_node_state=lambda nid, patch: saved.append((nid, patch)),
    )
    ctrl._hardware_manager = SimpleNamespace(devices={}, get_device=lambda i: None)
    bus = LiveSignalBus()
    if behaviors is not None:
        bus.set_behaviors(behaviors)
    ctrl._core = SimpleNamespace(live_signals=bus)
    ctrl._zone_config = None
    captured: dict = {}
    ctrl._properties_dock = SimpleNamespace(setWidget=lambda w: captured.__setitem__("w", w))
    return ctrl, captured, saved


def _widgets(captured):
    panel = captured["w"]
    combo = next(c for c in panel.findChildren(QComboBox))
    spin = next(s for s in panel.findChildren(QSpinBox))
    return combo, spin


# --- placement ----------------------------------------------------------------


def test_the_library_offers_a_behavior_input_button(qtbot):
    """Registered but unplaceable is not a feature."""
    from glider.gui.panels.node_library_panel import DraggableNodeButton, NodeLibraryPanel

    panel = NodeLibraryPanel(lambda: None, SimpleNamespace())
    qtbot.addWidget(panel)

    types = {b._node_type for b in panel.findChildren(DraggableNodeButton)}
    assert "BehaviorInput" in types


def test_it_styles_as_an_interface_node(qtbot):
    assert node_category_for_type("BehaviorInput") == "interface"


# --- ports --------------------------------------------------------------------


class _NodeItem:
    def __init__(self):
        self.inputs: list[tuple] = []
        self.outputs: list[tuple] = []

    def add_input_port(self, name, port_type):
        self.inputs.append((name, port_type))

    def add_output_port(self, name, port_type):
        self.outputs.append((name, port_type))


def _ports(node_type):
    # Ports for a type outside the hand-written table are read off its
    # NodeDefinition, so the registry has to be populated to find it.
    from glider.core.flow_engine import FlowEngine
    from glider.nodes.hardware import register_hardware_nodes
    from glider.nodes.vision import register_behavior_nodes

    register_hardware_nodes(FlowEngine)
    register_behavior_nodes(FlowEngine)

    ctrl = NodeEditorController.__new__(NodeEditorController)
    item = _NodeItem()
    ctrl.setup_node_ports(item, node_type)
    return item


def test_behavior_input_draws_its_real_ports(qtbot):
    """The generic fallback gave it one input and one output. It has none and
    four -- and On Enter has to land at index 2 to match the node definition,
    or a connection drawn on the canvas would fire the wrong output."""
    item = _ports("BehaviorInput")

    assert item.inputs == []
    assert [name for name, _ in item.outputs] == ["Active", "Behavior", "On Enter", "On Exit"]
    assert item.outputs[2] == ("On Enter", PortType.EXEC)
    assert item.outputs[3] == ("On Exit", PortType.EXEC)
    assert item.outputs[0][1] == PortType.DATA


def test_a_type_outside_the_table_draws_its_declared_ports(qtbot):
    """Device Action is registered and absent from the hand-written port table.
    The old generic fallback gave it one input and one output; it declares
    three and two, so wiring drawn on the canvas addressed the wrong ports."""
    item = _ports("DeviceAction")

    assert [name for name, _ in item.inputs] == ["exec", "arg1", "arg2"]
    assert [name for name, _ in item.outputs] == ["exec", "result"]
    assert item.inputs[0][1] == PortType.EXEC
    assert item.inputs[1][1] == PortType.DATA
    assert item.outputs[1][1] == PortType.DATA


def test_an_unregistered_type_still_gets_something_wireable(qtbot):
    """A file naming a node this install does not have must still draw."""
    item = _ports("NotARealNodeType")

    assert item.inputs == [("in", PortType.EXEC)]
    assert item.outputs == [("out", PortType.EXEC)]


# --- properties ---------------------------------------------------------------


def test_the_behavior_list_comes_from_the_loaded_model(qtbot):
    """A free-text box would let a typo mean 'never fires', which is the worst
    failure available to a closed-loop stimulus."""
    ctrl, captured, _ = _controller({}, behaviors=["darting", "freezing", "grooming"])

    ctrl._update_properties_panel("b1")

    combo, _spin = _widgets(captured)
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "darting",
        "freezing",
        "grooming",
    ]


def test_the_saved_behavior_is_shown(qtbot):
    ctrl, captured, _ = _controller(
        {"target_behavior": "darting", "min_frames": 8}, behaviors=["darting", "freezing"]
    )

    ctrl._update_properties_panel("b1")

    combo, spin = _widgets(captured)
    assert combo.currentText() == "darting"
    assert spin.value() == 8


def test_a_behavior_saved_before_the_model_loaded_survives(qtbot):
    """Authoring a graph without a model loaded is normal; the panel must not
    drop a label just because it cannot currently offer it."""
    ctrl, captured, _ = _controller({"target_behavior": "head dips"}, behaviors=[])

    ctrl._update_properties_panel("b1")

    combo, _spin = _widgets(captured)
    assert combo.isEditable()
    assert combo.currentText() == "head dips"


def test_choosing_a_behavior_persists(qtbot):
    ctrl, captured, saved = _controller({}, behaviors=["darting", "freezing"])
    ctrl._update_properties_panel("b1")
    combo, _spin = _widgets(captured)

    combo.setCurrentText("freezing")

    assert ("b1", {"target_behavior": "freezing"}) in saved


def test_editing_the_confirmation_window_persists(qtbot):
    ctrl, captured, saved = _controller({"min_frames": 5}, behaviors=["darting"])
    ctrl._update_properties_panel("b1")
    _combo, spin = _widgets(captured)

    spin.setValue(12)

    assert ("b1", {"min_frames": 12}) in saved


def test_confirmation_cannot_be_set_below_one(qtbot):
    """One frame is no confirmation at all -- the configuration the node exists
    to prevent."""
    ctrl, captured, _ = _controller({}, behaviors=["darting"])
    ctrl._update_properties_panel("b1")
    _combo, spin = _widgets(captured)

    assert spin.minimum() == 1


def test_a_missing_bus_does_not_break_the_panel(qtbot):
    """The properties panel must open whether or not vision is running."""
    ctrl, captured, _ = _controller({"target_behavior": "darting"})
    ctrl._core = SimpleNamespace()  # no live_signals at all

    ctrl._update_properties_panel("b1")

    combo, _spin = _widgets(captured)
    assert combo.currentText() == "darting"


# --- the vocabulary reaches the bus -------------------------------------------


def test_the_bus_carries_the_model_vocabulary():
    bus = LiveSignalBus()
    assert bus.behaviors == []

    bus.set_behaviors(["darting", "freezing"])
    assert bus.behaviors == ["darting", "freezing"]

    bus.set_behaviors(None)
    assert bus.behaviors == []


def test_the_published_vocabulary_is_a_copy():
    """A caller mutating what it got back must not edit the bus's state."""
    bus = LiveSignalBus()
    bus.set_behaviors(["darting"])

    bus.behaviors.append("nonsense")

    assert bus.behaviors == ["darting"]


def test_loading_a_model_publishes_its_vocabulary_to_the_bus(qtbot):
    """The link that makes the dropdown non-empty: the camera panel knows the
    model's classes, and the flow editor needs them."""
    from glider.gui.panels.camera_panel import CameraPanel

    panel = CameraPanel.__new__(CameraPanel)  # skip the heavy __init__
    bus = LiveSignalBus()
    panel._live_signals = bus
    panel._behavior_worker = SimpleNamespace(classes=["darting", "freezing", "grooming"])
    panel._preview = SimpleNamespace(set_behavior_vocab=lambda names: None)
    panel._behavior_running = False
    panel._live_behavior_btn = SimpleNamespace(setText=lambda t: None, setEnabled=lambda e: None)
    panel._rehearse_btn = SimpleNamespace(setEnabled=lambda e: None)
    panel._rehearse_status = SimpleNamespace(setText=lambda t: None)

    panel._on_behavior_ready()

    assert bus.behaviors == ["darting", "freezing", "grooming"]


def test_a_panel_with_no_bus_still_goes_live(qtbot):
    """Vision must not depend on a flow being attached."""
    from glider.gui.panels.camera_panel import CameraPanel

    panel = CameraPanel.__new__(CameraPanel)
    panel._live_signals = None
    panel._behavior_worker = SimpleNamespace(classes=["darting"])
    panel._preview = SimpleNamespace(set_behavior_vocab=lambda names: None)
    panel._behavior_running = False
    panel._live_behavior_btn = SimpleNamespace(setText=lambda t: None, setEnabled=lambda e: None)
    panel._rehearse_btn = SimpleNamespace(setEnabled=lambda e: None)
    panel._rehearse_status = SimpleNamespace(setText=lambda t: None)

    panel._on_behavior_ready()

    assert panel._behavior_running is True
