"""Behavior Input can follow one named animal."""

from glider.core.live_signals import BehaviorEvent
from glider.gui.panels.node_editor_controller import NodeEditorController
from glider.nodes.vision.behavior_nodes import BehaviorInputNode


def _node(animal="", min_frames=2):
    node = BehaviorInputNode()
    node.set_state({"target_behavior": "groom", "min_frames": min_frames, "animal": animal})
    node.fired = []
    node._dispatch_exec = node.fired.append
    return node


def _ev(label="groom", animal=None, visible=True):
    return BehaviorEvent(behavior=label, animal=animal, visible=visible)


def test_named_node_ignores_other_animals_and_untagged():
    node = _node("A")
    for animal in ("B", None, "B", None):
        node.handle_behavior_event(_ev(animal=animal))
    assert node.active is False
    assert node.get_output(1) == ""
    assert node.fired == []

    node.handle_behavior_event(_ev(animal="A"))
    node.handle_behavior_event(_ev(animal="A"))
    assert node.active is True
    assert node.fired == ["On Enter"]


def test_any_accepts_only_untagged_events():
    node = _node("")
    for _ in range(3):
        node.handle_behavior_event(_ev(animal="A"))
    assert node.active is False
    node.handle_behavior_event(_ev())
    node.handle_behavior_event(_ev())
    assert node.active is True
    assert node.get_output(4) is True


def test_hidden_event_changes_nothing_but_visible():
    node = _node("A")
    node.handle_behavior_event(_ev(animal="A"))
    node.handle_behavior_event(_ev(animal="A"))
    assert node.active is True
    for _ in range(5):
        node.handle_behavior_event(_ev("rest", animal="A", visible=False))
    assert node.active is True
    assert node.get_output(0) is True
    assert node.get_output(1) == "groom"
    assert node._miss_run == 0 and node._match_run == 2
    assert node.fired == ["On Enter"]
    assert node.get_output(4) is False

    node.handle_behavior_event(_ev(animal="A"))
    assert node.get_output(4) is True


def test_state_round_trips_and_old_state_means_any():
    node = BehaviorInputNode()
    node.set_state({"target_behavior": "groom", "animal": "A"})
    assert node.get_state()["animal"] == "A"
    old = BehaviorInputNode()
    old.set_state({"target_behavior": "groom"})
    assert old.get_state()["animal"] == ""


def test_visible_port_is_appended_after_on_exit():
    class Item:
        def __init__(self):
            self.outputs = []

        def add_input_port(self, n, t):
            pass

        def add_output_port(self, n, t):
            self.outputs.append(n)

    item = Item()
    NodeEditorController.__new__(NodeEditorController).setup_node_ports(item, "BehaviorInput")
    assert item.outputs == ["Active", "Behavior", "On Enter", "On Exit", "Visible"]
