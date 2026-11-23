"""Zone Input can follow one named animal."""

import asyncio

from glider.core.live_signals import LiveSignalBus
from glider.gui.panels.node_editor_controller import NodeEditorController
from glider.nodes.vision.zone_nodes import ZoneInputNode
from glider.vision.zones import ZoneState


def _state(animals=(), visible=(), count=None):
    animals = set(animals)
    n = len(animals) if count is None else count
    return ZoneState(
        "nest",
        "Nest",
        occupied=n > 0,
        object_count=n,
        animals=animals,
        visible_animals=frozenset(visible),
    )


async def _started(animal=""):
    bus = LiveSignalBus()
    node = ZoneInputNode()
    node.set_state({"zone_id": "nest", "zone_name": "Nest", "animal": animal})
    node.set_live_signals(bus)
    node.fired = []

    async def _record(name, value=True):
        node.fired.append(name)

    node._fire_exec_output = _record
    await node.start()
    return bus, node


async def _publish(bus, state):
    bus.publish_zones({"nest": state})
    for _ in range(5):
        await asyncio.sleep(0)


async def test_follows_only_its_animal():
    bus, node = await _started("A")

    await _publish(bus, _state({"B"}, {"B"}))
    assert node.get_output(0) is False
    assert node.get_output(1) == 0
    assert node.fired == []

    await _publish(bus, _state({"A", "B"}, {"A", "B"}))
    assert node.get_output(0) is True
    assert node.get_output(1) == 1
    assert node.get_output(4) is True
    assert node.fired == ["On Enter"]


async def test_hidden_inside_keeps_occupied_and_does_not_exit():
    bus, node = await _started("A")
    await _publish(bus, _state({"A"}, {"A"}))
    await _publish(bus, _state({"A"}, set()))

    assert node.get_output(0) is True
    assert node.get_output(4) is False
    assert node.fired == ["On Enter"]


async def test_leaving_fires_on_exit():
    bus, node = await _started("A")
    await _publish(bus, _state({"A"}, {"A"}))
    await _publish(bus, _state(set(), {"A"}))
    assert node.fired == ["On Enter", "On Exit"]


async def test_any_behaves_as_before_and_visible_is_true():
    bus, node = await _started("")
    await _publish(bus, _state(set(), set(), count=2))
    assert node.get_output(0) is True
    assert node.get_output(1) == 2
    assert node.get_output(4) is True
    assert node.fired == ["On Enter"]


def test_state_round_trips_and_old_state_means_any():
    node = ZoneInputNode()
    node.set_state({"zone_id": "z", "animal": "A"})
    assert node.get_state()["animal"] == "A"
    old = ZoneInputNode()
    old.set_state({"zone_id": "z"})
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
    NodeEditorController.__new__(NodeEditorController).setup_node_ports(item, "ZoneInput")
    assert item.outputs == ["Occupied", "Object Count", "On Enter", "On Exit", "Visible"]
