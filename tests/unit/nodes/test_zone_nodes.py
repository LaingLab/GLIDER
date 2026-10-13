"""Zone Input node: hearing the tracker's zone updates.

Nothing used to call ZoneInputNode.update_zone_state, so On Enter / On Exit
could never fire -- from a camera or a rehearsal. These pin the hop from the
CV processor's zone update, over the live signal bus, to the node.
"""

import asyncio

from glider.core.live_signals import LiveSignalBus
from glider.nodes.vision.zone_nodes import ZoneInputNode
from glider.vision.zones import ZoneState


async def _started(bus, zone_id="nest"):
    node = ZoneInputNode()
    node.zone_id = zone_id
    node.set_live_signals(bus)
    node.fired = []

    async def _record(name, value=True):
        node.fired.append(name)

    node._fire_exec_output = _record
    await node.start()
    return node


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def test_entering_its_zone_fires_on_enter():
    bus = LiveSignalBus()
    node = await _started(bus)

    bus.publish_zones({"nest": ZoneState("nest", "Nest", True, 1, entered=True)})
    await _settle()

    assert node.fired == ["On Enter"]
    assert node.occupied


async def test_leaving_its_zone_fires_on_exit():
    bus = LiveSignalBus()
    node = await _started(bus)

    bus.publish_zones({"nest": ZoneState("nest", "Nest", False, 0, exited=True)})
    await _settle()

    assert node.fired == ["On Exit"]


async def test_another_zones_events_are_not_its_own():
    bus = LiveSignalBus()
    node = await _started(bus)

    bus.publish_zones({"corner": ZoneState("corner", "Corner", True, 1, entered=True)})
    await _settle()

    assert node.fired == []
    assert not node.occupied


async def test_a_stopped_node_stops_listening():
    bus = LiveSignalBus()
    node = await _started(bus)
    await node.stop()

    assert bus.zone_subscriber_count == 0


async def test_the_cv_processors_zone_updates_reach_the_bus():
    """The CV processor works out enter/exit every frame and offers a callback
    for it; the core is what has to connect that to the flow."""
    from glider.core.glider_core import GliderCore

    core = GliderCore()
    seen = []
    core.live_signals.subscribe_zones(seen.append)
    states = {"nest": ZoneState("nest", "Nest", True, 1, entered=True)}

    for callback in core.cv_processor._zone_callbacks:
        callback(states)

    assert seen == [states]
