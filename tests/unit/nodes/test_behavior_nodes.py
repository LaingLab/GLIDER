"""Behavior Input node: debouncing, subscription lifecycle and persistence.

The debounce tests are the point of the file. A behavior node wired to hardware
fires stimuli, and the classifier's raw per-frame output is noisy enough that
without confirmation a single misclassified frame would trigger one.
"""

import asyncio

import pytest

from glider.core.live_signals import BehaviorEvent, LiveSignalBus
from glider.nodes.vision.behavior_nodes import BehaviorInputNode


def make_node(target="grooming", min_frames=3, bus=None):
    node = BehaviorInputNode()
    node.target_behavior = target
    node.min_frames = min_frames
    if bus is not None:
        node.set_live_signals(bus)
    return node


def feed(node, labels):
    for label in labels:
        node.handle_behavior_event(BehaviorEvent(behavior=label))


class TestDebounce:
    def test_single_stray_frame_does_not_fire(self):
        """One misclassified frame must not trigger anything."""
        node = make_node(min_frames=3)
        fired = []
        node._dispatch_exec = fired.append

        feed(node, ["locomote", "grooming", "locomote", "locomote"])

        assert fired == []
        assert node.active is False

    def test_fires_after_min_frames(self):
        node = make_node(min_frames=3)
        fired = []
        node._dispatch_exec = fired.append

        feed(node, ["grooming", "grooming"])
        assert fired == [], "fired before reaching min_frames"

        feed(node, ["grooming"])
        assert fired == ["On Enter"]
        assert node.active is True

    def test_fires_once_not_every_frame(self):
        """On Enter is an edge, not a level."""
        node = make_node(min_frames=2)
        fired = []
        node._dispatch_exec = fired.append

        feed(node, ["grooming"] * 10)

        assert fired == ["On Enter"]

    def test_exit_also_debounced(self):
        node = make_node(min_frames=3)
        fired = []
        node._dispatch_exec = fired.append

        feed(node, ["grooming"] * 3)
        assert fired == ["On Enter"]

        # A single non-matching frame mid-bout must not end it.
        feed(node, ["locomote", "grooming", "grooming"])
        assert fired == ["On Enter"]
        assert node.active is True

        feed(node, ["locomote"] * 3)
        assert fired == ["On Enter", "On Exit"]
        assert node.active is False

    def test_min_frames_one_fires_immediately(self):
        """Explicitly opting out of debouncing is allowed."""
        node = make_node(min_frames=1)
        fired = []
        node._dispatch_exec = fired.append

        feed(node, ["grooming"])
        assert fired == ["On Enter"]

    def test_min_frames_floored_at_one(self):
        node = make_node()
        node.min_frames = 0
        assert node.min_frames == 1
        node.min_frames = -5
        assert node.min_frames == 1

    def test_empty_label_is_a_miss_not_a_match(self):
        """The classifier emits '' while its rolling window fills."""
        node = make_node(target="grooming", min_frames=2)
        fired = []
        node._dispatch_exec = fired.append

        feed(node, ["", "", "", ""])
        assert fired == []
        assert node.active is False

    def test_no_target_never_fires(self):
        node = make_node(target="", min_frames=1)
        fired = []
        node._dispatch_exec = fired.append

        feed(node, ["grooming", "locomote", "grooming"])
        assert fired == []

    def test_changing_target_resets_progress(self):
        """A part-accumulated run for the old target must not count for the new one."""
        node = make_node(target="grooming", min_frames=3)
        fired = []
        node._dispatch_exec = fired.append

        feed(node, ["grooming", "grooming"])
        node.target_behavior = "dig"
        feed(node, ["dig"])

        assert fired == [], "carried a run across a target change"
        feed(node, ["dig", "dig"])
        assert fired == ["On Enter"]


class TestOutputs:
    def test_behavior_output_tracks_every_frame(self):
        """The label output reports what was seen, regardless of the target."""
        node = make_node(target="grooming")
        feed(node, ["locomote"])
        assert node._outputs[1] == "locomote"

    def test_active_output_follows_state(self):
        node = make_node(min_frames=2)
        node._dispatch_exec = lambda _: None

        feed(node, ["grooming", "grooming"])
        assert node._outputs[0] is True

        feed(node, ["dig", "dig"])
        assert node._outputs[0] is False


class TestSubscription:
    def test_set_live_signals_subscribes_when_running(self):
        bus = LiveSignalBus()
        node = make_node()
        node._main_loop = object()  # stand in for a running loop
        node.set_live_signals(bus)
        assert bus.behavior_subscriber_count == 1

    def test_swapping_bus_moves_the_subscription(self):
        a, b = LiveSignalBus(), LiveSignalBus()
        node = make_node()
        node._main_loop = object()
        node.set_live_signals(a)
        node.set_live_signals(b)
        assert a.behavior_subscriber_count == 0
        assert b.behavior_subscriber_count == 1

    def test_same_bus_twice_subscribes_once(self):
        bus = LiveSignalBus()
        node = make_node()
        node._main_loop = object()
        node.set_live_signals(bus)
        node.set_live_signals(bus)
        assert bus.behavior_subscriber_count == 1

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self):
        bus = LiveSignalBus()
        node = make_node(bus=bus)
        await node.start()
        assert bus.behavior_subscriber_count == 1

        await node.stop()
        assert bus.behavior_subscriber_count == 0

    @pytest.mark.asyncio
    async def test_stop_clears_active_state(self):
        bus = LiveSignalBus()
        node = make_node(min_frames=1, bus=bus)
        await node.start()
        node._dispatch_exec = lambda _: None
        feed(node, ["grooming"])
        assert node.active is True

        await node.stop()
        assert node.active is False
        assert node._outputs[0] is False

    @pytest.mark.asyncio
    async def test_start_without_bus_does_not_raise(self):
        """A node dropped on the canvas before a camera exists must still start."""
        node = make_node()
        await node.start()  # warns, does not raise
        assert node.active is False

    @pytest.mark.asyncio
    async def test_end_to_end_through_the_bus(self):
        bus = LiveSignalBus()
        node = make_node(min_frames=2, bus=bus)
        await node.start()
        fired = []
        node._dispatch_exec = fired.append

        for label in ["grooming", "grooming"]:
            bus.publish_behavior(BehaviorEvent(behavior=label))

        assert fired == ["On Enter"]


class TestPersistence:
    def test_round_trip(self):
        node = make_node(target="dig", min_frames=7)
        state = node.get_state()

        restored = BehaviorInputNode()
        restored.set_state(state)

        assert restored.target_behavior == "dig"
        assert restored.min_frames == 7

    def test_restore_defaults_min_frames(self):
        """A graph saved before this field existed must still load."""
        node = BehaviorInputNode()
        node.set_state({"target_behavior": "grooming"})
        assert node.min_frames >= 1

    def test_display_name(self):
        node = BehaviorInputNode()
        assert node.get_display_name() == "Behavior Input"
        node.target_behavior = "grooming"
        assert node.get_display_name() == "Behavior: grooming"


class TestDispatchThreadSafety:
    def test_dispatch_without_loop_is_dropped_not_raised(self):
        """Frames can arrive before start() captured a loop."""
        node = make_node(min_frames=1)
        node._main_loop = None
        feed(node, ["grooming"])  # must not raise
        assert node.active is True

    @pytest.mark.asyncio
    async def test_dispatch_schedules_onto_captured_loop(self):
        node = make_node(min_frames=1)
        await node.start()

        calls = []

        async def fake_fire(name, value=True):
            calls.append(name)

        node._fire_exec_output = fake_fire
        feed(node, ["grooming"])
        # run_coroutine_threadsafe goes via call_soon_threadsafe, so the
        # coroutine is not queued until the loop next wakes; a bare sleep(0)
        # yields once and can land before it has been scheduled at all.
        for _ in range(10):
            await asyncio.sleep(0.01)
            if calls:
                break

        assert calls == ["On Enter"]
