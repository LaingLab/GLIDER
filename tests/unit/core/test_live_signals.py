"""LiveSignalBus: fan-out, isolation between subscribers, and thread safety."""

import threading

from glider.core.live_signals import BehaviorEvent, LiveSignalBus


class TestSubscription:
    def test_publish_reaches_subscriber(self):
        bus = LiveSignalBus()
        seen = []
        bus.subscribe_behavior(seen.append)

        bus.publish_behavior(BehaviorEvent(behavior="grooming"))

        assert len(seen) == 1
        assert seen[0].behavior == "grooming"

    def test_duplicate_subscribe_is_idempotent(self):
        bus = LiveSignalBus()
        seen = []
        bus.subscribe_behavior(seen.append)
        bus.subscribe_behavior(seen.append)

        bus.publish_behavior(BehaviorEvent(behavior="dig"))

        assert len(seen) == 1, "one callback delivered twice"

    def test_unsubscribe_stops_delivery(self):
        bus = LiveSignalBus()
        seen = []
        bus.subscribe_behavior(seen.append)
        bus.unsubscribe_behavior(seen.append)

        bus.publish_behavior(BehaviorEvent(behavior="dig"))

        assert seen == []

    def test_unsubscribe_unknown_callback_is_not_an_error(self):
        bus = LiveSignalBus()
        bus.unsubscribe_behavior(lambda _: None)  # must not raise

    def test_publish_with_no_subscribers(self):
        bus = LiveSignalBus()
        assert bus.publish_behavior(BehaviorEvent(behavior="dig")) == 0

    def test_clear_removes_everything(self):
        bus = LiveSignalBus()
        bus.subscribe_behavior(lambda _: None)
        bus.clear()
        assert bus.behavior_subscriber_count == 0


class TestIsolation:
    def test_one_raising_subscriber_does_not_starve_the_others(self):
        """A bad node must not stop the rest of the graph seeing the frame."""
        bus = LiveSignalBus()
        seen = []

        def boom(_event):
            raise RuntimeError("node blew up")

        bus.subscribe_behavior(boom)
        bus.subscribe_behavior(seen.append)

        count = bus.publish_behavior(BehaviorEvent(behavior="grooming"))

        assert count == 2
        assert len(seen) == 1

    def test_unsubscribing_during_delivery_is_safe(self):
        """A node that stops itself mid-callback must not corrupt iteration."""
        bus = LiveSignalBus()
        seen = []

        def self_removing(event):
            seen.append(event)
            bus.unsubscribe_behavior(self_removing)

        bus.subscribe_behavior(self_removing)
        bus.subscribe_behavior(seen.append)

        bus.publish_behavior(BehaviorEvent(behavior="dig"))  # must not raise
        assert len(seen) == 2

        bus.publish_behavior(BehaviorEvent(behavior="dig"))
        assert len(seen) == 3, "self-removed subscriber was still called"


class TestThreadSafety:
    def test_concurrent_publish_and_subscribe(self):
        """Vision publishes while the flow starts and stops nodes."""
        bus = LiveSignalBus()
        stop = threading.Event()
        errors = []

        def publisher():
            try:
                while not stop.is_set():
                    bus.publish_behavior(BehaviorEvent(behavior="grooming"))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        def churn():
            try:
                for _ in range(200):
                    cb = lambda _e: None  # noqa: E731 - distinct object each time
                    bus.subscribe_behavior(cb)
                    bus.unsubscribe_behavior(cb)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        t1 = threading.Thread(target=publisher)
        t2 = threading.Thread(target=churn)
        t1.start()
        t2.start()
        t2.join(timeout=10)
        stop.set()
        t1.join(timeout=10)

        assert errors == []


class TestBehaviorEvent:
    def test_confidence_defaults_to_none_not_zero(self):
        """Unmeasured must stay distinguishable from measured-as-zero."""
        assert BehaviorEvent(behavior="dig").confidence is None

    def test_carries_optional_context(self):
        event = BehaviorEvent(behavior="dig", frame_index=42, timestamp=1.5)
        assert event.frame_index == 42
        assert event.timestamp == 1.5
