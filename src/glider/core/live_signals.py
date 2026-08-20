"""Delivery of live vision results to nodes in the running flow.

Vision produces a result per frame on a worker thread; nodes that react to it
live in the flow and run on the asyncio loop. Nothing connected the two: nodes
such as :class:`~glider.nodes.vision.zone_nodes.ZoneInputNode` carry an update
method that nothing in the codebase ever calls, so their exec outputs could
never fire.

This is the missing hop. Nodes subscribe when they start and unsubscribe when
they stop; whichever component is running inference publishes to the bus and
stays ignorant of what is listening. The alternative -- having the camera panel
reach into the flow and look for nodes by type -- would make the GUI import the
node package and re-implement the lookup for every new node type.

Delivery is deliberately forgiving. A subscriber that raises is logged and
skipped rather than allowed to kill the frame, because the publisher is usually
a vision thread whose job is not to care about the flow graph, and one bad node
should not stop the others from seeing the frame.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BehaviorEvent:
    """One frame's behavior classification.

    ``confidence`` is ``None`` when the classifier does not report one, which
    is distinct from a confidence of zero and must not be silently coerced to
    it: a node gating on confidence has to be able to tell "not measured" from
    "measured as hopeless".
    """

    behavior: str
    confidence: float | None = None
    frame_index: int | None = None
    timestamp: float | None = None
    keypoints: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class LiveSignalBus:
    """Fan-out of live vision results to whatever is listening.

    Subscription is by callable rather than by node, so anything can listen --
    a node, a recorder, a test. Callables are held strongly: a node that stops
    unsubscribes itself, and one that does not is a leak worth seeing rather
    than one silently swallowed by a weak reference.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._behavior_subs: list[Callable[[BehaviorEvent], None]] = []

    # -- behavior ---------------------------------------------------------
    def subscribe_behavior(self, callback: Callable[[BehaviorEvent], None]) -> None:
        """Register *callback* to receive every behavior classification."""
        with self._lock:
            if callback not in self._behavior_subs:
                self._behavior_subs.append(callback)
                logger.debug(
                    "live bus: behavior subscriber added (%d total)", len(self._behavior_subs)
                )

    def unsubscribe_behavior(self, callback: Callable[[BehaviorEvent], None]) -> None:
        """Remove *callback*. Removing one that was never added is not an error."""
        with self._lock:
            if callback in self._behavior_subs:
                self._behavior_subs.remove(callback)
                logger.debug(
                    "live bus: behavior subscriber removed (%d left)", len(self._behavior_subs)
                )

    def publish_behavior(self, event: BehaviorEvent) -> int:
        """Deliver *event* to every subscriber. Returns how many were called.

        Called from the vision worker thread on every frame, so it copies the
        subscriber list under the lock and then releases it: a subscriber that
        blocks must not hold off subscription changes, and one that
        unsubscribes from inside its own callback must not mutate the list
        being iterated.
        """
        with self._lock:
            subs = list(self._behavior_subs)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                logger.exception("live bus: behavior subscriber raised, continuing")
        return len(subs)

    @property
    def behavior_subscriber_count(self) -> int:
        with self._lock:
            return len(self._behavior_subs)

    def clear(self) -> None:
        """Drop every subscriber. For teardown between sessions."""
        with self._lock:
            self._behavior_subs.clear()
