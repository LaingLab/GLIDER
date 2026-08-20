"""Behavior Input Node - trigger flow execution from live behavior classification.

Watches the live classifier for one behavior and fires execution outputs as the
animal enters and leaves it, which is what turns GLIDER from a recorder into a
closed-loop instrument: a stimulus can be delivered because the animal is
grooming, not because a timer expired.

Two properties of the live signal shape this node.

It is noisy per frame. The classifier emits a label every frame, and on held-out
sessions raw frame-wise macro F1 is 0.777 against 0.811 for the same frames
under a 25-frame centred vote. Wiring a raw label straight to hardware means one
misclassified frame fires the stimulus. So entering a behavior requires
``min_frames`` consecutive frames carrying it, and leaving requires the same
number without it; a single stray frame changes nothing.

The centred vote cannot be used here. It reads frames that have not happened
yet, which is legitimate offline and impossible live. Consecutive-frame
confirmation is the causal equivalent, and its cost is explicit: at 30 fps a
``min_frames`` of 5 delays the trigger by about 167 ms on top of inference.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from glider.core.live_signals import BehaviorEvent
from glider.nodes.base_node import (
    InterfaceNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_FRAMES = 5


class BehaviorInputNode(InterfaceNode):
    """Fires when the live classifier reports a chosen behavior.

    Outputs
        Active      True while the behavior is confirmed present
        Behavior    the most recent label, whatever it is
        On Enter    exec, once, when the behavior becomes confirmed
        On Exit     exec, once, when it stops being confirmed

    There is no confidence output. The live path's ``LiveResult`` carries a
    label and keypoints only, so a confidence port would read zero on every
    frame and invite gating on a number that was never measured.
    :class:`~glider.core.live_signals.BehaviorEvent` keeps an optional
    ``confidence`` field for publishers that do have one.
    """

    definition = NodeDefinition(
        name="Behavior Input",
        category=NodeCategory.INTERFACE,
        description=(
            "Triggers on live behavior classification. Requires N consecutive "
            "frames before firing, so a single misclassified frame cannot "
            "trigger hardware."
        ),
        inputs=[],
        outputs=[
            PortDefinition(
                name="Active",
                data_type=bool,
                description="True while the target behavior is confirmed",
            ),
            PortDefinition(
                name="Behavior",
                data_type=str,
                description="Most recent classified behavior label",
            ),
            PortDefinition(
                name="On Enter",
                port_type=PortType.EXEC,
                description="Triggered once when the target behavior is confirmed",
            ),
            PortDefinition(
                name="On Exit",
                port_type=PortType.EXEC,
                description="Triggered once when the target behavior ends",
            ),
        ],
        color="#5a4a2d",  # Orange - interface color
    )

    def __init__(self) -> None:
        super().__init__()
        self._target_behavior: str = ""
        self._min_frames: int = DEFAULT_MIN_FRAMES

        self._active = False
        self._last_behavior = ""
        self._match_run = 0  # consecutive frames matching the target
        self._miss_run = 0  # consecutive frames not matching

        self._bus: Any = None
        self._subscribed = False
        self._main_loop: asyncio.AbstractEventLoop | None = None

        self._outputs = [False, "", None, None]

    # -- configuration ----------------------------------------------------
    @property
    def target_behavior(self) -> str:
        """Behavior label this node watches for."""
        return self._target_behavior

    @target_behavior.setter
    def target_behavior(self, value: str) -> None:
        value = (value or "").strip()
        if value != self._target_behavior:
            self._target_behavior = value
            self._reset_run_state()

    @property
    def min_frames(self) -> int:
        """Consecutive frames required before entering or leaving the behavior."""
        return self._min_frames

    @min_frames.setter
    def min_frames(self, value: int) -> None:
        # One frame means no confirmation at all, which is the configuration
        # this node exists to prevent, but forbidding it outright would block
        # deliberate low-latency use. Floor at 1 and let the docs argue.
        self._min_frames = max(1, int(value))

    @property
    def active(self) -> bool:
        """Whether the target behavior is currently confirmed."""
        return self._active

    def _reset_run_state(self) -> None:
        self._match_run = 0
        self._miss_run = 0

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        """Capture the running loop and subscribe to the live signal bus.

        The loop reference has to be taken here rather than when a frame
        arrives: ``asyncio.get_event_loop()`` called from the vision thread
        returns a fresh, non-running loop on Python 3.12+, and scheduling onto
        it drops the exec output silently.
        """
        self._main_loop = asyncio.get_running_loop()
        self._reset_run_state()
        self._active = False
        self.set_output(0, False)

        if self._bus is None:
            logger.warning(
                "Behavior Input '%s': no live signal bus attached; the node will "
                "not fire. The flow engine injects one via set_live_signals.",
                self._target_behavior or "(unset)",
            )
            return
        self._subscribe()

    async def stop(self) -> None:
        """Unsubscribe so a stopped flow stops receiving frames."""
        self._unsubscribe()
        self._reset_run_state()
        if self._active:
            self._active = False
            self.set_output(0, False)

    def set_live_signals(self, bus: Any) -> None:
        """Injected by the flow engine at node creation.

        Re-subscribes when the bus is swapped while running, so a node created
        before the bus existed still starts hearing frames once it does.
        """
        if bus is self._bus:
            return
        was_subscribed = self._subscribed
        self._unsubscribe()
        self._bus = bus
        if was_subscribed or self._main_loop is not None:
            self._subscribe()

    def _subscribe(self) -> None:
        if self._bus is not None and not self._subscribed:
            self._bus.subscribe_behavior(self.handle_behavior_event)
            self._subscribed = True

    def _unsubscribe(self) -> None:
        if self._bus is not None and self._subscribed:
            self._bus.unsubscribe_behavior(self.handle_behavior_event)
        self._subscribed = False

    # -- frame handling ---------------------------------------------------
    def handle_behavior_event(self, event: BehaviorEvent) -> None:
        """Called once per classified frame, from the vision thread."""
        label = (event.behavior or "").strip()

        self._last_behavior = label
        self.set_output(1, label)

        if not self._target_behavior:
            return

        # An empty label is the classifier's warm-up state, not a behavior, and
        # must count as a miss rather than a match against an empty target --
        # the guard above already returns when no target is configured.
        matched = bool(label) and label == self._target_behavior

        if matched:
            self._match_run += 1
            self._miss_run = 0
        else:
            self._miss_run += 1
            self._match_run = 0

        if not self._active and self._match_run >= self._min_frames:
            self._active = True
            self.set_output(0, True)
            self._dispatch_exec("On Enter")
        elif self._active and self._miss_run >= self._min_frames:
            self._active = False
            self.set_output(0, False)
            self._dispatch_exec("On Exit")

    def _dispatch_exec(self, port_name: str) -> None:
        """Hand an exec output to the flow's loop from the vision thread."""
        loop = self._main_loop
        if loop is None or loop.is_closed():
            logger.debug(
                "Behavior Input '%s': no running loop, dropping %s",
                self._target_behavior,
                port_name,
            )
            return
        asyncio.run_coroutine_threadsafe(self._fire_exec_output(port_name), loop)
        logger.debug("Behavior Input '%s': %s triggered", self._target_behavior, port_name)

    def update_event(self) -> None:
        """Driven by the classifier, not by node inputs."""

    # -- persistence ------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["target_behavior"] = self._target_behavior
        state["min_frames"] = self._min_frames
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._target_behavior = state.get("target_behavior", "")
        self.min_frames = state.get("min_frames", DEFAULT_MIN_FRAMES)
        self._reset_run_state()

    def get_display_name(self) -> str:
        if not self._target_behavior:
            return "Behavior Input"
        return f"Behavior: {self._target_behavior}"


def register_behavior_nodes(flow_engine) -> None:
    """Register behavior nodes with the flow engine."""
    flow_engine.register_node("BehaviorInput", BehaviorInputNode)
    logger.info("Registered behavior nodes")
