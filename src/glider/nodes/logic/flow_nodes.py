"""
Flow Control Nodes - Execution flow control, delays, and timers.
"""

import asyncio
import logging
from typing import Any

from glider.core import flow_clock
from glider.nodes.base_node import (
    ExecNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)


class SequenceNode(ExecNode):
    """Execute multiple outputs in sequence."""

    definition = NodeDefinition(
        name="Sequence",
        category=NodeCategory.LOGIC,
        description="Execute outputs in sequence",
        inputs=[
            PortDefinition(name="exec", port_type=PortType.EXEC),
        ],
        outputs=[
            PortDefinition(name="Then 0", port_type=PortType.EXEC),
            PortDefinition(name="Then 1", port_type=PortType.EXEC),
            PortDefinition(name="Then 2", port_type=PortType.EXEC),
            PortDefinition(name="Then 3", port_type=PortType.EXEC),
        ],
        color="#2d4a5a",
    )

    async def execute(self) -> None:
        # Each output is scheduled for when the one before it finished on the
        # flow clock, so a Delay in "Then 0" does not leak into "Then 1".
        at = flow_clock.scheduled_or_now()
        for output_def in self.definition.outputs:
            at = await self._fire_exec_output(output_def.name, at=at)


class DelayNode(ExecNode):
    """Delay execution for a specified time, on the flow clock.

    The wait is held by the flow clock's timing thread, not by
    ``asyncio.sleep`` on the qasync loop, which Qt repaints and the recorders'
    ticks hold up by tens of ms (a 10 s delay used to return in 10.01-10.27 s).
    See :mod:`glider.core.flow_clock`.

    **What the duration counts from.** By default, from the time this node's
    input was *scheduled* for, not from when it happened to arrive: in
    ``Loop -> Maimu -> Delay 10 s`` the BLE write takes ~130 ms, and counting
    from its return added that to every iteration (101.34 s for a 100 s run).
    "End of previous step" restores the old behaviour, for a delay that is a
    minimum gap after an action really finished -- a valve's settle time.

    If the input arrives later than the whole duration -- the upstream step
    overran the schedule -- the delay counts from now instead and the timing
    log notes the overrun. Firing at once would collapse a gap the protocol
    asked for to zero.
    """

    definition = NodeDefinition(
        name="Delay",
        category=NodeCategory.LOGIC,
        description="Delay execution for specified duration (seconds or milliseconds)",
        inputs=[
            PortDefinition(name="exec", port_type=PortType.EXEC),
            PortDefinition(name="Duration", data_type=float, default_value=1.0),
        ],
        outputs=[
            PortDefinition(name="Completed", port_type=PortType.EXEC),
        ],
        color="#2d4a5a",
    )

    PROPERTIES_SCHEMA = [
        {
            "key": "duration",
            "label": "Duration",
            "type": "float",
            "default": 1.0,
            "min": 0,
            "help": "How long to wait, in `unit`. Overrides the Duration input when set.",
        },
        {
            "key": "unit",
            "label": "Unit",
            "type": "enum",
            "default": "seconds",
            "choices": [["seconds", "sec"], ["milliseconds", "ms"]],
            "help": "Unit of `duration`.",
        },
        {
            "key": "count_from",
            "label": "Count from",
            "type": "enum",
            "default": "schedule",
            "choices": [["schedule", "Schedule"], ["previous", "End of previous step"]],
            "help": (
                "Schedule keeps a fixed timeline: time the previous steps took is "
                "absorbed. End of previous step waits the full duration after the "
                "step before this one finished, e.g. for a settle time."
            ),
        },
    ]

    def __init__(self):
        super().__init__()
        self._wait: flow_clock.ClockWait | None = None

    def _duration_s(self) -> float:
        """The configured duration in seconds."""
        # Priority: 1) state "duration", 2) input port, 3) default 1.0
        if "duration" in self._state:
            duration = float(self._state["duration"])
        else:
            duration = float(self.get_input(1) or self.setting_default("duration"))

        if self.setting("unit") == "milliseconds":
            duration = duration / 1000.0

        return max(0, duration)

    async def execute(self) -> None:
        duration = self._duration_s()
        started = flow_clock.now()
        scheduled = flow_clock.inherited_time()
        if scheduled is None or self.setting("count_from") == "previous":
            anchor = started
        else:
            anchor = scheduled

        deadline = anchor + duration
        note = ""
        if duration > 0 and deadline < started:
            note = f"overrun: input {(started - anchor) * 1000:.1f} ms behind schedule"
            logger.warning(
                "Delay %s: %s, longer than its %.3f s; counting from now",
                self._glider_id,
                note,
                duration,
            )
            deadline = started + duration

        self._wait = flow_clock.wait_until(deadline)
        try:
            fired = await self._wait
        finally:
            self._wait = None
        if fired is None:
            return  # stopped

        flow_clock.record(self, "delay", deadline, clock=fired, started=flow_clock.now(), note=note)
        await self._fire_exec_output("Completed", at=deadline)

    async def stop(self) -> None:
        # Ends an in-flight wait at once; execute() returns without firing.
        if self._wait is not None:
            self._wait.cancel()


class TimerNode(ExecNode):
    """Periodic timer that triggers at intervals."""

    definition = NodeDefinition(
        name="Timer",
        category=NodeCategory.LOGIC,
        description="Trigger execution at regular intervals",
        inputs=[
            PortDefinition(
                name="Interval",
                data_type=float,
                default_value=1.0,
                description="Interval (seconds or milliseconds, per Unit setting)",
            ),
            PortDefinition(name="Enabled", data_type=bool, default_value=True),
        ],
        outputs=[
            PortDefinition(name="Tick", port_type=PortType.EXEC),
            PortDefinition(name="Count", data_type=int),
        ],
        color="#2d4a5a",
    )

    PROPERTIES_SCHEMA = [
        {
            "key": "interval",
            "label": "Interval",
            "type": "float",
            "default": 1.0,
            "min": 0,
            "help": "Time between ticks, in `unit` (at least 10 ms is used). Overrides the "
            "Interval input when set.",
        },
        {
            "key": "unit",
            "label": "Unit",
            "type": "enum",
            "default": "seconds",
            "choices": [["seconds", "sec"], ["milliseconds", "ms"]],
            "help": "Unit of `interval`.",
        },
        {
            "key": "count",
            "label": "Count",
            "type": "int",
            "default": 0,
            "min": 0,
            "help": "Ticks so far. Saved by GLIDER; leave unset.",
        },
    ]

    def __init__(self):
        super().__init__()
        self._timer_task: asyncio.Task | None = None
        self._count = self.setting("count")
        self._paused = False

    @property
    def count(self) -> int:
        return self._count

    async def start(self) -> None:
        """Start the timer."""
        from glider.core.async_utils import log_task_exception

        self._count = 0
        self._paused = False
        self._timer_task = asyncio.create_task(self._timer_loop())
        self._timer_task.add_done_callback(log_task_exception)

    async def stop(self) -> None:
        """Stop the timer."""
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
            self._timer_task = None

    async def pause(self) -> None:
        """Pause the timer."""
        self._paused = True

    async def resume(self) -> None:
        """Resume the timer."""
        self._paused = False

    def _effective_interval(self) -> float:
        """Return the interval in seconds, honoring the state 'interval' override and unit."""
        if "interval" in self._state:
            raw = self._state["interval"]
        else:
            raw = self.get_input(0)
        if raw is None:
            raw = self.setting_default("interval")
        interval = float(raw)
        if self.setting("unit") == "milliseconds":
            interval = interval / 1000.0
        return max(0.01, interval)

    async def _timer_loop(self) -> None:
        """Tick at fixed points on the flow clock: start + k * interval.

        A tick whose downstream runs longer than the interval does not cause
        a burst of catch-up ticks afterwards: the schedule restarts from now.
        """
        next_tick = flow_clock.scheduled_or_now()
        while True:
            try:
                interval = self._effective_interval()
                enabled = bool(self.get_input(1) if self.get_input(1) is not None else True)

                next_tick += interval
                note = ""
                current = flow_clock.now()
                if next_tick < current:
                    note = f"overrun: {(current - next_tick) * 1000:.1f} ms behind; restarted"
                    next_tick = current
                    fired = None
                else:
                    fired = await flow_clock.wait_until(next_tick)

                if enabled and not self._paused:
                    flow_clock.record(
                        self, "timer", next_tick, clock=fired, started=flow_clock.now(), note=note
                    )
                    self._count += 1
                    self.set_output(1, self._count)
                    await self._fire_exec_output("Tick", at=next_tick)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.set_error(str(e))

    async def execute(self) -> None:
        """Manual execution not used for timers."""
        pass

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["count"] = self._count
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._count = self.setting("count")


logger = logging.getLogger(__name__)


def register_logic_nodes(flow_engine) -> None:
    """Register all logic/flow control nodes with the flow engine."""
    flow_engine.register_node("Delay", DelayNode)
    flow_engine.register_node("Sequence", SequenceNode)
    flow_engine.register_node("Timer", TimerNode)
    logger.info("Registered logic nodes")
