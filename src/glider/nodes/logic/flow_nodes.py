"""
Flow Control Nodes - Execution flow control, delays, and timers.
"""

import asyncio
import logging
import threading
import time
from typing import Any

from glider.core.timer_resolution import high_resolution_timers
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
        for _i, output_def in enumerate(self.definition.outputs):
            await self._fire_exec_output(output_def.name)


class DelayNode(ExecNode):
    """Delay execution for a specified time.

    Sleep is performed on a worker thread via
    ``asyncio.to_thread(threading.Event.wait, duration)`` rather than
    ``asyncio.sleep`` for two reasons:

    1. **Timing precision.** ``asyncio.sleep`` resolves on the qasync
       event loop, which under typical GLIDER load (data recorder
       sampling at 100 ms, runner timer ticking at 50 ms, Qt repaints,
       device polls) is too congested for sub-100ms wakeup accuracy.
       Field reports showed a 10-second delay returning in 10.01-10.27s
       (260 ms jitter). ``threading.Event.wait(timeout)`` is backed by
       an OS condition variable and runs on a thread that isn't
       competing with Qt for loop time, so a 10s delay reliably
       returns close to 10s. The condition variable cannot resolve
       finer than the system timer tick, which is ~1 ms on macOS and
       Linux but ~15.6 ms on Windows by default, so the wait is taken
       under :func:`~glider.core.timer_resolution.high_resolution_timers`.

    2. **Clean cancellation.** ``Event.set()`` returns control to the
       waiting thread immediately, which then unblocks the ``to_thread``
       wrapper, which delivers the result back to the asyncio side.
       ``DelayNode.stop()`` simply sets the event and the in-flight
       sleep returns promptly — no need to cancel an asyncio task and
       hope its sleep wakeup observes the cancellation in time.

    The event is replaced on each ``execute()`` so a node reused across
    multiple runs starts each delay with a fresh, unset event.
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

    def __init__(self):
        super().__init__()
        # threading.Event (not asyncio.Event) — must be settable from the
        # asyncio side AND waited on from the worker thread. stop() will
        # set this to break out of an in-flight wait.
        self._stop_event: threading.Event = threading.Event()

    async def execute(self) -> None:
        # Priority: 1) state "duration", 2) input port, 3) default 1.0
        if "duration" in self._state:
            duration = float(self._state["duration"])
        else:
            duration = float(self.get_input(1) or 1.0)

        unit = self._state.get("unit", "seconds")
        if unit == "milliseconds":
            duration = duration / 1000.0

        duration = max(0, duration)

        # Fresh event per execute() so a previous stop() doesn't leak
        # into a new delay (and so the node is safely reusable across
        # multiple flow runs without re-init).
        self._stop_event = threading.Event()

        # Wait on a worker thread. Event.wait(timeout) returns True if
        # the event was set (cancellation), False on timeout (delay
        # completed normally). Either way we just unblock and fall
        # through — only fire Completed on the normal path.
        # Hold a 1 ms system timer tick for the wait. Event.wait cannot
        # resolve finer than the tick, and Windows' default is ~15.6 ms, so
        # without this a delay overshoots by a median of 31 ms and up to 63 ms
        # -- quantised to the tick, not scattered. A no-op off Windows.
        with high_resolution_timers():
            cancelled = await asyncio.to_thread(self._stop_event.wait, duration)
        if cancelled:
            return

        await self._fire_exec_output("Completed")

    async def stop(self) -> None:
        # Wakes any in-flight Event.wait on the worker thread within
        # condition-variable precision (~µs). The asyncio side picks
        # up the to_thread completion within a normal loop tick.
        self._stop_event.set()


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

    def __init__(self):
        super().__init__()
        self._timer_task: asyncio.Task | None = None
        self._count = 0
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
            raw = 1.0
        interval = float(raw)
        if self._state.get("unit", "seconds") == "milliseconds":
            interval = interval / 1000.0
        return max(0.01, interval)

    async def _timer_loop(self) -> None:
        """Timer loop that triggers at intervals."""
        next_tick = time.monotonic()
        while True:
            try:
                interval = self._effective_interval()
                enabled = bool(self.get_input(1) if self.get_input(1) is not None else True)

                next_tick += interval
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                if enabled and not self._paused:
                    self._count += 1
                    self.set_output(1, self._count)
                    await self._fire_exec_output("Tick")

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
        self._count = state.get("count", 0)


logger = logging.getLogger(__name__)


def register_logic_nodes(flow_engine) -> None:
    """Register all logic/flow control nodes with the flow engine."""
    flow_engine.register_node("Delay", DelayNode)
    flow_engine.register_node("Sequence", SequenceNode)
    flow_engine.register_node("Timer", TimerNode)
    logger.info("Registered logic nodes")
