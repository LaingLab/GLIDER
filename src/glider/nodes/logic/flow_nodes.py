"""
Flow Control Nodes - Execution flow control, delays, and timers.
"""

import asyncio
from typing import Any

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
    """Delay execution for a specified time."""

    definition = NodeDefinition(
        name="Delay",
        category=NodeCategory.LOGIC,
        description="Delay execution for specified seconds",
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
        self._delay_task: asyncio.Task | None = None

    async def execute(self) -> None:
        duration = float(self.get_input(1) or 1.0)
        duration = max(0, duration)

        await asyncio.sleep(duration)
        await self._fire_exec_output("Completed")

    async def stop(self) -> None:
        if self._delay_task:
            self._delay_task.cancel()


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
                description="Interval in seconds",
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
        self._count = 0
        self._paused = False
        self._timer_task = asyncio.create_task(self._timer_loop())

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

    async def _timer_loop(self) -> None:
        """Timer loop that triggers at intervals."""
        while True:
            try:
                interval = float(self.get_input(0) or 1.0)
                enabled = bool(self.get_input(1) if self.get_input(1) is not None else True)

                await asyncio.sleep(max(0.01, interval))

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
