"""
Control Nodes - PID controller, counter, toggle, and state machines.
"""

import time
from typing import Any

from glider.nodes.base_node import (
    ExecNode,
    LogicNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)


class PIDNode(LogicNode):
    """PID Controller for feedback control loops."""

    definition = NodeDefinition(
        name="PID Controller",
        category=NodeCategory.LOGIC,
        description="Proportional-Integral-Derivative controller",
        inputs=[
            PortDefinition(name="Setpoint", data_type=float, default_value=0.0),
            PortDefinition(name="Process Value", data_type=float, default_value=0.0),
            PortDefinition(name="Kp", data_type=float, default_value=1.0),
            PortDefinition(name="Ki", data_type=float, default_value=0.0),
            PortDefinition(name="Kd", data_type=float, default_value=0.0),
        ],
        outputs=[
            PortDefinition(name="Output", data_type=float),
            PortDefinition(name="Error", data_type=float),
        ],
        color="#2d4a5a",
    )

    def __init__(self):
        super().__init__()
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()
        self._output_min = -255.0
        self._output_max = 255.0

    def reset(self) -> None:
        """Reset the PID state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()

    def process(self) -> None:
        setpoint = float(self.get_input(0) or 0)
        process_value = float(self.get_input(1) or 0)
        kp = float(self.get_input(2) or 1)
        ki = float(self.get_input(3) or 0)
        kd = float(self.get_input(4) or 0)

        # Calculate time delta
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0:
            dt = 0.001
        self._last_time = current_time

        # Calculate error
        error = setpoint - process_value
        self.set_output(1, error)

        # Proportional term
        p_term = kp * error

        # Integral term (with anti-windup)
        self._integral += error * dt
        i_term = ki * self._integral

        # Derivative term
        derivative = (error - self._last_error) / dt
        d_term = kd * derivative
        self._last_error = error

        # Calculate output
        output = p_term + i_term + d_term

        # Clamp output
        output = max(self._output_min, min(self._output_max, output))

        # Anti-windup: if output is saturated, don't accumulate more integral
        if output >= self._output_max or output <= self._output_min:
            self._integral -= error * dt

        self.set_output(0, output)

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["integral"] = self._integral
        state["last_error"] = self._last_error
        state["output_min"] = self._output_min
        state["output_max"] = self._output_max
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._integral = state.get("integral", 0.0)
        self._last_error = state.get("last_error", 0.0)
        self._output_min = state.get("output_min", -255.0)
        self._output_max = state.get("output_max", 255.0)


class ToggleNode(ExecNode):
    """Toggle between on/off states."""

    definition = NodeDefinition(
        name="Toggle",
        category=NodeCategory.LOGIC,
        description="Toggle state on each trigger",
        inputs=[
            PortDefinition(name="Toggle", port_type=PortType.EXEC),
            PortDefinition(name="Set On", port_type=PortType.EXEC),
            PortDefinition(name="Set Off", port_type=PortType.EXEC),
        ],
        outputs=[
            PortDefinition(name="State", data_type=bool),
            PortDefinition(name="On", port_type=PortType.EXEC),
            PortDefinition(name="Off", port_type=PortType.EXEC),
        ],
        color="#2d4a5a",
    )

    def __init__(self):
        super().__init__()
        self._toggle_state = False

    @property
    def state(self) -> bool:
        return self._toggle_state

    async def toggle(self) -> None:
        """Toggle the state."""
        self._toggle_state = not self._toggle_state
        await self._update_outputs()

    async def set_on(self) -> None:
        """Set state to on."""
        self._toggle_state = True
        await self._update_outputs()

    async def set_off(self) -> None:
        """Set state to off."""
        self._toggle_state = False
        await self._update_outputs()

    async def _update_outputs(self) -> None:
        """Update outputs based on current state."""
        self.set_output(0, self._toggle_state)
        if self._toggle_state:
            await self._fire_exec_output("On")
        else:
            await self._fire_exec_output("Off")

    async def execute(self) -> None:
        """Execute based on which input was triggered."""
        pass

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["toggle_state"] = self._toggle_state
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._toggle_state = state.get("toggle_state", False)
