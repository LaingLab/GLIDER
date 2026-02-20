"""
Analog I/O Nodes

Nodes for analog input and PWM output operations.
"""

import asyncio
from typing import Any

from glider.nodes.base_node import (
    HardwareNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)


class AnalogReadNode(HardwareNode):
    """Read an analog value from a pin."""

    definition = NodeDefinition(
        name="Analog Read",
        category=NodeCategory.HARDWARE,
        description="Read analog value (0-1023 for 10-bit ADC)",
        inputs=[
            PortDefinition(
                name="exec",
                port_type=PortType.EXEC,
                description="Trigger read operation",
            ),
        ],
        outputs=[
            PortDefinition(
                name="exec",
                port_type=PortType.EXEC,
                description="Triggered after read completes",
            ),
            PortDefinition(
                name="value",
                port_type=PortType.DATA,
                data_type=int,
                description="Raw analog value",
            ),
            PortDefinition(
                name="voltage",
                port_type=PortType.DATA,
                data_type=float,
                description="Voltage value",
            ),
            PortDefinition(
                name="threshold_exceeded",
                port_type=PortType.DATA,
                data_type=bool,
                description="True if value exceeds threshold",
            ),
        ],
        color="#2d5a2d",
    )

    def __init__(self):
        super().__init__()
        self._pin: int = 0
        self._reference_voltage = 5.0
        self._resolution = 10  # bits
        self._continuous = False
        self._poll_interval = 0.05  # 20Hz default
        self._polling_task = None
        self._threshold = 512  # Default threshold (mid-range)
        self._threshold_enabled = False
        self.visible_in_runner = False  # Can be enabled for live display

    @property
    def pin(self) -> int:
        return self._pin

    @pin.setter
    def pin(self, value: int) -> None:
        self._pin = value

    async def hardware_operation(self) -> None:
        """Read analog value from the device."""
        if self._device:
            raw_value = await self._device.execute_action("read")
            self.set_output(1, raw_value)

            # Calculate voltage
            max_value = 2**self._resolution - 1
            voltage = (raw_value / max_value) * self._reference_voltage
            self.set_output(2, voltage)

            # Check threshold
            threshold_exceeded = False
            if self._threshold_enabled:
                threshold_exceeded = raw_value > self._threshold
            self.set_output(3, threshold_exceeded)
        else:
            self.set_error("No device bound")
            return

        # Trigger exec output and await downstream
        await self._fire_exec_output("exec")

    def get_display_value(self) -> str:
        """Get formatted value for display in dashboard."""
        if len(self._outputs) > 1 and self._outputs[1] is not None:
            raw = self._outputs[1]
            if len(self._outputs) > 2 and self._outputs[2] is not None:
                voltage = self._outputs[2]
                return f"{raw} ({voltage:.2f}V)"
            return f"{raw}"
        return "---"

    async def start(self) -> None:
        """Start continuous reading if enabled."""
        if self._continuous:
            self._polling_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop continuous reading."""
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None

    async def _poll_loop(self) -> None:
        """Continuous polling loop."""
        while True:
            try:
                await self.hardware_operation()
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.set_error(str(e))
                await asyncio.sleep(self._poll_interval)

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["pin"] = self._pin
        state["reference_voltage"] = self._reference_voltage
        state["resolution"] = self._resolution
        state["continuous"] = self._continuous
        state["poll_interval"] = self._poll_interval
        state["threshold"] = self._threshold
        state["threshold_enabled"] = self._threshold_enabled
        state["visible_in_runner"] = self.visible_in_runner
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._pin = state.get("pin", 0)
        self._reference_voltage = state.get("reference_voltage", 5.0)
        self._resolution = state.get("resolution", 10)
        self._continuous = state.get("continuous", False)
        self._poll_interval = state.get("poll_interval", 0.05)
        self._threshold = state.get("threshold", 512)
        self._threshold_enabled = state.get("threshold_enabled", False)
        self.visible_in_runner = state.get("visible_in_runner", False)


class PWMWriteNode(HardwareNode):
    """Write a PWM value to a pin."""

    definition = NodeDefinition(
        name="PWM Write",
        category=NodeCategory.HARDWARE,
        description="Write PWM value (0-255 for 8-bit PWM)",
        inputs=[
            PortDefinition(
                name="exec",
                port_type=PortType.EXEC,
                description="Trigger write operation",
            ),
            PortDefinition(
                name="value",
                port_type=PortType.DATA,
                data_type=int,
                default_value=0,
                description="PWM value (0-255)",
            ),
        ],
        outputs=[
            PortDefinition(
                name="exec",
                port_type=PortType.EXEC,
                description="Triggered after write completes",
            ),
        ],
        color="#2d5a2d",
    )

    def __init__(self):
        super().__init__()
        self._pin: int = 0

    @property
    def pin(self) -> int:
        return self._pin

    @pin.setter
    def pin(self, value: int) -> None:
        self._pin = value

    async def hardware_operation(self) -> None:
        """Write PWM value to the device."""
        value = int(self.get_input(1))  # Input 0 is exec, 1 is value
        value = max(0, min(255, value))

        if self._device:
            await self._device.execute_action("set", value)
        else:
            self.set_error("No device bound")
            return

        # Trigger exec output and await downstream
        await self._fire_exec_output("exec")

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["pin"] = self._pin
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._pin = state.get("pin", 0)
