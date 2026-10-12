"""
Analog I/O Nodes

Nodes for analog input and PWM output operations.
"""

import asyncio
from typing import Any

from glider.hal.value_spec import clamp_to_spec
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

    PROPERTIES_SCHEMA = [
        {
            "key": "pin",
            "label": "Pin",
            "type": "int",
            "default": 0,
            "min": 0,
            "help": "Analog pin to read when no device is bound.",
        },
        {
            "key": "reference_voltage",
            "label": "Reference (V)",
            "type": "float",
            "default": 5.0,
            "min": 0,
            "help": "ADC reference voltage, for the voltage output.",
        },
        {
            "key": "resolution",
            "label": "Resolution (bits)",
            "type": "int",
            "default": 10,
            "min": 1,
            "help": "ADC resolution in bits.",
        },
        {
            "key": "continuous",
            "label": "Continuous",
            "type": "bool",
            "default": False,
            "help": "Keep reading while the flow runs instead of once per exec.",
        },
        {
            "key": "poll_interval",
            "label": "Poll interval (s)",
            "type": "float",
            "default": 0.05,
            "min": 0,
            "help": "Seconds between continuous reads.",
        },
        {
            "key": "threshold",
            "label": "Threshold",
            "type": "float",
            "default": 512,
            "min": 0,
            "help": "Raw value that sets threshold_exceeded.",
        },
        {
            "key": "threshold_enabled",
            "label": "Use threshold",
            "type": "bool",
            "default": False,
            "help": "Report threshold_exceeded.",
        },
        {
            "key": "visible_in_runner",
            "label": "Show in runner",
            "type": "bool",
            "default": False,
            "help": "Show this reading on the runner dashboard.",
        },
    ]

    def __init__(self):
        super().__init__()
        self._pin: int = self.setting("pin")
        self._reference_voltage = self.setting("reference_voltage")
        self._resolution = self.setting("resolution")  # bits
        self._continuous = self.setting("continuous")
        self._poll_interval = self.setting("poll_interval")  # 20Hz default
        self._polling_task = None
        self._threshold = self.setting("threshold")  # Default threshold (mid-range)
        self._threshold_enabled = self.setting("threshold_enabled")
        self.visible_in_runner = self.setting(
            "visible_in_runner"
        )  # Can be enabled for live display

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
            # Full-scale reading from the bound device's declared range rather
            # than a hardcoded 10-bit assumption; fall back to the node's own
            # resolution when unbound or the device declares no read range.
            read_spec = self._device.value_spec("read")
            self.set_output(1, raw_value)

            # Calculate voltage from the device's full-scale range.
            max_value = read_spec.max if read_spec is not None else (2**self._resolution - 1)
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
            from glider.core.async_utils import log_task_exception

            self._polling_task = asyncio.create_task(self._poll_loop())
            self._polling_task.add_done_callback(log_task_exception)

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
        self._pin = self.setting("pin")
        self._reference_voltage = self.setting("reference_voltage")
        self._resolution = self.setting("resolution")
        self._continuous = self.setting("continuous")
        self._poll_interval = self.setting("poll_interval")
        self._threshold = self.setting("threshold")
        self._threshold_enabled = self.setting("threshold_enabled")
        self.visible_in_runner = self.setting("visible_in_runner")


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

    PROPERTIES_SCHEMA = [
        {
            "key": "pin",
            "label": "Pin",
            "type": "int",
            "default": 0,
            "min": 0,
            "help": "Pin to write when no device is bound.",
        },
    ]

    def __init__(self):
        super().__init__()
        self._pin: int = self.setting("pin")

    @property
    def pin(self) -> int:
        return self._pin

    @pin.setter
    def pin(self, value: int) -> None:
        self._pin = value

    async def hardware_operation(self) -> None:
        """Write PWM value to the device."""
        value = self.get_input(1)  # Input 0 is exec, 1 is value

        if self._device:
            # Clamp to the bound device's declared range rather than a hardcoded
            # 0-255: an 8-bit board still yields 0-255, a 12-bit one yields
            # 0-4095. The device remains the single authority for its range.
            spec = self._device.value_spec("set")
            if spec is not None:
                value, _ = clamp_to_spec(value, spec)
            else:
                # No declared range: keep the historical 8-bit ceiling.
                value = max(0, min(255, int(value)))
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
        self._pin = self.setting("pin")
