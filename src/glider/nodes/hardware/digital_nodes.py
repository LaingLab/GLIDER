"""
Digital I/O Nodes

Nodes for digital input and output operations.
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


class DigitalWriteNode(HardwareNode):
    """Write a digital value to a pin."""

    definition = NodeDefinition(
        name="Digital Write",
        category=NodeCategory.HARDWARE,
        description="Write HIGH or LOW to a digital output pin",
        inputs=[
            PortDefinition(
                name="exec",
                port_type=PortType.EXEC,
                description="Trigger write operation",
            ),
            PortDefinition(
                name="value",
                port_type=PortType.DATA,
                data_type=bool,
                default_value=False,
                description="Value to write (True=HIGH, False=LOW)",
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
        """Write digital value to the device."""
        value = bool(self.get_input(1))  # Input 0 is exec, 1 is value

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


class DigitalReadNode(HardwareNode):
    """Read a digital value from a pin."""

    definition = NodeDefinition(
        name="Digital Read",
        category=NodeCategory.HARDWARE,
        description="Read HIGH or LOW from a digital input pin",
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
                data_type=bool,
                description="Read value (True=HIGH, False=LOW)",
            ),
        ],
        color="#2d5a2d",
    )

    def __init__(self):
        super().__init__()
        self._pin: int = 0
        self._continuous = False
        self._poll_interval = 0.1  # seconds
        self._polling_task = None

    @property
    def pin(self) -> int:
        return self._pin

    @pin.setter
    def pin(self, value: int) -> None:
        self._pin = value

    @property
    def continuous(self) -> bool:
        """Whether to continuously poll the input."""
        return self._continuous

    @continuous.setter
    def continuous(self, value: bool) -> None:
        self._continuous = value

    async def hardware_operation(self) -> None:
        """Read digital value from the device."""
        if self._device:
            value = await self._device.execute_action("read")
            self.set_output(1, value)  # Output 0 is exec, 1 is value
        else:
            self.set_error("No device bound")
            return

        # Trigger exec output and await downstream
        await self._fire_exec_output("exec")

    async def start(self) -> None:
        """Start continuous polling if enabled."""
        if self._continuous:
            self._polling_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop continuous polling."""
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
        state["continuous"] = self._continuous
        state["poll_interval"] = self._poll_interval
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._pin = state.get("pin", 0)
        self._continuous = state.get("continuous", False)
        self._poll_interval = state.get("poll_interval", 0.1)
