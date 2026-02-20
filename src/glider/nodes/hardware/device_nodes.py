"""
Device Nodes

Generic nodes for interacting with any device type through
their action interfaces.
"""

from typing import Any

from glider.nodes.base_node import (
    HardwareNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)


class DeviceActionNode(HardwareNode):
    """Execute an action on a device."""

    definition = NodeDefinition(
        name="Device Action",
        category=NodeCategory.HARDWARE,
        description="Execute a named action on a device",
        inputs=[
            PortDefinition(
                name="exec",
                port_type=PortType.EXEC,
                description="Trigger action execution",
            ),
            PortDefinition(
                name="arg1",
                port_type=PortType.DATA,
                description="Optional argument 1",
            ),
            PortDefinition(
                name="arg2",
                port_type=PortType.DATA,
                description="Optional argument 2",
            ),
        ],
        outputs=[
            PortDefinition(
                name="exec",
                port_type=PortType.EXEC,
                description="Triggered after action completes",
            ),
            PortDefinition(
                name="result",
                port_type=PortType.DATA,
                description="Action result",
            ),
        ],
        color="#2d5a2d",
    )

    def __init__(self):
        super().__init__()
        self._action_name = ""

    @property
    def action_name(self) -> str:
        return self._action_name

    @action_name.setter
    def action_name(self, value: str) -> None:
        self._action_name = value

    async def hardware_operation(self) -> None:
        """Execute the device action."""
        if not self._action_name:
            self.set_error("No action specified")
            return

        if not self._device:
            self.set_error("No device bound")
            return

        # Collect arguments
        args = []
        if self.get_input(1) is not None:
            args.append(self.get_input(1))
        if self.get_input(2) is not None:
            args.append(self.get_input(2))

        # Execute action
        result = await self._device.execute_action(self._action_name, *args)
        self.set_output(1, result)

        # Trigger exec output and await downstream
        await self._fire_exec_output("exec")

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["action_name"] = self._action_name
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._action_name = state.get("action_name", "")


class DeviceReadNode(HardwareNode):
    """Read a value from a device."""

    definition = NodeDefinition(
        name="Device Read",
        category=NodeCategory.HARDWARE,
        description="Read a value from a device",
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
                description="Read value",
            ),
        ],
        color="#2d5a2d",
    )

    def __init__(self):
        super().__init__()
        self._read_action = "read"

    @property
    def read_action(self) -> str:
        return self._read_action

    @read_action.setter
    def read_action(self, value: str) -> None:
        self._read_action = value

    async def hardware_operation(self) -> None:
        """Read value from the device."""
        if not self._device:
            self.set_error("No device bound")
            return

        value = await self._device.execute_action(self._read_action)
        self.set_output(1, value)

        # Trigger exec output and await downstream
        await self._fire_exec_output("exec")

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["read_action"] = self._read_action
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._read_action = state.get("read_action", "read")
