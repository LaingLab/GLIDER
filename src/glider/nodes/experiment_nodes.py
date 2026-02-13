"""
Experiment Flow Nodes - Basic nodes for experiment control.

These nodes provide the core functionality for running experiments:
- StartExperiment: Entry point
- EndExperiment: Exit point
- Delay: Wait for a duration
- Output: Write to a device
- Input: Read from a device
"""

import asyncio
import logging
from typing import Optional

from glider.nodes.base_node import (
    GliderNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

logger = logging.getLogger(__name__)


class StartExperimentNode(GliderNode):
    """Entry point for the experiment flow."""

    definition = NodeDefinition(
        name="StartExperiment",
        category=NodeCategory.LOGIC,
        description="Entry point - begins the experiment flow",
        inputs=[],
        outputs=[
            PortDefinition("next", PortType.EXEC, description="Triggers the next node"),
        ],
    )

    def update_event(self) -> None:
        """Called when inputs change - not used for start node."""
        pass

    async def start(self) -> None:
        """Called when experiment starts - triggers the flow."""
        logger.info(f"StartExperiment.start() called, node ID: {self._glider_id}")
        logger.info(f"  Registered callbacks: {len(self._update_callbacks)}")
        self.exec_output(0)

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        logger.info(
            f"StartExperiment.exec_output({index}) called, callbacks: {len(self._update_callbacks)}"
        )
        for i, callback in enumerate(self._update_callbacks):
            logger.debug(f"  Calling callback {i}")
            callback("next", True)


class EndExperimentNode(GliderNode):
    """Exit point for the experiment flow."""

    definition = NodeDefinition(
        name="EndExperiment",
        category=NodeCategory.LOGIC,
        description="Exit point - ends the experiment",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
        ],
        outputs=[],
    )

    def update_event(self) -> None:
        """Called when inputs change."""
        pass

    async def execute(self) -> None:
        """Called when this node is triggered."""
        logger.info(f"EndExperimentNode.execute() called, node ID: {self._glider_id}")
        logger.info("Experiment ended")


class DelayNode(GliderNode):
    """Wait for a specified duration."""

    definition = NodeDefinition(
        name="Delay",
        category=NodeCategory.LOGIC,
        description="Wait for a specified duration",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
            PortDefinition("seconds", PortType.DATA, float, 1.0, "Duration in seconds"),
        ],
        outputs=[
            PortDefinition("next", PortType.EXEC, description="Triggers after delay"),
        ],
    )

    def update_event(self) -> None:
        """Called when inputs change."""
        pass

    async def execute(self) -> None:
        """Wait for the specified duration then trigger output."""
        logger.info(f"DelayNode.execute() called, node ID: {self._glider_id}")
        logger.info(f"  Node state: {self._state}")

        # Priority: 1) Saved state, 2) Default (1.0 seconds)
        # The state is set by the properties panel when user changes duration
        if "duration" in self._state:
            duration = float(self._state["duration"])
            logger.info(f"  Using duration from state: {duration}")
        else:
            # No saved state, default to 1 second
            duration = 1.0
            logger.info(f"  Using default duration: {duration}")

        logger.info(f"Delay: waiting {duration} seconds")
        await asyncio.sleep(duration)
        logger.info("Delay: complete")
        self.exec_output(0)

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        logger.info(
            f"DelayNode.exec_output({index}) called, callbacks: {len(self._update_callbacks)}"
        )
        for callback in self._update_callbacks:
            callback("next", True)


class OutputNode(GliderNode):
    """Write a value to a device (digital HIGH/LOW or PWM 0-255)."""

    definition = NodeDefinition(
        name="Output",
        category=NodeCategory.HARDWARE,
        description="Write to a device (digital HIGH/LOW or PWM 0-255)",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
            PortDefinition("value", PortType.DATA, description="Value to write"),
        ],
        outputs=[
            PortDefinition("next", PortType.EXEC, description="Triggers after write"),
        ],
    )

    def update_event(self) -> None:
        """Called when inputs change."""
        pass

    async def execute(self) -> None:
        """Write the value to the bound device."""
        logger.info(f"OutputNode.execute() called, node ID: {self._glider_id}")
        logger.info(f"  Node state: {self._state}")

        # Priority: 1) Saved state, 2) Default (1 = HIGH)
        # The state is set by the properties panel when user selects HIGH/LOW or PWM value
        if "value" in self._state:
            value = self._state["value"]
            logger.info(f"  Using value from state: {value}")
        else:
            # No saved state, default to HIGH
            value = 1
            logger.info(f"  Using default value: {value}")

        if self._device is not None:
            try:
                device_type = getattr(self._device, "device_type", "")
                if device_type == "PWMOutput":
                    # PWM device: send integer value 0-255
                    pwm_value = max(0, min(255, int(value)))
                    logger.info(f"Output: setting PWM device to {pwm_value}")
                    if hasattr(self._device, "set_value"):
                        await self._device.set_value(pwm_value)
                    elif hasattr(self._device, "board"):
                        pin = list(self._device.pins.values())[0] if self._device.pins else 0
                        await self._device.board.write_analog(pin, pwm_value)
                else:
                    # Digital device: send bool HIGH/LOW
                    bool_value = bool(value)
                    logger.info(f"Output: setting device to {'HIGH' if bool_value else 'LOW'}")
                    if hasattr(self._device, "set_state"):
                        await self._device.set_state(bool_value)
                    elif hasattr(self._device, "turn_on") and hasattr(self._device, "turn_off"):
                        if bool_value:
                            await self._device.turn_on()
                        else:
                            await self._device.turn_off()
            except Exception as e:
                logger.error(f"Output error: {e}")
                self._error = str(e)
        else:
            logger.warning("Output: no device bound")

        self.exec_output(0)

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        logger.info(
            f"OutputNode.exec_output({index}) called, callbacks: {len(self._update_callbacks)}"
        )
        for callback in self._update_callbacks:
            callback("next", True)


class InputNode(GliderNode):
    """Read from a device."""

    definition = NodeDefinition(
        name="Input",
        category=NodeCategory.HARDWARE,
        description="Read from a device (digital or analog)",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
        ],
        outputs=[
            PortDefinition("value", PortType.DATA, description="Read value"),
            PortDefinition("next", PortType.EXEC, description="Triggers after read"),
        ],
    )

    def update_event(self) -> None:
        """Called when inputs change."""
        pass

    async def execute(self) -> None:
        """Read the value from the bound device."""
        value = None

        if self._device is not None:
            try:
                if hasattr(self._device, "read"):
                    value = await self._device.read()
                elif hasattr(self._device, "get_state"):
                    value = await self._device.get_state()
                logger.info(f"Input: read value = {value}")
            except Exception as e:
                logger.error(f"Input error: {e}")
                self._error = str(e)
        else:
            logger.warning("Input: no device bound")

        # Set output value
        if len(self._outputs) > 0:
            self._outputs[0] = value

        # Notify callbacks
        for callback in self._update_callbacks:
            callback("value", value)

        self.exec_output(1)

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        for callback in self._update_callbacks:
            callback("next", True)


class MotorGovernorNode(GliderNode):
    """Control a motor governor device (up/down/stop)."""

    definition = NodeDefinition(
        name="MotorGovernor",
        category=NodeCategory.HARDWARE,
        description="Control a motor governor (up/down/stop)",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
        ],
        outputs=[
            PortDefinition("next", PortType.EXEC, description="Triggers after action"),
        ],
    )

    def update_event(self) -> None:
        """Called when inputs change."""
        pass

    async def execute(self) -> None:
        """Execute the motor governor action."""
        logger.info(f"MotorGovernorNode.execute() called, node ID: {self._glider_id}")
        logger.info(f"  Node state: {self._state}")

        # Get action from state (up, down, stop)
        action = self._state.get("action", "stop")
        logger.info(f"  Action: {action}")

        if self._device is not None:
            try:
                if action == "up":
                    logger.info("MotorGovernor: moving up")
                    if hasattr(self._device, "move_up"):
                        await self._device.move_up()
                elif action == "down":
                    logger.info("MotorGovernor: moving down")
                    if hasattr(self._device, "move_down"):
                        await self._device.move_down()
                elif action == "stop":
                    logger.info("MotorGovernor: stopping")
                    if hasattr(self._device, "stop"):
                        await self._device.stop()
                else:
                    logger.warning(f"MotorGovernor: unknown action '{action}'")
            except Exception as e:
                logger.error(f"MotorGovernor error: {e}")
                self._error = str(e)
        else:
            logger.warning("MotorGovernor: no device bound")

        self.exec_output(0)

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        logger.info(f"MotorGovernorNode.exec_output({index}) called")
        for callback in self._update_callbacks:
            callback("next", True)


class CustomDeviceNode(GliderNode):
    """
    Operates on a specific pin of a custom device.

    This node writes or reads from a pin on the custom device.
    Configure the pin and value in the properties panel.
    """

    definition = NodeDefinition(
        name="CustomDevice",
        category=NodeCategory.HARDWARE,
        description="Operate on a custom device pin",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
        ],
        outputs=[
            PortDefinition("value", PortType.DATA, description="Read value (for input pins)"),
            PortDefinition("next", PortType.EXEC, description="Triggers next node"),
        ],
    )

    def __init__(self):
        super().__init__()
        self._custom_device_runner = None
        self._definition_id: Optional[str] = None

    def update_event(self) -> None:
        """Called when inputs change."""
        pass

    def set_custom_device_context(self, runner, definition_id: str) -> None:
        """Set the custom device runner and definition ID."""
        self._custom_device_runner = runner
        self._definition_id = definition_id

    async def execute(self) -> None:
        """Execute the pin operation."""
        logger.info(f"CustomDeviceNode.execute() called, node ID: {self._glider_id}")
        logger.info(f"  Node state: {self._state}")

        pin_name = self._state.get("pin", "")
        value = self._state.get("value", 0)

        if not pin_name:
            logger.warning("CustomDeviceNode: no pin selected")
            self.exec_output(0)
            return

        if self._custom_device_runner is not None:
            try:
                # Determine if this is an input or output pin
                pin_def = self._custom_device_runner.definition.get_pin(pin_name)
                if pin_def is None:
                    logger.warning(f"CustomDeviceNode: pin '{pin_name}' not found")
                else:
                    from glider.core.custom_device import PinType

                    if pin_def.pin_type in (PinType.DIGITAL_INPUT, PinType.ANALOG_INPUT):
                        # Read from input pin
                        read_value = await self._custom_device_runner.read_pin(pin_name)
                        logger.info(f"CustomDeviceNode: read {read_value} from pin '{pin_name}'")
                        # Set output value
                        if len(self._outputs) > 0:
                            self._outputs[0] = read_value
                        for callback in self._update_callbacks:
                            callback("value", read_value)
                    else:
                        # Write to output pin
                        await self._custom_device_runner.write_pin(pin_name, value)
                        logger.info(f"CustomDeviceNode: wrote {value} to pin '{pin_name}'")
            except Exception as e:
                logger.error(f"CustomDeviceNode error: {e}")
                self._error = str(e)
        else:
            logger.warning("CustomDeviceNode: no device runner bound")

        self.exec_output(0)

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        logger.info(f"CustomDeviceNode.exec_output({index}) called")
        for callback in self._update_callbacks:
            callback("next", True)


def register_experiment_nodes(flow_engine) -> None:
    """Register all experiment nodes with the flow engine."""
    flow_engine.register_node("StartExperiment", StartExperimentNode)
    flow_engine.register_node("EndExperiment", EndExperimentNode)
    flow_engine.register_node("Delay", DelayNode)
    flow_engine.register_node("Output", OutputNode)
    flow_engine.register_node("Input", InputNode)
    flow_engine.register_node("MotorGovernor", MotorGovernorNode)
    flow_engine.register_node("CustomDevice", CustomDeviceNode)
    # Keep CustomDeviceAction as alias for backward compatibility
    flow_engine.register_node("CustomDeviceAction", CustomDeviceNode)
    logger.info("Registered experiment nodes")
