"""
Control Flow Nodes - Logic and looping for experiments.

These nodes provide control flow functionality:
- Loop: Repeat actions N times or indefinitely
- WaitForInput: Wait for input trigger before continuing
"""

import asyncio
import logging

from glider.nodes.base_node import (
    GliderNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

logger = logging.getLogger(__name__)


class LoopNode(GliderNode):
    """
    Loop node - repeats connected actions.

    Can loop:
    - A specific number of times (count > 0)
    - Indefinitely until stopped (count = 0)
    """

    definition = NodeDefinition(
        name="Loop",
        category=NodeCategory.LOGIC,
        description="Repeat actions N times (0 = infinite)",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Start the loop"),
        ],
        outputs=[
            PortDefinition("body", PortType.EXEC, description="Executes each iteration"),
            PortDefinition("done", PortType.EXEC, description="Executes when loop completes"),
        ],
    )

    def __init__(self):
        super().__init__()
        self._running = False
        self._current_index = 0

    def update_event(self) -> None:
        pass

    async def execute(self) -> None:
        """Execute the loop."""
        logger.info(f"LoopNode.execute() called, node ID: {self._glider_id}")

        # Get parameters from state
        count = self._state.get("count", 0)
        delay = self._state.get("delay", 1.0)

        logger.info(f"  Loop count: {count} (0=infinite), delay: {delay}s")

        self._running = True
        self._current_index = 0

        iteration = 0
        while self._running:
            # Check if we've completed the requested iterations
            if count > 0 and iteration >= count:
                break

            self._current_index = iteration

            logger.info(f"  Loop iteration {iteration}")

            # Trigger body execution and AWAIT completion
            await self._exec_body_async()

            iteration += 1

            # Delay between iterations (only if body completed and we're continuing)
            if delay > 0 and self._running:
                await asyncio.sleep(delay)

        logger.info(f"  Loop completed after {iteration} iterations")

        # Trigger done output
        await self._exec_done_async()

    async def _exec_body_async(self) -> None:
        """Trigger the body execution output and await completion."""
        tasks = []
        for callback in self._update_callbacks:
            result = callback("body", True)
            # Callbacks may return tasks that we should await
            if result is not None and asyncio.isfuture(result):
                tasks.append(result)

        # Wait for all body tasks to complete
        if tasks:
            logger.info(f"  Awaiting {len(tasks)} body task(s)...")
            await asyncio.gather(*tasks)
            logger.info("  Body execution complete")

    async def _exec_done_async(self) -> None:
        """Trigger the done execution output and await completion."""
        tasks = []
        for callback in self._update_callbacks:
            result = callback("done", True)
            if result is not None and asyncio.isfuture(result):
                tasks.append(result)

        if tasks:
            await asyncio.gather(*tasks)

    def _exec_body(self) -> None:
        """Trigger the body execution output (sync version for compatibility)."""
        for callback in self._update_callbacks:
            callback("body", True)

    def _exec_done(self) -> None:
        """Trigger the done execution output (sync version for compatibility)."""
        for callback in self._update_callbacks:
            callback("done", True)

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output by index."""
        if index == 0:
            self._exec_body()
        elif index == 1:
            self._exec_done()

    async def stop(self) -> None:
        """Stop the loop."""
        logger.info(f"LoopNode.stop() called, node ID: {self._glider_id}")
        self._running = False


class WaitForInputNode(GliderNode):
    """
    Wait for Input node - pauses until an input is received.

    Supports two modes:
    - Digital: Wait for HIGH (True) signal
    - Analog: Wait until value crosses threshold
    """

    definition = NodeDefinition(
        name="WaitForInput",
        category=NodeCategory.LOGIC,
        description="Wait for input trigger (digital or analog threshold)",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Start waiting"),
        ],
        outputs=[
            PortDefinition("triggered", PortType.EXEC, description="Executes when triggered"),
            PortDefinition("timeout", PortType.EXEC, description="Executes on timeout"),
            PortDefinition(
                "value", PortType.DATA, data_type=int, description="Read value when triggered"
            ),
        ],
    )

    def __init__(self):
        super().__init__()
        self._waiting = False
        self._trigger_value = None
        self._poll_interval = 0.05  # 50ms polling interval
        self._threshold_mode = "digital"  # "digital" or "analog"
        self._threshold = 512  # Default analog threshold
        self._threshold_direction = "above"  # "above" or "below"

    def update_event(self) -> None:
        """Called when inputs change."""
        pass

    async def execute(self) -> None:
        """Wait for input from bound device."""
        logger.info(f"WaitForInputNode.execute() called, node ID: {self._glider_id}")

        if self._device is None:
            logger.error("  No device bound to WaitForInput node")
            return

        timeout = self._state.get("timeout", 0.0)
        poll_interval = self._state.get("poll_interval", 0.05)
        self._poll_interval = poll_interval

        # Get threshold settings from state
        self._threshold_mode = self._state.get("threshold_mode", "digital")
        self._threshold = self._state.get("threshold", 512)
        self._threshold_direction = self._state.get("threshold_direction", "above")

        logger.info(f"  Waiting for input (timeout: {timeout}s, mode: {self._threshold_mode})")
        if self._threshold_mode == "analog":
            logger.info(f"  Threshold: {self._threshold_direction} {self._threshold}")

        self._waiting = True
        self._trigger_value = None

        try:
            await self._poll_device(timeout)

            # Triggered successfully
            logger.info(f"  Input received: {self._trigger_value}")
            # Set output value
            if len(self._outputs) > 0:
                self._outputs[0] = self._trigger_value
            # Fire value callbacks (non-exec notification)
            for callback in self._update_callbacks:
                callback("value", self._trigger_value)
            # Fire triggered exec output and await downstream chain
            await self._fire_exec_output("triggered")

        except asyncio.TimeoutError:
            logger.info("  Timeout waiting for input")
            await self._fire_exec_output("timeout")

        finally:
            self._waiting = False

    async def _poll_device(self, timeout: float) -> None:
        """Poll the bound device until condition is met or timeout."""
        import time

        start_time = time.time()
        poll_count = 0
        error_count = 0
        max_errors = 3  # Stop after 3 consecutive errors
        last_value = None  # Track previous value for edge detection

        logger.info(f"  Starting device poll loop, device type: {type(self._device).__name__}")

        while self._waiting:
            # Check timeout
            if timeout > 0 and (time.time() - start_time) >= timeout:
                raise asyncio.TimeoutError()

            # Read from device
            try:
                value = None
                if hasattr(self._device, "read"):
                    value = await self._device.read()
                elif hasattr(self._device, "get_state"):
                    value = await self._device.get_state()

                # Reset error count on successful read
                error_count = 0
                poll_count += 1

                # Log every 20 polls (~1 second at 50ms interval)
                if poll_count % 20 == 1:
                    logger.info(f"  Poll #{poll_count}: value = {value}")

                # Check trigger condition based on mode
                triggered = False

                if self._threshold_mode == "digital":
                    # Digital mode: detect rising edge (LOW to HIGH)
                    if not last_value and value:
                        logger.info("  TRIGGERED! Rising edge detected")
                        triggered = True

                elif self._threshold_mode == "analog":
                    # Analog mode: check threshold crossing
                    if isinstance(value, (int, float)):
                        if self._threshold_direction == "above":
                            if value > self._threshold:
                                logger.info(
                                    f"  TRIGGERED! Value {value} > threshold {self._threshold}"
                                )
                                triggered = True
                        else:  # below
                            if value < self._threshold:
                                logger.info(
                                    f"  TRIGGERED! Value {value} < threshold {self._threshold}"
                                )
                                triggered = True

                if triggered:
                    self._trigger_value = value
                    return

                last_value = value

            except Exception as e:
                error_count += 1
                if error_count <= max_errors:
                    logger.error(f"  Error polling device ({error_count}/{max_errors}): {e}")
                if error_count >= max_errors:
                    logger.error("  Too many polling errors - stopping poll loop")
                    raise RuntimeError(f"Device polling failed: {e}") from e

            # Wait before next poll
            await asyncio.sleep(self._poll_interval)

    def _exec_triggered(self) -> None:
        """Trigger the triggered output."""
        # First send the value
        for callback in self._update_callbacks:
            callback("value", self._trigger_value)
        # Then trigger execution
        for callback in self._update_callbacks:
            callback("triggered", True)

    def get_state(self) -> dict:
        """Get node state for serialization."""
        state = super().get_state()
        state["threshold_mode"] = self._threshold_mode
        state["threshold"] = self._threshold
        state["threshold_direction"] = self._threshold_direction
        return state

    def set_state(self, state: dict) -> None:
        """Set node state from deserialization."""
        super().set_state(state)
        self._threshold_mode = state.get("threshold_mode", "digital")
        self._threshold = state.get("threshold", 512)
        self._threshold_direction = state.get("threshold_direction", "above")

    def _exec_timeout(self) -> None:
        """Trigger the timeout output."""
        for callback in self._update_callbacks:
            callback("timeout", True)

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output by index."""
        if index == 0:
            self._exec_triggered()
        elif index == 1:
            self._exec_timeout()

    async def stop(self) -> None:
        """Stop waiting."""
        self._waiting = False


def register_control_nodes(flow_engine) -> None:
    """Register all control flow nodes with the flow engine."""
    flow_engine.register_node("Loop", LoopNode)
    flow_engine.register_node("WaitForInput", WaitForInputNode)
    logger.info("Registered control flow nodes")
