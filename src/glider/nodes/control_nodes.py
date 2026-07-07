"""
Control Flow Nodes - Logic and looping for experiments.

These nodes provide control flow functionality:
- Loop: Repeat actions N times or indefinitely
- WaitForInput: Wait for input trigger before continuing
"""

import asyncio
import logging
import time

from glider.hal import revolution_tracking as rt
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

            body_start = time.monotonic()

            try:
                # Trigger body execution and AWAIT completion
                await self._exec_body_async()
            except Exception as e:
                logger.error(f"Error in Loop body (iteration {iteration}): {e}", exc_info=True)
                if not self._running:
                    break

            iteration += 1

            # Delay between iterations, compensating for body execution time
            # to prevent cumulative drift
            if delay > 0 and self._running:
                body_elapsed = time.monotonic() - body_start
                remaining = delay - body_elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)

        logger.info(f"  Loop completed after {iteration} iterations")

        # Trigger done output
        try:
            await self._exec_done_async()
        except Exception as e:
            logger.error(f"Error in Loop done execution: {e}", exc_info=True)

    async def _exec_body_async(self) -> None:
        """Trigger the body execution output and await completion."""
        tasks = []
        for callback in self._update_callbacks:
            try:
                result = callback("body", True)
                # Callbacks may return tasks that we should await
                if result is not None and asyncio.isfuture(result):
                    tasks.append(result)
            except Exception as e:
                logger.error(f"Error triggering body callback: {e}", exc_info=True)

        # Wait for all body tasks to complete
        if tasks:
            logger.info(f"  Awaiting {len(tasks)} body task(s)...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in body task {i}: {result}", exc_info=True)
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
        self._threshold_mode = "digital"  # "digital", "analog", or "revolution"
        self._threshold = 512  # Default analog threshold
        self._threshold_direction = "above"  # "above" or "below"
        self._turns_target = 1  # Revolution mode: fire after this many turns
        self._counts_per_turn = 4096  # Revolution mode: sensor full-scale range
        self._counts_target = 400  # Counts mode: move this many counts then stop
        # Revolution-mode "ramp down to landing": decelerate a motor PWM as the
        # angle approaches the wrap so it coasts almost nothing and lands on ~0.
        self._ramp_down = False
        self._ramp_device_id = None
        self._drive_pwm = 100  # speed before the deceleration zone
        self._creep_pwm = 30  # minimum speed at the wrap point
        self._ramp_zone = 512  # counts before the wrap where ramping begins
        self._land_tolerance = 0  # stop within this many counts of 0 (0 = off)
        self._landing_armed = False  # armed after the angle passes mid-range
        self._ramp_direction = 1  # +1 angle rising toward top wrap, -1 falling to 0
        self._hardware_manager = None  # set by the flow engine for device lookup
        self._ramp_device = None  # resolved PWM device, looked up at execute time

    def set_hardware_manager(self, hardware_manager) -> None:
        """Give the node access to other devices (e.g. the ramp-down motor).

        The flow engine calls this so revolution-mode ramp-down can drive a
        PWM device that is not the node's own bound (encoder) device.
        """
        self._hardware_manager = hardware_manager

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
        self._turns_target = self._state.get("turns_target", 1)
        self._counts_per_turn = self._state.get("counts_per_turn", 4096)
        self._counts_target = self._state.get("counts_target", 400)
        self._ramp_down = self._state.get("ramp_down", False)
        self._ramp_device_id = self._state.get("ramp_device_id")
        self._drive_pwm = self._state.get("drive_pwm", 100)
        self._creep_pwm = self._state.get("creep_pwm", 30)
        self._ramp_zone = self._state.get("ramp_zone", 512)
        self._land_tolerance = self._state.get("land_tolerance", 0)
        self._landing_armed = False  # re-arm fresh each run
        self._ramp_direction = 1  # assume rising until motion proves otherwise

        # Resolve the PWM device to ramp (revolution/counts modes). If it can't
        # be found, ramping is silently skipped — the wait still works.
        self._ramp_device = None
        if self._threshold_mode in ("revolution", "counts") and self._ramp_down:
            if self._ramp_device_id and self._hardware_manager is not None:
                self._ramp_device = self._hardware_manager.get_device(self._ramp_device_id)
            if self._ramp_device is None:
                logger.warning(
                    f"  Ramp-down enabled but ramp device '{self._ramp_device_id}' "
                    "not found; continuing without ramp"
                )

        logger.info(f"  Waiting for input (timeout: {timeout}s, mode: {self._threshold_mode})")
        if self._threshold_mode == "analog":
            logger.info(f"  Threshold: {self._threshold_direction} {self._threshold}")
        elif self._threshold_mode == "revolution":
            logger.info(
                f"  Revolution: {self._turns_target} turn(s), "
                f"{self._counts_per_turn} counts/turn"
            )
            if self._ramp_device is not None:
                logger.info(
                    f"  Ramp-down on '{self._ramp_device_id}': "
                    f"drive={self._drive_pwm}, creep={self._creep_pwm}, zone={self._ramp_zone}"
                )
        elif self._threshold_mode == "counts":
            logger.info(f"  Move counts: target {self._counts_target} (signed displacement)")

        self._waiting = True
        self._trigger_value = None

        try:
            await self._poll_device(timeout)

            # Triggered successfully
            logger.info(f"  Input received: {self._trigger_value}")
            # Set output value on the DATA port (index 2: "value")
            if len(self._outputs) > 2:
                self._outputs[2] = self._trigger_value
            # Fire value callbacks (non-exec notification)
            for callback in self._update_callbacks:
                callback("value", self._trigger_value)
            # Fire triggered exec output and await downstream chain
            await self._fire_exec_output("triggered")

        except TimeoutError:
            logger.info("  Timeout waiting for input")
            # De-energize the ramp-driven motor: the triggered-path stop never
            # ran, and downstream stop nodes hang off "triggered", not "timeout".
            await self._set_ramp_pwm(0)
            await self._fire_exec_output("timeout")

        except Exception:
            # Polling failed (e.g. repeated device read errors). Stop the
            # ramp-driven motor before propagating — the error halts the exec
            # chain, so no downstream node will ever write PWM 0.
            await self._set_ramp_pwm(0)
            raise

        finally:
            self._waiting = False

    async def _poll_device(self, timeout: float) -> None:
        """Poll the bound device until condition is met or timeout."""
        import time

        start_time = time.time()
        poll_count = 0
        error_count = 0
        max_errors = 3  # Stop after 3 consecutive errors
        last_value = None  # digital-mode rising-edge detection
        # Revolution/counts wrap math is delegated to the shared pure helper.
        # `st` carries last_value/turn_count/accumulated/landing_armed/ramp_direction
        # across poll iterations; `settings` mirrors the node's current attributes.
        st = rt.new_state()
        settings = {
            "turns_target": self._turns_target,
            "counts_per_turn": self._counts_per_turn,
            "counts_target": self._counts_target,
            "land_tolerance": self._land_tolerance,
            "drive_pwm": self._drive_pwm,
            "creep_pwm": self._creep_pwm,
            "ramp_zone": self._ramp_zone,
        }

        logger.info(f"  Starting device poll loop, device type: {type(self._device).__name__}")

        while self._waiting:
            # Check timeout
            if timeout > 0 and (time.time() - start_time) >= timeout:
                raise TimeoutError()

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

                elif self._threshold_mode == "revolution" and isinstance(value, (int, float)):
                    # Revolution mode: count wrap-arounds of a sawtooth sensor
                    # (e.g. AS5600 raw angle 0-4095). The turn-counting, rotation
                    # direction, mid-range arming and landing-tolerance math live
                    # in the shared pure helper; `st` carries state across polls.
                    triggered = rt.revolution_triggered(value, settings, st)
                    # Mirror the helper's rotation direction and arm gate back
                    # onto the node (tests assert `node._ramp_direction`).
                    self._ramp_direction = st["ramp_direction"]
                    self._landing_armed = st["landing_armed"]

                    # Ramp down to landing: on the final turn, ease the motor
                    # PWM toward a creep as the angle nears the wrap so it
                    # coasts almost nothing and lands on ~0.
                    if (
                        not triggered
                        and self._ramp_device is not None
                        and st["turn_count"] >= self._turns_target - 1
                    ):
                        remaining = (
                            self._counts_per_turn - value if self._ramp_direction >= 0 else value
                        )
                        await self._set_ramp_pwm(rt.ramp_pwm(remaining, settings))

                elif self._threshold_mode == "counts" and isinstance(value, (int, float)):
                    # Move-counts mode: accumulate signed displacement from the
                    # start (wrap-corrected) and stop once the magnitude reaches
                    # the target. Direction is whichever way the motor drives, so
                    # this is bidirectional via the absolute displacement.
                    triggered = rt.counts_triggered(value, settings, st)
                    if not triggered and self._ramp_device is not None:
                        # Clamp the deceleration zone to the move length, so a
                        # zone larger than the target still starts at drive_pwm
                        # instead of opening partway down the ramp.
                        await self._set_ramp_pwm(
                            rt.ramp_pwm(
                                self._counts_target - abs(st["accumulated"]),
                                settings,
                                span=min(self._ramp_zone, self._counts_target),
                            )
                        )

                if triggered:
                    self._trigger_value = value
                    # Stop the motor immediately at the landing point.
                    if self._ramp_device is not None:
                        await self._set_ramp_pwm(0)
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

    async def _set_ramp_pwm(self, pwm: float) -> None:
        """Write a 0-255 PWM value to the resolved ramp device."""
        device = self._ramp_device
        if device is None:
            return
        pwm_value = max(0, min(255, int(round(pwm))))
        try:
            if hasattr(device, "set_value"):
                await device.set_value(pwm_value)
            elif hasattr(device, "board"):
                pin = list(device.pins.values())[0] if getattr(device, "pins", None) else 0
                await device.board.write_analog(pin, pwm_value)
        except Exception as e:
            logger.error(f"  Ramp PWM write failed: {e}")

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
        state["turns_target"] = self._turns_target
        state["counts_per_turn"] = self._counts_per_turn
        state["counts_target"] = self._counts_target
        state["ramp_down"] = self._ramp_down
        state["ramp_device_id"] = self._ramp_device_id
        state["drive_pwm"] = self._drive_pwm
        state["creep_pwm"] = self._creep_pwm
        state["ramp_zone"] = self._ramp_zone
        state["land_tolerance"] = self._land_tolerance
        return state

    def set_state(self, state: dict) -> None:
        """Set node state from deserialization."""
        super().set_state(state)
        self._threshold_mode = state.get("threshold_mode", "digital")
        self._threshold = state.get("threshold", 512)
        self._threshold_direction = state.get("threshold_direction", "above")
        self._turns_target = state.get("turns_target", 1)
        self._counts_per_turn = state.get("counts_per_turn", 4096)
        self._counts_target = state.get("counts_target", 400)
        self._ramp_down = state.get("ramp_down", False)
        self._ramp_device_id = state.get("ramp_device_id")
        self._drive_pwm = state.get("drive_pwm", 100)
        self._creep_pwm = state.get("creep_pwm", 30)
        self._ramp_zone = state.get("ramp_zone", 512)
        self._land_tolerance = state.get("land_tolerance", 0)

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
