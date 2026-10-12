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

    PROPERTIES_SCHEMA = [
        {
            "key": "count",
            "label": "Iterations",
            "type": "int",
            "default": 0,
            "min": 0,
            "help": "How many times to run the loop body; 0 repeats forever.",
        },
        {
            "key": "delay",
            "label": "Delay (s)",
            "type": "float",
            "default": 1.0,
            "min": 0,
            "help": "Seconds to wait between iterations.",
        },
        {
            "key": "prompt_count",
            "label": "Prompt for iterations",
            "type": "bool",
            "default": False,
            "help": "Ask the operator for the iteration count at Start.",
        },
    ]

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
        count = self.setting("count")
        delay = self.setting("delay")

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

    PROPERTIES_SCHEMA = [
        {
            "key": "input_behavior",
            "label": "Input behaviour",
            "type": "str",
            "default": None,
            "help": "Name of the bound device's input behaviour to wait on.",
        },
        {
            "key": "timeout",
            "label": "Timeout (s)",
            "type": "float",
            "default": 0.0,
            "min": 0,
            "help": "Give up after this many seconds; 0 waits forever.",
        },
        {
            "key": "poll_interval",
            "label": "Poll interval (s)",
            "type": "float",
            "default": 0.05,
            "min": 0,
            "help": "Seconds between input checks.",
        },
        {
            "key": "behavior_settings",
            "label": "Behaviour settings",
            "type": "object",
            "default": {},
            "help": "Per-behaviour settings, keyed by behaviour name.",
        },
        {
            "key": "threshold_mode",
            "label": "Mode",
            "type": "enum",
            "default": "digital",
            "choices": [
                ["digital", "Digital (rising edge)"],
                ["analog", "Analog (threshold)"],
                ["revolution", "Revolution (turns)"],
                ["counts", "Move counts"],
            ],
            "help": "Used when the bound device offers no input behaviours.",
        },
        {
            "key": "threshold",
            "label": "Threshold",
            "type": "float",
            "default": 512,
            "min": 0,
            "help": "Analog mode: the value to cross.",
        },
        {
            "key": "threshold_direction",
            "label": "Direction",
            "type": "enum",
            "default": "above",
            "choices": [["above", "Above"], ["below", "Below"]],
            "help": "Analog mode: fire when the value goes above or below the threshold.",
        },
        {
            "key": "turns_target",
            "label": "Turns",
            "type": "float",
            "default": 1,
            "min": 0,
            "help": "Revolution mode: turns to wait for.",
        },
        {
            "key": "counts_per_turn",
            "label": "Counts per turn",
            "type": "int",
            "default": 4096,
            "min": 1,
            "help": "Revolution mode: encoder counts per turn.",
        },
        {
            "key": "counts_target",
            "label": "Counts",
            "type": "int",
            "default": 400,
            "min": 0,
            "help": "Counts mode: encoder counts to wait for.",
        },
        {
            "key": "ramp_down",
            "label": "Ramp down",
            "type": "bool",
            "default": False,
            "help": "Slow a motor as the target approaches.",
        },
        {
            "key": "ramp_device_id",
            "label": "Ramp device",
            "type": "device_ref",
            "default": None,
            "help": "The motor device to slow when ramping down.",
        },
        {
            "key": "drive_pwm",
            "label": "Drive PWM",
            "type": "int",
            "default": 100,
            "min": 0,
            "max": 255,
            "help": "Motor PWM before the ramp zone.",
        },
        {
            "key": "creep_pwm",
            "label": "Creep PWM",
            "type": "int",
            "default": 30,
            "min": 0,
            "max": 255,
            "help": "Motor PWM inside the ramp zone.",
        },
        {
            "key": "ramp_zone",
            "label": "Ramp zone",
            "type": "int",
            "default": 512,
            "min": 0,
            "help": "Counts before the target where the motor slows.",
        },
        {
            "key": "land_tolerance",
            "label": "Land tolerance",
            "type": "int",
            "default": 0,
            "min": 0,
            "help": "Counts short of the target that count as arrived.",
        },
    ]

    def __init__(self):
        super().__init__()
        self._waiting = False
        self._trigger_value = None
        self._poll_interval = self.setting("poll_interval")
        self._threshold_mode = self.setting("threshold_mode")
        self._threshold = self.setting("threshold")
        self._threshold_direction = self.setting("threshold_direction")
        self._turns_target = self.setting("turns_target")
        self._counts_per_turn = self.setting("counts_per_turn")
        self._counts_target = self.setting("counts_target")
        # Revolution-mode "ramp down to landing": decelerate a motor PWM as the
        # angle approaches the wrap so it coasts almost nothing and lands on ~0.
        self._ramp_down = self.setting("ramp_down")
        self._ramp_device_id = self.setting("ramp_device_id")
        self._drive_pwm = self.setting("drive_pwm")
        self._creep_pwm = self.setting("creep_pwm")
        self._ramp_zone = self.setting("ramp_zone")
        self._land_tolerance = self.setting("land_tolerance")
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

        # Route to a selected device input behavior when one is chosen and
        # available; otherwise fall through to the legacy poll path unchanged.
        behavior_key = self.setting("input_behavior")
        behaviors = list(getattr(self._device, "input_behaviors", []) or [])
        # A behavior-declaring device with no explicit selection defaults to its
        # first behavior — matching the editor's displayed default. Guarded on the
        # absence of a legacy "threshold_mode" key so previously-saved .glider files
        # (which always carry it) still load on the legacy path.
        if behaviors and not behavior_key and "threshold_mode" not in self._state:
            behavior_key = behaviors[0].key

        behavior = None
        if behavior_key:
            for b in behaviors:
                if b.key == behavior_key:
                    behavior = b
                    break

        if behavior is not None:
            await self._run_behavior(behavior)
            return

        timeout = self.setting("timeout")
        poll_interval = self.setting("poll_interval")
        self._poll_interval = poll_interval

        # Get threshold settings from state
        self._threshold_mode = self.setting("threshold_mode")
        self._threshold = self.setting("threshold")
        self._threshold_direction = self.setting("threshold_direction")
        self._turns_target = self.setting("turns_target")
        self._counts_per_turn = self.setting("counts_per_turn")
        self._counts_target = self.setting("counts_target")
        self._ramp_down = self.setting("ramp_down")
        self._ramp_device_id = self.setting("ramp_device_id")
        self._drive_pwm = self.setting("drive_pwm")
        self._creep_pwm = self.setting("creep_pwm")
        self._ramp_zone = self.setting("ramp_zone")
        self._land_tolerance = self.setting("land_tolerance")
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

    async def _run_behavior(self, behavior) -> None:
        """Wait via a device-declared input behavior instead of the legacy loop.

        Cleanup on timeout/error is the behavior's responsibility (its
        ``wait_for_input`` runs ``cleanup`` in a ``finally``), so the node does
        not stop a ramp motor on this path.

        Note: unlike the legacy loop, this path does not use ``self._waiting`` to
        break out. Aborting a wait relies on asyncio task cancellation, which
        propagates through ``wait_for_input``'s ``finally`` -> ``cleanup``. Do
        not "fix" the absence of a ``self._waiting`` flag here.
        """
        from glider.hal.input_behavior import BehaviorContext

        timeout = self.setting("timeout")
        poll = self.setting("poll_interval")
        all_settings = self.setting("behavior_settings")
        # schema defaults <- saved values (saved wins)
        settings = {f["key"]: f.get("default") for f in behavior.settings}
        settings.update(all_settings.get(behavior.key, {}))

        logger.info(f"  Waiting via input behavior '{behavior.key}' (timeout: {timeout}s)")

        ctx = BehaviorContext(
            device=self._device,
            hardware_manager=self._hardware_manager,
            poll_interval=poll,
        )
        try:
            value = await behavior.wait_for_input(settings, ctx, timeout)
        except TimeoutError:
            logger.info(f"  Timeout waiting for input behavior '{behavior.key}'")
            await self._fire_exec_output("timeout")
            return
        logger.info(f"  Behavior '{behavior.key}' triggered, value: {value}")
        self._trigger_value = value
        if len(self._outputs) > 2:
            self._outputs[2] = value
        for callback in self._update_callbacks:
            callback("value", value)
        await self._fire_exec_output("triggered")

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
        # across poll iterations; `settings` is a one-time snapshot of the node's
        # attributes taken at poll start (not re-read each iteration).
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
                    # Mirror the helper's rotation direction back onto the node;
                    # the ramp branch below reads it and a test asserts it.
                    self._ramp_direction = st["ramp_direction"]

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
        self._threshold_mode = self.setting("threshold_mode")
        self._threshold = self.setting("threshold")
        self._threshold_direction = self.setting("threshold_direction")
        self._turns_target = self.setting("turns_target")
        self._counts_per_turn = self.setting("counts_per_turn")
        self._counts_target = self.setting("counts_target")
        self._ramp_down = self.setting("ramp_down")
        self._ramp_device_id = self.setting("ramp_device_id")
        self._drive_pwm = self.setting("drive_pwm")
        self._creep_pwm = self.setting("creep_pwm")
        self._ramp_zone = self.setting("ramp_zone")
        self._land_tolerance = self.setting("land_tolerance")

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
