"""A4988 stepper driver device.

Drives an A4988 stepper driver's STEP/DIR/ENABLE/MS1-3 pins directly via
gpiozero, bypassing the board abstraction (same precedent as ADS1115Device /
GenericI2CDevice: lazy-import in ``initialize()``, blocking work via
``asyncio.to_thread``). Control is open-loop: a move pulses an exact step
count and returns when the motor has stopped.

Sign convention: positive steps/turns drive DIR high, negative drive it low.
Which physical direction that is depends on how the motor coils are wired.

The A4988's ENABLE input is active-low (LOW = outputs energized). The pin is
claimed with ``active_high=False`` so "on" always means "energized" in code,
and the ``energize``/``de_energize`` actions speak the same language. They are
deliberately NOT named ``enable``/``disable``: ``BaseDevice.enable/disable``
already exist and gate ``execute_action`` via ``self._enabled`` — overriding
them would make every action raise "Device is disabled" after de-energizing.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from glider.hal.base_device import BaseDevice, DeviceConfig

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard

logger = logging.getLogger(__name__)

# steptype -> (steps multiplier, MS1, MS2, MS3) per the A4988 datasheet.
MICROSTEP_MODES: dict[str, tuple[int, bool, bool, bool]] = {
    "Full": (1, False, False, False),
    "Half": (2, True, False, False),
    "1/4": (4, False, True, False),
    "1/8": (8, True, True, False),
    "1/16": (16, True, True, True),
}


class StepperA4988Device(BaseDevice):
    """Stepper motor behind an A4988 driver, all six control pins on GPIO.

    Settings:
    - steps_per_rev: full steps per motor revolution (default 200)
    - steptype: default microstep mode, one of MICROSTEP_MODES (default "Full")
    - step_delay: seconds between STEP pulse edges (default 0.005,
        ~100 full steps/sec; one step = 2 edges)
    - auto_disable: de-energize after each move (default True — cool and
        quiet, but no holding torque between moves)
    """

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        s = config.settings
        self._steps_per_rev = int(s.get("steps_per_rev", 200))
        if self._steps_per_rev <= 0:
            raise ValueError(f"steps_per_rev must be positive, got {self._steps_per_rev}")
        self._steptype = self._validate_steptype(s.get("steptype", "Full"))
        self._step_delay = float(s.get("step_delay", 0.005))
        if self._step_delay < 0:
            raise ValueError(f"step_delay must be >= 0, got {self._step_delay}")
        self._auto_disable = bool(s.get("auto_disable", True))
        # pin name -> gpiozero DigitalOutputDevice. NOT named ``_pins``:
        # HardwareManager.add_device_multi_pin and the edit-device dialog
        # overwrite ``device._pins`` with a legacy list of pin ints, which
        # would clobber the live GPIO handles.
        self._gpio: dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._move_lock = asyncio.Lock()
        self._energized = False

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "StepperA4988"

    @property
    def required_pins(self) -> list[str]:
        return ["step", "dir", "enable", "ms1", "ms2", "ms3"]

    @property
    def steps_per_rev(self) -> int:
        """Full steps per motor revolution."""
        return self._steps_per_rev

    @property
    def steptype(self) -> str:
        """Default microstep mode."""
        return self._steptype

    @property
    def step_delay(self) -> float:
        """Seconds between STEP pulse edges."""
        return self._step_delay

    @property
    def auto_disable(self) -> bool:
        """Whether the driver is de-energized after each move."""
        return self._auto_disable

    @property
    def is_energized(self) -> bool:
        """Whether the driver outputs are currently energized."""
        return self._energized

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "move_steps": self.move_steps,
            "move_turns": self.move_turns,
            "stop": self.stop,
            "energize": self.energize,
            "de_energize": self.de_energize,
        }

    # --- validation ---

    @staticmethod
    def _validate_steptype(steptype: Any) -> str:
        if steptype not in MICROSTEP_MODES:
            raise ValueError(f"Unknown steptype {steptype!r}. Valid: {', '.join(MICROSTEP_MODES)}")
        # Membership in MICROSTEP_MODES guarantees this is one of its str keys.
        return str(steptype)

    # --- lifecycle (Task 2) ---

    async def initialize(self) -> None:
        """Claim the six GPIO pins via gpiozero and apply the default steptype."""

        def _claim():
            try:
                import gpiozero
            except ImportError as e:
                raise RuntimeError(
                    "gpiozero not installed. Run: pip install 'GLIDER[rpi]' "
                    "(or pip install gpiozero)"
                ) from e
            pins = {}
            for pin_name in self.required_pins:
                bcm = self._config.pins[pin_name]
                if pin_name == "enable":
                    # A4988 ENABLE is active-low; invert here so .on() always
                    # means "energized" everywhere else in this class.
                    pins[pin_name] = gpiozero.DigitalOutputDevice(bcm, active_high=False)
                else:
                    pins[pin_name] = gpiozero.DigitalOutputDevice(bcm)
            return pins

        self._gpio = await asyncio.to_thread(_claim)
        self._apply_steptype(self._steptype)
        self._energized = False  # gpiozero initial_value=False -> de-energized
        self._initialized = True
        logger.info("StepperA4988 initialized on pins %s", self._config.pins)

    async def shutdown(self) -> None:
        """E-stop safe state: clear the initialized flag, set the stop event,
        wait for any in-flight move (under the move lock), de-energize, then
        release the pins.

        The flag is cleared FIRST, before taking the lock: ``asyncio.Lock``
        wakes waiters FIFO, so a move queued behind the in-flight one acquires
        the lock before this method does — its under-lock initialized check
        must already see False so it raises instead of running (and clearing
        the e-stop's stop event). ``initialize()`` re-sets the flag at its
        end, so re-init after shutdown still works.

        The lock acquisition then bounds the wait to <= 2 x step_delay: the
        pulse loop exits at its next stop-event check, and holding the lock
        guarantees the worker thread is done before the pins close. No
        deadlock risk: ``stop()`` never takes the lock, and moves run in
        worker threads, not in this task.
        """
        self._initialized = False
        try:
            self._stop_event.set()
            async with self._move_lock:
                pins, self._gpio = self._gpio, {}
                if pins:

                    def _release():
                        enable = pins.get("enable")
                        if enable is not None:
                            enable.off()  # de-energize (active-low handled at claim)
                        for dev in pins.values():
                            try:
                                dev.close()
                            except Exception:  # close is best-effort
                                pass

                    await asyncio.to_thread(_release)
        finally:
            self._energized = False
            self._initialized = False

    def _apply_steptype(self, steptype: str) -> None:
        """Drive MS1-3 to select the microstep resolution."""
        _, ms1, ms2, ms3 = MICROSTEP_MODES[steptype]
        for pin_name, state in (("ms1", ms1), ("ms2", ms2), ("ms3", ms3)):
            (self._gpio[pin_name].on if state else self._gpio[pin_name].off)()

    def _set_energized(self, on: bool) -> None:
        """Drive the (inverted) ENABLE pin and track state."""
        (self._gpio["enable"].on if on else self._gpio["enable"].off)()
        self._energized = on

    # --- actions (Tasks 3-5) ---

    async def move_steps(self, steps: Any = None, steptype: Any = None) -> int:
        """Pulse ``abs(steps)`` steps; the sign selects DIR. Returns signed
        steps actually completed (a ``stop()`` mid-move returns fewer).

        ``steptype`` optionally overrides the device's default microstep mode
        for this move only.
        """
        if steps is None:
            raise ValueError("steps is required")
        steps = int(round(float(steps)))
        mode = self._validate_steptype(steptype) if steptype is not None else self._steptype
        async with self._move_lock:
            # Checked under the lock: a move queued behind an in-flight one
            # must see a shutdown() that ran in between, not enter _run_move
            # with the pins already released.
            if not self._initialized:
                raise RuntimeError("StepperA4988 not initialized")
            return await asyncio.to_thread(self._run_move, steps, mode)

    def _run_move(self, steps: int, steptype: str) -> int:
        """Blocking pulse loop; runs in a worker thread under the move lock."""
        # Clear the stop flag on entry so a move issued after a stop() runs
        # normally instead of exiting immediately.
        self._stop_event.clear()
        self._apply_steptype(steptype)
        if steps >= 0:
            self._gpio["dir"].on()
        else:
            self._gpio["dir"].off()
        self._set_energized(True)
        done = 0
        try:
            step_pin = self._gpio["step"]
            for _ in range(abs(steps)):
                if self._stop_event.is_set():
                    break
                step_pin.on()
                time.sleep(self._step_delay)
                step_pin.off()
                time.sleep(self._step_delay)
                done += 1
        finally:
            if self._auto_disable:
                try:
                    self._set_energized(False)
                except Exception:
                    # Belt-and-braces: shutdown() waits on the move lock, so it
                    # can no longer release the pins mid-move — but if the
                    # de-energize write fails anyway, the completed step count
                    # must still be returned.
                    pass
        return done if steps >= 0 else -done

    async def move_turns(self, turns: Any = None, steptype: Any = None) -> int:
        """Turn the shaft ``turns`` revolutions (signed) via ``move_steps``.

        Steps = turns x steps_per_rev x microstep factor, rounded to the
        nearest whole step.
        """
        if turns is None:
            raise ValueError("turns is required")
        mode = self._validate_steptype(steptype) if steptype is not None else self._steptype
        factor = MICROSTEP_MODES[mode][0]
        steps = round(float(turns) * self._steps_per_rev * factor)
        return await self.move_steps(steps, mode)

    async def stop(self) -> None:
        """Interrupt a move in progress; the pulse loop exits within one step.

        Deliberately does NOT take the move lock: the lock is held by the very
        move this is stopping, so acquiring it here would deadlock. Setting the
        event is atomic and safe from any task. Safe to call when idle or
        uninitialized (the next move clears the flag on entry).
        """
        self._stop_event.set()

    async def energize(self) -> None:
        """Energize the driver outputs (ENABLE low) without moving."""
        if not self._initialized:
            raise RuntimeError("StepperA4988 not initialized")
        await asyncio.to_thread(self._set_energized, True)

    async def de_energize(self) -> None:
        """De-energize the driver outputs (ENABLE high); no holding torque."""
        if not self._initialized:
            raise RuntimeError("StepperA4988 not initialized")
        await asyncio.to_thread(self._set_energized, False)

    # --- serialization ---

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "StepperA4988Device":
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance
