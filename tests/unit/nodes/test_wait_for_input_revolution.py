"""
Tests for WaitForInputNode's "revolution" mode.

Revolution mode counts full turns of a wrap-around sensor such as the
AS5600 magnetic encoder, whose 12-bit raw angle sawtooths 0..4095 and
back. A poll-to-poll jump larger than half the full-scale range can only
be the 4095<->0 boundary being crossed, i.e. one completed turn. Either
direction counts, and the node fires after ``turns_target`` of them.
"""

from __future__ import annotations

import pytest

from glider.nodes.control_nodes import WaitForInputNode


class _ScriptedDevice:
    """Async-readable device that yields a fixed sequence of values.

    Once the sequence is exhausted it keeps returning the final value, so
    a poll loop that never triggers stalls on a steady reading (and is
    caught by the test's timeout) rather than erroring out.
    """

    def __init__(self, values: list[int]):
        self._values = list(values)
        self._i = 0
        self.id = "scripted"

    async def read(self) -> int:
        value = self._values[self._i]
        if self._i < len(self._values) - 1:
            self._i += 1
        return value


def _make_node(values: list[int], turns: int = 1, counts: int = 4096) -> WaitForInputNode:
    node = WaitForInputNode()
    node._threshold_mode = "revolution"
    node._turns_target = turns
    node._counts_per_turn = counts
    node._poll_interval = 0.0  # don't slow the test down
    node._device = _ScriptedDevice(values)
    node._waiting = True
    return node


# A forward sweep (0..4000), a wrap down to ~100 (= turn 1), another
# sweep, then a wrap down to ~200 (= turn 2).
_TWO_TURN_SWEEP = [0, 1000, 2000, 3000, 4000, 100, 1100, 2100, 3100, 4090, 200]


async def test_fires_after_one_completed_turn():
    node = _make_node(_TWO_TURN_SWEEP, turns=1)
    await node._poll_device(timeout=0.0)
    # First wrap is 4000 -> 100, so the trigger value is the post-wrap reading.
    assert node._trigger_value == 100


async def test_waits_for_target_turn_count():
    node = _make_node(_TWO_TURN_SWEEP, turns=2)
    await node._poll_device(timeout=0.0)
    # Should not fire on the first wrap; the second wrap is 4090 -> 200.
    assert node._trigger_value == 200


async def test_counts_either_direction():
    # Decreasing sweep then a jump UP across 0->4095 is also one turn.
    node = _make_node([4000, 3000, 2000, 1000, 4090], turns=1)
    await node._poll_device(timeout=0.0)
    assert node._trigger_value == 4090


async def test_no_phantom_trigger_on_first_read_or_small_steps():
    # Starts high and only ever takes small steps — never wraps, so the
    # node must keep waiting and ultimately time out (no phantom turn from
    # the very first reading having no predecessor).
    node = _make_node([4095, 4090, 4080, 4070], turns=1)
    with pytest.raises(TimeoutError):
        await node._poll_device(timeout=0.2)
    assert node._trigger_value is None


def test_revolution_state_round_trips():
    node = WaitForInputNode()
    node.set_state(
        {
            "threshold_mode": "revolution",
            "turns_target": 5,
            "counts_per_turn": 1024,
            "ramp_down": True,
            "ramp_device_id": "right_pwm",
            "drive_pwm": 120,
            "creep_pwm": 25,
            "ramp_zone": 300,
        }
    )
    assert node._threshold_mode == "revolution"
    assert node._turns_target == 5
    assert node._counts_per_turn == 1024
    assert node._ramp_down is True
    assert node._ramp_device_id == "right_pwm"
    assert node._drive_pwm == 120
    assert node._creep_pwm == 25
    assert node._ramp_zone == 300

    state = node.get_state()
    assert state["threshold_mode"] == "revolution"
    assert state["turns_target"] == 5
    assert state["counts_per_turn"] == 1024
    assert state["ramp_down"] is True
    assert state["ramp_device_id"] == "right_pwm"
    assert state["drive_pwm"] == 120
    assert state["creep_pwm"] == 25
    assert state["ramp_zone"] == 300


class _RecordingPWM:
    """Mock PWMOutput device that records every set_value() write."""

    device_type = "PWMOutput"

    def __init__(self):
        self.writes: list[int] = []

    async def set_value(self, value: int) -> None:
        self.writes.append(value)


async def test_ramp_down_decelerates_then_stops_at_landing():
    # Sweep approaching the wrap: full drive outside the 512-count zone, then
    # easing down as the angle nears 4096, then a wrap to 50 (= turn complete).
    values = [3000, 3600, 3800, 4000, 4090, 50]
    node = _make_node(values, turns=1)
    pwm = _RecordingPWM()
    node._ramp_down = True
    node._ramp_device = pwm
    node._drive_pwm = 100
    node._creep_pwm = 30
    node._ramp_zone = 512

    await node._poll_device(timeout=0.0)

    # Triggered on the wrap; value is the post-wrap reading.
    assert node._trigger_value == 50
    # Outside the deceleration zone the motor runs at full drive speed.
    assert pwm.writes[0] == 100
    # Through the zone the PWM only ever decreases (monotonic ramp down).
    ramp_writes = pwm.writes[:-1]  # exclude the final stop
    assert ramp_writes == sorted(ramp_writes, reverse=True)
    # It is near the creep speed just before the wrap, and stops at the landing.
    assert pwm.writes[-2] <= node._creep_pwm + 5
    assert pwm.writes[-1] == 0


async def test_ramp_down_decelerates_in_decreasing_direction():
    # Reverse rotation: the angle falls toward the 0 wrap. The ramp must slow
    # the motor as the angle approaches 0 (not 4096), then stop on the wrap.
    values = [3000, 2500, 2000, 1000, 500, 100, 4090]
    node = _make_node(values, turns=1)
    pwm = _RecordingPWM()
    node._ramp_down = True
    node._ramp_device = pwm
    node._drive_pwm = 100
    node._creep_pwm = 30
    node._ramp_zone = 512

    await node._poll_device(timeout=0.0)

    assert node._trigger_value == 4090  # wrapped 100 -> 4090
    assert node._ramp_direction == -1  # detected falling rotation
    assert pwm.writes[0] == 100  # full drive far from the 0 wrap
    ramp_writes = pwm.writes[:-1]
    assert ramp_writes == sorted(ramp_writes, reverse=True)  # eases down toward 0
    assert pwm.writes[-2] <= node._creep_pwm + 20  # near creep just before the wrap
    assert pwm.writes[-1] == 0


async def test_counts_mode_stops_at_target_forward():
    # Move-counts mode: accumulate displacement until |displacement| >= target.
    node = _make_node([0, 100, 200, 300, 400, 500], turns=1)
    node._threshold_mode = "counts"
    node._counts_target = 400
    await node._poll_device(timeout=0.0)
    # First reading where cumulative displacement reaches 400 is value 400.
    assert node._trigger_value == 400


async def test_counts_mode_stops_at_target_reverse():
    # Reverse motion (decreasing angle) accumulates negative displacement; the
    # magnitude is what counts, so it still stops after 400 counts of travel.
    node = _make_node([1000, 900, 800, 700, 600], turns=1)
    node._threshold_mode = "counts"
    node._counts_target = 400
    await node._poll_device(timeout=0.0)
    assert node._trigger_value == 600  # moved 1000 -> 600 = 400 counts


async def test_counts_mode_ramps_toward_target():
    node = _make_node([0, 1000, 2000, 3600, 3900, 4000], turns=1)
    node._threshold_mode = "counts"
    node._counts_target = 4000
    pwm = _RecordingPWM()
    node._ramp_down = True
    node._ramp_device = pwm
    node._drive_pwm = 100
    node._creep_pwm = 30
    node._ramp_zone = 512
    await node._poll_device(timeout=0.0)
    assert node._trigger_value == 4000  # 4000 counts of displacement reached
    assert pwm.writes[0] == 100  # full drive far from the target
    ramp_writes = pwm.writes[:-1]
    assert ramp_writes == sorted(ramp_writes, reverse=True)  # eases down to the target
    assert pwm.writes[-1] == 0


async def test_counts_mode_zone_larger_than_target_still_starts_at_drive():
    # ramp_zone (3500) exceeds the move target (300): the effective zone must
    # clamp to the move length so the motor starts at full drive_pwm instead
    # of opening partway down the ramp (which can stall short moves).
    node = _make_node([0, 50, 100, 200, 300], turns=1)
    node._threshold_mode = "counts"
    node._counts_target = 300
    pwm = _RecordingPWM()
    node._ramp_down = True
    node._ramp_device = pwm
    node._drive_pwm = 150
    node._creep_pwm = 24
    node._ramp_zone = 3500
    await node._poll_device(timeout=0.0)
    assert node._trigger_value == 300
    assert pwm.writes[0] == 150  # full drive at the start, not mid-ramp
    assert pwm.writes[-1] == 0


class _FailingDevice:
    """Device whose every read raises, like a flaky I2C encoder."""

    id = "flaky"

    async def read(self):
        raise OSError("I2C read failed")


class _RecordingHardwareManager:
    """Resolves one known device id to a recording PWM."""

    def __init__(self, device_id: str, device):
        self._device_id = device_id
        self._device = device

    def get_device(self, device_id):
        return self._device if device_id == self._device_id else None


def _ramping_counts_state(**overrides) -> dict:
    state = {
        "threshold_mode": "counts",
        "counts_target": 400,
        "ramp_down": True,
        "ramp_device_id": "left_pwm",
        "poll_interval": 0.0,
    }
    state.update(overrides)
    return state


async def test_polling_failure_stops_the_ramp_motor():
    # Repeated device read errors abort the wait with RuntimeError. Since the
    # error halts the exec chain, no downstream stop node ever runs — so the
    # node itself must de-energize the motor it was ramp-driving.
    node = WaitForInputNode()
    node._device = _FailingDevice()
    pwm = _RecordingPWM()
    node.set_hardware_manager(_RecordingHardwareManager("left_pwm", pwm))
    node.set_state(_ramping_counts_state())

    with pytest.raises(RuntimeError):
        await node.execute()
    assert pwm.writes[-1] == 0


async def test_timeout_stops_the_ramp_motor():
    # A wait that times out fires the "timeout" exec path; downstream stop
    # nodes hang off "triggered", so the node must stop the motor itself.
    node = WaitForInputNode()
    node._device = _ScriptedDevice([0, 10, 20])  # never reaches the target
    pwm = _RecordingPWM()
    node.set_hardware_manager(_RecordingHardwareManager("left_pwm", pwm))
    node.set_state(_ramping_counts_state(timeout=0.2))

    await node.execute()
    assert pwm.writes[-1] == 0


async def test_no_ramp_writes_when_ramp_device_absent():
    # Revolution mode without a ramp device must not attempt any PWM writes.
    node = _make_node([3000, 4090, 50], turns=1)
    node._ramp_down = True
    node._ramp_device = None  # not resolved -> ramp is skipped
    await node._poll_device(timeout=0.0)
    assert node._trigger_value == 50


async def test_landing_tolerance_stops_before_the_wrap():
    # Motor climbs toward the wrap and stalls at 4060 (never reaches 4095).
    # With a 50-count landing tolerance it should still stop, because 4060 is
    # within 50 of 0 (4096 - 4060 = 36 <= 50). This is the stall case from the
    # field log, where the old wrap-only logic ran forever.
    node = _make_node([2000, 3000, 4000, 4060], turns=1)
    node._land_tolerance = 50
    # Drive the loop a few iterations; it stalls at 4060 (repeats forever).
    await node._poll_device(timeout=1.0)
    assert node._trigger_value == 4060


async def test_landing_tolerance_not_armed_when_starting_near_zero():
    # If the angle starts within tolerance of 0 (e.g. 20) but has NOT yet
    # passed mid-range, the landing must not trigger immediately. Here it
    # climbs through the mid-range, arms, then lands near the wrap at 4070.
    node = _make_node([20, 30, 2048, 4000, 4070], turns=1)
    node._land_tolerance = 40
    await node._poll_device(timeout=1.0)
    assert node._trigger_value == 4070


async def test_landing_tolerance_off_uses_pure_wrap_detection():
    # land_tolerance = 0 -> behaves exactly like wrap-only revolution mode.
    node = _make_node([0, 1500, 3000, 4090, 60], turns=1)
    node._land_tolerance = 0
    await node._poll_device(timeout=0.0)
    assert node._trigger_value == 60  # only the wrap (4090 -> 60) fires
