# tests/unit/hal/test_revolution_tracking.py
from glider.hal import revolution_tracking as rt

_TWO_TURN_SWEEP = [0, 1000, 2000, 3000, 4000, 100, 1100, 2100, 3100, 4090, 200]


def _run_rev(values, **settings):
    settings.setdefault("counts_per_turn", 4096)
    settings.setdefault("turns_target", 1)
    settings.setdefault("land_tolerance", 0)
    st = rt.new_state()
    for v in values:
        if rt.revolution_triggered(v, settings, st):
            return v
    return None


def test_fires_after_one_turn():
    assert _run_rev(_TWO_TURN_SWEEP, turns_target=1) == 100


def test_waits_for_two_turns():
    assert _run_rev(_TWO_TURN_SWEEP, turns_target=2) == 200


def test_counts_either_direction():
    assert _run_rev([4000, 3000, 2000, 1000, 4090], turns_target=1) == 4090


def test_landing_tolerance_stops_before_the_wrap():
    # Motor climbs through mid-range (arming the gate) and stalls at 4060,
    # never reaching the 4095->0 wrap. With a 50-count landing tolerance it
    # still fires because 4096 - 4060 = 36 <= 50. Mirrors the node's
    # test_landing_tolerance_stops_before_the_wrap.
    assert _run_rev([2000, 3000, 4000, 4060], turns_target=1, land_tolerance=50) == 4060


def test_landing_tolerance_arm_gate_blocks_start_near_zero():
    # The sweep STARTS within tolerance of 0 (20, 30) but has not yet passed
    # mid-range, so the arm gate must suppress an instant trigger. It only
    # fires after climbing through mid-range (arming) and landing near the
    # wrap at 4070. Mirrors the node's
    # test_landing_tolerance_not_armed_when_starting_near_zero.
    assert _run_rev([20, 30, 2048, 4000, 4070], turns_target=1, land_tolerance=40) == 4070


def _run_counts(values, target):
    st = rt.new_state()
    settings = {"counts_target": target, "counts_per_turn": 4096}
    for v in values:
        if rt.counts_triggered(v, settings, st):
            return v
    return None


def test_counts_forward_and_reverse():
    assert _run_counts([0, 100, 200, 300, 400, 500], 400) == 400
    assert _run_counts([1000, 900, 800, 700, 600], 400) == 600


def test_ramp_monotonic_decrease_into_zone():
    settings = {"drive_pwm": 100, "creep_pwm": 30, "ramp_zone": 512}
    outs = [rt.ramp_pwm(r, settings) for r in (1000, 400, 200, 50, 0)]
    assert outs[0] == 100  # full drive outside the zone
    assert outs == sorted(outs, reverse=True)  # monotonic ramp down


def test_ramp_span_override_clamps_and_eases():
    # Covers the counts-mode span=min(ramp_zone, counts_target) path.
    settings = {"drive_pwm": 100, "creep_pwm": 30, "ramp_zone": 512}
    # remaining=150 is inside the default 512 zone, so it would already ease...
    assert rt.ramp_pwm(150, settings) < 100
    # ...but a small span (100) narrows the zone: remaining >= span clamps to
    # full drive_pwm, keeping the motor at speed until much closer to landing.
    assert rt.ramp_pwm(150, settings, span=100) == 100
    # Within the small span it eases monotonically from drive to creep.
    inside = [rt.ramp_pwm(r, settings, span=100) for r in (100, 60, 20, 0)]
    assert inside[0] == 100  # remaining == span -> still full drive
    assert inside == sorted(inside, reverse=True)  # monotonic ease down
    assert inside[-1] == 30  # remaining 0 -> creep_pwm
