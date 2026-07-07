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
