"""Pure wrap-around encoder math shared by the legacy node path and behaviors.

This module holds the turn-counting, landing-tolerance, counts-accumulation and
ramp-curve math that originally lived inline in
``glider.nodes.control_nodes.WaitForInputNode._poll_device``. The functions here
are pure (no I/O, no device writes): they operate on a caller-owned mutable
``state`` dict so both the legacy ``WaitForInputNode`` path and the ported input
behaviors can share exactly the same logic.

The math is a byte-for-byte extraction of the node implementation -- no constant
or comparison has been changed.
"""

from __future__ import annotations


def new_state() -> dict:
    """Return a fresh mutable tracking-state container.

    Keys:
        last_value: last reading seen (``None`` before the first read).
        turn_count: completed wrap-around turns (revolution mode).
        accumulated: signed wrap-corrected displacement from start (counts mode).
        landing_armed: whether the mid-range arm gate has been passed.
        ramp_direction: +1 rising toward the wrap, -1 falling toward 0.
    """
    return {
        "last_value": None,
        "turn_count": 0,
        "accumulated": 0.0,
        "landing_armed": False,
        "ramp_direction": 1,
    }


def shortest_delta(value: float, last: float, counts_per_turn: float) -> float:
    """Return the wrap-corrected shortest-path delta between two readings.

    A raw delta larger than half the full-scale range can only be the boundary
    being crossed, so it is folded back into the ``±half`` window. Sign gives the
    rotation direction (+1 rising toward the wrap, -1 falling toward 0).
    """
    step = value - last
    half = counts_per_turn / 2
    if step > half:
        step -= counts_per_turn
    elif step < -half:
        step += counts_per_turn
    return step


def revolution_triggered(value: float, settings: dict, st: dict) -> bool:
    """Update ``st`` for one revolution-mode reading; return whether to stop.

    Counts a wrap when the raw delta magnitude exceeds half the full-scale range
    (either direction), records the rotation direction, arms the landing gate
    once the angle has been seen mid-range, and -- on the final turn -- fires the
    landing tolerance the moment the angle lands within ``land_tolerance`` of 0.
    Returns True on turns-target reached OR landing-tolerance fired.

    Settings keys:
        turns_target: turns to complete before stopping (default 1).
        counts_per_turn: sensor full-scale range (default 4096).
        land_tolerance: stop within this many counts of 0; 0 = off (default 0).
    """
    counts_per_turn = settings.get("counts_per_turn", 4096)
    turns_target = settings.get("turns_target", 1)
    land_tolerance = settings.get("land_tolerance", 0)

    triggered = False
    last_value = st["last_value"]

    if last_value is not None:
        raw_delta = value - last_value
        if abs(raw_delta) > counts_per_turn / 2:
            st["turn_count"] += 1
            if st["turn_count"] >= turns_target:
                triggered = True

        # Signed shortest-path movement -> rotation direction, so the ramp knows
        # which boundary (0 or counts_per_turn) the motor is approaching.
        # +1 = angle rising toward the wrap, -1 = angle falling toward 0.
        step = shortest_delta(value, last_value, counts_per_turn)
        if step > 0:
            st["ramp_direction"] = 1
        elif step < 0:
            st["ramp_direction"] = -1

    # Arm tolerance landing once the angle has been seen in the mid-range, so
    # starting near 0 can't trigger it instantly.
    if 0.25 * counts_per_turn <= value <= 0.75 * counts_per_turn:
        st["landing_armed"] = True

    # Landing tolerance: on the final turn, stop the moment the angle is within
    # `land_tolerance` counts of 0 (either just before the wrap,
    # value >= counts - tol, or just after it, value <= tol).
    if (
        not triggered
        and land_tolerance > 0
        and st["landing_armed"]
        and st["turn_count"] >= turns_target - 1
        and (value <= land_tolerance or value >= counts_per_turn - land_tolerance)
    ):
        triggered = True

    st["last_value"] = value
    return triggered


def counts_triggered(value: float, settings: dict, st: dict) -> bool:
    """Update ``st`` for one counts-mode reading; return whether to stop.

    Accumulates wrap-corrected signed displacement from the start and stops once
    the magnitude reaches the target (less any ``land_tolerance`` to pre-empt
    coast). Bidirectional via the absolute displacement.

    Settings keys:
        counts_target: signed displacement to travel before stopping (default 400).
        counts_per_turn: sensor full-scale range (default 4096).
        land_tolerance: stop this many counts short of the target; 0 = off (default 0).
    """
    counts_per_turn = settings.get("counts_per_turn", 4096)
    counts_target = settings.get("counts_target", 400)
    land_tolerance = settings.get("land_tolerance", 0)

    last_value = st["last_value"]
    if last_value is not None:
        st["accumulated"] += shortest_delta(value, last_value, counts_per_turn)

    # Stop at the target, less any land_tolerance to pre-empt coast.
    stop_at = max(0, counts_target - land_tolerance)
    triggered = abs(st["accumulated"]) >= stop_at

    st["last_value"] = value
    return triggered


def ramp_pwm(remaining: float, settings: dict, span: float | None = None) -> int:
    """Return the decelerating PWM (0-255 int) for ``remaining`` counts to land.

    Outside the deceleration zone the motor runs at ``drive_pwm``; within the
    last ``ramp_zone`` (or ``span``) counts the PWM eases linearly down to
    ``creep_pwm``. ``span`` overrides the zone width (e.g. clamped to the move
    length for short counts-mode moves); a span larger than the move still starts
    at full drive because ``remaining`` is clamped to non-negative. Returns the
    PWM value that would be written rather than writing it.

    Settings keys:
        drive_pwm: full-speed PWM outside the zone (default 100).
        creep_pwm: PWM at the landing point (default 30).
        ramp_zone: deceleration zone width when ``span`` is None (default 512).
    """
    drive_pwm = settings.get("drive_pwm", 100)
    creep_pwm = settings.get("creep_pwm", 30)
    ramp_zone = settings.get("ramp_zone", 512)

    span = max(1, ramp_zone if span is None else span)
    remaining = max(0, remaining)
    if remaining >= span:
        pwm = drive_pwm
    else:
        frac = remaining / span  # 1.0 at zone entry -> 0.0 at the wrap
        pwm = creep_pwm + (drive_pwm - creep_pwm) * frac
    return max(0, min(255, int(round(pwm))))
