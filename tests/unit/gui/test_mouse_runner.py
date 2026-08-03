"""Tests for the running-mouse loading indicator.

The interesting properties are geometric, and they are the ones that were
actually wrong during development: feet passing through the wheel rim, the
foot teleporting at toe-off, and the knee whipping when the IK went
ill-conditioned. Those are all checkable without rendering anything.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QProgressBar  # noqa: E402

from glider.gui.widgets.mouse_runner import (  # noqa: E402
    _FORE,
    _HIND,
    GAITS,
    BusyIndicator,
    BusyProgress,
    MouseRunner,
    _ankle_at,
    _knee,
    _tail_points,
)

# The wheel's inner radius in mouse-local units, as _draw_wheel computes it.
RIM_LOCAL = 41.4


def rim_ground(x: float) -> float:
    """The wheel arc, as a height above the bottom contact point."""
    return -(RIM_LOCAL - math.sqrt(max(0.0, RIM_LOCAL**2 - x**2)))


def flat_ground(_x: float) -> float:
    return 0.0


# --- limb geometry ----------------------------------------------------


@pytest.mark.parametrize("limb", [_HIND, _FORE], ids=["hind", "fore"])
def test_ankle_is_continuous_across_the_whole_cycle(limb):
    """No teleports. Sampling densely, no step exceeds a small multiple of
    the average — a discontinuity at toe-off is what this catches."""
    stance = GAITS["bound"].stance
    n = 2000
    pts = [_ankle_at(limb, limb.x, i / n, stance, rim_ground)[0] for i in range(n)]

    steps = [math.hypot(b.x() - a.x(), b.y() - a.y()) for a, b in zip(pts, pts[1:], strict=False)]
    # Include the wrap from the last sample back to the first.
    steps.append(math.hypot(pts[0].x() - pts[-1].x(), pts[0].y() - pts[-1].y()))
    assert max(steps) < 8 * (sum(steps) / len(steps))


@pytest.mark.parametrize("limb", [_HIND, _FORE], ids=["hind", "fore"])
def test_foot_angle_is_continuous_across_the_whole_cycle(limb):
    """The toe-off jump was in the angle as well as the height."""
    stance = GAITS["bound"].stance
    n = 2000
    angs = [_ankle_at(limb, limb.x, i / n, stance, rim_ground)[1] for i in range(n)]
    deltas = [abs(b - a) for a, b in zip(angs, angs[1:], strict=False)]
    deltas.append(abs(angs[0] - angs[-1]))
    assert max(deltas) < 8 * (sum(deltas) / len(deltas))


@pytest.mark.parametrize("limb", [_HIND, _FORE], ids=["hind", "fore"])
def test_foot_never_passes_through_the_rim(limb):
    """Solved against the wheel arc, no part of the foot leaves the wheel."""
    stance = GAITS["bound"].stance
    centre_y = -RIM_LOCAL
    for i in range(1000):
        ankle, ang = _ankle_at(limb, limb.x, i / 1000, stance, rim_ground)
        toe_x = ankle.x() + math.cos(ang) * limb.lf
        toe_y = ankle.y() + math.sin(ang) * limb.lf
        for x, y in ((ankle.x(), ankle.y()), (toe_x, toe_y)):
            # The paw rests exactly on the arc in stance and is lifted
            # clear of it in swing, so nothing should cross at all.
            assert math.hypot(x, y - centre_y) <= RIM_LOCAL + 1e-6


@pytest.mark.parametrize("limb", [_HIND, _FORE], ids=["hind", "fore"])
def test_planted_foot_travels_backwards_only(limb):
    """During stance the foot is gripping the wheel, so it must not slide
    forwards relative to the body."""
    stance = GAITS["bound"].stance
    xs = [
        _ankle_at(limb, limb.x, (i / 500) * stance * 0.99, stance, rim_ground)[0].x()
        for i in range(500)
    ]
    assert all(b <= a + 1e-9 for a, b in zip(xs, xs[1:], strict=False))


def test_knee_lies_on_both_bones():
    """The IK solution has to actually respect the bone lengths."""
    knee = _knee(0.0, -11.5, -3.0, -2.0, _HIND.l1, _HIND.l2, _HIND.bend)
    assert math.hypot(knee.x() - 0.0, knee.y() - (-11.5)) == pytest.approx(_HIND.l1, abs=1e-6)
    assert math.hypot(knee.x() - (-3.0), knee.y() - (-2.0)) == pytest.approx(_HIND.l2, abs=1e-6)


def test_knee_stays_put_when_the_ankle_is_unreachable():
    """Beyond full extension the solve clamps rather than producing NaN."""
    knee = _knee(0.0, 0.0, 500.0, 0.0, _HIND.l1, _HIND.l2, _HIND.bend)
    assert math.isfinite(knee.x()) and math.isfinite(knee.y())


@pytest.mark.parametrize("gait", sorted(GAITS))
def test_knee_speed_stays_bounded_in_every_gait(gait):
    """The whip: bound and gallop drove the ankle close to the hip, where a
    tiny change in foot position swung the knee through a huge arc."""
    g = GAITS[gait]
    n = 1500
    knees = []
    for i in range(n):
        t = i / n
        fx = math.sin(2 * math.pi * (t + g.flex_ph))
        bob = -g.bob * math.sin(2 * math.pi * g.bob_f * t)
        hip_y = _HIND.y - g.flex * fx + bob
        ankle, _ = _ankle_at(_HIND, _HIND.x, t + g.hind[0], g.stance, rim_ground)
        dx, dy = ankle.x() - _HIND.x, ankle.y() - hip_y
        d = math.hypot(dx, dy)
        min_d = abs(_HIND.l2 - _HIND.l1) + (_HIND.l1 + _HIND.l2) * 0.10
        if 1e-4 < d < min_d:
            ankle_x = _HIND.x + (dx / d) * min_d
            ankle_y = hip_y + (dy / d) * min_d
        else:
            ankle_x, ankle_y = ankle.x(), ankle.y()
        knees.append(_knee(_HIND.x, hip_y, ankle_x, ankle_y, _HIND.l1, _HIND.l2, _HIND.bend))

    steps = [
        math.hypot(b.x() - a.x(), b.y() - a.y()) for a, b in zip(knees, knees[1:], strict=False)
    ]
    steps.append(math.hypot(knees[0].x() - knees[-1].x(), knees[0].y() - knees[-1].y()))
    assert max(steps) < 8 * (sum(steps) / len(steps))


# --- tail -------------------------------------------------------------


def test_tail_stays_inside_the_cage():
    """It is in a wheel; it cannot leave one."""
    from PyQt6.QtCore import QPointF

    centre_y = -RIM_LOCAL
    for i in range(200):
        pts = _tail_points(i / 200, QPointF(-19.5, -16), math.pi * 1.05, RIM_LOCAL)
        for p in pts:
            assert math.hypot(p.x(), p.y() - centre_y) <= RIM_LOCAL


def test_tail_base_moves_less_than_its_tip():
    """Amplitude ramps toward the tip so the tail trails rather than whips."""
    from PyQt6.QtCore import QPointF

    samples = [
        _tail_points(i / 60, QPointF(-19.5, -16), math.pi * 1.05, RIM_LOCAL) for i in range(60)
    ]

    def spread(joint: int) -> float:
        xs = [s[joint].x() for s in samples]
        ys = [s[joint].y() for s in samples]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    assert spread(2) < spread(len(samples[0]) - 1)


# --- widgets ----------------------------------------------------------


def test_rejects_an_unknown_gait(qtbot):
    with pytest.raises(ValueError, match="unknown gait"):
        MouseRunner(gait="moonwalk")


def test_runs_only_while_visible(qtbot):
    """A spinner in a hidden tab should not burn CPU."""
    w = MouseRunner()
    qtbot.addWidget(w)
    assert not w.is_animating()
    w.show()
    qtbot.waitExposed(w)
    assert w.is_animating()
    w.hide()
    assert not w.is_animating()


def test_advance_moves_the_animation(qtbot):
    w = MouseRunner()
    qtbot.addWidget(w)
    before = w._t
    w.advance(1.0)
    assert w._t > before


def test_paints_without_error_at_spinner_sizes(qtbot):
    """Exercises both the full wheel and the plain-rim path used when small."""
    from PyQt6.QtGui import QPixmap

    for size in (24, 32, 48, 96):
        w = MouseRunner()
        qtbot.addWidget(w)
        w.resize(size, size)
        w.grab()  # renders through paintEvent
        assert isinstance(w.grab(), QPixmap)


def test_busy_indicator_carries_its_caption(qtbot):
    w = BusyIndicator("Fitting…")
    qtbot.addWidget(w)
    assert w.text() == "Fitting…"
    w.setText("Nearly there…")
    assert w.text() == "Nearly there…"


def test_busy_progress_swaps_on_range(qtbot):
    """(0, 0) is the indeterminate signal these call sites use; a real total
    must hand back to the bar so the percentage is not thrown away."""
    w = BusyProgress()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)

    assert w.is_indeterminate()

    w.setRange(0, 250)
    w.setValue(125)
    assert not w.is_indeterminate()
    assert w.maximum() == 250
    assert w.value() == 125

    w.setRange(0, 0)
    assert w.is_indeterminate()


def test_busy_progress_covers_the_progress_bar_api_used(qtbot):
    """Guards the drop-in claim: every QProgressBar member the call sites
    touch has to exist here too."""
    w = BusyProgress()
    qtbot.addWidget(w)
    for name in ("setRange", "setValue", "value", "maximum", "setFormat", "setVisible"):
        assert hasattr(w, name), name
        assert hasattr(QProgressBar, name), name


def test_busy_progress_stops_animating_when_determinate(qtbot):
    w = BusyProgress()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    assert w.runner.is_animating()
    w.setRange(0, 10)
    assert not w.runner.is_animating()
