"""Tests for the live speed-axis detector (src/glider/analysis/behavior/classify/speed_state.py)."""

from __future__ import annotations

import numpy as np

from glider.analysis.behavior.classify.speed_state import (
    CausalSpeed,
    FreezeDartDetector,
    calibrate_speed_thresholds,
)

# ---------------------------------------------------------------------------
# FreezeDartDetector: online duration-enforced state machine
# ---------------------------------------------------------------------------


def test_freezing_confirmed_only_after_min_frames():
    d = FreezeDartDetector(
        freeze_threshold=1.0, dart_threshold=10.0, freeze_min_frames=30, dart_min_frames=3
    )
    labels = [d.push(0.5) for _ in range(35)]
    assert labels[:29] == [""] * 29  # not yet confirmed
    assert all(lbl == "freezing" for lbl in labels[29:])  # 30th frame onward


def test_darting_confirmed_after_min_frames():
    d = FreezeDartDetector(1.0, 10.0, freeze_min_frames=30, dart_min_frames=3)
    labels = [d.push(50.0) for _ in range(5)]
    assert labels == ["", "", "darting", "darting", "darting"]


def test_speed_between_thresholds_is_none():
    d = FreezeDartDetector(1.0, 10.0, 30, 3)
    assert all(d.push(5.0) == "" for _ in range(10))


def test_short_run_then_break_resets_then_confirms():
    d = FreezeDartDetector(1.0, 10.0, freeze_min_frames=30, dart_min_frames=3)
    seq = [0.5] * 20 + [5.0] + [0.5] * 30  # short, break, full freeze
    labels = [d.push(s) for s in seq]
    assert labels[:50] == [""] * 50  # 20 + break + first 29
    assert labels[50] == "freezing"  # 30th frame of the 2nd run


def test_nan_speed_breaks_run():
    d = FreezeDartDetector(1.0, 10.0, freeze_min_frames=5, dart_min_frames=3)
    seq = [0.5] * 4 + [float("nan")] + [0.5] * 5
    labels = [d.push(s) for s in seq]
    assert labels[:9] == [""] * 9  # 4 + nan + first 4
    assert labels[9] == "freezing"  # 5th frame of the post-nan run


def test_freeze_and_dart_are_mutually_exclusive():
    d = FreezeDartDetector(1.0, 10.0, 3, 3)
    assert "freezing" not in [d.push(50.0) for _ in range(5)]


# ---------------------------------------------------------------------------
# CausalSpeed: streaming causal mean-keypoint speed
# ---------------------------------------------------------------------------


def test_causal_speed_first_frame_is_zero():
    cs = CausalSpeed(coord_smooth=1, speed_smooth=1)
    assert cs.push(np.array([[0.0, 0.0]])) == 0.0


def test_causal_speed_constant_motion():
    cs = CausalSpeed(coord_smooth=1, speed_smooth=1)
    cs.push(np.array([[0.0, 0.0]]))
    assert cs.push(np.array([[3.0, 4.0]])) == 5.0  # 3-4-5 triangle
    assert cs.push(np.array([[6.0, 8.0]])) == 5.0


def test_causal_speed_means_over_keypoints():
    cs = CausalSpeed(coord_smooth=1, speed_smooth=1)
    cs.push(np.array([[0.0, 0.0], [0.0, 0.0]]))
    # kp0 moves 5, kp1 moves 0 -> mean 2.5
    assert cs.push(np.array([[3.0, 4.0], [0.0, 0.0]])) == 2.5


def test_causal_speed_median_suppresses_single_teleport():
    cs = CausalSpeed(coord_smooth=3, speed_smooth=1)
    pts = [
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[100.0, 0.0]]),
        np.array([[0.0, 0.0]]),
    ]
    speeds = [cs.push(p) for p in pts]
    assert max(speeds) < 50.0  # raw teleport would be ~100; median rejects it


def test_causal_speed_full_dropout_is_nan():
    cs = CausalSpeed(coord_smooth=1, speed_smooth=1)
    cs.push(np.array([[0.0, 0.0]]))
    s = cs.push(np.array([[np.nan, np.nan]]))
    assert np.isnan(s)


# ---------------------------------------------------------------------------
# calibrate_speed_thresholds: once-per-rig absolute thresholds
# ---------------------------------------------------------------------------


def test_calibrate_thresholds_are_speed_percentiles():
    # 1 keypoint walking +1,+2,+3,+4 px → causal speeds [1,2,3,4] (frame 0 dropped)
    xs = [0.0, 1.0, 3.0, 6.0, 10.0]
    frames = [np.array([[x, 0.0]]) for x in xs]
    fz, dt = calibrate_speed_thresholds(
        frames, freeze_pct=25, dart_pct=75, coord_smooth=1, speed_smooth=1
    )
    assert fz == np.percentile([1.0, 2.0, 3.0, 4.0], 25)
    assert dt == np.percentile([1.0, 2.0, 3.0, 4.0], 75)
    assert fz < dt


class TestDropoutAtTheStart:
    """A dropped opening frame must not blank the whole session's speed axis.

    Found on a real 45,000-frame session whose frame 0 was undetected: every
    subsequent speed came back NaN, so freeze/dart was silently empty for that
    animal while the other 29 in the cohort looked fine.
    """

    def _walker(self, n=40, step=3.0):
        return [np.full((5, 2), i * step) for i in range(n)]

    def test_an_all_nan_first_frame_still_yields_speeds(self):
        from glider.analysis.behavior.classify.speed_state import CausalSpeed

        frames = self._walker()
        frames[0] = np.full((5, 2), np.nan)
        causal = CausalSpeed()
        speeds = [causal.push(f) for f in frames]
        assert np.isnan(speeds[0])
        finite = [s for s in speeds[1:] if not np.isnan(s)]
        assert len(finite) > len(frames) // 2
        assert max(finite) > 0

    def test_a_run_of_dropped_opening_frames_recovers(self):
        from glider.analysis.behavior.classify.speed_state import CausalSpeed

        frames = self._walker()
        for i in range(6):
            frames[i] = np.full((5, 2), np.nan)
        causal = CausalSpeed()
        speeds = [causal.push(f) for f in frames]
        assert all(np.isnan(s) for s in speeds[:6])
        assert any(not np.isnan(s) for s in speeds[6:])

    def test_a_clean_first_frame_is_unchanged(self):
        from glider.analysis.behavior.classify.speed_state import CausalSpeed

        assert CausalSpeed().push(np.zeros((5, 2))) == 0.0

    def test_dropout_in_the_middle_still_only_gaps(self):
        from glider.analysis.behavior.classify.speed_state import CausalSpeed

        frames = self._walker()
        frames[20] = np.full((5, 2), np.nan)
        causal = CausalSpeed()
        speeds = [causal.push(f) for f in frames]
        assert not np.isnan(speeds[-1])
