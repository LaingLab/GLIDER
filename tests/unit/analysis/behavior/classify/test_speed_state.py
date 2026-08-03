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
# causal_speed_series: the whole trace, one value per frame
#
# The annotator's speed trace indexes by frame, so it needs every frame kept
# in place -- including frame 0 and the NaN dropouts that the threshold
# pooling drops. One definition of the signal, two consumers.
# ---------------------------------------------------------------------------


def test_causal_speed_series_is_one_value_per_frame():
    from glider.analysis.behavior.classify.speed_state import causal_speed_series

    frames = np.zeros((7, 2, 2))
    assert causal_speed_series(frames).shape == (7,)


def test_causal_speed_series_matches_pushing_frames_one_at_a_time():
    """The trace must BE the streamed signal, not an approximation of it."""
    from glider.analysis.behavior.classify.speed_state import causal_speed_series

    rng = np.random.default_rng(0)
    frames = rng.normal(size=(40, 3, 2)) * 5.0

    cs = CausalSpeed()
    expected = np.array([cs.push(f) for f in frames])

    np.testing.assert_allclose(causal_speed_series(frames), expected)


def test_causal_speed_series_keeps_frame_zero():
    """Frame 0 is 0.0 by construction and must stay at index 0, not be dropped."""
    from glider.analysis.behavior.classify.speed_state import causal_speed_series

    frames = np.array([[[0.0, 0.0]], [[3.0, 4.0]], [[6.0, 8.0]]])
    series = causal_speed_series(frames, coord_smooth=1, speed_smooth=1)
    assert series[0] == 0.0
    assert series[1] == 5.0


def test_causal_speed_series_keeps_dropouts_in_place():
    """A dropout must stay a NaN at its own index so the trace shows a gap."""
    from glider.analysis.behavior.classify.speed_state import causal_speed_series

    frames = np.array(
        [[[0.0, 0.0]], [[1.0, 0.0]], [[np.nan, np.nan]], [[3.0, 0.0]]],
    )
    series = causal_speed_series(frames, coord_smooth=1, speed_smooth=1)
    assert np.isnan(series[2])
    assert not np.isnan(series[1])
    assert series.size == 4


def test_causal_speed_series_handles_no_frames():
    from glider.analysis.behavior.classify.speed_state import causal_speed_series

    assert causal_speed_series(np.zeros((0, 3, 2))).shape == (0,)


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


class TestOfflineLabellingKeepsRunsWhole:
    """The online detector reports a run of n frames as n - min + 1.

    Unavoidable live — you cannot know a freeze began until it has lasted a
    second — and wrong for a recording, where every frame is known before
    anything is written. On a real 30-animal cohort it lost 119% of the
    freezing time: 470 s reported against 1031 s actually below threshold.
    """

    FT, DT = 1.0, 10.0

    def _trace(self, kind, n, pad=5):
        value = {"freeze": 0.1, "dart": 20.0}[kind]
        return [5.0] * pad + [value] * n + [5.0] * pad

    def _online(self, speeds, fmin=30, dmin=3):
        from glider.analysis.behavior.classify.speed_state import FreezeDartDetector

        detector = FreezeDartDetector(
            self.FT, self.DT, freeze_min_frames=fmin, dart_min_frames=dmin
        )
        return [detector.push(s) for s in speeds]

    def _offline(self, speeds, **kw):
        from glider.analysis.behavior.classify.speed_state import speed_axis_offline

        return speed_axis_offline(speeds, self.FT, self.DT, **kw)

    def test_a_qualifying_freeze_is_reported_in_full(self):
        labels = self._offline(self._trace("freeze", 45))
        assert labels.count("freezing") == 45

    def test_the_online_detector_reports_the_same_run_short(self):
        """Pins the discrepancy this function exists to remove."""
        speeds = self._trace("freeze", 45)
        assert self._online(speeds).count("freezing") == 45 - 30 + 1

    def test_a_run_below_the_minimum_is_still_excluded(self):
        """The minimum is a filter on which runs count, not a haircut."""
        assert self._offline(self._trace("freeze", 29)).count("freezing") == 0

    def test_exactly_the_minimum_qualifies_whole(self):
        assert self._offline(self._trace("freeze", 30)).count("freezing") == 30

    def test_darts_behave_the_same_way(self):
        assert self._offline(self._trace("dart", 3)).count("darting") == 3
        assert self._offline(self._trace("dart", 2)).count("darting") == 0

    def test_labels_land_on_the_frames_that_qualified(self):
        labels = self._offline(self._trace("freeze", 40))
        assert labels[4] == ""
        assert labels[5] == "freezing"  # the run's first frame, not its 30th
        assert labels[44] == "freezing"
        assert labels[45] == ""

    def test_a_dropout_breaks_a_run(self):
        speeds = [0.1] * 20 + [float("nan")] + [0.1] * 20
        # Neither half reaches 30 frames on its own.
        assert self._offline(speeds).count("freezing") == 0

    def test_a_dropout_does_not_end_the_session(self):
        speeds = [0.1] * 20 + [float("nan")] + [0.1] * 40
        assert self._offline(speeds).count("freezing") == 40

    def test_freezing_and_darting_cannot_both_hold(self):
        labels = self._offline(self._trace("freeze", 40) + self._trace("dart", 5))
        assert set(labels) <= {"", "freezing", "darting"}

    def test_dart_bursts_are_not_merged_by_default(self):
        """Merging changes what counts as one dart — a scoring decision."""
        speeds = [20.0] * 4 + [5.0] * 6 + [20.0] * 4
        labels = self._offline(speeds)
        assert labels[4:10] == [""] * 6

    def test_merging_joins_bursts_within_the_gap(self):
        speeds = [20.0] * 4 + [5.0] * 6 + [20.0] * 4
        labels = self._offline(speeds, dart_merge_gap=24)
        assert labels.count("darting") == 14  # one dart, gap included

    def test_merging_leaves_distant_bursts_apart(self):
        speeds = [20.0] * 4 + [5.0] * 30 + [20.0] * 4
        labels = self._offline(speeds, dart_merge_gap=24)
        assert labels.count("darting") == 8

    def test_an_empty_trace_is_empty(self):
        assert self._offline([]) == []

    def test_both_paths_agree_on_which_runs_qualify(self):
        """Only the durations differ; a bout either happened or it did not."""
        import numpy as np

        rng = np.random.default_rng(0)
        speeds = rng.choice([0.1, 5.0, 20.0], size=2000, p=[0.3, 0.5, 0.2]).tolist()
        online = self._online(speeds)
        offline = self._offline(speeds)
        for state in ("freezing", "darting"):
            assert (state in online) == (state in offline)
