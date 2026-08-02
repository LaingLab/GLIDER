"""Tests for analysis.ethogram: behavioral_state → interval RLE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from glider.analysis import Session, compute_intervals

from .conftest import RecordingSpec, write_synthetic_recording


def test_intervals_collapse_consecutive_states(synthetic_recording: Path):
    """Default fixture has resting → active → resting; expect 3 intervals."""
    s = Session.load(synthetic_recording)
    intervals = s.ethogram()

    assert list(intervals["state"]) == ["resting", "active", "resting"]
    assert len(intervals) == 3


def test_interval_boundaries_match_state_schedule(synthetic_recording: Path):
    """The state-change boundaries in the fixture appear at known flow times."""
    s = Session.load(synthetic_recording)
    intervals = s.ethogram()

    # Default schedule: state transitions at 0ms, 2000ms, 3500ms.
    # Fixture is 30 FPS so frame boundaries align to ~33ms granularity.
    first = intervals.iloc[0]
    second = intervals.iloc[1]
    third = intervals.iloc[2]

    assert first["start_ms"] == 0.0
    assert abs(second["start_ms"] - 2000.0) < 50  # within one frame of 2s
    assert abs(third["start_ms"] - 3500.0) < 50


def test_intervals_have_positive_durations(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    intervals = s.ethogram()
    assert (intervals["duration_ms"] >= 0).all()
    # All but the last interval span more than one frame.
    assert (intervals["duration_ms"].iloc[:-1] > 30).all()


def test_pre_flow_frames_excluded_by_default(synthetic_recording: Path):
    """Frames with NaN flow_elapsed_ms (pre-flow) must not appear in intervals."""
    s = Session.load(synthetic_recording)
    intervals = s.ethogram()

    # Default fixture labels pre-flow frames as 'unknown'; if they leaked
    # in we'd see an extra interval at the start.
    assert "unknown" not in set(
        intervals["state"]
    ), "Pre-flow rows ('unknown') leaked into the ethogram"


def test_include_pre_flow_keeps_unknown_state(synthetic_recording: Path):
    """Explicit include_pre_flow=True surfaces the pre-flow setup rows."""
    s = Session.load(synthetic_recording)
    intervals = compute_intervals(s.tracking, include_pre_flow=True)
    assert "unknown" in set(intervals["state"])


def test_empty_tracking_returns_empty_intervals(tmp_path: Path):
    write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_tracking=False))
    s = Session.load(tmp_path / "rec")
    intervals = s.ethogram()
    assert intervals.empty
    # Schema is still present so callers can do .empty checks without KeyError.
    assert list(intervals.columns) == [
        "object_id",
        "state",
        "start_ms",
        "end_ms",
        "duration_ms",
    ]


def test_object_not_present_returns_empty(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    intervals = s.ethogram(object_id=999)
    assert intervals.empty


def test_handles_single_state_recording(tmp_path: Path):
    """A flow where the subject never changes state should produce one row."""
    spec = RecordingSpec(state_schedule=((0.0, "resting"),))
    write_synthetic_recording(tmp_path / "rec", spec)
    s = Session.load(tmp_path / "rec")
    intervals = s.ethogram()

    assert len(intervals) == 1
    assert intervals.iloc[0]["state"] == "resting"


def test_compute_intervals_directly_with_synthetic_frame():
    """Smoke test on a hand-crafted minimal DataFrame."""
    df = pd.DataFrame(
        {
            "object_id": [0, 0, 0, 0, 0],
            "behavioral_state": ["a", "a", "b", "b", "a"],
            "flow_elapsed_ms": [0.0, 100.0, 200.0, 300.0, 400.0],
        }
    )
    intervals = compute_intervals(df)
    assert list(intervals["state"]) == ["a", "b", "a"]
    assert list(intervals["start_ms"]) == [0.0, 200.0, 400.0]
    # Half-open [start, end): end is when the run ended, not the timestamp of
    # its last sample. The final run gets one median sampling period.
    assert list(intervals["end_ms"]) == [200.0, 400.0, 500.0]
    assert list(intervals["duration_ms"]) == [200.0, 200.0, 100.0]


class TestDurationsAccountForEveryFrame:
    """A run of n samples must be charged n sampling periods, not n-1.

    The old off-by-one lost one frame per bout: single-frame bouts reported
    0 ms, and across a 45,000-frame session with ~2,250 bouts the state
    fractions summed to ~0.95 instead of 1.0.
    """

    def _tracking(self, states, period=100.0):
        return pd.DataFrame(
            {
                "object_id": 0,
                "behavioral_state": states,
                "flow_elapsed_ms": [i * period for i in range(len(states))],
            }
        )

    def test_durations_sum_to_the_session_span(self):
        states = ["a", "a", "b", "c", "c", "c", "a"]
        intervals = compute_intervals(self._tracking(states))
        assert intervals["duration_ms"].sum() == pytest.approx(len(states) * 100.0)

    def test_a_single_sample_run_is_one_period_not_zero(self):
        intervals = compute_intervals(self._tracking(["a", "b", "a"]))
        assert list(intervals["duration_ms"]) == [100.0, 100.0, 100.0]

    def test_duration_always_equals_end_minus_start(self):
        intervals = compute_intervals(self._tracking(["a", "b", "b", "c"]))
        span = intervals["end_ms"] - intervals["start_ms"]
        assert list(span) == list(intervals["duration_ms"])

    def test_runs_are_contiguous_with_no_gaps(self):
        intervals = compute_intervals(self._tracking(["a", "b", "b", "c"]))
        assert list(intervals["start_ms"])[1:] == list(intervals["end_ms"])[:-1]

    def test_jittered_sampling_uses_the_median_period(self):
        """One long gap must not stretch the final run."""
        df = pd.DataFrame(
            {
                "object_id": 0,
                "behavioral_state": ["a", "a", "a", "b"],
                "flow_elapsed_ms": [0.0, 100.0, 900.0, 1000.0],
            }
        )
        intervals = compute_intervals(df)
        # median gap is 100 ms, so the final single-sample run gets 100 ms.
        assert intervals.iloc[-1]["duration_ms"] == pytest.approx(100.0)

    def test_one_sample_total_cannot_invent_a_period(self):
        intervals = compute_intervals(self._tracking(["a"]))
        assert intervals.iloc[0]["duration_ms"] == 0.0


def test_intervals_use_flow_elapsed_ms_by_default(synthetic_recording: Path):
    """Default time_col is flow_elapsed_ms; first interval starts at 0."""
    s = Session.load(synthetic_recording)
    intervals = s.ethogram()
    assert intervals.iloc[0]["start_ms"] == 0.0


class TestUnscoredIsNotABehavior:
    """A classifier's blank label is the absence of an answer.

    Counted as a state it answers "how long does grooming last" partly with
    "how often did tracking drop out", and it appears in the transition
    matrix as if the animal transitioned into nothing.
    """

    def _intervals(self, states):
        df = pd.DataFrame(
            {
                "object_id": 0,
                "behavioral_state": states,
                "flow_elapsed_ms": [i * 100.0 for i in range(len(states))],
            }
        )
        return compute_intervals(df)

    def test_scored_only_drops_the_blank_runs(self):
        from glider.analysis.ethogram import scored_only

        intervals = self._intervals(["a", "", "b"])
        assert list(scored_only(intervals)["state"]) == ["a", "b"]

    def test_a_gap_is_not_bridged(self):
        """Merging across a gap would assert the behavior continued through
        frames the classifier refused to label."""
        from glider.analysis.ethogram import scored_only

        intervals = self._intervals(["a", "a", "", "a", "a"])
        assert list(scored_only(intervals)["state"]) == ["a", "a"]

    def test_transitions_through_a_gap_are_dropped_not_invented(self):
        from glider.analysis.ethogram import compute_state_transitions

        intervals = self._intervals(["a", "", "b"])
        t = compute_state_transitions(intervals)
        assert t.empty  # a->b never observed; a->"" and ""->b are not behavior

    def test_real_transitions_still_counted(self):
        from glider.analysis.ethogram import compute_state_transitions

        intervals = self._intervals(["a", "b", "", "c", "a"])
        pairs = {
            (r.from_state, r.to_state) for r in compute_state_transitions(intervals).itertuples()
        }
        assert pairs == {("a", "b"), ("c", "a")}

    def test_hand_scored_ethograms_can_keep_every_pair(self):
        from glider.analysis.ethogram import compute_state_transitions

        intervals = self._intervals(["a", "", "b"])
        t = compute_state_transitions(intervals, unscored=None)
        assert len(t) == 2

    def test_scored_only_on_an_empty_frame_is_safe(self):
        from glider.analysis.ethogram import scored_only

        empty = compute_intervals(pd.DataFrame())
        assert scored_only(empty).empty
