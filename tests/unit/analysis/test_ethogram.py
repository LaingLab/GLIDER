"""Tests for analysis.ethogram: behavioral_state → interval RLE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    assert list(intervals["end_ms"]) == [100.0, 300.0, 400.0]


def test_intervals_use_flow_elapsed_ms_by_default(synthetic_recording: Path):
    """Default time_col is flow_elapsed_ms; first interval starts at 0."""
    s = Session.load(synthetic_recording)
    intervals = s.ethogram()
    assert intervals.iloc[0]["start_ms"] == 0.0
