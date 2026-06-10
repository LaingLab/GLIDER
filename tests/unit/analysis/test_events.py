"""Tests for event log queries + event-triggered analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from glider.analysis import Session, event_triggered, find_events

from .conftest import RecordingSpec, write_synthetic_recording


def test_find_events_filters_by_source(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    markers = s.find_events(source="flow_marker")
    assert len(markers) == 2
    assert set(markers["value"]) == {"start", "end"}


def test_find_events_filters_by_value(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    starts = s.find_events(value="start")
    assert len(starts) == 1


def test_find_events_filters_by_both(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    end = s.find_events(source="flow_marker", value="end")
    assert len(end) == 1
    assert end.iloc[0]["value"] == "end"


def test_find_events_empty_for_no_match(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    none = s.find_events(source="output_write")
    assert none.empty


def test_events_with_flow_ms_adds_flow_column(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    e = s.events_with_flow_ms
    assert "flow_elapsed_ms" in e.columns
    # flow_marker start row is at flow time 0.
    start_row = e[(e["source"] == "flow_marker") & (e["value"] == "start")]
    assert abs(start_row["flow_elapsed_ms"].iloc[0]) < 1.0


def test_events_with_flow_ms_nan_when_no_flow_marker(tmp_path: Path):
    write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_events=False))
    s = Session.load(tmp_path / "rec")
    e = s.events_with_flow_ms
    # No events at all → empty frame with the column present.
    assert "flow_elapsed_ms" in e.columns
    assert e.empty


def test_event_triggered_extracts_per_trial_window(tmp_path: Path):
    """An LED-on event at flow t=1500ms should yield a window of tracking
    frames spanning [-1000, 5000] ms around it."""
    spec = RecordingSpec(
        extra_events=((1500.0, "output_write", "board0", "5", "1"),),
    )
    write_synthetic_recording(tmp_path / "rec", spec)
    s = Session.load(tmp_path / "rec")

    eta = s.event_triggered(
        source="output_write",
        value="1",
        window_ms=(-500.0, 1000.0),
        value_col="velocity_px_frame",
    )
    # All offsets within window.
    assert (eta["time_offset_ms"] >= -500.0).all()
    assert (eta["time_offset_ms"] <= 1000.0).all()
    assert (eta["trial_id"] == 0).all()
    assert (eta["event_time_ms"] == 1500.0).all()


def test_event_triggered_handles_multiple_trials(tmp_path: Path):
    spec = RecordingSpec(
        extra_events=(
            (1000.0, "output_write", "board0", "5", "1"),
            (2500.0, "output_write", "board0", "5", "1"),
        ),
    )
    write_synthetic_recording(tmp_path / "rec", spec)
    s = Session.load(tmp_path / "rec")

    eta = s.event_triggered(source="output_write", window_ms=(-100.0, 100.0))
    assert set(eta["trial_id"]) == {0, 1}


def test_event_triggered_empty_when_no_matching_events(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    eta = s.event_triggered(source="output_write", value="1")
    assert eta.empty


def test_event_triggered_direct_with_explicit_times():
    tracking = pd.DataFrame(
        {
            "object_id": [0] * 10,
            "flow_elapsed_ms": list(range(0, 1000, 100)),
            "velocity_px_frame": list(range(10)),
        }
    )
    eta = event_triggered(
        tracking,
        event_times_ms=[500.0],
        window_ms=(-200.0, 200.0),
        value_col="velocity_px_frame",
    )
    assert len(eta) > 0
    assert (eta["trial_id"] == 0).all()
    # offsets should bracket 0
    assert eta["time_offset_ms"].min() <= 0
    assert eta["time_offset_ms"].max() >= 0


def test_find_events_direct():
    df = pd.DataFrame(
        {
            "source": ["a", "b", "a"],
            "value": ["1", "2", "3"],
        }
    )
    out = find_events(df, source="a")
    assert len(out) == 2
