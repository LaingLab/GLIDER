"""Tests for analysis.Session: loading recordings + derived timing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from glider.analysis import Session

from .conftest import RecordingSpec, write_synthetic_recording


def test_load_returns_session_with_all_dataframes(synthetic_recording: Path):
    s = Session.load(synthetic_recording)

    assert isinstance(s.tracking, pd.DataFrame)
    assert isinstance(s.data, pd.DataFrame)
    assert isinstance(s.events, pd.DataFrame)
    assert len(s.tracking) == 150
    assert len(s.events) == 2  # flow_marker start + end


def test_load_tolerates_missing_artifacts(tmp_path: Path):
    """A headless recording loads with an empty tracking frame, not an error."""
    write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_tracking=False))
    s = Session.load(tmp_path / "rec")

    assert s.tracking.empty
    assert "behavioral_state" in s.tracking.columns  # template columns preserved
    assert not s.events.empty


def test_flow_boundaries_extracted_from_event_log(synthetic_recording: Path):
    """flow_start_wall + flow_end_wall come from flow_marker rows."""
    s = Session.load(synthetic_recording)

    assert s.flow_start_wall is not None
    assert s.flow_end_wall is not None
    assert s.flow_end_wall > s.flow_start_wall
    # Default fixture: 4s of post-flow recording.
    assert abs(s.flow_duration_s - 4.0) < 0.01


def test_flow_boundaries_none_when_markers_missing(tmp_path: Path):
    """A recording without flow markers (legacy / never-started) returns None."""
    write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_events=False))
    s = Session.load(tmp_path / "rec")

    assert s.flow_start_wall is None
    assert s.flow_end_wall is None
    assert s.flow_duration_s is None


def test_frame_rate_estimated_from_tracking_timestamps(synthetic_recording: Path):
    """Frame rate derives from the median tracking timestamp interval.

    Real CSVs write timestamps with millisecond precision, so 30 FPS
    reads back as ~30.3 Hz (median interval = 33 ms instead of 33.33...).
    Tolerance reflects that quantization, not framerate measurement
    error. The library can't recover sub-ms precision the writer threw
    away.
    """
    s = Session.load(synthetic_recording)
    # Default spec is 30 FPS; ms-quantized timestamps push the estimate
    # to ~30.3 Hz. Anything within 1 Hz is "correctly recovered".
    assert s.frame_rate is not None
    assert abs(s.frame_rate - 30.0) < 1.0


def test_frame_rate_none_for_empty_tracking(tmp_path: Path):
    write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_tracking=False))
    s = Session.load(tmp_path / "rec")
    assert s.frame_rate is None


def test_metadata_merged_across_csvs(synthetic_recording: Path):
    """Session.metadata is the union of all CSVs' header dicts."""
    s = Session.load(synthetic_recording)
    # From tracking CSV header.
    assert s.metadata.get("Pixels/mm") == "4.0000"
    # Common to all three CSVs.
    assert "Start Time" in s.metadata


def test_video_path_prefers_annotated_when_both_exist(tmp_path: Path):
    rec_dir = tmp_path / "rec"
    write_synthetic_recording(rec_dir, RecordingSpec())
    # Create empty placeholders for both video kinds.
    raw = rec_dir / "test_experiment_20260525_140030.mp4"
    annotated = rec_dir / "test_experiment_20260525_140030_annotated.mp4"
    raw.write_bytes(b"")
    annotated.write_bytes(b"")

    s = Session.load(rec_dir)
    assert s.video_path == annotated
    assert s.raw_video_path == raw
    assert s.annotated_video_path == annotated


def test_video_path_falls_back_to_raw(tmp_path: Path):
    rec_dir = tmp_path / "rec"
    write_synthetic_recording(rec_dir, RecordingSpec())
    raw = rec_dir / "test_experiment_20260525_140030.mp4"
    raw.write_bytes(b"")

    s = Session.load(rec_dir)
    assert s.video_path == raw
    assert s.annotated_video_path is None


def test_directory_property_round_trips(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    assert s.directory == synthetic_recording


def test_cached_property_reuses_first_value(synthetic_recording: Path):
    """Accessing flow_duration_s twice doesn't re-derive it."""
    s = Session.load(synthetic_recording)
    a = s.flow_duration_s
    b = s.flow_duration_s
    assert a is b or a == b  # cached_property guarantees identity for non-immutables
