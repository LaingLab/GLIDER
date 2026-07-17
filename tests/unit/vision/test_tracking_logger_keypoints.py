"""TrackingDataLogger contract: live pose keypoints must reach disk.

Before this, keypoints made it as far as ``TrackedObject.keypoints`` and were
drawn on the live preview, then dropped — a pose session produced no keypoint
data at all. They now go to a sibling long-format ``*_keypoints.csv``, one row
per keypoint per object per frame, joinable to the tracking CSV on
``(frame, object_id)``.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import pytest

from glider.vision.cv_processor import TrackedObject
from glider.vision.tracking_logger import TrackingDataLogger


def _obj(track_id: int = 1, keypoints=None) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name="mouse",
        bbox=(10, 20, 30, 40),
        confidence=0.9,
        centroid=(25, 40),
        keypoints=None if keypoints is None else np.asarray(keypoints, dtype=float),
    )


def _read_keypoints(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def _run_session(tmp_path, objs, *, names=None, name="kp_test"):
    """Log one frame carrying ``objs`` and return the parsed keypoint rows."""
    tracker = TrackingDataLogger(output_dir=tmp_path)
    if names is not None:
        tracker.set_keypoint_names(names)
    await tracker.start(name)
    try:
        tracker.log_frame(time.time(), objs)
    finally:
        await tracker.stop()
    return tracker


@pytest.mark.asyncio
async def test_keypoints_written_to_sibling_file(tmp_path):
    """A pose object produces one row per keypoint, with names applied."""
    obj = _obj(keypoints=[[10.0, 20.0, 0.9], [30.0, 40.0, 0.8], [50.0, 60.0, 0.7]])
    tracker = await _run_session(tmp_path, [obj], names=["nose", "left_ear", "right_ear"])

    assert tracker.keypoints_file_path is not None
    assert tracker.keypoints_file_path.exists()
    rows = _read_keypoints(tracker.keypoints_file_path)

    assert [r["keypoint"] for r in rows] == ["nose", "left_ear", "right_ear"]
    assert [r["x"] for r in rows] == ["10.00", "30.00", "50.00"]
    assert [r["y"] for r in rows] == ["20.00", "40.00", "60.00"]
    assert [r["confidence"] for r in rows] == ["0.900", "0.800", "0.700"]
    # Joinable back to the tracking CSV.
    assert all(r["object_id"] == "1" for r in rows)
    assert all(r["frame"] == "1" for r in rows)
    assert tracker.keypoint_row_count == 3


@pytest.mark.asyncio
async def test_keypoints_file_sits_next_to_tracking_csv(tmp_path):
    obj = _obj(keypoints=[[1.0, 2.0, 0.5]])
    tracker = await _run_session(tmp_path, [obj], names=["nose"])

    assert tracker.keypoints_file_path.parent == tracker.file_path.parent
    assert tracker.file_path.name.endswith("_tracking.csv")
    assert tracker.keypoints_file_path.name.endswith("_keypoints.csv")


@pytest.mark.asyncio
async def test_no_keypoints_file_when_session_has_no_pose_data(tmp_path):
    """Background-subtraction sessions must not litter an empty file."""
    tracker = await _run_session(tmp_path, [_obj(keypoints=None)])

    assert tracker.keypoints_file_path is None
    assert list(tmp_path.glob("*_keypoints.csv")) == []


@pytest.mark.asyncio
async def test_tracking_csv_is_unchanged_by_pose_data(tmp_path):
    """The existing tracking CSV keeps its exact header — no new columns."""
    obj = _obj(keypoints=[[10.0, 20.0, 0.9]])
    tracker = await _run_session(tmp_path, [obj], names=["nose"])

    with open(tracker.file_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = next(r for r in rows if r and r[0] == "frame")
    assert header == [
        "frame",
        "timestamp",
        "elapsed_ms",
        "flow_elapsed_ms",
        "object_id",
        "class",
        "x",
        "y",
        "w",
        "h",
        "confidence",
        "center_x",
        "center_y",
        "distance_px",
        "distance_mm",
        "cumulative_mm",
        "zone_ids",
        "behavioral_state",
        "velocity_px_frame",
    ]


@pytest.mark.asyncio
async def test_unnamed_keypoints_fall_back_to_index(tmp_path):
    """No names supplied: data is still recorded, labelled positionally."""
    obj = _obj(keypoints=[[1.0, 2.0, 0.5], [3.0, 4.0, 0.6]])
    tracker = await _run_session(tmp_path, [obj], names=None)

    rows = _read_keypoints(tracker.keypoints_file_path)
    assert [r["keypoint"] for r in rows] == ["0", "1"]


@pytest.mark.asyncio
async def test_short_name_list_falls_back_per_keypoint(tmp_path):
    """A names list shorter than the model's output must not drop keypoints."""
    obj = _obj(keypoints=[[1.0, 2.0, 0.5], [3.0, 4.0, 0.6], [5.0, 6.0, 0.7]])
    tracker = await _run_session(tmp_path, [obj], names=["nose"])

    rows = _read_keypoints(tracker.keypoints_file_path)
    assert [r["keypoint"] for r in rows] == ["nose", "1", "2"]


@pytest.mark.asyncio
async def test_undetected_keypoint_recorded_as_empty_not_origin(tmp_path):
    """(0,0) placeholders must not be logged as a real coordinate.

    Pose models emit (0, 0) for undetected keypoints. Writing that verbatim
    would put a spurious point at the frame origin; the row is kept (so the
    frame x keypoint grid stays complete) with empty x/y instead.
    """
    obj = _obj(keypoints=[[0.0, 0.0, 0.0], [5.0, 6.0, 0.8]])
    tracker = await _run_session(tmp_path, [obj], names=["nose", "tail"])

    rows = _read_keypoints(tracker.keypoints_file_path)
    assert rows[0]["keypoint"] == "nose"
    assert rows[0]["x"] == ""
    assert rows[0]["y"] == ""
    assert rows[0]["confidence"] == "0.000"
    assert rows[1]["x"] == "5.00"


@pytest.mark.asyncio
async def test_low_confidence_keypoints_are_still_logged(tmp_path):
    """keypoint_min_confidence is a *rendering* setting, not a data filter.

    The renderer hides low-confidence points; the CSV must still record them
    so the analyst can choose their own threshold. Silently dropping data at
    a display setting is the class of bug this file exists to prevent.
    """
    obj = _obj(keypoints=[[1.0, 2.0, 0.01], [3.0, 4.0, 0.99]])
    tracker = await _run_session(tmp_path, [obj], names=["a", "b"])

    rows = _read_keypoints(tracker.keypoints_file_path)
    assert len(rows) == 2
    assert rows[0]["confidence"] == "0.010"


@pytest.mark.asyncio
async def test_keypoints_without_confidence_column(tmp_path):
    """Nx2 (x, y) models leave confidence blank rather than inventing one."""
    obj = _obj(keypoints=[[1.0, 2.0], [3.0, 4.0]])
    tracker = await _run_session(tmp_path, [obj], names=["a", "b"])

    rows = _read_keypoints(tracker.keypoints_file_path)
    assert [r["confidence"] for r in rows] == ["", ""]
    assert [r["x"] for r in rows] == ["1.00", "3.00"]


@pytest.mark.asyncio
async def test_multiple_objects_are_distinguished_by_object_id(tmp_path):
    a = _obj(track_id=7, keypoints=[[1.0, 2.0, 0.9]])
    b = _obj(track_id=9, keypoints=[[3.0, 4.0, 0.8]])
    tracker = await _run_session(tmp_path, [a, b], names=["nose"])

    rows = _read_keypoints(tracker.keypoints_file_path)
    assert [r["object_id"] for r in rows] == ["7", "9"]


@pytest.mark.asyncio
async def test_leading_instance_dim_is_collapsed(tmp_path):
    """Ultralytics hands back (1, K, 3); it must not be logged as one row."""
    obj = _obj(keypoints=[[[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]]])
    tracker = await _run_session(tmp_path, [obj], names=["a", "b"])

    rows = _read_keypoints(tracker.keypoints_file_path)
    assert [r["keypoint"] for r in rows] == ["a", "b"]


@pytest.mark.asyncio
async def test_keypoint_names_rejected_mid_recording(tmp_path):
    """Changing names mid-file would make the keypoint column inconsistent."""
    tracker = TrackingDataLogger(output_dir=tmp_path)
    tracker.set_keypoint_names(["nose"])
    await tracker.start("mid_recording")
    try:
        tracker.set_keypoint_names(["tail"])
        tracker.log_frame(time.time(), [_obj(keypoints=[[1.0, 2.0, 0.9]])])
    finally:
        await tracker.stop()

    rows = _read_keypoints(tracker.keypoints_file_path)
    assert rows[0]["keypoint"] == "nose"


@pytest.mark.asyncio
async def test_state_resets_between_sessions(tmp_path):
    """A second run must not inherit the first run's keypoint path/counts."""
    obj = _obj(keypoints=[[1.0, 2.0, 0.9]])
    first = TrackingDataLogger(output_dir=tmp_path)
    await first.start("run_one")
    first.log_frame(time.time(), [obj])
    await first.stop()
    first_path = first.keypoints_file_path
    assert first.keypoint_row_count == 1

    await first.start("run_two")
    try:
        assert first.keypoints_file_path is None
        assert first.keypoint_row_count == 0
        first.log_frame(time.time(), [_obj(keypoints=None)])
    finally:
        await first.stop()

    assert first.keypoints_file_path is None
    assert first_path.exists()
