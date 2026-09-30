"""Tracker output and lifecycle: no stale tracks, toggles and resets take effect."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from glider.vision.cv_processor import CVProcessor, CVSettings, Detection, ObjectTracker


def _det(x: int) -> Detection:
    return Detection(class_id=0, class_name="mouse", confidence=0.9, bbox=(x, 10, 10, 10))


def test_a_track_that_lost_its_detection_is_not_reported_live():
    tracker = ObjectTracker(max_disappeared=50)
    assert [t.track_id for t in tracker.update([_det(10)])] == [0]
    assert tracker.update([]) == []  # gone this frame: not logged at its last bbox
    # ...but the ID survives the occlusion.
    assert [t.track_id for t in tracker.update([_det(12)])] == [0]


def test_enabling_tracking_later_builds_a_tracker():
    proc = CVProcessor(CVSettings(tracking_enabled=False))
    assert proc._tracker is None
    settings = proc.settings.copy()
    settings.tracking_enabled = True
    proc.update_settings(settings)
    assert isinstance(proc._tracker, ObjectTracker)


def test_reset_clears_persisted_ultralytics_trackers():
    proc = CVProcessor(CVSettings())
    bytetrack = MagicMock()
    proc._yolo_model = SimpleNamespace(predictor=SimpleNamespace(trackers=[bytetrack]))
    proc.reset()
    bytetrack.reset.assert_called_once()


def test_skipped_frames_are_flagged_as_cached():
    proc = CVProcessor(CVSettings(process_every_n_frames=2))
    proc._initialized = True
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    proc.process_frame(frame, 0.0)
    assert proc.last_result_cached is False
    proc.process_frame(frame, 0.033)
    assert proc.last_result_cached is True
