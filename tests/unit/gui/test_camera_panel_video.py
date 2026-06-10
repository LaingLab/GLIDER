"""Tests for the camera panel's video-tracking integration (worker + panel)."""

import pytest

pytest.importorskip("PyQt6")

from pathlib import Path

from glider.gui.panels.video_tracking_worker import VideoTrackingWorker
from glider.vision.cv_processor import MotionResult, TrackedObject
from glider.vision.video_tracking_runner import VideoTrackingConfig


class FakeCV:
    is_initialized = True

    def initialize(self):
        return True

    def process_frame(self, frame, timestamp):
        obj = TrackedObject(1, "subject", (0, 0, 8, 8), 0.9, (4, 4))
        return [], [obj], MotionResult(False, 0.0)


def test_worker_emits_finished(qtbot, synthetic_clip: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=None,
        write_zone_events=False,
        write_annotated=False,
    )
    worker = VideoTrackingWorker(cfg, cv_processor=FakeCV())
    with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
        worker.run()
    assert Path(blocker.args[0]) == out
