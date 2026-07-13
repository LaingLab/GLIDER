"""Tests for VideoTrackingWorker — Qt signal adapter over VideoTrackingRunner."""

from pathlib import Path

import numpy as np

from glider.gui.panels.video_tracking_worker import VideoTrackingWorker
from glider.vision.video_tracking_runner import VideoTrackingConfig


class _FakeRunner:
    """Runner stub that drives the callbacks run() would fire, then returns."""

    def run(self, progress_cb=None, cancel_cb=None, frame_cb=None):
        if frame_cb is not None:
            frame_cb(np.zeros((4, 4, 3), dtype=np.uint8), 0)
            frame_cb(np.ones((4, 4, 3), dtype=np.uint8), 3)
        if progress_cb is not None:
            progress_cb(4, 4)
        return Path("out")


def test_worker_re_emits_preview_frames(qtbot, tmp_path):
    cfg = VideoTrackingConfig(source_path=tmp_path / "clip.avi", output_dir=tmp_path / "out")
    worker = VideoTrackingWorker(cfg)
    worker._runner = _FakeRunner()

    seen: list[int] = []
    worker.preview.connect(lambda frame, n: seen.append(n))
    finished: list[str] = []
    worker.finished.connect(finished.append)

    worker.run()

    assert seen == [0, 3]
    assert finished == ["out"]
