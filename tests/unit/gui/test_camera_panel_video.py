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


def _make_panel(qtbot):
    from glider.gui.panels.camera_panel import CameraPanel
    from glider.vision.camera_manager import CameraManager
    from glider.vision.cv_processor import CVProcessor

    panel = CameraPanel(CameraManager(), CVProcessor())
    qtbot.addWidget(panel)
    panel.show()
    return panel


def test_video_mode_toggles_controls(qtbot):
    panel = _make_panel(qtbot)

    panel._video_radio.setChecked(True)
    assert panel._video_mode is True
    assert panel._video_controls.isVisible() is True
    assert panel._camera_combo.isEnabled() is False
    assert panel._browse_btn.isEnabled() is True

    # Toggling back to Live restores the live-camera controls.
    panel._live_radio.setChecked(True)
    assert panel._video_mode is False
    assert panel._video_controls.isVisible() is False
    assert panel._camera_combo.isEnabled() is True


def test_seek_updates_frame_label_and_preview(qtbot, monkeypatch):
    import numpy as np

    panel = _make_panel(qtbot)
    # Stub the scrub source so no real video file is needed.
    monkeypatch.setattr(type(panel._video_source), "is_loaded", property(lambda self: True))
    monkeypatch.setattr(type(panel._video_source), "frame_count", property(lambda self: 12))
    monkeypatch.setattr(
        panel._video_source, "read_frame", lambda n: np.zeros((48, 64, 3), np.uint8)
    )

    panel._video_radio.setChecked(True)
    panel._seek_slider.setRange(0, 11)
    panel._on_seek(3)
    assert panel._frame_label.text() == "3 / 11"
    assert panel._video_current_frame == 3
