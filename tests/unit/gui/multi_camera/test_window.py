"""The Multi-Camera Recording tool window.

Standalone rather than a panel checkbox because a sixteen-camera grid needs a
whole window - and often a second monitor - while the node graph keeps the
first.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from glider.gui.multi_camera.window import MultiCameraWindow


def _manager(count: int = 4):
    """A MultiCameraManager stand-in with *count* cameras."""
    manager = MagicMock()
    ids = [f"cam_{i}" for i in range(count)]
    manager.cameras = {i: MagicMock() for i in ids}
    manager.camera_count = count
    manager.primary_camera_id = ids[0] if ids else None
    manager.get_camera_fps.return_value = 30.0
    manager.start_all_streaming.return_value = dict.fromkeys(ids, True)
    manager.enumerate_all_cameras.return_value = []
    return manager


def _recorder():
    """A MultiVideoRecorder stand-in that is idle.

    is_recording has to be set explicitly: a bare MagicMock attribute is
    truthy, which would leave the window correctly believing a run is already
    in flight and disabling Record.
    """
    rec = MagicMock()
    rec.is_recording = False
    rec.frames_dropped = {}
    return rec


@pytest.fixture
def window(qtbot):
    win = MultiCameraWindow(_manager(4), recorder=_recorder(), parent=None)
    qtbot.addWidget(win)
    return win


class TestLayout:
    def test_it_lists_every_camera(self, window):
        assert window.status_table.rowCount() == 4

    def test_it_shows_a_tile_per_camera(self, window):
        assert window.preview.camera_count == 4

    def test_it_scales_to_sixteen(self, qtbot):
        win = MultiCameraWindow(_manager(16), recorder=_recorder(), parent=None)
        qtbot.addWidget(win)
        assert win.status_table.rowCount() == 16
        assert win.preview.camera_count == 16

    def test_it_opens_with_no_cameras(self, qtbot):
        win = MultiCameraWindow(_manager(0), recorder=_recorder(), parent=None)
        qtbot.addWidget(win)
        assert win.status_table.rowCount() == 0
        assert not win.record_button.isEnabled()


class TestRecording:
    def test_record_starts_every_camera(self, window):
        window.record_button.click()
        assert window._recorder.start_recording.called

    def test_stop_stops_recording(self, window):
        window._recorder.is_recording = True
        window._refresh_controls()
        window.stop_button.click()
        assert window._recorder.stop_recording.called

    def test_stop_is_disabled_until_recording(self, window):
        window._recorder.is_recording = False
        window._refresh_controls()
        assert not window.stop_button.isEnabled()
        assert window.record_button.isEnabled()

    def test_record_is_disabled_while_recording(self, window):
        window._recorder.is_recording = True
        window._refresh_controls()
        assert not window.record_button.isEnabled()
        assert window.stop_button.isEnabled()


class TestStatusPolling:
    def test_a_poll_fills_the_table(self, window):
        window._recorder.is_recording = True
        window._recorder.frames_dropped = {"cam_0": 0, "cam_1": 0, "cam_2": 0, "cam_3": 0}
        window._poll_status()
        assert window.status_table.item(0, 1).text() == "30.0"

    def test_dropped_frames_reach_the_table(self, window):
        window._recorder.is_recording = True
        window._recorder.frames_dropped = {"cam_0": 0, "cam_1": 7, "cam_2": 0, "cam_3": 0}
        window._poll_status()
        assert window.status_table.is_flagged("cam_1")
        assert not window.status_table.is_flagged("cam_0")

    def test_a_drop_surfaces_a_warning(self, window):
        window._recorder.is_recording = True
        window._recorder.frames_dropped = {"cam_0": 4}
        window._poll_status()
        assert "cam_0" in window.warning_label.text()

    def test_no_warning_when_healthy(self, window):
        window._recorder.is_recording = True
        window._recorder.frames_dropped = dict.fromkeys(["cam_0", "cam_1"], 0)
        window._poll_status()
        assert window.warning_label.text() == ""

    def test_polling_survives_a_recorder_that_has_gone_away(self, window):
        # A poll can land after teardown; it must not raise into the timer.
        window._recorder = None
        window._poll_status()


class TestFrames:
    def test_a_frame_reaches_its_tile(self, window):
        frame = np.full((48, 64, 3), 128, np.uint8)
        window.on_frame("cam_2", frame, 0.0)
        assert window.preview._tiles["cam_2"]._preview.pixmap() is not None

    def test_a_frame_for_an_unknown_camera_is_ignored(self, window):
        window.on_frame("cam_99", np.zeros((48, 64, 3), np.uint8), 0.0)
