"""Timer, focus, naming and per-camera exposure in the recording window.

The exposure half is not decoration. Eight cameras on one rig had eight
different exposures and nothing said so: one ran auto-exposure out to its
one-second cap and stalled, another clipped 64% of its frame to white. Both
were found afterwards, from the files, when nothing could be done. The readout
here exists to put that number in front of someone while a dial can still be
turned.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from glider.gui.multi_camera.camera_names import CameraLabels
from glider.gui.multi_camera.camera_properties import (
    CLIPPED_BAD,
    CameraPropertiesPanel,
    exposure_verdict,
    frame_exposure,
)
from glider.gui.multi_camera.window import MultiCameraWindow
from glider.vision.camera_manager import CameraSettings
from glider.vision.multi_camera_manager import MultiCameraManager
from glider.vision.multi_video_recorder import MultiVideoRecorder


def _manager(count: int = 4):
    manager = MagicMock(spec=MultiCameraManager)
    ids = [f"cam_{i}" for i in range(count)]
    manager.cameras = {i: MagicMock() for i in ids}
    manager.camera_count = count
    manager.primary_camera_id = ids[0] if ids else None
    manager.get_camera_fps.return_value = 30.0
    manager.start_all_streaming.return_value = dict.fromkeys(ids, True)
    manager.enumerate_all_cameras.return_value = []
    manager.camera_id_from_index.side_effect = lambda i: f"cam_{i}"
    manager.get_camera_settings.side_effect = lambda cid: CameraSettings(
        camera_index=int(cid.split("_")[1])
    )
    # One stable mock per camera. Returning a fresh MagicMock each call would
    # mean the window and the assertion look at different objects, and every
    # "did it reach the camera" test would be vacuous.
    devices = {cid: MagicMock() for cid in ids}
    manager.get_camera.side_effect = devices.get
    manager.devices = devices
    return manager


def _recorder():
    rec = MagicMock(spec=MultiVideoRecorder)
    rec.is_recording = False
    rec.frames_dropped = {}
    rec.duration = 0.0
    return rec


@pytest.fixture
def window(qtbot, tmp_path):
    win = MultiCameraWindow(
        _manager(4),
        recorder=_recorder(),
        labels=CameraLabels(tmp_path / "labels.json"),
        parent=None,
    )
    qtbot.addWidget(win)
    return win


class TestElapsedTimer:
    def test_it_reads_blank_before_a_run(self, window):
        assert window.elapsed_label.text() == "--:--"

    def test_it_counts_while_recording(self, window):
        window._recorder.is_recording = True
        window._recorder.duration = 75.0
        window._refresh_elapsed()
        assert window.elapsed_label.text() == "01:15"

    def test_it_grows_an_hours_field_when_it_needs_one(self, window):
        assert window._format_elapsed(3675) == "1:01:15"
        assert window._format_elapsed(59) == "00:59"

    def test_it_holds_the_final_time_after_stopping(self, window):
        # The last thing anyone wants to know when a run ends is how long it
        # was, so stopping must not blank the clock.
        window._recorder.is_recording = True
        window._recorder.duration = 42.0
        window._refresh_elapsed()
        window._recorder.is_recording = False
        window._refresh_elapsed()
        assert window.elapsed_label.text() == "00:42"


class TestFocus:
    def test_the_grid_is_the_default(self, window):
        assert window.preview.focused_camera_id is None

    def test_clicking_the_selected_camera_focuses_it(self, window):
        window.select_camera("cam_2")
        window.select_camera("cam_2")
        assert window.preview.focused_camera_id == "cam_2"

    def test_clicking_it_again_returns_to_the_grid(self, window):
        window.select_camera("cam_2")
        window.select_camera("cam_2")
        window.select_camera("cam_2")
        assert window.preview.focused_camera_id is None

    def test_selecting_a_different_camera_does_not_rearrange_the_window(self, window):
        # Selecting to read a camera's exposure should not reflow everything.
        window.select_camera("cam_1")
        assert window.preview.focused_camera_id is None

    def test_every_camera_stays_on_screen_while_one_is_focused(self, window):
        # The point of focusing is to look closely at one arena without losing
        # sight of the others.
        window.preview.set_focus("cam_0")
        assert len(window.preview._tiles) == 4
        assert all(t.isVisibleTo(window.preview) for t in window.preview._tiles.values())

    def test_focusing_an_unknown_camera_is_ignored(self, window):
        window.preview.set_focus("cam_99")
        assert window.preview.focused_camera_id is None


class TestNaming:
    def test_a_name_reaches_the_tile(self, window):
        window._labels.set_label("cam_1", "arena B")
        window._refresh_names()
        assert window.preview.get_tile("cam_1")._camera_label.text() == "arena B"

    def test_names_are_handed_to_the_recorder_for_its_filenames(self, window):
        window._labels.set_label("cam_1", "arena B")
        window._refresh_names()
        handed = window._recorder.set_camera_labels.call_args.args[0]
        assert handed["cam_1"] == "arena_B"

    def test_an_unnamed_camera_shows_its_id(self, window):
        assert window.preview.get_tile("cam_2")._camera_label.text().endswith("2")


class TestProperties:
    def test_a_camera_is_selected_on_open(self, window):
        # An empty panel beside eight cameras is a panel nobody uses.
        assert window._selected_id is not None

    def test_selecting_loads_that_cameras_settings(self, window):
        window.select_camera("cam_3")
        assert window.properties._camera_id == "cam_3"

    def test_edits_reach_the_camera(self, window):
        window.select_camera("cam_1")
        window.properties._auto_exposure.setChecked(False)
        window.properties._exposure.setValue(-6)
        window.properties._emit_settings()
        camera = window._manager.get_camera("cam_1")
        assert camera.apply_settings.called

    def test_loading_a_camera_does_not_write_back_to_it(self, window):
        # Filling the form must not read as the operator editing it, or
        # selecting a camera would push its own values back at it.
        window.select_camera("cam_1")
        assert not window.properties._apply_timer.isActive()

    def test_a_camera_that_rejects_a_property_does_not_take_the_window_down(self, window):
        camera = MagicMock()
        camera.apply_settings.side_effect = RuntimeError("DirectShow said no")
        window._manager.get_camera.side_effect = lambda cid: camera
        window._apply_camera_settings("cam_1", CameraSettings())  # must not raise


class TestExposureReadout:
    def test_a_blown_out_frame_is_called_out(self):
        frame = np.full((480, 640), 255, np.uint8)
        mean, sd, clipped = frame_exposure(frame)
        assert clipped == pytest.approx(100.0)
        message, _ = exposure_verdict(mean, sd, clipped)
        assert "pure white" in message

    def test_a_well_exposed_frame_reads_as_fine(self):
        rng = np.random.default_rng(0)
        frame = np.clip(rng.normal(130, 40, (480, 640)), 0, 255).astype(np.uint8)
        message, _ = exposure_verdict(*frame_exposure(frame))
        assert "looks good" in message

    def test_a_flat_frame_is_called_out_even_though_nothing_clips(self):
        # Nothing is clipped, and an animal still will not separate from it.
        frame = np.full((480, 640), 120, np.uint8)
        message, _ = exposure_verdict(*frame_exposure(frame))
        assert "contrast" in message

    def test_a_very_dark_frame_names_the_frame_rate_cost(self):
        frame = np.clip(np.random.default_rng(1).normal(20, 18, (480, 640)), 0, 255).astype(
            np.uint8
        )
        message, _ = exposure_verdict(*frame_exposure(frame))
        assert "dark" in message or "contrast" in message

    def test_the_real_failure_would_have_been_flagged(self):
        # Test 7 of the eight-camera run: 64% of the frame at full white.
        frame = np.full((480, 640), 100, np.uint8)
        frame[: int(480 * 0.64)] = 255
        _, _, clipped = frame_exposure(frame)
        assert clipped >= CLIPPED_BAD
        message, _ = exposure_verdict(*frame_exposure(frame))
        assert "detail is being lost" in message

    def test_an_empty_frame_is_not_a_crash(self):
        assert frame_exposure(np.zeros((0, 0), np.uint8)) == (0.0, 0.0, 0.0)


class TestPanelStandalone:
    def test_it_is_disabled_with_no_camera(self, qtbot):
        panel = CameraPropertiesPanel()
        qtbot.addWidget(panel)
        assert not panel._exposure.isEnabled()

    def test_manual_exposure_enables_the_exposure_field(self, qtbot):
        panel = CameraPropertiesPanel()
        qtbot.addWidget(panel)
        panel.set_camera("cam_0", CameraSettings(auto_exposure=True), "arena A")
        assert not panel._exposure.isEnabled()
        panel._auto_exposure.setChecked(False)
        assert panel._exposure.isEnabled()

    def test_it_emits_a_copy_not_the_managers_own_settings(self, qtbot):
        # Mutating the manager's object in place would change what every other
        # camera sharing it records at.
        panel = CameraPropertiesPanel()
        qtbot.addWidget(panel)
        original = CameraSettings(camera_index=2, brightness=100)
        panel.set_camera("cam_2", original, "arena C")
        panel._brightness.setValue(200)
        panel._emit_settings()
        assert original.brightness == 100
