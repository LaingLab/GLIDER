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


def _manager(count: int = 4, *, available: int = 0):
    """A MultiCameraManager stand-in with *count* cameras already registered.

    add_camera actually populates ``cameras``, as the real one does. A bare
    MagicMock does not, which leaves camera_ids() empty and silently makes any
    test of the subscribe path vacuous.

    ``available`` is how many enumeration reports, for the case where the
    window has to register them itself.
    """
    manager = MagicMock()
    ids = [f"cam_{i}" for i in range(count)]
    manager.cameras = {i: MagicMock() for i in ids}
    manager.camera_count = count
    manager.primary_camera_id = ids[0] if ids else None
    manager.get_camera_fps.return_value = 30.0
    manager.start_all_streaming.return_value = dict.fromkeys(ids, True)
    manager.enumerate_all_cameras.return_value = [MagicMock(index=i) for i in range(available)]
    manager.camera_id_from_index.side_effect = lambda i: f"cam_{i}"

    def _add(camera_id, settings):
        manager.cameras[camera_id] = MagicMock()
        return True

    manager.add_camera.side_effect = _add
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


class TestCameraWiring:
    """The window must connect cameras itself, not assume someone else did.

    The first version only read manager.cameras and defined an on_frame that
    nothing called. Against a mock with cameras pre-populated every test
    passed; against real hardware the grid was empty until the camera panel
    happened to have been opened first.
    """

    def test_connect_adds_every_enumerated_camera(self, qtbot):
        manager = _manager(0, available=3)
        win = MultiCameraWindow(manager, recorder=_recorder(), parent=None)
        qtbot.addWidget(win)
        assert manager.add_camera.call_count == 3
        assert win.preview.camera_count == 3

    def test_connect_subscribes_to_frames(self, qtbot):
        manager = _manager(0, available=2)
        win = MultiCameraWindow(manager, recorder=_recorder(), parent=None)
        qtbot.addWidget(win)
        subscribed = {c.args[0] for c in manager.on_frame.call_args_list}
        assert subscribed == {"cam_0", "cam_1"}, "nothing would ever deliver a frame"

    def test_connect_starts_streaming(self, qtbot):
        manager = _manager(0, available=1)
        win = MultiCameraWindow(manager, recorder=_recorder(), parent=None)
        qtbot.addWidget(win)
        assert manager.start_all_streaming.called

    def test_subscribing_happens_once_per_camera(self, qtbot):
        # connect_cameras runs at construction and again on demand; a second
        # subscribe would deliver every frame twice.
        manager = _manager(0, available=2)
        win = MultiCameraWindow(manager, recorder=_recorder(), parent=None)
        qtbot.addWidget(win)
        before = manager.on_frame.call_count
        win.connect_cameras()
        assert manager.on_frame.call_count == before

    def test_it_does_not_re_add_cameras_the_panel_already_set_up(self, window):
        # The camera panel and this window share the core's manager. Adding a
        # camera twice would stack a second frame callback on it.
        window._manager.add_camera.reset_mock()
        window.connect_cameras()
        assert not window._manager.add_camera.called

    def test_a_camera_that_refuses_to_add_gets_no_tile(self, qtbot):
        manager = _manager(0, available=2)

        def _add(camera_id, settings):
            if camera_id == "cam_1":
                return False
            manager.cameras[camera_id] = MagicMock()
            return True

        manager.add_camera.side_effect = _add
        win = MultiCameraWindow(manager, recorder=_recorder(), parent=None)
        qtbot.addWidget(win)
        assert win.preview.camera_count == 1

    def test_a_frame_from_a_capture_thread_reaches_the_tile(self, window, qtbot):
        # The delivery path the real cameras use, signal marshalling included.
        frame = np.full((48, 64, 3), 200, np.uint8)
        window._on_camera_frame("cam_1", frame, 0.0)
        qtbot.wait(50)
        assert window.preview._tiles["cam_1"]._preview.pixmap() is not None
