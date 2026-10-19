"""Switching cameras off, and recording them one at a time, in the window.

A rig is rarely every camera the computer can see: the laptop's own webcam
enumerates beside the arena cameras, and a four-animal session on a
sixteen-camera rig has twelve cameras spending USB bandwidth on empty arenas.
And arenas are not always run in lockstep - one starts when its animal goes in,
the next ten minutes later.
"""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt

from glider.gui.multi_camera.camera_names import CameraLabels
from glider.gui.multi_camera.camera_selection import CameraSelection
from glider.gui.multi_camera.window import MultiCameraWindow, camera_sort_key
from glider.vision.camera_manager import CameraInfo, CameraSettings
from glider.vision.multi_camera_manager import MultiCameraManager
from glider.vision.multi_video_recorder import MultiVideoRecorder


def _manager(available=(0, 1, 2), connected=()):
    """A manager stand-in whose add/remove really change what it holds.

    ``available`` is what enumeration can find; ``connected`` is what is
    already registered when the window opens, as when the camera panel got
    there first.
    """
    manager = MagicMock(spec=MultiCameraManager)
    manager.available = list(available)
    manager.cameras = {f"cam_{i}": MagicMock() for i in connected}
    manager.primary_camera_id = f"cam_{connected[0]}" if connected else None
    manager.get_camera_fps.return_value = 30.0
    manager.camera_id_from_index.side_effect = lambda i: f"cam_{i}"
    manager.get_camera_settings.side_effect = lambda cid: (
        CameraSettings(camera_index=int(cid.split("_")[1])) if cid in manager.cameras else None
    )

    def _enumerate(max_cameras=16, skip_indices=()):
        return [
            CameraInfo(index=i, name=f"Camera {i}")
            for i in manager.available
            if i not in set(skip_indices)
        ]

    def _add(camera_id, settings):
        manager.cameras[camera_id] = MagicMock()
        if manager.primary_camera_id is None:
            manager.primary_camera_id = camera_id
        return True

    def _remove(camera_id):
        manager.cameras.pop(camera_id, None)
        if manager.primary_camera_id == camera_id:
            manager.primary_camera_id = next(iter(manager.cameras), None)

    manager.enumerate_all_cameras.side_effect = _enumerate
    manager.add_camera.side_effect = _add
    manager.remove_camera.side_effect = _remove
    return manager


def _recorder():
    rec = MagicMock(spec=MultiVideoRecorder)
    rec.is_recording = False
    rec.frames_dropped = {}
    rec.duration = 0.0
    rec.recording_camera_ids = []
    rec.saving_camera_ids = []
    rec.camera_duration.return_value = 0.0
    return rec


@pytest.fixture
def selection(tmp_path):
    return CameraSelection(tmp_path / "selection.json")


@pytest.fixture
def make_window(qtbot, tmp_path, selection):
    def _make(manager=None, recorder=None, **kwargs):
        win = MultiCameraWindow(
            manager if manager is not None else _manager(),
            recorder=recorder if recorder is not None else _recorder(),
            labels=CameraLabels(tmp_path / "labels.json"),
            selection=selection,
            parent=None,
            **kwargs,
        )
        qtbot.addWidget(win)
        return win

    return _make


def _row(window, camera_id: str) -> int:
    return window.status_table._rows[camera_id]


def _state(window, camera_id: str) -> str:
    headers = [
        window.status_table.horizontalHeaderItem(c).text()
        for c in range(window.status_table.columnCount())
    ]
    return window.status_table.item(_row(window, camera_id), headers.index("State")).text()


def _added(manager) -> list[str]:
    return [c.args[0] for c in manager.add_camera.call_args_list]


class TestSwitchingCamerasOff:
    def test_every_camera_starts_switched_on(self, make_window):
        manager = _manager()
        win = make_window(manager)
        assert _added(manager) == ["cam_0", "cam_1", "cam_2"]
        assert all(win.status_table.is_enabled(c) for c in ("cam_0", "cam_1", "cam_2"))

    def test_a_camera_switched_off_last_session_is_not_connected(self, make_window, selection):
        selection.set_enabled("cam_1", False)
        manager = _manager()
        win = make_window(manager)
        assert _added(manager) == ["cam_0", "cam_2"]
        assert win.preview.camera_count == 2

    def test_it_is_still_listed_so_it_can_be_switched_back_on(self, make_window, selection):
        selection.set_enabled("cam_1", False)
        win = make_window(_manager())
        assert win.status_table.rowCount() == 3
        assert not win.status_table.is_enabled("cam_1")
        assert _state(win, "cam_1") == "off"

    def test_switching_a_camera_off_disconnects_it(self, make_window, tmp_path):
        # Off means disconnected, not hidden: the bandwidth is usually why.
        manager = _manager()
        win = make_window(manager)

        win.set_camera_enabled("cam_1", False)

        manager.remove_camera.assert_called_once_with("cam_1")
        assert "cam_1" not in win.preview._tiles
        assert not win.status_table.is_enabled("cam_1")

    def test_it_is_remembered(self, make_window, tmp_path):
        win = make_window(_manager())
        win.set_camera_enabled("cam_1", False)
        assert not CameraSelection(tmp_path / "selection.json").is_enabled("cam_1")

    def test_unticking_its_box_switches_it_off(self, make_window, qtbot):
        manager = _manager()
        win = make_window(manager)
        item = win.status_table.item(_row(win, "cam_2"), 0)

        item.setCheckState(Qt.CheckState.Unchecked)

        qtbot.waitUntil(lambda: manager.remove_camera.called, timeout=1000)
        assert manager.remove_camera.call_args.args == ("cam_2",)

    def test_a_recording_camera_cannot_be_switched_off(self, make_window, selection):
        # Disconnecting it would cut its file off unfinished.
        manager = _manager()
        recorder = _recorder()
        recorder.recording_camera_ids = ["cam_1"]
        win = make_window(manager, recorder)

        win.set_camera_enabled("cam_1", False)

        assert not manager.remove_camera.called
        assert win.status_table.is_enabled("cam_1")
        assert selection.is_enabled("cam_1")


class TestSwitchingCamerasBackOn:
    def test_it_reconnects_at_the_configured_settings(self, make_window):
        configured = CameraSettings(camera_index=0, resolution=(1280, 720), fps=60)
        manager = _manager()
        win = make_window(manager, base_settings=configured)
        win.set_camera_enabled("cam_1", False)
        manager.add_camera.reset_mock()

        win.set_camera_enabled("cam_1", True)

        (camera_id, settings), _ = manager.add_camera.call_args
        assert camera_id == "cam_1"
        assert settings.camera_index == 1
        assert settings.resolution == (1280, 720) and settings.fps == 60
        assert "cam_1" in win.preview._tiles

    def test_it_is_subscribed_and_streaming_again(self, make_window):
        # The manager forgets a removed camera's callbacks entirely.
        manager = _manager()
        win = make_window(manager)
        win.set_camera_enabled("cam_1", False)
        manager.on_frame.reset_mock()

        win.set_camera_enabled("cam_1", True)

        assert [c.args[0] for c in manager.on_frame.call_args_list] == ["cam_1"]
        manager.start_streaming.assert_called_with("cam_1")

    def test_a_camera_that_will_not_connect_stays_unticked(self, make_window):
        manager = _manager()
        win = make_window(manager)
        win.set_camera_enabled("cam_1", False)
        manager.add_camera.side_effect = lambda cid, s: False

        win.set_camera_enabled("cam_1", True)

        assert not win.status_table.is_enabled("cam_1")
        assert "cam_1" not in win.preview._tiles


class TestFindingCameras:
    def test_streaming_cameras_are_not_probed(self, make_window):
        # A streaming camera can refuse a second open and read as absent.
        manager = _manager(available=(0, 1, 2), connected=(0, 1))
        make_window(manager)
        skipped = set(manager.enumerate_all_cameras.call_args.kwargs["skip_indices"])
        assert skipped == {0, 1}

    def test_a_camera_off_beside_ones_the_panel_connected_is_still_listed(
        self, make_window, selection
    ):
        selection.set_enabled("cam_2", False)
        manager = _manager(available=(0, 1, 2), connected=(0, 1))
        win = make_window(manager)
        assert win.all_camera_ids() == ["cam_0", "cam_1", "cam_2"]
        assert not manager.add_camera.called

    def test_refresh_finds_a_camera_plugged_in_later(self, make_window):
        manager = _manager(available=(0, 1))
        win = make_window(manager)
        manager.available = [0, 1, 2]

        win.rediscover_cameras()

        assert "cam_2" in win.preview._tiles
        assert win.status_table.is_enabled("cam_2")

    def test_refresh_leaves_a_switched_off_camera_off(self, make_window):
        manager = _manager()
        win = make_window(manager)
        win.set_camera_enabled("cam_1", False)
        manager.add_camera.reset_mock()

        win.rediscover_cameras()

        assert "cam_1" not in _added(manager)

    def test_reopening_does_not_probe_again(self, make_window):
        manager = _manager()
        win = make_window(manager)
        calls = manager.enumerate_all_cameras.call_count
        win.close()
        win.show()
        assert manager.enumerate_all_cameras.call_count == calls

    def test_cameras_are_listed_in_number_order(self, make_window):
        win = make_window(_manager(available=range(12)))
        assert win.all_camera_ids() == [f"cam_{i}" for i in range(12)]
        assert sorted(["cam_10", "cam_2", "cam_1"], key=camera_sort_key) == [
            "cam_1",
            "cam_2",
            "cam_10",
        ]


class TestRecordingOneCamera:
    def test_a_rows_button_records_that_camera_alone(self, make_window):
        win = make_window()
        win.name_edit.setText("trh day 3")

        win.status_table.record_button("cam_2").click()

        win._recorder.start_camera.assert_called_once_with("cam_2", "trh day 3")
        assert not win._recorder.start.called

    def test_it_stops_a_camera_that_is_recording(self, make_window):
        recorder = _recorder()
        recorder.is_recording = True
        recorder.recording_camera_ids = ["cam_2"]
        win = make_window(recorder=recorder)

        win.status_table.record_button("cam_2").click()

        recorder.stop_camera.assert_called_once_with("cam_2")
        assert not recorder.stop.called

    def test_its_row_is_live_again_the_moment_its_file_is_saved(self, make_window):
        # Found driving the window against virtual cameras: the row read
        # "Saving…", button disabled, until the next poll, so a quick click on
        # Record after a Stop was swallowed.
        recorder = _recorder()
        recorder.saving_camera_ids = ["cam_2"]
        win = make_window(recorder=recorder)
        win._poll_status()
        assert not win.status_table.record_button("cam_2").isEnabled()

        recorder.saving_camera_ids = []
        done = Future()
        done.set_result(None)
        win._timer.stop()
        win._recorder_task_done(done, "stop cam_2")

        button = win.status_table.record_button("cam_2")
        assert button.isEnabled() and button.text() == "Record"

    def test_a_camera_still_saving_is_not_restarted(self, make_window):
        recorder = _recorder()
        recorder.saving_camera_ids = ["cam_1"]
        win = make_window(recorder=recorder)

        win.toggle_camera_recording("cam_1")

        assert not recorder.start_camera.called

    def test_a_switched_off_camera_is_not_recorded(self, make_window):
        win = make_window()
        win.set_camera_enabled("cam_1", False)
        win.toggle_camera_recording("cam_1")
        assert not win._recorder.start_camera.called


class TestRecordAll:
    def test_from_idle_it_starts_one_run(self, make_window):
        win = make_window()
        win.record_button.click()
        assert win._recorder.start.call_count == 1
        assert not win._recorder.start_camera.called

    def test_with_some_recording_it_adds_the_rest(self, make_window):
        # Rather than doing nothing, or restarting takes already under way.
        recorder = _recorder()
        recorder.is_recording = True
        recorder.recording_camera_ids = ["cam_0"]
        win = make_window(recorder=recorder)

        win.record_button.click()

        started = [c.args[0] for c in recorder.start_camera.call_args_list]
        assert started == ["cam_1", "cam_2"]
        assert not recorder.start.called

    def test_it_leaves_switched_off_cameras_out(self, make_window, selection):
        selection.set_enabled("cam_1", False)
        recorder = _recorder()
        recorder.is_recording = True
        recorder.recording_camera_ids = ["cam_0"]
        win = make_window(recorder=recorder)

        win.record_button.click()

        assert [c.args[0] for c in recorder.start_camera.call_args_list] == ["cam_2"]

    def test_stop_all_is_live_while_any_camera_records(self, make_window):
        recorder = _recorder()
        recorder.recording_camera_ids = ["cam_1"]
        win = make_window(recorder=recorder)
        win._refresh_controls()
        assert win.stop_button.isEnabled()


class TestPerCameraStatus:
    def test_only_the_recording_camera_reads_as_recording(self, make_window):
        recorder = _recorder()
        recorder.is_recording = True
        recorder.recording_camera_ids = ["cam_1"]
        win = make_window(recorder=recorder)

        win._poll_status()

        assert _state(win, "cam_1") == "recording"
        assert _state(win, "cam_0") == "idle"

    def test_only_its_tile_shows_rec(self, make_window):
        recorder = _recorder()
        recorder.is_recording = True
        recorder.recording_camera_ids = ["cam_1"]
        win = make_window(recorder=recorder)

        win._poll_status()

        assert win.preview.get_tile("cam_1")._is_recording
        assert not win.preview.get_tile("cam_0")._is_recording

    def test_each_camera_shows_its_own_time(self, make_window):
        recorder = _recorder()
        recorder.is_recording = True
        recorder.recording_camera_ids = ["cam_1"]
        recorder.camera_duration.side_effect = lambda cid: 75.0 if cid == "cam_1" else 0.0
        win = make_window(recorder=recorder)

        win._poll_status()

        headers = [
            win.status_table.horizontalHeaderItem(c).text()
            for c in range(win.status_table.columnCount())
        ]
        time_col = headers.index("Time")
        assert win.status_table.item(_row(win, "cam_1"), time_col).text() == "01:15"
        assert win.status_table.item(_row(win, "cam_0"), time_col).text() == "—"

    def test_closing_while_one_camera_records_leaves_the_cameras_streaming(self, make_window):
        manager = _manager()
        recorder = _recorder()
        recorder.recording_camera_ids = ["cam_1"]
        win = make_window(manager, recorder)

        win.close()

        assert not manager.stop_all_streaming.called
