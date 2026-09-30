"""Recorder lifecycle: finalize from PAUSED/ERROR, callbacks, fps, frame size."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from glider.vision.camera_manager import CameraSettings
from glider.vision.frame_writer import FrameWriterThread
from glider.vision.multi_camera_manager import MultiCameraManager
from glider.vision.multi_video_recorder import MultiVideoRecorder
from glider.vision.video_recorder import FrameClock, RecordingState, VideoRecorder

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _writer():
    w = MagicMock()
    w.isOpened.return_value = True
    return w


def _camera():
    cam = MagicMock()
    cam.settings = CameraSettings()
    cam.current_fps = 0.0
    return cam


async def _started(tmp_path):
    rec = VideoRecorder(_camera())
    rec.set_output_directory(tmp_path)
    writer = _writer()
    with patch("glider.vision.video_recorder.open_video_writer", return_value=(writer, "avc1")):
        await rec.start("t")
    return rec, writer


async def test_pause_then_stop_finalizes(tmp_path):
    rec, writer = await _started(tmp_path)
    await rec.pause()
    assert await rec.stop() is not None
    writer.release.assert_called_once()
    assert rec.state == RecordingState.IDLE


async def test_stop_from_error_releases_and_idle_stop_is_a_noop(tmp_path):
    rec, writer = await _started(tmp_path)
    rec._on_writer_error(OSError("disk full"))
    assert rec.state == RecordingState.ERROR
    await rec.stop()
    writer.release.assert_called_once()
    assert await rec.stop() is None


async def test_start_after_pause_finalizes_the_old_writer_first(tmp_path):
    rec, writer = await _started(tmp_path)
    await rec.pause()
    with patch("glider.vision.video_recorder.open_video_writer", return_value=(_writer(), "avc1")):
        await rec.start("t2")
    writer.release.assert_called_once()
    await rec.stop()


def test_frame_clock_excludes_paused_time():
    clock = FrameClock()
    for i in range(10):  # 27.5 fps
        clock.tick(100.0 + i / 27.5)
    clock.break_()  # a 60 s pause
    for i in range(10):
        clock.tick(200.0 + i / 27.5)
    assert abs(clock.fps - 27.5) < 1e-6


async def test_stop_corrects_fps_from_capture_timestamps(tmp_path):
    # 8% slow camera: under the old 10% threshold nothing was corrected.
    rec, _ = await _started(tmp_path)
    for i in range(30):
        rec._on_frame(FRAME, 100.0 + i / 27.5)
    with patch("glider.vision.video_recorder.fix_video_fps") as fix:
        await rec.stop()
    (path, fps, _codec), _ = fix.call_args
    assert path == rec.file_path
    assert abs(fps - 27.5) < 1e-6


def test_writer_rejects_frames_of_the_wrong_size():
    fwt = FrameWriterThread(MagicMock(), frame_size=(1280, 720))
    assert fwt.enqueue(FRAME) is False
    assert fwt.frames_dropped == 1


async def test_multi_recorder_reregisters_after_camera_is_re_added(tmp_path):
    manager = MultiCameraManager()
    settings = CameraSettings()

    def add():  # what add_camera does, minus a real device
        cam = MagicMock()
        cam.current_fps = 0.0
        manager._cameras["cam_0"] = cam
        manager._camera_settings["cam_0"] = settings
        manager._frame_callbacks["cam_0"] = []
        manager._primary_camera_id = "cam_0"

    rec = MultiVideoRecorder(manager)
    rec.set_output_directory(tmp_path)
    with patch(
        "glider.vision.multi_video_recorder.open_video_writer",
        side_effect=lambda *a, **k: (_writer(), "avc1"),
    ):
        add()
        await rec.start("a")
        await rec.stop()
        manager.remove_camera("cam_0")  # Stop Preview
        add()  # preview again: fresh, empty callback list
        await rec.start("b")
    assert manager._frame_callbacks["cam_0"] == [rec._on_frame]
    manager._on_camera_frame("cam_0", FRAME, 1.0)
    assert rec.get_frame_count("cam_0") == 1
    await rec.stop()
