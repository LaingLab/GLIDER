"""Recording cameras one at a time.

The recorder used to know one verb for a camera array: all of them, together.
A rig is often run otherwise - one arena started when its animal goes in, the
next ten minutes later, each stopped when its trial ends - and every one of
those has to leave the others' files alone.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from glider.vision.camera_manager import CameraSettings
from glider.vision.multi_camera_manager import MultiCameraManager
from glider.vision.multi_video_recorder import MultiVideoRecorder, RecordingState

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _manager(*camera_ids: str) -> MultiCameraManager:
    """A real manager holding fake cameras: what add_camera does, minus a device."""
    manager = MultiCameraManager()
    for camera_id in camera_ids:
        camera = MagicMock()
        camera.current_fps = 0.0
        manager._cameras[camera_id] = camera
        manager._camera_settings[camera_id] = CameraSettings(
            camera_index=int(camera_id.split("_")[1])
        )
        manager._frame_callbacks[camera_id] = []
        if manager._primary_camera_id is None:
            manager._primary_camera_id = camera_id
    return manager


@pytest.fixture
def writers():
    """Every writer the recorder opens, by the path it was opened for."""
    opened: dict[Path, MagicMock] = {}

    def _open(path, codec, fps, resolution):
        writer = MagicMock()
        opened[Path(path)] = writer
        return writer, codec

    with patch("glider.vision.multi_video_recorder.open_video_writer", side_effect=_open):
        yield opened


@pytest.fixture
def manager():
    return _manager("cam_0", "cam_1", "cam_2")


@pytest.fixture
def recorder(manager, tmp_path, writers):
    rec = MultiVideoRecorder(manager)
    rec.set_output_directory(tmp_path)
    yield rec


class TestStartingOneCamera:
    async def test_one_camera_records_on_its_own(self, recorder, manager):
        path = await recorder.start_camera("cam_1", "solo")
        assert recorder.is_recording
        assert recorder.recording_camera_ids == ["cam_1"]
        assert list(recorder.file_paths) == ["cam_1"]
        assert path == recorder.file_paths["cam_1"]

        manager._on_camera_frame("cam_0", FRAME, 1.0)
        manager._on_camera_frame("cam_1", FRAME, 1.0)
        assert recorder.get_frame_count("cam_1") == 1
        assert recorder.get_frame_count("cam_0") == 0
        await recorder.stop()

    async def test_a_camera_can_join_a_run_already_going(self, recorder, writers):
        await recorder.start_camera("cam_0", "run")
        first = recorder.file_paths["cam_0"]

        await recorder.start_camera("cam_2", "run")

        assert recorder.recording_camera_ids == ["cam_0", "cam_2"]
        assert recorder.file_paths["cam_0"] == first, "joining must not reopen the others"
        assert not writers[first].release.called
        await recorder.stop()

    async def test_starting_a_camera_twice_keeps_its_file(self, recorder):
        first = await recorder.start_camera("cam_1", "run")
        assert await recorder.start_camera("cam_1", "run") == first
        await recorder.stop()

    async def test_an_unknown_camera_is_refused(self, recorder):
        assert await recorder.start_camera("cam_7", "run") is None
        assert recorder.state == RecordingState.IDLE

    async def test_nothing_starts_after_a_writer_error(self, recorder):
        await recorder.start_camera("cam_0", "run")
        recorder._on_writer_error(OSError("disk full"))
        assert await recorder.start_camera("cam_1", "run") is None
        await recorder.stop()

    async def test_a_camera_still_saving_cannot_be_restarted(self, recorder):
        pending = asyncio.get_running_loop().create_future()
        recorder._saving["cam_1"] = pending
        assert await recorder.start_camera("cam_1", "run") is None
        pending.set_result(None)


class TestStoppingOneCamera:
    async def test_the_others_keep_recording(self, recorder, manager, writers):
        await recorder.start("run")
        stopped = recorder.file_paths["cam_1"]

        saved = await recorder.stop_camera("cam_1")

        assert saved == stopped
        writers[stopped].release.assert_called_once()
        assert recorder.is_recording
        assert recorder.recording_camera_ids == ["cam_0", "cam_2"]
        for camera_id in ("cam_0", "cam_2"):
            assert not writers[recorder.file_paths[camera_id]].release.called
        await recorder.stop()

    async def test_a_stopped_camera_takes_no_more_frames(self, recorder, manager):
        await recorder.start("run")
        await recorder.stop_camera("cam_1")

        manager._on_camera_frame("cam_1", FRAME, 2.0)
        manager._on_camera_frame("cam_0", FRAME, 2.0)

        assert recorder.get_frame_count("cam_1") == 0
        assert recorder.get_frame_count("cam_0") == 1
        await recorder.stop()

    async def test_stopping_the_last_camera_ends_the_run(self, recorder):
        await recorder.start_camera("cam_1", "solo")
        await recorder.stop_camera("cam_1")
        assert recorder.state == RecordingState.IDLE
        assert not recorder.is_recording
        assert await recorder.stop() == {}

    async def test_a_camera_that_is_not_recording_is_left_alone(self, recorder):
        await recorder.start_camera("cam_0", "solo")
        assert await recorder.stop_camera("cam_2") is None
        assert recorder.recording_camera_ids == ["cam_0"]
        await recorder.stop()

    async def test_its_frame_rate_is_corrected_from_its_own_frames(self, recorder, manager):
        await recorder.start("run")
        for i in range(30):  # cam_1 really ran at 27.5 fps
            manager._on_camera_frame("cam_1", FRAME, 100.0 + i / 27.5)
        with patch("glider.vision.multi_video_recorder.correct_video_fps") as correct:
            await recorder.stop_camera("cam_1")
        (path, _recorded, measured, _codec), _ = correct.call_args
        assert path == recorder.file_paths["cam_1"]
        assert measured == pytest.approx(27.5)
        await recorder.stop()

    async def test_the_annotated_video_stops_with_the_primary(self, recorder, writers):
        # It is drawn from the primary's frames and its frame rate is corrected
        # from them, so it cannot outlive them.
        await recorder.start("run", record_annotated=True)
        annotated = recorder.annotated_file_path
        assert recorder.write_annotated_frame(FRAME)

        await recorder.stop_camera("cam_0")

        writers[annotated].release.assert_called_once()
        assert not recorder.write_annotated_frame(FRAME)
        await recorder.stop()
        writers[annotated].release.assert_called_once()


class TestStopAll:
    async def test_it_reports_every_file_of_the_run(self, recorder):
        await recorder.start("run")
        early = recorder.file_paths["cam_1"]
        await recorder.stop_camera("cam_1")

        saved = await recorder.stop()

        assert set(saved) == {"cam_0", "cam_1", "cam_2"}
        assert saved["cam_1"] == early

    async def test_it_finishes_each_file_once(self, recorder, writers):
        # A second release, or a second frame-rate correction, of a file
        # already finished would re-encode a good video at the wrong rate.
        await recorder.start("run")
        early = recorder.file_paths["cam_1"]
        with patch("glider.vision.multi_video_recorder.correct_video_fps") as correct:
            await recorder.stop_camera("cam_1")
            saved = await recorder.stop()
        writers[early].release.assert_called_once()
        corrected = sorted(str(c.args[0]) for c in correct.call_args_list)
        assert corrected == sorted(str(p) for p in saved.values())

    async def test_it_waits_for_a_camera_still_saving(self, recorder):
        await recorder.start("run")
        pending = asyncio.get_running_loop().create_future()
        recorder._saving["cam_9"] = pending

        stopping = asyncio.ensure_future(recorder.stop())
        await asyncio.sleep(0.05)
        assert not stopping.done(), "callers treat what stop() returns as finished"

        pending.set_result(None)
        await stopping
        assert recorder.state == RecordingState.IDLE


class TestTakes:
    async def test_a_quick_restart_does_not_overwrite_the_first_take(self, recorder):
        # Filenames carry a one-second timestamp; the second take inside that
        # second used to open the first one's file for writing.
        with patch.object(recorder, "_generate_filename", return_value="solo_x.mp4"):
            first = await recorder.start_camera("cam_1", "solo")
            first.touch()
            await recorder.stop_camera("cam_1")
            second = await recorder.start_camera("cam_1", "solo")
        assert first.name == "solo_x.mp4"
        assert second.name == "solo_x_take2.mp4"
        await recorder.stop()

    async def test_each_camera_keeps_its_own_clock(self, recorder):
        await recorder.start_camera("cam_0", "run")
        recorder._camera_started["cam_0"] = datetime.now() - timedelta(seconds=90)
        await recorder.start_camera("cam_1", "run")

        assert recorder.camera_duration("cam_0") >= 90
        assert recorder.camera_duration("cam_1") < 5
        assert recorder.camera_duration("cam_2") == 0.0
        await recorder.stop()

    async def test_a_stopped_take_holds_its_length(self, recorder):
        await recorder.start_camera("cam_0", "run")
        await recorder.start_camera("cam_1", "run")
        await recorder.stop_camera("cam_0")
        held = recorder.camera_duration("cam_0")
        await asyncio.sleep(0.05)
        assert recorder.camera_duration("cam_0") == held
        await recorder.stop()


class TestNamesMidRun:
    async def test_a_camera_renamed_mid_run_starts_under_its_new_name(self, recorder):
        await recorder.start_camera("cam_0", "run")
        recorder.set_camera_labels({"cam_2": "arena_C"})
        path = await recorder.start_camera("cam_2", "run")
        assert "arena_C" in path.name
        await recorder.stop()

    async def test_renaming_does_not_rename_an_open_file(self, recorder):
        await recorder.start_camera("cam_2", "run")
        before = recorder.file_paths["cam_2"]
        recorder.set_camera_labels({"cam_2": "arena_C"})
        assert recorder.file_paths["cam_2"] == before
        await recorder.stop()
