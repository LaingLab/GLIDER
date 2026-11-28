from types import SimpleNamespace
from unittest.mock import MagicMock

from glider.vision.video_recorder import VideoRecorder


def test_frames_dropped_zero_without_writer():
    assert VideoRecorder(MagicMock()).frames_dropped == 0


def test_frames_dropped_reads_writer_thread():
    rec = VideoRecorder(MagicMock())
    rec._writer_thread = SimpleNamespace(frames_dropped=7)
    assert rec.frames_dropped == 7


def test_first_frame_timestamp_and_measured_fps():
    from glider.vision.video_recorder import RecordingState

    rec = VideoRecorder(MagicMock())
    assert rec.first_frame_timestamp is None and rec.fps == 30.0
    rec._state = RecordingState.RECORDING
    rec._writer_thread = SimpleNamespace(enqueue=lambda f: True, frames_dropped=0)
    frame = MagicMock()
    for ts in (10.0, 10.1, 10.2):
        rec._on_frame(frame, ts)
    assert rec.first_frame_timestamp == 10.0
    assert rec.frame_timestamps == [10.0, 10.1, 10.2]
    assert abs(rec.fps - 10.0) < 1e-9
