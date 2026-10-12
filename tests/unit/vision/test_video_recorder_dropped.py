from types import SimpleNamespace
from unittest.mock import MagicMock

from glider.vision.video_recorder import VideoRecorder


def test_frames_dropped_zero_without_writer():
    assert VideoRecorder(MagicMock()).frames_dropped == 0


def test_frames_dropped_reads_writer_thread():
    rec = VideoRecorder(MagicMock())
    rec._writer_thread = SimpleNamespace(frames_dropped=7)
    assert rec.frames_dropped == 7
