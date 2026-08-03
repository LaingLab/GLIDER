"""Tests for VideoFileSource — offline video frame access (scrub + sequential)."""

from pathlib import Path

import numpy as np

from glider.vision.video_source import VideoFileSource, video_resolution


def test_video_resolution_reads_the_header(synthetic_clip: Path):
    assert video_resolution(synthetic_clip) == (64, 48)


def test_video_resolution_is_none_when_unreadable(tmp_path: Path):
    """None is meaningful: callers size an arena with it, and a wrong size is
    worse than an absent one."""
    assert video_resolution(tmp_path / "nope.avi") is None


def test_load_reports_metadata(synthetic_clip: Path):
    src = VideoFileSource()
    assert src.load(synthetic_clip) is True
    assert src.frame_count == 12
    assert src.fps == 10.0
    assert src.resolution == (64, 48)
    src.release()


def test_load_rejects_missing_file(tmp_path: Path):
    src = VideoFileSource()
    assert src.load(tmp_path / "nope.avi") is False


def test_read_frame_returns_array(synthetic_clip: Path):
    src = VideoFileSource()
    src.load(synthetic_clip)
    frame = src.read_frame(5)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (48, 64, 3)
    src.release()


def test_frames_yields_every_frame_in_order(synthetic_clip: Path):
    src = VideoFileSource()
    src.load(synthetic_clip)
    indices = [n for n, _frame in src.frames()]
    assert indices == list(range(12))
    src.release()


def test_release_resets_state(synthetic_clip: Path):
    src = VideoFileSource()
    src.load(synthetic_clip)
    src.release()
    assert src.is_loaded is False
    assert src.frame_count == 0
    assert src.resolution == (0, 0)
    assert src.path is None


class _SeekRecorder:
    """Delegates to a real VideoCapture, noting every seek.

    cv2.VideoCapture attributes are read-only, so counting seeks means
    standing in front of the object rather than patching it.
    """

    def __init__(self, cap):
        self._cap = cap
        self.seeks: list[float] = []

    def set(self, prop, value):
        self.seeks.append(value)
        return self._cap.set(prop, value)

    def __getattr__(self, name):
        return getattr(self._cap, name)


def test_reading_forward_does_not_reseek(synthetic_clip: Path):
    """Playback is a run of consecutive reads; on a long-GOP codec a seek per
    frame re-decodes from the previous keyframe."""
    src = VideoFileSource()
    src.load(synthetic_clip)
    recorder = _SeekRecorder(src._cap)
    src._cap = recorder

    for n in range(5):
        assert src.read_frame(n) is not None
    assert recorder.seeks == []  # frame 0 onward is already where the decoder sits
    src._cap = recorder._cap
    src.release()


def test_jumping_backwards_still_seeks(synthetic_clip: Path):
    src = VideoFileSource()
    src.load(synthetic_clip)
    src.read_frame(5)
    recorder = _SeekRecorder(src._cap)
    src._cap = recorder

    src.read_frame(2)
    assert recorder.seeks == [2]
    src._cap = recorder._cap
    src.release()


def test_the_fast_path_returns_the_same_frames_as_seeking(synthetic_clip: Path):
    """The optimisation must be invisible in the output."""
    sequential = VideoFileSource()
    sequential.load(synthetic_clip)
    walked = [sequential.read_frame(n) for n in range(6)]
    sequential.release()

    sought = VideoFileSource()
    sought.load(synthetic_clip)
    jumped = []
    for n in range(6):
        sought.read_frame(11)  # force a seek away every time
        jumped.append(sought.read_frame(n))
    sought.release()

    for a, b in zip(walked, jumped, strict=True):
        assert np.array_equal(a, b)
