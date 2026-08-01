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
