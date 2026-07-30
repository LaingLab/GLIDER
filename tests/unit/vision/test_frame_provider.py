"""FrameProvider adapters: live camera and recorded video behind one interface."""

from __future__ import annotations

import numpy as np

from glider.vision.frame_provider import CameraFrameProvider, VideoFrameProvider

# tests/unit/ has no __init__.py, so `from ..conftest import ...` is not a
# usable relative import from tests/unit/vision/. These mirror the constants
# in tests/unit/conftest.py (the synthetic_clip fixture is autodiscovered
# from there and needs no import).
CLIP_FRAMES = 12
CLIP_W = 64
CLIP_H = 48
CLIP_FPS = 10.0


class _FakeCamera:
    """The two members of CameraManager that the dialog actually uses."""

    def __init__(self, connected=True, frame=None):
        self.is_connected = connected
        self._frame = frame
        self.calls = 0

    def get_frame(self):
        self.calls += 1
        return None if self._frame is None else (self._frame, 123.0)


class TestCameraFrameProvider:
    def test_forwards_connection_state(self):
        assert CameraFrameProvider(_FakeCamera(connected=False)).is_connected is False
        assert CameraFrameProvider(_FakeCamera(connected=True)).is_connected is True

    def test_forwards_frames(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        provider = CameraFrameProvider(_FakeCamera(frame=frame))
        got = provider.get_frame()
        assert got is not None
        assert got[0].shape == (4, 4, 3)

    def test_passes_through_no_frame(self):
        assert CameraFrameProvider(_FakeCamera(frame=None)).get_frame() is None

    def test_is_not_seekable(self):
        # The dialog keys its scrubber off this.
        assert not hasattr(CameraFrameProvider(_FakeCamera()), "seek")


class TestVideoFrameProvider:
    def test_loads_a_clip(self, synthetic_clip):
        provider = VideoFrameProvider(synthetic_clip)
        try:
            assert provider.is_connected is True
            assert provider.frame_count == CLIP_FRAMES
        finally:
            provider.release()

    def test_missing_file_is_not_connected(self, tmp_path):
        provider = VideoFrameProvider(tmp_path / "nope.mp4")
        try:
            assert provider.is_connected is False
            assert provider.get_frame() is None
        finally:
            provider.release()

    def test_get_frame_returns_frame_and_timeline_timestamp(self, synthetic_clip):
        provider = VideoFrameProvider(synthetic_clip)
        try:
            provider.seek(5)
            got = provider.get_frame()
            assert got is not None
            frame, timestamp = got
            assert frame.shape == (CLIP_H, CLIP_W, 3)
            # Video timeline, not wall clock.
            assert timestamp == 5 / CLIP_FPS
        finally:
            provider.release()

    def test_seek_is_clamped_into_range(self, synthetic_clip):
        provider = VideoFrameProvider(synthetic_clip)
        try:
            provider.seek(-10)
            assert provider.position == 0
            provider.seek(9999)
            assert provider.position == CLIP_FRAMES - 1
            assert provider.get_frame() is not None
        finally:
            provider.release()

    def test_seeking_changes_the_frame(self, synthetic_clip):
        """The fixture's square slides, so distinct frames differ."""
        provider = VideoFrameProvider(synthetic_clip)
        try:
            provider.seek(0)
            first = provider.get_frame()[0].copy()
            provider.seek(CLIP_FRAMES - 1)
            last = provider.get_frame()[0]
            assert not np.array_equal(first, last)
        finally:
            provider.release()
