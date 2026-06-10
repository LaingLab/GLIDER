"""Tests for CameraManager.enumerate_cameras index discovery.

Regression coverage for the enumeration loop terminating after the first
camera (a `break` that exited the index loop instead of advancing to the
next index), which left multi-camera rigs reporting only one device.
"""

from unittest.mock import patch

from glider.vision.camera_manager import CameraManager


class _FakeCapture:
    """Stand-in for cv2.VideoCapture: a fixed set of indices are "present"."""

    PRESENT = {0, 1}

    def __init__(self, index, backend=None):
        self._index = index

    def isOpened(self):
        return self._index in self.PRESENT

    def get(self, prop):
        # Return 0 so the production code falls back to its default w/h/fps.
        return 0

    def release(self):
        pass


def test_enumerate_cameras_finds_all_present_indices():
    """Every openable index is reported, not just the first one found."""
    with (
        patch("glider.vision.camera_manager.cv2.VideoCapture", _FakeCapture),
        patch("glider.vision.camera_manager._get_windows_camera_names", return_value=[]),
    ):
        cameras = CameraManager.enumerate_cameras(max_cameras=4)

    assert sorted(c.index for c in cameras) == [0, 1]


def test_enumerate_cameras_empty_when_none_open():
    """No present indices -> no cameras (loop completes without error)."""

    class _NoneOpen(_FakeCapture):
        PRESENT: set[int] = set()

    with (
        patch("glider.vision.camera_manager.cv2.VideoCapture", _NoneOpen),
        patch("glider.vision.camera_manager._get_windows_camera_names", return_value=[]),
    ):
        cameras = CameraManager.enumerate_cameras(max_cameras=4)

    assert cameras == []
