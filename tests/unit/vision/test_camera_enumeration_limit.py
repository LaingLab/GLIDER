"""How many camera indices enumeration is willing to look at.

The rigs this is for run up to sixteen cameras. The old defaults stopped at
four (``CameraManager``) and ten (``MultiCameraManager``), so cameras past that
were invisible in the UI even though every layer below could drive them.

Scanning costs real time - each index opens a device - so the cap is a
deliberate number, not an accident, and it lives in one place.
"""

from __future__ import annotations

from unittest.mock import patch

from glider.vision.camera_manager import MAX_CAMERAS, CameraManager
from glider.vision.multi_camera_manager import MultiCameraManager


class TestDefaults:
    def test_the_cap_covers_a_sixteen_camera_rig(self):
        assert MAX_CAMERAS >= 16

    def test_camera_manager_defaults_to_the_shared_cap(self):
        import inspect

        default = (
            inspect.signature(CameraManager.enumerate_cameras).parameters["max_cameras"].default
        )
        assert default == MAX_CAMERAS

    def test_multi_camera_manager_defaults_to_the_shared_cap(self):
        import inspect

        default = (
            inspect.signature(MultiCameraManager.enumerate_all_cameras)
            .parameters["max_cameras"]
            .default
        )
        assert default == MAX_CAMERAS

    def test_the_two_agree(self):
        import inspect

        a = inspect.signature(CameraManager.enumerate_cameras).parameters["max_cameras"].default
        b = (
            inspect.signature(MultiCameraManager.enumerate_all_cameras)
            .parameters["max_cameras"]
            .default
        )
        assert a == b, "the two enumeration entry points must not drift apart"


class TestScanRange:
    def test_it_scans_up_to_the_cap(self):
        seen = []

        class FakeCap:
            def __init__(self, index, *_a, **_k):
                seen.append(index)

            def isOpened(self):
                return False

            def release(self):
                pass

            def get(self, *_a):
                return 0

            def set(self, *_a):
                return False

        with patch("cv2.VideoCapture", FakeCap):
            CameraManager.enumerate_cameras()
        assert max(seen) == MAX_CAMERAS - 1
        assert len(set(seen)) == MAX_CAMERAS

    def test_an_explicit_limit_still_wins(self):
        seen = []

        class FakeCap:
            def __init__(self, index, *_a, **_k):
                seen.append(index)

            def isOpened(self):
                return False

            def release(self):
                pass

            def get(self, *_a):
                return 0

            def set(self, *_a):
                return False

        with patch("cv2.VideoCapture", FakeCap):
            CameraManager.enumerate_cameras(max_cameras=3)
        assert max(seen) == 2
