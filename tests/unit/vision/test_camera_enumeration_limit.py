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
    def test_it_scans_up_to_the_cap_when_asked_to(self):
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
            # The default gives up early once a run of indices comes back
            # empty; this asserts the ceiling, so it opts out of that.
            CameraManager.enumerate_cameras(stop_after_misses=None)
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


class TestEarlyStop:
    """Probing every index to the cap is slow and noisy when few exist.

    Raising the cap from 4 to 16 made startup probe fifteen absent devices,
    which on macOS costs about a second and prints a wall of OpenCV errors.
    """

    @staticmethod
    def _fake(present):
        seen = []

        class FakeCap:
            def __init__(self, index, *_a, **_k):
                seen.append(index)
                self._ok = index in present

            def isOpened(self):
                return self._ok

            def release(self):
                pass

            def get(self, *_a):
                return 0

            def set(self, *_a):
                return False

        return FakeCap, seen

    def test_it_gives_up_after_a_run_of_empty_indices(self):
        fake_cap, seen = self._fake({0})
        with patch("cv2.VideoCapture", fake_cap):
            CameraManager.enumerate_cameras()
        assert max(seen) < MAX_CAMERAS - 1, "should not have probed all the way up"

    def test_a_lone_camera_is_still_found(self):
        fake_cap, seen = self._fake({0})
        with patch("cv2.VideoCapture", fake_cap):
            found = CameraManager.enumerate_cameras()
        assert [c.index for c in found] == [0]

    def test_a_contiguous_block_is_found_whole(self):
        fake_cap, _ = self._fake(set(range(8)))
        with patch("cv2.VideoCapture", fake_cap):
            found = CameraManager.enumerate_cameras()
        assert [c.index for c in found] == list(range(8))

    def test_a_small_gap_does_not_end_the_scan(self):
        # Indices are not guaranteed contiguous; one missing slot must not
        # hide the cameras above it.
        fake_cap, _ = self._fake({0, 1, 3, 4})
        with patch("cv2.VideoCapture", fake_cap):
            found = CameraManager.enumerate_cameras()
        assert [c.index for c in found] == [0, 1, 3, 4]

    def test_the_early_stop_can_be_turned_off(self):
        fake_cap, seen = self._fake({0, 15})
        with patch("cv2.VideoCapture", fake_cap):
            found = CameraManager.enumerate_cameras(stop_after_misses=None)
        assert max(seen) == MAX_CAMERAS - 1
        assert [c.index for c in found] == [0, 15]
