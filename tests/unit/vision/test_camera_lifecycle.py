"""CameraManager: real frame size, unplug surfaced, clean reconnect/restart."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np

from glider.vision.camera_manager import CameraManager, CameraState


class _UnpluggingCapture:
    """Delivers a few 320x240 frames, then grab() fails forever."""

    def __init__(self, frames: int):
        self._left = frames

    def isOpened(self):
        return True

    def grab(self):
        self._left -= 1
        return self._left >= 0

    def retrieve(self):
        return True, np.zeros((240, 320, 3), dtype=np.uint8)

    def release(self):
        pass


def test_unplug_is_surfaced_and_real_frame_size_is_adopted():
    cam = CameraManager()
    cam._settings.resolution = (640, 480)
    cam._settings.fps = 120
    cam._STALL_TIMEOUT_S = 0.2
    cam._capture = _UnpluggingCapture(frames=3)
    errors: list[str] = []
    stamps: list[float] = []
    cam.on_error(errors.append)
    cam.on_frame(lambda frame, ts: stamps.append(ts))

    before = time.time()
    assert cam.start_streaming()
    deadline = time.monotonic() + 3.0
    while not errors and time.monotonic() < deadline:
        time.sleep(0.02)
    cam.stop_streaming()

    assert cam.settings.resolution == (320, 240)
    assert len(stamps) == 3 and all(before <= ts <= time.time() for ts in stamps)
    assert errors and "no frames" in errors[0]
    assert cam.state == CameraState.ERROR
    assert cam._capture_thread is None


def test_connect_drops_a_previous_picamera2():
    cam = CameraManager()
    old_pi = MagicMock()
    cam._picamera2, cam._using_picamera2 = old_pi, True
    with (
        patch.object(cam, "_try_connect_with_backend", return_value=True),
        patch("glider.vision.camera_manager.sys.platform", "linux"),
        patch("glider.vision.camera_manager._is_raspberry_pi", return_value=False),
    ):
        assert cam.connect()
    old_pi.close.assert_called_once()
    assert cam._picamera2 is None and cam._using_picamera2 is False


def test_restart_refused_while_the_old_loop_is_still_blocked():
    cam = CameraManager()
    cam._capture = _UnpluggingCapture(frames=0)
    stuck = MagicMock()
    stuck.is_alive.return_value = True
    cam._capture_thread = stuck
    assert cam.start_streaming() is False
    assert cam._capture_thread is stuck
