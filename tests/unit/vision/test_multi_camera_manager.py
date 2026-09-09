"""MultiCameraManager: the streaming claims that keep two windows apart."""

from __future__ import annotations

from unittest.mock import MagicMock

from glider.vision.multi_camera_manager import MultiCameraManager


class TestSharedStreamingClaims:
    """Two windows drive these cameras through one manager.

    The camera panel's multi-camera preview and the Multi-Camera window both
    hold this object, so "stop" from either cannot mean "stop" outright.
    """

    @staticmethod
    def _manager_with_camera():
        manager = MultiCameraManager()
        camera = MagicMock()
        manager._cameras["cam_0"] = camera
        return manager, camera

    def test_one_owner_releasing_stops_the_cameras(self):
        manager, camera = self._manager_with_camera()
        manager.start_all_streaming(owner="a")

        manager.stop_all_streaming(owner="a")

        camera.stop_streaming.assert_called_once()

    def test_a_second_owner_keeps_them_on(self):
        manager, camera = self._manager_with_camera()
        manager.start_all_streaming(owner="a")
        manager.start_all_streaming(owner="b")

        manager.stop_all_streaming(owner="a")

        assert not camera.stop_streaming.called
        assert manager.stream_owners() == {"b"}

        manager.stop_all_streaming(owner="b")
        camera.stop_streaming.assert_called_once()

    def test_an_unowned_stop_is_unconditional(self):
        """The shutdown path, and the historical behaviour: a caller that never
        claimed anything must still be able to stop everything."""
        manager, camera = self._manager_with_camera()
        manager.start_all_streaming(owner="a")

        manager.stop_all_streaming()

        camera.stop_streaming.assert_called_once()
        assert manager.stream_owners() == set()

    def test_releasing_an_owner_that_never_claimed_is_harmless(self):
        manager, camera = self._manager_with_camera()
        manager.stop_all_streaming(owner="never-started")
        camera.stop_streaming.assert_called_once()
