"""Cameras the window registers must use the operator's configured settings.

The first version registered every camera with CameraSettings(camera_index=...)
defaults - 640x480 at 30 fps - regardless of what was set in the Settings
dialog. On a rig where resolution or frame rate has been chosen deliberately
that silently records the wrong thing, and the recording is the artifact you
cannot go back and redo.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from glider.gui.multi_camera.window import MultiCameraWindow
from glider.vision.camera_manager import CameraSettings


def _manager(available: int = 2):
    manager = MagicMock()
    manager.cameras = {}
    manager.primary_camera_id = None
    manager.get_camera_fps.return_value = 30.0
    manager.enumerate_all_cameras.return_value = [MagicMock(index=i) for i in range(available)]
    manager.camera_id_from_index.side_effect = lambda i: f"cam_{i}"

    def _add(camera_id, settings):
        manager.cameras[camera_id] = MagicMock()
        return True

    manager.add_camera.side_effect = _add
    return manager


def _recorder():
    rec = MagicMock()
    rec.is_recording = False
    rec.frames_dropped = {}
    return rec


@pytest.fixture
def configured():
    return CameraSettings(camera_index=0, resolution=(1280, 720), fps=60)


class TestConfiguredSettings:
    def test_registered_cameras_use_the_configured_resolution(self, qtbot, configured):
        manager = _manager(2)
        win = MultiCameraWindow(
            manager, recorder=_recorder(), base_settings=configured, parent=None
        )
        qtbot.addWidget(win)
        used = [c.args[1] for c in manager.add_camera.call_args_list]
        assert all(s.resolution == (1280, 720) for s in used)
        assert all(s.fps == 60 for s in used)

    def test_each_camera_keeps_its_own_index(self, qtbot, configured):
        manager = _manager(3)
        win = MultiCameraWindow(
            manager, recorder=_recorder(), base_settings=configured, parent=None
        )
        qtbot.addWidget(win)
        used = [c.args[1] for c in manager.add_camera.call_args_list]
        assert [s.camera_index for s in used] == [0, 1, 2]

    def test_the_configured_settings_are_not_mutated(self, qtbot, configured):
        # replace() rather than mutating: the same object is shared with the
        # camera panel, and stamping an index onto it would retarget that too.
        manager = _manager(2)
        win = MultiCameraWindow(
            manager, recorder=_recorder(), base_settings=configured, parent=None
        )
        qtbot.addWidget(win)
        assert configured.camera_index == 0

    def test_it_still_works_with_no_settings_given(self, qtbot):
        manager = _manager(1)
        win = MultiCameraWindow(manager, recorder=_recorder(), parent=None)
        qtbot.addWidget(win)
        used = [c.args[1] for c in manager.add_camera.call_args_list]
        assert used and isinstance(used[0], CameraSettings)

    def test_settings_are_a_copy_per_camera(self, qtbot, configured):
        manager = _manager(2)
        win = MultiCameraWindow(
            manager, recorder=_recorder(), base_settings=configured, parent=None
        )
        qtbot.addWidget(win)
        used = [c.args[1] for c in manager.add_camera.call_args_list]
        assert used[0] is not used[1]
        assert replace(used[0], camera_index=1) == used[1]
