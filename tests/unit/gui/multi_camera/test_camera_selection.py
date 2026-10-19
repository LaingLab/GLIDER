"""Which cameras are switched on, remembered between sessions."""

from __future__ import annotations

from glider.gui.multi_camera.camera_selection import CameraSelection


class TestSelection:
    def test_every_camera_starts_switched_on(self, tmp_path):
        # Including ones never seen before: a camera plugged in later should
        # appear, as every camera did before cameras could be switched off.
        selection = CameraSelection(tmp_path / "selection.json")
        assert selection.is_enabled("cam_0")
        assert selection.is_enabled("cam_15")

    def test_switching_off_is_remembered(self, tmp_path):
        path = tmp_path / "selection.json"
        CameraSelection(path).set_enabled("cam_3", False)
        assert not CameraSelection(path).is_enabled("cam_3")
        assert CameraSelection(path).is_enabled("cam_2")

    def test_switching_back_on_is_remembered(self, tmp_path):
        path = tmp_path / "selection.json"
        CameraSelection(path).set_enabled("cam_3", False)
        CameraSelection(path).set_enabled("cam_3", True)
        assert CameraSelection(path).is_enabled("cam_3")

    def test_an_unreadable_file_means_every_camera_is_on(self, tmp_path):
        path = tmp_path / "selection.json"
        path.write_text("{not json", encoding="utf-8")
        assert CameraSelection(path).disabled() == set()

    def test_a_file_of_the_wrong_shape_means_every_camera_is_on(self, tmp_path):
        path = tmp_path / "selection.json"
        path.write_text('["cam_1"]', encoding="utf-8")
        assert CameraSelection(path).disabled() == set()

    def test_a_save_that_fails_does_not_raise(self, tmp_path):
        # A read-only home directory must not stop anyone recording.
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("", encoding="utf-8")
        selection = CameraSelection(blocker / "selection.json")
        selection.set_enabled("cam_1", False)
        assert not selection.is_enabled("cam_1")
        assert selection.save() is False
