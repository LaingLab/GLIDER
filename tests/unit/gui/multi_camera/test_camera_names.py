"""Naming the cameras in a rig.

``cam_3`` is an enumeration order, not a place, and it is not stable across
replugging. The name is only worth typing if it reaches the recordings -
otherwise the operator names a camera in the window and still gets ``_cam5`` on
disk, which answers nothing a month later.
"""

from __future__ import annotations

import json

import pytest

from glider.gui.multi_camera.camera_names import CameraLabels, slug


@pytest.fixture
def labels(tmp_path):
    return CameraLabels(tmp_path / "camera_labels.json")


class TestNaming:
    def test_a_named_camera_reports_its_name(self, labels):
        labels.set_label("cam_3", "arena 3")
        assert labels.label("cam_3") == "arena 3"
        assert labels.display_name("cam_3") == "arena 3"

    def test_an_unnamed_camera_falls_back_to_its_id(self, labels):
        # A tile with no caption is worse than one captioned cam_3.
        assert labels.label("cam_3") == ""
        assert labels.display_name("cam_3") == "cam_3"

    def test_clearing_a_name_restores_the_id(self, labels):
        labels.set_label("cam_3", "arena 3")
        labels.set_label("cam_3", "")
        assert labels.display_name("cam_3") == "cam_3"

    def test_whitespace_is_not_a_name(self, labels):
        labels.set_label("cam_3", "   ")
        assert labels.display_name("cam_3") == "cam_3"

    def test_a_long_name_is_truncated_not_refused(self, labels):
        kept = labels.set_label("cam_3", "x" * 200)
        assert 0 < len(kept) <= 48


class TestPersistence:
    def test_names_survive_a_restart(self, tmp_path):
        path = tmp_path / "camera_labels.json"
        CameraLabels(path).set_label("cam_0", "arena A")
        assert CameraLabels(path).display_name("cam_0") == "arena A"

    def test_a_missing_file_is_simply_no_names(self, tmp_path):
        assert len(CameraLabels(tmp_path / "nope.json")) == 0

    def test_a_corrupt_file_is_no_names_not_a_crash(self, tmp_path):
        # The window has to open regardless.
        path = tmp_path / "camera_labels.json"
        path.write_text("{ not json")
        assert len(CameraLabels(path)) == 0

    def test_an_unwritable_location_reports_rather_than_raises(self, tmp_path):
        # A read-only home directory must not stop anyone recording; the cost
        # is retyping names, not losing data.
        blocked = tmp_path / "afile"
        blocked.write_text("x")
        store = CameraLabels(blocked / "labels.json")
        assert store.set_label("cam_0", "arena A") == "arena A"
        assert store.display_name("cam_0") == "arena A"

    def test_the_file_is_readable_json(self, tmp_path):
        path = tmp_path / "camera_labels.json"
        CameraLabels(path).set_label("cam_1", "arena B")
        assert json.loads(path.read_text()) == {"cam_1": "arena B"}


class TestFilenames:
    """The half that makes naming worth doing."""

    def test_a_name_becomes_the_filename_fragment(self, labels):
        labels.set_label("cam_3", "arena 3")
        assert labels.file_fragment("cam_3") == "arena_3"

    def test_an_unnamed_camera_still_gets_its_id(self, labels):
        assert labels.file_fragment("cam_3") == "cam_3"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("arena 3 / back", "arena_3_back"),
            ("A:B", "A_B"),
            ("  spaced  ", "spaced"),
            ("...", ""),
            ("", ""),
        ],
    )
    def test_slug_produces_something_windows_will_accept(self, raw, expected):
        # A label like "arena 3 / back" is a perfectly good thing to type and
        # cannot appear in a filename.
        assert slug(raw) == expected

    def test_a_name_that_slugs_to_nothing_falls_back_to_the_id(self, labels):
        labels.set_label("cam_2", "///")
        assert labels.file_fragment("cam_2") == "cam_2"
