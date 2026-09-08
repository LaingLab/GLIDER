"""Camera names reach the filenames the recorder writes.

Without this the operator names a camera in the window and still gets
``_cam5`` on disk, which is an enumeration order rather than a place -- and not
a stable one, since replugging renumbers it. Eight files a run, and the name is
the only thing that says which arena each came from.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glider.vision.multi_video_recorder import MultiVideoRecorder, RecordingState


@pytest.fixture
def recorder():
    multi = MagicMock()
    multi.primary_camera_id = "cam_0"
    return MultiVideoRecorder(multi)


def _name(recorder, camera_id: str) -> str:
    return recorder._generate_filename("trh_d2", camera_id)


class TestLabelledFilenames:
    def test_a_named_camera_appears_by_name(self, recorder):
        recorder.set_camera_labels({"cam_3": "arena_C"})
        assert "arena_C" in _name(recorder, "cam_3")

    def test_an_unnamed_camera_keeps_the_old_scheme(self, recorder):
        # Nothing changes for a rig nobody has named.
        recorder.set_camera_labels({"cam_3": "arena_C"})
        assert _name(recorder, "cam_5").endswith("_cam5.mp4")

    def test_no_labels_at_all_is_the_old_behaviour(self, recorder):
        assert _name(recorder, "cam_2").endswith("_cam2.mp4")

    def test_the_primary_is_still_named_plainly(self, recorder):
        # Backwards compatibility: downstream tooling finds the primary by the
        # unsuffixed name.
        recorder.set_camera_labels({"cam_0": "arena_A"})
        name = _name(recorder, "cam_0")
        assert "arena_A" not in name
        assert name.startswith("trh_d2_")

    def test_the_experiment_name_still_leads(self, recorder):
        recorder.set_camera_labels({"cam_1": "arena_B"})
        assert _name(recorder, "cam_1").startswith("trh_d2_")

    def test_a_blank_label_is_ignored(self, recorder):
        recorder.set_camera_labels({"cam_1": "   "})
        assert _name(recorder, "cam_1").endswith("_cam1.mp4")


class TestWhenLabelsMayChange:
    def test_labels_are_refused_mid_recording(self, recorder):
        # The files are already open; renaming half a set partway through is
        # worse than not renaming.
        recorder.set_camera_labels({"cam_1": "before"})
        recorder._state = RecordingState.RECORDING
        recorder.set_camera_labels({"cam_1": "after"})
        assert "before" in _name(recorder, "cam_1")

    def test_labels_can_be_cleared_between_runs(self, recorder):
        recorder.set_camera_labels({"cam_1": "arena_B"})
        recorder.set_camera_labels({})
        assert _name(recorder, "cam_1").endswith("_cam1.mp4")
