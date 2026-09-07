"""classify_session: one object supplies the recording, the pose and the folder.

Phase 2 of the project-structure work. The point is not convenience - it is
that the pose CSV cannot be named without its sidecar coming with it, because
neither is named at all. Passing them separately is how an exp-7 CSV came to be
scored against metadata parked in another folder, which left eleven recordings
with no speed axis and nothing to say so.

These tests check what the adapter resolves, not what classify does with it.
"""

from __future__ import annotations

import json

import pytest

from glider.analysis.behavior import classify as classify_module
from glider.core.session import Session


@pytest.fixture
def calls(monkeypatch):
    """Capture what classify_session hands to classify."""
    recorded: list[tuple[tuple, dict]] = []

    def fake_classify(*args, **kwargs):
        recorded.append((args, kwargs))
        return "result"

    monkeypatch.setattr(classify_module, "classify", fake_classify)
    return recorded


@pytest.fixture
def session(tmp_path):
    folder = tmp_path / "sessions" / "Test 1"
    folder.mkdir(parents=True)
    (folder / "Test 1.mp4").touch()
    csv = folder / "Test 1DLC_train-6.csv"
    csv.write_text(
        "scorer,yolo,yolo,yolo\nbodyparts,nose,nose,nose\ncoords,x,y,likelihood\n0,1,1,1\n"
    )
    (folder / "Test 1DLC_train-6.meta.json").write_text(
        json.dumps({"fps": 30.0, "resolution": [640, 480]})
    )
    return Session(tmp_path, "Test 1")


def _kwargs(calls) -> dict:
    return calls[0][1]


class TestResolution:
    def test_it_passes_the_sessions_recording(self, session, calls):
        classify_module.classify_session(session, None, None, ["nose"])
        assert calls[0][0][0] == session.video

    def test_it_passes_the_sessions_pose_track(self, session, calls):
        classify_module.classify_session(session, None, None, ["nose"])
        assert _kwargs(calls)["pose_csv_in"] == session.pose_csv

    def test_the_pose_it_passes_has_its_sidecar(self, session, calls):
        # The relationship the whole design exists to protect: whatever CSV is
        # handed over, its metadata is beside it.
        classify_module.classify_session(session, None, None, ["nose"])
        from glider.vision.pose.dlc import meta_path

        assert meta_path(_kwargs(calls)["pose_csv_in"]).exists()

    def test_outputs_default_into_the_session_folder(self, session, calls):
        classify_module.classify_session(session, None, None, ["nose"])
        assert _kwargs(calls)["output_dir"] == session.folder / "analysis"

    def test_it_does_not_write_a_second_copy_of_the_poses(self, session, calls):
        # Two pose tracks for one session, with nothing recording which
        # produced the numbers, is the ambiguity this is meant to remove.
        classify_module.classify_session(session, None, None, ["nose"])
        assert _kwargs(calls)["write_pose_csv"] is False


class TestOverrides:
    def test_an_explicit_output_dir_wins(self, session, calls, tmp_path):
        elsewhere = tmp_path / "somewhere"
        classify_module.classify_session(session, None, None, ["nose"], output_dir=elsewhere)
        assert _kwargs(calls)["output_dir"] == elsewhere

    def test_an_explicit_pose_csv_wins(self, session, calls, tmp_path):
        other = tmp_path / "other.csv"
        classify_module.classify_session(session, None, None, ["nose"], pose_csv_in=other)
        assert _kwargs(calls)["pose_csv_in"] == other

    def test_other_options_pass_straight_through(self, session, calls):
        classify_module.classify_session(session, None, None, ["nose"], freeze_cm_s=3.0)
        assert _kwargs(calls)["freeze_cm_s"] == 3.0


class TestUntrackedSession:
    def test_a_session_with_no_poses_names_none(self, tmp_path, calls):
        # Nothing on disk to reuse: the run tracks the video itself, and
        # naming a CSV that is not there would only fail the run.
        folder = tmp_path / "sessions" / "Test 9"
        folder.mkdir(parents=True)
        (folder / "Test 9.mp4").touch()
        classify_module.classify_session(Session(tmp_path, "Test 9"), None, "y.pt", ["nose"])
        assert "pose_csv_in" not in _kwargs(calls)
        assert "write_pose_csv" not in _kwargs(calls)

    def test_a_session_with_no_recording_is_refused(self, tmp_path, calls):
        with pytest.raises(FileNotFoundError, match="Test 9"):
            classify_module.classify_session(Session(tmp_path, "Test 9"), None, None, ["nose"])
        assert calls == []
