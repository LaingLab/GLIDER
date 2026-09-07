"""Session: one recording and everything derived from it.

Phase 1 of the project-structure work is resolution only. It has to answer
"where is this session's pose CSV" against the canonical layout *and* against
the flat folders that already exist, because nothing has moved yet and real
cohorts cannot be asked to reorganise before they can be read.

The relationship that matters most is pose_csv/pose_meta. Those two drifting
apart is what silently erased freezing from eleven VMHAHA recordings: the
sidecar carries the resolution, and without it classify scores no speed axis at
all. Resolving them from one object is the point.
"""

from __future__ import annotations

import json

import pytest

from glider.core.session import Session


def _pose_csv(path, *, frames: int = 3, resolution=(640, 480)) -> None:
    """A minimal DLC-format CSV plus the sidecar that belongs beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = ["nose", "body_center"]
    path.write_text(
        "scorer," + ",".join(["yolo_exp-7"] * len(parts) * 3) + "\n"
        "bodyparts," + ",".join(p for p in parts for _ in range(3)) + "\n"
        "coords,"
        + ",".join(["x", "y", "likelihood"] * len(parts))
        + "\n"
        + "".join(f"{i}," + ",".join(["1.0"] * len(parts) * 3) + "\n" for i in range(frames))
    )
    sidecar = path.parent / (path.stem + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "source": "yolo_exp-7",
                "keypoint_names": parts,
                "n_frames": frames,
                "resolution": list(resolution),
            }
        )
    )


@pytest.fixture
def canonical(tmp_path):
    """The layout the spec proposes."""
    root = tmp_path / "experiment"
    folder = root / "sessions" / "Test 1"
    folder.mkdir(parents=True)
    (folder / "Test 1.mp4").touch()
    _pose_csv(folder / "Test 1DLC_exp-7.csv")
    (folder / "Test 1_arena.json").write_text("{}")
    (folder / "Test 1_zone.json").write_text("{}")
    (folder / "analysis").mkdir()
    (folder / "analysis" / "stats.csv").write_text("state,fraction\n")
    (folder / "analysis" / "run.json").write_text('{"schema_version": 1}')
    return root


@pytest.fixture
def flat(tmp_path):
    """A cohort folder as they actually exist today."""
    root = tmp_path / "videos"
    root.mkdir()
    (root / "Test 1.mp4").touch()
    _pose_csv(root / "Test 1DLC_exp-7.csv")
    (root / "test1_zone.json").write_text("{}")  # note the spelling drift
    (root / "Test 1").mkdir()
    (root / "Test 1" / "stats.csv").write_text("state,fraction\n")
    return root


class TestCanonicalLayout:
    def test_it_finds_the_video(self, canonical):
        assert Session(canonical, "Test 1").video.name == "Test 1.mp4"

    def test_it_finds_the_pose_csv(self, canonical):
        assert Session(canonical, "Test 1").pose_csv.name == "Test 1DLC_exp-7.csv"

    def test_the_sidecar_resolves_beside_its_csv(self, canonical):
        session = Session(canonical, "Test 1")
        assert session.pose_meta.parent == session.pose_csv.parent
        assert session.pose_meta.exists()

    def test_it_finds_the_arena_and_zone(self, canonical):
        session = Session(canonical, "Test 1")
        assert session.arena.name == "Test 1_arena.json"
        assert session.zone.name == "Test 1_zone.json"

    def test_it_finds_the_analysis_outputs(self, canonical):
        session = Session(canonical, "Test 1")
        assert session.stats.exists()
        assert session.run_manifest.exists()


class TestFlatLayout:
    def test_it_still_finds_the_video(self, flat):
        assert Session(flat, "Test 1").video.name == "Test 1.mp4"

    def test_it_still_finds_the_pose_csv(self, flat):
        assert Session(flat, "Test 1").pose_csv.name == "Test 1DLC_exp-7.csv"

    def test_it_tolerates_the_naming_drift(self, flat):
        # test1_zone.json for a session called "Test 1" - four spellings of one
        # session is what these folders actually contain.
        assert Session(flat, "Test 1").zone is not None

    def test_it_finds_outputs_in_a_sibling_folder(self, flat):
        assert Session(flat, "Test 1").stats.exists()


class TestMissingArtifacts:
    def test_an_absent_artifact_is_none_not_a_guess(self, tmp_path):
        # A path that does not exist is worse than None: it reads as an answer
        # and fails later, somewhere else.
        (tmp_path / "Test 9.mp4").touch()
        session = Session(tmp_path, "Test 9")
        assert session.pose_csv is None
        assert session.arena is None

    def test_a_pose_csv_with_no_sidecar_reports_none(self, tmp_path):
        # The VMHAHA failure exactly: the CSV is there, the sidecar is not, and
        # nothing downstream can compute a speed axis.
        (tmp_path / "Test 9.mp4").touch()
        _pose_csv(tmp_path / "Test 9DLC_exp-7.csv")
        (tmp_path / "Test 9DLC_exp-7.meta.json").unlink()
        session = Session(tmp_path, "Test 9")
        assert session.pose_csv is not None
        assert session.pose_meta is None

    def test_resolution_comes_back_from_the_sidecar(self, canonical):
        assert Session(canonical, "Test 1").resolution == (640, 480)

    def test_resolution_is_none_without_a_sidecar(self, tmp_path):
        (tmp_path / "Test 9.mp4").touch()
        _pose_csv(tmp_path / "Test 9DLC_exp-7.csv")
        (tmp_path / "Test 9DLC_exp-7.meta.json").unlink()
        assert Session(tmp_path, "Test 9").resolution is None


class TestIdentity:
    def test_sessions_compare_by_root_and_id(self, tmp_path):
        assert Session(tmp_path, "a") == Session(tmp_path, "a")
        assert Session(tmp_path, "a") != Session(tmp_path, "b")

    def test_it_is_hashable_so_it_can_key_a_map(self, tmp_path):
        assert len({Session(tmp_path, "a"), Session(tmp_path, "a")}) == 1

    def test_repr_names_the_session(self, tmp_path):
        assert "Test 1" in repr(Session(tmp_path, "Test 1"))


class TestRealWorldLayouts:
    """Two more shapes that exist in real cohorts today.

    Neither is the canonical layout, and neither should have to be reorganised
    before the data can be read.
    """

    def test_analysis_in_a_final_outputs_subfolder(self, tmp_path):
        # What the reorganised TRH cohort looks like.
        folder = tmp_path / "Test 1"
        (folder / "final_outputs").mkdir(parents=True)
        (folder / "Test 1.mp4").touch()
        (folder / "final_outputs" / "stats.csv").write_text("state,fraction\n")
        (folder / "final_outputs" / "run.json").write_text("{}")
        session = Session(tmp_path, "Test 1")
        assert session.stats is not None
        assert session.run_manifest is not None

    def test_video_in_a_media_subfolder_beside_the_analysis(self, tmp_path):
        # What VMHAHA looks like: analysis in <root>/<id>/, video in
        # <root>/males/.
        (tmp_path / "males").mkdir()
        (tmp_path / "males" / "Test 17.mp4").touch()
        _pose_csv(tmp_path / "males" / "Test 17DLC_train-6.csv")
        (tmp_path / "Test 17").mkdir()
        (tmp_path / "Test 17" / "stats.csv").write_text("state,fraction\n")
        session = Session(tmp_path, "Test 17")
        assert session.video is not None, "video is one level down, in males/"
        assert session.pose_csv is not None
        assert session.pose_meta is not None
        assert session.stats is not None

    def test_an_empty_analysis_folder_does_not_count_as_found(self, tmp_path):
        # analysis_dir must not name a folder just because it exists; a folder
        # with no outputs in it is not where the outputs are.
        (tmp_path / "Test 9.mp4").touch()
        assert Session(tmp_path, "Test 9").stats is None

    def test_a_subfolder_search_does_not_grab_another_session(self, tmp_path):
        (tmp_path / "males").mkdir()
        (tmp_path / "males" / "Test 17.mp4").touch()
        (tmp_path / "males" / "Test 23.mp4").touch()
        assert Session(tmp_path, "Test 17").video.name == "Test 17.mp4"
        assert Session(tmp_path, "Test 23").video.name == "Test 23.mp4"
