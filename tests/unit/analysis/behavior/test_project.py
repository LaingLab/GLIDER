"""Tests for the project manifest (paths + settings)."""

from __future__ import annotations

import pytest

pytest.importorskip("yaml")


def _write(path, text):
    path.write_text(text)
    return path


def test_load_minimal_applies_defaults(tmp_path):
    from glider.analysis.behavior.project import Project

    (tmp_path / "videos").mkdir()
    p = _write(tmp_path / "project.yaml", "videos_dir: videos\n")
    proj = Project.load(p)
    assert proj.videos_dir == (tmp_path / "videos").resolve()
    # poses_dir defaults to videos_dir.
    assert proj.poses_dir == (tmp_path / "videos").resolve()
    assert proj.fps == 30.0
    assert proj.window == 30
    assert proj.body_axis == (0, -1)
    assert proj.vocab is None
    assert proj.holdout == []
    assert proj.merge == {}


def test_load_resolves_paths_relative_to_file(tmp_path):
    from glider.analysis.behavior.project import Project

    sub = tmp_path / "proj"
    (sub / "vids").mkdir(parents=True)
    (sub / "ps").mkdir()
    p = _write(
        sub / "project.yaml",
        "videos_dir: vids\nposes_dir: ps\nvocab: behaviors.yaml\n",
    )
    proj = Project.load(p)
    assert proj.videos_dir == (sub / "vids").resolve()
    assert proj.poses_dir == (sub / "ps").resolve()
    assert proj.vocab == (sub / "behaviors.yaml").resolve()


def test_resolve_sessions_pairs_by_stem(tmp_path):
    from glider.analysis.behavior.project import Project

    vids = tmp_path / "videos"
    poses = tmp_path / "poses"
    vids.mkdir()
    poses.mkdir()
    for stem in ("t1_d2", "t2_d2"):
        (vids / f"{stem}.mp4").write_bytes(b"")
        (poses / f"{stem}.csv").write_text("x")
    p = _write(tmp_path / "project.yaml", "videos_dir: videos\nposes_dir: poses\n")
    proj = Project.load(p)
    sessions = proj.resolve_sessions()
    assert [v.stem for v, _pose in sessions] == ["t1_d2", "t2_d2"]
    assert all(pose == poses.resolve() / f"{v.stem}.csv" for v, pose in sessions)


def test_resolve_holdout_maps_stems(tmp_path):
    from glider.analysis.behavior.project import Project

    vids = tmp_path / "videos"
    poses = tmp_path / "poses"
    vids.mkdir()
    poses.mkdir()
    for stem in ("t1_d2", "t2_d2", "t3_d2"):
        (vids / f"{stem}.mp4").write_bytes(b"")
        (poses / f"{stem}.csv").write_text("x")
    p = _write(
        tmp_path / "project.yaml",
        "videos_dir: videos\nposes_dir: poses\nholdout: [t2_d2]\n",
    )
    proj = Project.load(p)
    holdout = proj.resolve_holdout()
    assert holdout == [poses.resolve() / "t2_d2.csv"]


def test_resolve_holdout_unknown_stem_raises(tmp_path):
    from glider.analysis.behavior.project import Project, ProjectError

    vids = tmp_path / "videos"
    poses = tmp_path / "poses"
    vids.mkdir()
    poses.mkdir()
    (vids / "t1_d2.mp4").write_bytes(b"")
    (poses / "t1_d2.csv").write_text("x")
    p = _write(
        tmp_path / "project.yaml",
        "videos_dir: videos\nposes_dir: poses\nholdout: [nope]\n",
    )
    proj = Project.load(p)
    with pytest.raises(ProjectError):
        proj.resolve_holdout()


def test_merge_specs_from_mapping(tmp_path):
    from glider.analysis.behavior.project import Project

    (tmp_path / "videos").mkdir()
    p = _write(
        tmp_path / "project.yaml",
        "videos_dir: videos\n"
        "merge:\n"
        "  groom: [grooming, flank groom]\n"
        "  explore: [sniff, rearing]\n",
    )
    proj = Project.load(p)
    specs = sorted(proj.merge_specs())
    assert specs == ["explore=sniff,rearing", "groom=grooming,flank groom"]


def test_body_axis_parsed_from_list(tmp_path):
    from glider.analysis.behavior.project import Project

    (tmp_path / "videos").mkdir()
    p = _write(tmp_path / "project.yaml", "videos_dir: videos\nbody_axis: [0, 4]\n")
    proj = Project.load(p)
    assert proj.body_axis == (0, 4)


def test_load_missing_videos_dir_key_raises(tmp_path):
    from glider.analysis.behavior.project import Project, ProjectError

    p = _write(tmp_path / "project.yaml", "fps: 30\n")
    with pytest.raises(ProjectError):
        Project.load(p)
