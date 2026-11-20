"""Stored train/val/test splits on every frame."""

from __future__ import annotations

import json

import pytest

from glider.vision.training.dataset import split_frames
from glider.vision.training.frames import extract_frames
from glider.vision.training.project import FrameRecord, StudioProject
from glider.vision.training.splits import (
    assign_splits,
    ensure_splits,
    leaky_videos,
    split_counts,
)
from tests.unit.vision.test_annotator_core import MOUSE, _clip, _label_all


@pytest.fixture
def project(tmp_path) -> StudioProject:
    return StudioProject.create(tmp_path / "proj", "Mice", MOUSE)


def _frames(video: str, n: int, start: int = 0) -> list[FrameRecord]:
    return [FrameRecord(f"{video}_{i}", video, i) for i in range(start, start + n)]


def _add(project, video: str, n: int, start: int = 0) -> list[FrameRecord]:
    project.videos[video] = f"/v/{video}.avi"
    new = _frames(video, n, start)
    project.frames.extend(new)
    assign_splits(project, new)
    return new


def test_split_round_trips(project):
    project.frames = [FrameRecord("a_0", "a", 0, split="val"), FrameRecord("a_1", "a", 1)]
    project.save()
    back = StudioProject.load(project.folder)
    assert [f.split for f in back.frames] == ["val", ""]
    assert "split" not in project.frames[1].to_dict()
    assert (
        FrameRecord.from_dict({"name": "x", "video": "a", "frame": 0, "split": "bogus"}).split == ""
    )


def test_one_video_splits_per_frame(project):
    a = _add(project, "a", 200)
    counts = split_counts(project)
    assert all(counts[s] for s in counts)
    assert 0.6 < counts["train"] / 200 < 0.8
    other = StudioProject.create(project.folder.parent / "p2", "Mice", MOUSE)
    b = _add(other, "a", 200)
    assert [f.split for f in a] == [f.split for f in b]


def test_second_video_goes_to_val_whole(project):
    a = _add(project, "a", 100)
    before = [f.split for f in a]
    b = _add(project, "b", 10)
    assert {f.split for f in b} == {"val"}
    assert [f.split for f in a] == before


def test_third_video_goes_to_train(project):
    _add(project, "a", 100)
    _add(project, "b", 10)
    assert {f.split for f in _add(project, "c", 10)} == {"train"}


def test_top_up_joins_its_video(project):
    _add(project, "a", 100)
    _add(project, "b", 10)
    assert {f.split for f in _add(project, "b", 5, start=10)} == {"val"}


def test_extract_frames_assigns_splits(project, tmp_path):
    extract_frames(project, _clip(tmp_path / "a.avi"), 6, "uniform")
    b = extract_frames(project, _clip(tmp_path / "b.avi"), 4, "uniform")
    assert all(f.split for f in project.frames)
    assert {f.split for f in b} == {"val"}
    saved = json.loads((project.folder / "studio.json").read_text())
    assert all(f["split"] for f in saved["frames"])


def _migrate_project(project):
    frames = []
    for v in "abcde":
        project.videos[v] = f"/v/{v}.avi"
        frames += _frames(v, 10)
    project.frames = list(frames)
    _label_all(project, frames)
    return frames


def test_migration_keeps_todays_export_split(project):
    frames = _migrate_project(project)
    expected = split_frames(frames, project.val_fraction, project.seed)
    assert ensure_splits(project) is True
    val = {f.name for f in expected[1]}
    assert all(f.split == ("val" if f.name in val else "train") for f in frames)
    assert ensure_splits(project) is False


def test_migration_unlabelled_frames_follow_their_video(project):
    frames = _migrate_project(project)
    extra = _frames("a", 3, start=10) + _frames("b", 3, start=10)
    project.frames += extra
    expected = split_frames(frames, project.val_fraction, project.seed)
    held = {f.video for f in expected[1]}
    ensure_splits(project)
    for f in extra:
        assert f.split == ("val" if f.video in held else "train")


def test_leaky_videos(project):
    _add(project, "a", 200)
    assert leaky_videos(project) == ["a"]
    for f in project.frames:
        f.split = "train"
    assert leaky_videos(project) == []
