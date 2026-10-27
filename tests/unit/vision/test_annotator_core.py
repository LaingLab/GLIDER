"""Animal Annotator core: labels, project file, frame extraction and dataset export (#224)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from glider.analysis.behavior.keypoint_schema import KeypointSchema
from glider.vision.training.dataset import DatasetError, export_dataset, flip_idx, split_frames
from glider.vision.training.frames import (
    ExtractionCancelledError,
    diverse_indices,
    extract_frames,
    uniform_indices,
)
from glider.vision.training.labels import (
    HIDDEN,
    MISSING,
    VISIBLE,
    LabelFormatError,
    PoseInstance,
    format_pose_line,
    parse_pose_line,
    read_labels,
    write_labels,
)
from glider.vision.training.project import (
    FrameRecord,
    StudioProject,
    StudioProjectError,
    flip_pairs_from_names,
)

MOUSE = KeypointSchema.default_mouse().names


def _clip(path: Path, frames: int = 60, size=(64, 48)) -> Path:
    """A clip whose square jumps between a few distinct places."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot open an MJPG writer")
    w, h = size
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = (i // 10) * 8  # six distinct positions, ten frames each
        cv2.rectangle(frame, (x, 10), (x + 12, 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def project(tmp_path) -> StudioProject:
    return StudioProject.create(tmp_path / "proj", "Mice", MOUSE)


# ----------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------


def test_pose_line_round_trips():
    inst = PoseInstance([(0.5, 0.25, VISIBLE), (0.0, 0.0, MISSING), (0.6, 0.3, HIDDEN)])
    line = format_pose_line(inst, padding=0.0)
    back = parse_pose_line(line, 3)
    assert back.class_id == 0
    assert back.keypoints[1] == (0.0, 0.0, MISSING)
    assert back.keypoints[2][2] == HIDDEN
    assert back.keypoints[0][:2] == pytest.approx((0.5, 0.25))


def test_box_wraps_the_points_with_padding_and_stays_on_the_image():
    inst = PoseInstance([(0.4, 0.4, VISIBLE), (0.6, 0.5, VISIBLE), (0.0, 0.0, MISSING)])
    cx, cy, w, h = inst.box(padding=0.25)
    assert (cx, cy) == pytest.approx((0.5, 0.45))
    assert w == pytest.approx(0.2 + 2 * 0.05)  # pad = 0.25 * longer side (0.2)
    cx, cy, w, h = PoseInstance([(0.0, 0.0, VISIBLE), (0.1, 0.1, VISIBLE)]).box(0.5)
    assert cx - w / 2 >= 0 and cy - h / 2 >= 0


def test_a_single_point_still_gets_a_usable_box():
    _cx, _cy, w, h = PoseInstance([(0.5, 0.5, VISIBLE)]).box()
    assert w > 0 and h > 0


def test_wrong_value_count_is_refused():
    with pytest.raises(LabelFormatError):
        parse_pose_line("0 0.5 0.5 0.1 0.1 0.5 0.5 2", 2)


def test_writing_no_placed_points_removes_the_label_file(tmp_path):
    path = tmp_path / "a.txt"
    assert write_labels(path, [PoseInstance([(0.5, 0.5, VISIBLE)])])
    assert path.exists()
    assert not write_labels(path, [PoseInstance.empty(1)])
    assert not path.exists()
    assert read_labels(path, 1) == []


def test_several_animals_per_image(tmp_path):
    path = tmp_path / "a.txt"
    a = PoseInstance([(0.2, 0.2, VISIBLE)])
    b = PoseInstance([(0.8, 0.8, VISIBLE)])
    write_labels(path, [a, b])
    assert len(read_labels(path, 1)) == 2


# ----------------------------------------------------------------------
# project
# ----------------------------------------------------------------------


def test_project_round_trips(project):
    project.frames.append(FrameRecord("v_f000001", "v", 1, skipped=True))
    project.save()
    again = StudioProject.load(project.folder)
    assert again.keypoint_names == MOUSE
    assert again.frames[0].skipped


def test_create_refuses_an_existing_project(project):
    with pytest.raises(StudioProjectError):
        StudioProject.create(project.folder, "again", MOUSE)


def test_create_refuses_duplicate_names(tmp_path):
    with pytest.raises(StudioProjectError):
        StudioProject.create(tmp_path / "p", "x", ["nose", "nose"])


def test_load_refuses_a_future_version(project):
    path = project.folder / "studio.json"
    data = json.loads(path.read_text())
    data["studio_version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(StudioProjectError):
        StudioProject.load(project.folder)


@pytest.mark.parametrize(
    "names, pair",
    [
        (["left_ear", "nose", "right_ear"], (0, 2)),
        (["ear_left", "ear_right"], (0, 1)),
        (["L_Paw", "R_Paw"], (0, 1)),
    ],
)
def test_flip_pairs_found_from_names(names, pair):
    assert flip_pairs_from_names(names) == [pair]


def test_flip_idx_swaps_pairs():
    assert flip_idx(4, [(0, 2)]) == [2, 1, 0, 3]


def test_same_stem_videos_get_distinct_keys(project, tmp_path):
    a = project.video_key(tmp_path / "one" / "s1.mp4")
    b = project.video_key(tmp_path / "two" / "s1.mp4")
    assert a == "s1" and b == "s1-2"
    assert project.video_key(tmp_path / "one" / "s1.mp4") == "s1"


def test_counts(project):
    project.frames = [FrameRecord(f"v_f{i:06d}", "v", i) for i in range(3)]
    project.frames[0].skipped = True
    project.write_instances(project.frames[1], [PoseInstance([(0.5, 0.5, VISIBLE)] * 7)])
    assert project.counts() == {"total": 3, "labelled": 1, "skipped": 1, "todo": 1}


# ----------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------


def test_uniform_indices_are_spread_and_in_range():
    idx = uniform_indices(100, 4)
    assert idx == [12, 37, 62, 87]
    assert uniform_indices(3, 10) == [0, 1, 2]
    assert uniform_indices(0, 5) == []


def test_diverse_prefers_frames_that_differ():
    thumbs = np.array([[0.0], [0.01], [0.02], [10.0]])
    picked = diverse_indices(thumbs, [0, 1, 2, 3], 2, seed=0)
    assert 3 in picked


@pytest.mark.parametrize("method", ["uniform", "diverse"])
def test_extract_writes_images_and_tops_up(project, tmp_path, method):
    video = _clip(tmp_path / "s1.avi")
    added = extract_frames(project, video, 5, method)
    assert len(added) == 5
    assert all(project.image_path(f).exists() for f in added)
    more = extract_frames(project, video, 5, "uniform")
    names = [f.name for f in project.frames]
    assert len(names) == len(set(names))  # nothing extracted twice
    assert StudioProject.load(project.folder).frames  # saved
    assert len(more) <= 5


def test_diverse_extraction_covers_the_distinct_scenes(project, tmp_path):
    video = _clip(tmp_path / "s1.avi")
    added = extract_frames(project, video, 6, "diverse")
    scenes = {f.frame // 10 for f in added}
    assert len(scenes) >= 5  # six distinct positions; near-duplicates avoided


def test_extract_can_be_cancelled_and_keeps_what_it_wrote(project, tmp_path):
    video = _clip(tmp_path / "s1.avi")
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(ExtractionCancelledError):
        extract_frames(project, video, 5, "uniform", cancel_cb=cancel)
    assert len(StudioProject.load(project.folder).frames) == 2


def test_extract_reports_a_missing_video(project, tmp_path):
    with pytest.raises(OSError):
        extract_frames(project, tmp_path / "nope.avi", 3)


def test_remove_video_deletes_its_files(project, tmp_path):
    video = _clip(tmp_path / "s1.avi")
    added = extract_frames(project, video, 3, "uniform")
    assert project.remove_video("s1") == 3
    assert not any(project.image_path(f).exists() for f in added)
    assert project.frames == [] and project.videos == {}


# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------


def _label_all(project, frames):
    for f in frames:
        project.write_instances(f, [PoseInstance([(0.5, 0.5, VISIBLE)] * len(MOUSE))])


def test_split_by_video_holds_whole_videos_out():
    frames = [FrameRecord(f"{v}_{i}", v, i) for v in "abcde" for i in range(10)]
    train, val, by_video = split_frames(frames, 0.2, seed=1)
    assert by_video
    assert {f.video for f in train}.isdisjoint({f.video for f in val})
    assert len(val) == 10


def test_split_with_one_video_falls_back_to_frames():
    frames = [FrameRecord(f"a_{i}", "a", i) for i in range(10)]
    train, val, by_video = split_frames(frames, 0.2, seed=1)
    assert not by_video
    assert len(val) == 2 and len(train) == 8


def test_split_never_holds_out_every_video():
    frames = [FrameRecord("a_0", "a", 0), FrameRecord("b_0", "b", 0)]
    train, val, _ = split_frames(frames, 0.9, seed=0)
    assert train and val


def test_export_needs_labels(project):
    with pytest.raises(DatasetError):
        export_dataset(project)


def test_export_writes_lists_and_yaml(project, tmp_path):
    a = extract_frames(project, _clip(tmp_path / "a.avi"), 3, "uniform")
    b = extract_frames(project, _clip(tmp_path / "b.avi"), 3, "uniform")
    _label_all(project, a + b)
    project.frames[0].skipped = True
    result = export_dataset(project)
    assert result.split_by_video
    listed = (project.folder / "train.txt").read_text().splitlines()
    listed += (project.folder / "val.txt").read_text().splitlines()
    assert len(listed) == 5  # the skipped frame is left out
    text = result.data_yaml.read_text()
    assert "kpt_shape: [7, 3]" in text
    left, right = MOUSE.index("left_ear"), MOUSE.index("right_ear")
    assert f"flip_idx: {json.dumps(flip_idx(7, [(left, right)] + [(4, 5)]))}" in text


def test_ultralytics_accepts_the_exported_dataset(project, tmp_path):
    utils = pytest.importorskip("ultralytics.data.utils")
    a = extract_frames(project, _clip(tmp_path / "a.avi"), 3, "uniform")
    b = extract_frames(project, _clip(tmp_path / "b.avi"), 3, "uniform")
    _label_all(project, a + b)
    result = export_dataset(project)
    data = utils.check_det_dataset(str(result.data_yaml), autodownload=False)
    assert data["kpt_shape"] == [7, 3]
    assert data["names"] == {0: "animal"}
    assert data["kpt_names"][0] == MOUSE
