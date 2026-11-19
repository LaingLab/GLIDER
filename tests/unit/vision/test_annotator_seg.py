"""Animal Annotator instance segmentation: labels, project, export and SAM helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from glider.vision.training.dataset import export_dataset
from glider.vision.training.labels import (
    LabelFormatError,
    SegInstance,
    format_seg_line,
    parse_seg_line,
    read_seg_labels,
    write_seg_labels,
)
from glider.vision.training.project import (
    TASK_SEGMENT,
    FrameRecord,
    StudioProject,
    StudioProjectError,
)
from glider.vision.training.sam import SamSegmenter, mask_to_polygon

SQUARE = [(0.1, 0.1), (0.4, 0.1), (0.4, 0.5), (0.1, 0.5)]


@pytest.fixture
def project(tmp_path) -> StudioProject:
    return StudioProject.create(
        tmp_path / "seg", "Pups", [], task=TASK_SEGMENT, class_names=["dam", "pup"]
    )


def _add_frames(project: StudioProject, n: int, videos: int = 1) -> list[FrameRecord]:
    frames = []
    for i in range(n):
        frame = FrameRecord(f"v{i % videos}_f{i:06d}", f"v{i % videos}", i)
        cv2.imwrite(str(project.image_path(frame)), np.zeros((40, 60, 3), np.uint8))
        project.frames.append(frame)
        frames.append(frame)
    project.save()
    return frames


# ----------------------------------------------------------------------
# label lines
# ----------------------------------------------------------------------


def test_seg_line_round_trips():
    inst = SegInstance(SQUARE, class_id=1)
    line = format_seg_line(inst)
    assert line.split()[0] == "1"
    assert len(line.split()) == 1 + 2 * len(SQUARE)
    back = parse_seg_line(line)
    assert back.class_id == 1
    assert np.allclose(back.polygon, SQUARE)


def test_seg_line_needs_three_vertices():
    assert format_seg_line(SegInstance([(0.1, 0.1), (0.2, 0.2)])) is None
    with pytest.raises(LabelFormatError):
        parse_seg_line("0 0.1 0.1 0.2 0.2")
    with pytest.raises(LabelFormatError):
        parse_seg_line("0 0.1 0.1 0.2 0.2 0.3")  # odd coordinate count


def test_seg_line_clips_to_the_image():
    line = format_seg_line(SegInstance([(-0.1, 0.5), (1.2, 0.5), (0.5, 1.0)]))
    values = [float(v) for v in line.split()[1:]]
    assert min(values) == 0.0 and max(values) == 1.0


def test_writing_no_outlines_removes_the_file(tmp_path):
    path = tmp_path / "labels" / "a.txt"
    assert write_seg_labels(path, [SegInstance(SQUARE)])
    assert np.allclose(read_seg_labels(path)[0].polygon, SQUARE)
    assert not write_seg_labels(path, [SegInstance(SQUARE[:2])])
    assert not path.exists()
    assert read_seg_labels(path) == []


def test_seg_box():
    assert SegInstance(SQUARE).box() == pytest.approx((0.25, 0.3, 0.3, 0.4))


# ----------------------------------------------------------------------
# project
# ----------------------------------------------------------------------


def test_segment_project_round_trips(project):
    again = StudioProject.load(project.folder)
    assert again.is_segment
    assert again.class_names == ["dam", "pup"]
    assert again.keypoint_names == []


def test_project_reads_and_writes_outlines(project):
    (frame,) = _add_frames(project, 1)
    project.write_instances(frame, [SegInstance(SQUARE, 1)])
    assert project.is_labelled(frame)
    (inst,) = project.read_instances(frame)
    assert isinstance(inst, SegInstance) and inst.class_id == 1


def test_pose_projects_still_need_keypoints(tmp_path):
    with pytest.raises(StudioProjectError):
        StudioProject.create(tmp_path / "p", "x", [])
    with pytest.raises(StudioProjectError):
        StudioProject.create(tmp_path / "q", "x", [], task="detect")


def test_class_names_checked(tmp_path):
    with pytest.raises(StudioProjectError):
        StudioProject.create(tmp_path / "a", "x", [], task=TASK_SEGMENT, class_names=[])
    with pytest.raises(StudioProjectError):
        StudioProject.create(tmp_path / "b", "x", [], task=TASK_SEGMENT, class_names=["a", "a"])


def test_classes_can_be_renamed_and_added_but_not_removed_once_labelled(project):
    project.set_class_names(["dam"])  # nothing labelled yet: fine
    project.set_class_names(["dam", "pup"])
    (frame,) = _add_frames(project, 1)
    project.write_instances(frame, [SegInstance(SQUARE, 1)])
    project.set_class_names(["mother", "pup", "nest"])
    assert StudioProject.load(project.folder).class_names == ["mother", "pup", "nest"]
    with pytest.raises(StudioProjectError):
        project.set_class_names(["mother"])


def test_unknown_task_on_disk_is_refused(project):
    import json

    path = project.folder / "studio.json"
    data = json.loads(path.read_text())
    data["task"] = "obb"
    path.write_text(json.dumps(data))
    with pytest.raises(StudioProjectError):
        StudioProject.load(project.folder)


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------


def test_export_writes_a_segmentation_yaml(project):
    for frame in _add_frames(project, 4, videos=2):
        project.write_instances(frame, [SegInstance(SQUARE, 0)])
    result = export_dataset(project)
    text = result.data_yaml.read_text()
    assert "kpt_shape" not in text and "flip_idx" not in text and "kpt_names" not in text
    assert '0: "dam"' in text and '1: "pup"' in text
    assert len(result.train) + len(result.val) == 4


def test_exported_dataset_loads_in_ultralytics(project):
    """The label files and yaml are what Ultralytics' segment loader expects."""
    pytest.importorskip("ultralytics")
    from ultralytics.data.utils import verify_image_label

    for frame in _add_frames(project, 2, videos=2):
        project.write_instances(frame, [SegInstance(SQUARE, 1)])
    export_dataset(project)
    frame = project.frames[0]
    out = verify_image_label(
        (
            str(project.image_path(frame)),
            str(project.label_path(frame)),
            "",
            False,  # keypoint
            2,  # number of classes
            0,
            0,
            False,
        )
    )
    _im, lb, _shape, segments, _kpt, nm, nf, ne, nc, msg = out
    assert nf == 1 and nc == 0, msg
    assert len(segments) == 1 and int(lb[0][0]) == 1


# ----------------------------------------------------------------------
# SAM helpers
# ----------------------------------------------------------------------


def test_mask_to_polygon_traces_the_largest_region():
    mask = np.zeros((100, 120), bool)
    mask[20:60, 30:90] = True  # the animal
    mask[90:92, 5:7] = True  # a speck
    poly = mask_to_polygon(mask)
    assert len(poly) >= 4
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    assert min(xs) == pytest.approx(30, abs=1) and max(xs) == pytest.approx(89, abs=1)
    assert min(ys) == pytest.approx(20, abs=1) and max(ys) == pytest.approx(59, abs=1)
    assert len(poly) < 40  # simplified, not one vertex per boundary pixel


def test_mask_to_polygon_empty():
    assert mask_to_polygon(np.zeros((10, 10), bool)) == []


class _FakeTensor:
    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __len__(self):
        return len(self._array)

    def __getitem__(self, i):
        return _FakeTensor(self._array[i])

    def __gt__(self, other):
        return self._array > other


class _FakeMasks:
    def __init__(self, array):
        self.data = _FakeTensor(array)


class _FakeResult:
    def __init__(self, array):
        self.masks = _FakeMasks(array)


class _FakePredictor:
    def __init__(self, mask):
        self.mask = mask
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(self.mask[None].astype(np.float32))]


def test_segmenter_passes_one_object_of_several_points():
    mask = np.zeros((40, 60), bool)
    mask[10:20, 10:30] = True
    seg = SamSegmenter("mobile_sam.pt")
    fake = _FakePredictor(mask)
    seg._predictor, seg._image_key, seg._image_shape = fake, "f", (40, 60)
    out = seg.segment([(15, 15), (50, 5)], [1, 0], box=(5, 5, 35, 25))
    assert out.shape == (40, 60) and out.sum() == mask.sum()
    call = fake.calls[0]
    assert call["points"] == [[[15, 15], [50, 5]]]  # one object, two clicks
    assert call["labels"] == [[1, 0]]
    assert call["bboxes"] == [[5, 5, 35, 25]]


def test_segmenter_without_a_prompt_returns_nothing():
    seg = SamSegmenter()
    seg._predictor, seg._image_key, seg._image_shape = _FakePredictor(None), "f", (4, 4)
    assert seg.segment([], [], None) is None


def test_segmenter_needs_an_image():
    with pytest.raises(RuntimeError):
        SamSegmenter().segment([(1, 1)], [1])


def test_bare_model_names_go_in_the_glider_weights_folder(monkeypatch, tmp_path):
    import glider.vision.training.sam as sam

    monkeypatch.setattr(sam, "sam_weights_dir", lambda: tmp_path / "sam")
    assert SamSegmenter("sam2.1_t.pt")._weights_path() == str(tmp_path / "sam" / "sam2.1_t.pt")
    custom = tmp_path / "mine.pt"
    assert SamSegmenter(str(custom))._weights_path() == str(custom)
