"""Telling the animal from a small object the model also calls an animal."""

from __future__ import annotations

import numpy as np

from glider.vision.pose import core
from glider.vision.pose.body_size import body_span, gate_body_size, pick_by_size

from .conftest import FakeResult, fake_yolo_streaming

NAMES = ["nose", "body", "tail"]
RAT = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])  # 200 px long
DISH = np.array([[0.0, 0.0], [20.0, 10.0], [40.0, 0.0]])  # ~40 px across


def _at(shape, x, y):
    return shape + [x, y]


def test_body_span_is_the_bounding_box_diagonal():
    assert body_span(_at(RAT, 50, 50)) == 200.0
    assert np.isclose(body_span(np.array([[0, 0], [30, 0], [30, 40]], dtype=float) + 1), 50.0)


def test_unplaced_keypoints_do_not_count():
    """Ultralytics writes a keypoint it could not place as (0, 0)."""
    xy = np.array([[0.0, 0.0], [100.0, 100.0], [110.0, 100.0]])
    assert body_span(xy) == 10.0
    assert np.isnan(body_span(np.array([[0.0, 0.0], [5.0, 5.0], [np.nan, np.nan]])))


def test_a_confident_object_gives_way_to_the_animal():
    rat, dish = _at(RAT, 300, 300), _at(DISH, 600, 300)
    frames = [[rat]] * 20 + [[dish, rat]] * 5 + [[dish]] * 3 + [[]]
    chosen, reference = pick_by_size(frames)
    assert reference == 200.0
    assert chosen[20:25] == [1] * 5  # the rat, though the dish ranked first
    assert chosen[25:28] == [0] * 3  # nothing better: re-ranked, never dropped
    assert chosen[:20] == [0] * 20 and chosen[28] == 0


def test_the_gate_blanks_only_frames_too_small_to_be_the_animal(synthetic_pose):
    pose = synthetic_pose.copy()
    rest = pose.xy[10].copy()
    pose.xy[10] = rest[0] + (rest - rest[0]) * 0.1  # shrunk to a tenth
    gated, report = gate_body_size(pose)
    assert report.frames_blanked == 1
    assert np.isnan(gated.xy[10]).all() and (gated.confidence[10] == 0).all()
    assert np.isfinite(gated.xy[np.arange(len(gated.xy)) != 10]).all()
    assert np.isfinite(pose.xy[10]).all()  # the input is untouched


def _infer(tmp_path, stub_ultralytics, results, **kwargs):
    stub_ultralytics.YOLO = fake_yolo_streaming(results)
    return core.infer_video(
        tmp_path / "model.pt",
        tmp_path / "video.mp4",
        NAMES,
        progress=False,
        echo_device=False,
        **kwargs,
    )


def test_infer_video_keeps_the_animal_over_a_more_confident_object(stub_ultralytics, tmp_path):
    rat, dish = _at(RAT, 300, 300), _at(DISH, 600, 300)
    both = FakeResult(np.stack([dish, rat]), boxes_conf=[0.9, 0.6])
    results = [FakeResult(rat[None], boxes_conf=[0.9])] * 10 + [both] * 3
    pose = _infer(tmp_path, stub_ultralytics, results)
    assert np.allclose(pose.xy[10:], rat)
    assert pose.metadata["body_size"]["frames_reranked"] == 3

    by_confidence = _infer(tmp_path, stub_ultralytics, results, min_body_fraction=None)
    assert np.allclose(by_confidence.xy[10:], dish)
    assert "body_size" not in by_confidence.metadata


def test_run_batch_blanks_the_objects_and_keeps_them_in_raw(tmp_path, monkeypatch):
    import glider.vision.pose.device as device_mod
    from glider.vision.pose import batch
    from glider.vision.pose.dlc import from_dlc_csv, read_pose_meta

    monkeypatch.setattr(device_mod, "resolve_device", lambda d, require_gpu=False: "cpu")
    video = tmp_path / "rat.avi"
    video.write_bytes(b"")
    xy = np.stack([_at(DISH, 600, 300)] * 5 + [_at(RAT, 300, 300)] * 20)

    def infer(**kwargs):
        return core.PoseData(xy=xy, confidence=np.ones(xy.shape[:2]), keypoint_names=NAMES)

    batch.run_batch([video], tmp_path / "rat-1.pt", NAMES, infer=infer)
    primary = from_dlc_csv(batch.dlc_output_path(video, tmp_path / "rat-1.pt"))
    raw = from_dlc_csv(batch.raw_output_path(video, tmp_path / "rat-1.pt"))
    assert np.isnan(primary.xy[:5]).all() and np.isfinite(primary.xy[5:]).all()
    assert np.allclose(raw.xy, xy)
    meta = read_pose_meta(batch.dlc_output_path(video, tmp_path / "rat-1.pt"))
    assert meta["body_size"]["frames_blanked"] == 5
