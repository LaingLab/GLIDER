from __future__ import annotations

import numpy as np
import pytest

from glider.vision.pose import PoseData, core, pose_from_array

from .conftest import FakeResult, fake_yolo_streaming


def test_posedata_validates_shape(kpt_names):
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((10, 4, 2)),  # 4 keypoints, but 5 names
            confidence=np.zeros((10, 4)),
            keypoint_names=kpt_names,
        )


def test_posedata_validates_confidence_shape(kpt_names):
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((10, 5, 2)),
            confidence=np.zeros((10, 4)),  # mismatched
            keypoint_names=kpt_names,
        )


def test_posedata_rejects_duplicate_names():
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((3, 2, 2)),
            confidence=np.zeros((3, 2)),
            keypoint_names=["a", "a"],
        )


def test_posedata_rejects_bad_xy_dim(kpt_names):
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((10, 5)),  # 2D not 3D
            confidence=np.zeros((10, 5)),
            keypoint_names=kpt_names,
        )


def test_posedata_rejects_bad_fps(kpt_names):
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((10, 5, 2)),
            confidence=np.zeros((10, 5)),
            keypoint_names=kpt_names,
            fps=0.0,
        )


def test_posedata_properties(synthetic_pose):
    assert synthetic_pose.n_frames == 200
    assert synthetic_pose.n_keypoints == 5


def test_posedata_kp_index(synthetic_pose):
    assert synthetic_pose.kp_index("snout") == 0
    with pytest.raises(KeyError):
        synthetic_pose.kp_index("nonexistent")


def test_posedata_slice(synthetic_pose):
    sliced = synthetic_pose.slice_frames(50, 100)
    assert sliced.n_frames == 50
    np.testing.assert_array_equal(sliced.xy, synthetic_pose.xy[50:100])


def test_posedata_copy_is_deep(synthetic_pose):
    cp = synthetic_pose.copy()
    cp.xy[0, 0, 0] = 9999
    assert synthetic_pose.xy[0, 0, 0] != 9999


def test_pose_from_array(kpt_names):
    xy = np.zeros((5, len(kpt_names), 2))
    cf = np.ones((5, len(kpt_names)))
    p = pose_from_array(xy, cf, kpt_names, fps=60.0, source="test")
    assert p.fps == 60.0
    assert p.source == "test"


# ---------------------------------------------------------------------------
# Arena-aware candidate selection
#
# infer_video used to keep argmax(boxes.conf) and throw the rest away, so a
# blob on the bench floor at 0.90 beat the real mouse at 0.85 and the correct
# detection never reached the CSV — no post-hoc pass can recover what was
# never written. These pin the re-ranking, and its argmax fallback, which is
# what keeps it a re-ranking rather than a filter.
# ---------------------------------------------------------------------------

ARENA_NAMES = ["a", "b", "c", "d"]

# The fronto-parallel square from tests/unit/vision/test_arena_gate.py: a real
# 400x400 px square in a 640x480 frame (13.3 px/cm on a 30 cm floor).
_L, _R, _T, _B = 120 / 640, 520 / 640, 40 / 480, 440 / 480
SQUARE = [(_L, _T), (_R, _T), (_R, _B), (_L, _B)]

INSIDE_PX = (320.0, 240.0)  # arena centre
BENCH_FLOOR_PX = (-900.0, -900.0)
FURTHER_OUT_PX = (-4000.0, -4000.0)
#: What Ultralytics writes for a keypoint it did not localize: a finite pixel
#: in the frame's top-left corner, at confidence 0. It sits 9 cm left of the
#: arena, past the default 7.5 cm margin, so it scores as *outside*.
PAD_PX = (0.0, 0.0)


def _arena():
    from glider.vision.arena import ArenaCalibration

    return ArenaCalibration(corners=SQUARE, width_cm=30.0, height_cm=30.0, frame_size=(640, 480))


def _is_inside(point) -> bool:
    x, y = point
    return 120.0 <= x <= 520.0 and 40.0 <= y <= 440.0


def _four(point):
    return [list(point)] * len(ARENA_NAMES)


def _inside_arena():
    return _four(INSIDE_PX)


def _on_the_bench_floor():
    return _four(BENCH_FLOOR_PX)


def _far_outside():
    return _four(BENCH_FLOOR_PX)


def _further_outside():
    return _four(FURTHER_OUT_PX)


def _inside_with_pads():
    """One localized keypoint in the arena; the rest are (0, 0) pads.

    Three pads, not the two the plan sketched: two of four scores exactly 0.5
    under xy-only judging, which clears ``min_inside_fraction`` anyway and so
    would leave the trap untested.
    """
    return [list(INSIDE_PX), list(PAD_PX), list(PAD_PX), list(PAD_PX)]


def _fake_result(*, boxes_conf, keypoints, keypoint_conf):
    return FakeResult(
        np.asarray(keypoints, dtype=float),
        boxes_conf=np.asarray(boxes_conf, dtype=float),
        keypoint_conf=np.asarray(keypoint_conf, dtype=float),
    )


def _infer_with(stub, tmp_path, result, **kwargs):
    stub.YOLO = fake_yolo_streaming([result])
    return core.infer_video(
        tmp_path / "model.pt",
        tmp_path / "video.mp4",
        ARENA_NAMES,
        progress=False,
        echo_device=False,
        **kwargs,
    )


@pytest.fixture
def sized_video(monkeypatch):
    """The stubbed video has no header, so hand the loop a real frame size."""
    import glider.vision.video_source as video_source

    monkeypatch.setattr(video_source, "video_resolution", lambda path: (640, 480))


def test_an_in_arena_candidate_beats_a_more_confident_outsider(
    stub_ultralytics, sized_video, tmp_path
):
    result = _fake_result(
        boxes_conf=[0.85, 0.90],
        keypoints=[_inside_arena(), _on_the_bench_floor()],
        keypoint_conf=[[0.9] * 4, [0.9] * 4],
    )
    pose = _infer_with(stub_ultralytics, tmp_path, result, arena=_arena())
    assert _is_inside(pose.xy[0, 0])


def test_it_falls_back_to_argmax_when_none_are_inside(stub_ultralytics, sized_video, tmp_path):
    """Never turns a frame that had a usable detection into a dropout."""
    result = _fake_result(
        boxes_conf=[0.4, 0.9],
        keypoints=[_far_outside(), _further_outside()],
        keypoint_conf=[[0.9] * 4, [0.9] * 4],
    )
    pose = _infer_with(stub_ultralytics, tmp_path, result, arena=_arena())
    np.testing.assert_allclose(pose.xy[0], _further_outside())


def test_a_padded_but_correct_detection_still_wins(stub_ultralytics, sized_video, tmp_path):
    """The (0, 0) trap at selection time: judging on xy alone scores a good
    detection with three pads at 1/4 and hands the frame to the blob."""
    result = _fake_result(
        boxes_conf=[0.85, 0.90],
        keypoints=[_inside_with_pads(), _on_the_bench_floor()],
        keypoint_conf=[[0.9, 0.0, 0.0, 0.0], [0.9] * 4],
    )
    pose = _infer_with(stub_ultralytics, tmp_path, result, arena=_arena())
    assert _is_inside(pose.xy[0, 0])


def test_without_an_arena_the_result_is_unchanged(stub_ultralytics, sized_video, tmp_path):
    """The guard that this cannot silently move existing results."""

    def run(**kwargs):
        result = _fake_result(
            boxes_conf=[0.85, 0.90],
            keypoints=[_inside_arena(), _on_the_bench_floor()],
            keypoint_conf=[[0.9] * 4, [0.9] * 4],
        )
        return _infer_with(stub_ultralytics, tmp_path, result, **kwargs)

    np.testing.assert_array_equal(run(arena=None).xy, run().xy)


def test_an_arena_without_settings_does_not_crash(stub_ultralytics, sized_video, tmp_path):
    """run_batch's `gating = gate is not None and arena is not None` means the
    arena can arrive without settings."""
    result = _fake_result(
        boxes_conf=[0.9],
        keypoints=[_inside_arena()],
        keypoint_conf=[[0.9] * 4],
    )
    _infer_with(stub_ultralytics, tmp_path, result, arena=_arena(), gate_settings=None)


def test_the_supplied_gate_settings_are_the_ones_applied(stub_ultralytics, sized_video, tmp_path):
    """Otherwise a hard-coded ``ArenaGateSettings()`` would pass every other
    test here, and inference would re-rank under a margin the gate does not
    share — the drift inside_fraction was factored out to prevent."""
    from glider.vision.arena_gate import ArenaGateSettings

    def run(**kwargs):
        result = _fake_result(
            # A rear 3 cm past the left wall: inside the default 7.5 cm margin,
            # outside a 1 cm one.
            boxes_conf=[0.85, 0.90],
            keypoints=[_four((80.0, 240.0)), _on_the_bench_floor()],
            keypoint_conf=[[0.9] * 4, [0.9] * 4],
        )
        return _infer_with(stub_ultralytics, tmp_path, result, arena=_arena(), **kwargs)

    np.testing.assert_allclose(run().xy[0], _four((80.0, 240.0)))
    np.testing.assert_allclose(
        run(gate_settings=ArenaGateSettings(margin_cm=1.0)).xy[0], _on_the_bench_floor()
    )


def test_an_unreadable_frame_size_falls_back_to_the_arenas_own(stub_ultralytics, tmp_path):
    """video_resolution returns None for a header it cannot read. Re-ranking a
    candidate must never be the thing that kills a multi-hour batch."""
    result = _fake_result(
        boxes_conf=[0.85, 0.90],
        keypoints=[_inside_arena(), _on_the_bench_floor()],
        keypoint_conf=[[0.9] * 4, [0.9] * 4],
    )
    pose = _infer_with(stub_ultralytics, tmp_path, result, arena=_arena())
    assert _is_inside(pose.xy[0, 0])


def test_the_backend_path_logs_the_no_op(stub_ultralytics, caplog, tmp_path, monkeypatch):
    """_infer_video_backend yields one detection per frame — nothing to
    re-rank. A documented no-op, logged so it is not read as a silent failure."""
    from glider.vision.pose import spec as spec_mod

    class _Spec:
        kind = "dlc"

    monkeypatch.setattr(spec_mod, "identify_pose_model", lambda path: _Spec())
    monkeypatch.setattr(
        core,
        "_infer_video_backend",
        lambda spec, video_path, **kw: pose_from_array(
            np.zeros((1, len(ARENA_NAMES), 2)), np.ones((1, len(ARENA_NAMES))), ARENA_NAMES
        ),
    )
    with caplog.at_level("INFO"):
        core.infer_video(
            tmp_path / "model",
            tmp_path / "video.mp4",
            ARENA_NAMES,
            progress=False,
            echo_device=False,
            arena=_arena(),
        )
    assert "no candidates to re-rank" in caplog.text
