"""Applying from an existing pose CSV instead of re-running the pose model.

Batch Pose Tracking has usually already produced these. Re-deriving them is
the single biggest avoidable cost in an apply run.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from glider.analysis.behavior.classify.threads import END_OF_STREAM, PoseReplay

NAMES = ["nose", "l_ear", "r_ear"]


def _pose_csv(path, n=6):
    """A CSV whose frame N carries the value N, so ordering is checkable."""
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    xy = np.stack([np.full((3, 2), float(i)) for i in range(n)])
    to_dlc_csv(PoseData(xy=xy, confidence=np.ones((n, 3)), keypoint_names=NAMES, fps=30.0), path)
    return path


def _drain(csv_path, frame_indices, names=NAMES):
    raw: queue.Queue = queue.Queue()
    tracked: queue.Queue = queue.Queue()
    display: queue.Queue = queue.Queue()
    replay = PoseReplay(raw, tracked, display, threading.Event(), csv_path, names)
    for i in frame_indices:
        raw.put((i, None))
    raw.put(END_OF_STREAM)
    replay.start()
    replay.join(timeout=10)

    out = []
    while not tracked.empty():
        item = tracked.get()
        if item is not END_OF_STREAM:
            out.append(item)
    return replay, out


def test_poses_come_from_the_csv_in_frame_order(tmp_path):
    csv = _pose_csv(tmp_path / "vDLC_m.csv")
    replay, out = _drain(csv, range(6))
    assert replay.error is None
    assert [idx for idx, _, _, _ in out] == list(range(6))
    # Frame N was written with value N, so the pairing is positional.
    assert [kp[0][0] for _, _, kp, _ in out] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_the_payload_matches_what_the_tracker_emits(tmp_path):
    """Drop-in: the feature engine must not be able to tell which ran."""
    csv = _pose_csv(tmp_path / "vDLC_m.csv")
    _, out = _drain(csv, range(3))
    for frame_idx, frame_bgr, keypoints, confidences in out:
        assert isinstance(frame_idx, int)
        assert frame_bgr is None  # exactly what the raw queue supplied
        assert keypoints.shape == (3, 2)
        assert confidences.shape == (3,)


def test_frames_beyond_the_csv_pass_through_as_undetected(tmp_path):
    """A shorter CSV must not truncate or misalign the run."""
    csv = _pose_csv(tmp_path / "vDLC_m.csv", n=3)
    replay, out = _drain(csv, range(5))
    assert len(out) == 5
    assert np.isnan(out[4][2]).all()
    assert replay.n_frames_without_pose == 2


def test_a_keypoint_count_mismatch_is_reported_not_guessed(tmp_path):
    csv = _pose_csv(tmp_path / "vDLC_m.csv")
    replay, out = _drain(csv, range(3), names=["a", "b", "c", "d"])
    assert replay.error is not None
    assert "4 names" in replay.error
    assert out == []


def test_an_unreadable_csv_is_reported_not_raised(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("nonsense")
    replay, out = _drain(bad, range(2))
    assert replay.error is not None
    assert out == []


def test_downstream_always_sees_end_of_stream(tmp_path):
    """Even on failure, or the pipeline would hang waiting for it."""
    bad = tmp_path / "bad.csv"
    bad.write_text("nonsense")
    raw: queue.Queue = queue.Queue()
    tracked: queue.Queue = queue.Queue()
    display: queue.Queue = queue.Queue()
    replay = PoseReplay(raw, tracked, display, threading.Event(), bad, NAMES)
    replay.start()
    replay.join(timeout=10)
    assert tracked.get(timeout=1) is END_OF_STREAM
    assert display.get(timeout=1) is END_OF_STREAM


class TestClassifyWiring:
    def test_reuse_picks_up_a_csv_beside_the_video(self, tmp_path, monkeypatch):
        """With poses on disk and no annotated video, the batch path takes it.

        Every frame is already known, so there is nothing to stream; the
        streaming pipeline must not even be constructed.
        """
        from glider.analysis.behavior import classify as classify_mod
        from glider.analysis.behavior.classify import batch as batch_mod
        from glider.analysis.behavior.classify import pipeline as pipeline_mod

        video = tmp_path / "T7_5.mp4"
        video.write_bytes(b"")
        _pose_csv(tmp_path / "T7_5DLC_exp-5.csv")

        seen = {}
        monkeypatch.setattr(pipeline_mod, "_load_behavior_model", lambda _p: object())

        def _fake_batch(config, _ethogram, _model):
            seen["pose_csv_in"] = config.pose_csv_in
            seen["pose_csv_out"] = config.pose_csv_out
            raise RuntimeError("stop before running")

        monkeypatch.setattr(batch_mod, "batch_apply", _fake_batch)
        monkeypatch.setattr(
            classify_mod,
            "LiveInferencePipeline",
            lambda *a, **k: pytest.fail("the streaming pipeline should not be built"),
        )
        with pytest.raises(RuntimeError):
            classify_mod.classify(
                video,
                model_path=tmp_path / "m.pkl",
                yolo_path=tmp_path / "exp-5.pt",
                keypoint_names=NAMES,
                output_dir=tmp_path / "out",
                reuse_existing_poses=True,
            )
        assert seen["pose_csv_in"] is not None
        # Nothing new was tracked, so nothing new should be written.
        assert seen["pose_csv_out"] is None

    def test_reuse_finds_poses_in_a_separate_folder(self, tmp_path, monkeypatch):
        """Batch Pose Tracking writes where it was pointed, not beside the
        videos; copying a CSV per video by hand is the thing to avoid."""
        from glider.analysis.behavior import classify as classify_mod
        from glider.analysis.behavior.classify import batch as batch_mod
        from glider.analysis.behavior.classify import pipeline as pipeline_mod

        videos = tmp_path / "videos"
        poses = tmp_path / "poses"
        videos.mkdir()
        poses.mkdir()
        video = videos / "T7_5.mp4"
        video.write_bytes(b"")
        _pose_csv(poses / "T7_5DLC_exp-5.csv")

        seen = {}
        monkeypatch.setattr(pipeline_mod, "_load_behavior_model", lambda _p: object())

        def _fake_batch(config, _ethogram, _model):
            seen["pose_csv_in"] = config.pose_csv_in
            raise RuntimeError("stop before running")

        monkeypatch.setattr(batch_mod, "batch_apply", _fake_batch)
        with pytest.raises(RuntimeError):
            classify_mod.classify(
                video,
                model_path=tmp_path / "m.pkl",
                yolo_path=tmp_path / "exp-5.pt",
                keypoint_names=NAMES,
                output_dir=tmp_path / "out",
                reuse_existing_poses=True,
                pose_dir=poses,
            )
        assert seen["pose_csv_in"] == poses / "T7_5DLC_exp-5.csv"

    def test_without_a_pose_dir_a_foreign_folder_is_not_searched(self, tmp_path, monkeypatch):
        """Reuse must not silently pick up an unrelated video's poses."""
        from glider.analysis.behavior import classify as classify_mod

        videos = tmp_path / "videos"
        poses = tmp_path / "poses"
        videos.mkdir()
        poses.mkdir()
        video = videos / "T7_5.mp4"
        video.write_bytes(b"")
        _pose_csv(poses / "T7_5DLC_exp-5.csv")

        seen = {}

        class _Pipeline:
            def __init__(self, config, **_kw):
                seen["pose_csv_in"] = config.pose_csv_in
                raise RuntimeError("stop")

        monkeypatch.setattr(classify_mod, "LiveInferencePipeline", _Pipeline)
        with pytest.raises(RuntimeError):
            classify_mod.classify(
                video,
                model_path=tmp_path / "m.pkl",
                yolo_path=tmp_path / "exp-5.pt",
                keypoint_names=NAMES,
                output_dir=tmp_path / "out",
                reuse_existing_poses=True,
            )
        assert seen["pose_csv_in"] is None

    def test_without_reuse_the_pose_model_still_runs(self, tmp_path, monkeypatch):
        from glider.analysis.behavior import classify as classify_mod

        video = tmp_path / "T7_5.mp4"
        video.write_bytes(b"")
        _pose_csv(tmp_path / "T7_5DLC_exp-5.csv")

        seen = {}

        class _Pipeline:
            def __init__(self, config):
                seen["pose_csv_in"] = config.pose_csv_in
                raise RuntimeError("stop")

        monkeypatch.setattr(classify_mod, "LiveInferencePipeline", _Pipeline)
        with pytest.raises(RuntimeError):
            classify_mod.classify(
                video,
                model_path=tmp_path / "m.pkl",
                yolo_path=tmp_path / "exp-5.pt",
                keypoint_names=NAMES,
                output_dir=tmp_path / "out",
            )
        assert seen["pose_csv_in"] is None

    def test_a_missing_explicit_csv_fails_before_any_work(self, tmp_path):
        from glider.analysis.behavior.classify import classify

        video = tmp_path / "v.mp4"
        video.write_bytes(b"")
        with pytest.raises(ValueError, match="pose CSV not found"):
            classify(
                video,
                model_path=tmp_path / "m.pkl",
                yolo_path=tmp_path / "y.pt",
                keypoint_names=NAMES,
                output_dir=tmp_path / "out",
                pose_csv_in=tmp_path / "absent.csv",
            )
