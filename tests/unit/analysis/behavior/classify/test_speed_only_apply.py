"""An apply run needs a behaviour model, or pose weights, only where it does.

Two independent relaxations, and the tests here are about not over-applying
either of them:

* A run with no behaviour bundle scores freezing and darting off the speed
  trace. That is the whole analysis for a fear-conditioning study, and it
  never needed a classifier — but it does need thresholds, and it cannot
  render an annotated video, because there are no predicted labels to draw.
* A run with no pose weights scores keypoints that are already on disk.
  Tracking is the only thing the weights are for. A video with no pose CSV
  and no weights has no source of coordinates at all, and must say so
  before it spends anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior.classify import classify
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import to_dlc_csv

KP = ["nose", "body_center", "tail_base"]


def _pose(n=120, step=0.0):
    """A still animal by default; ``step`` px/frame of steady travel."""
    t = np.arange(n, dtype=float)[:, None, None]
    base = np.array([[100.0, 100.0], [100.0, 110.0], [100.0, 120.0]])
    xy = base[None] + t * np.array([[step, 0.0]])
    return PoseData(xy=xy, confidence=np.ones((n, len(KP))), keypoint_names=KP, fps=30.0)


@pytest.fixture
def pose_csv(tmp_path):
    return to_dlc_csv(_pose(), tmp_path / "clipDLC_yolo.csv")


@pytest.fixture(autouse=True)
def no_streaming(monkeypatch):
    """The streaming pipeline must never be reached by these runs."""
    import glider.analysis.behavior.classify as mod

    def _explode(*_args, **_kwargs):
        raise AssertionError("the streaming pipeline should not have been used")

    monkeypatch.setattr(mod, "LiveInferencePipeline", _explode)


class TestSpeedOnly:
    def test_freezing_is_scored_with_no_model_at_all(self, tmp_path, pose_csv):
        out = tmp_path / "out"
        classify(
            video="clip.mp4",
            model_path=None,
            yolo_path=None,
            keypoint_names=KP,
            fps_override=30.0,
            output_dir=out,
            pose_csv_in=pose_csv,
            predict_every=1,
            freeze_threshold=1.0,
            dart_threshold=50.0,
            freeze_min_frames=5,
        )
        df = pd.read_csv(out / "ethogram_raw.csv", keep_default_na=False)
        assert set(df["behavior"]) == {"freezing"}
        assert list(df.columns) == ["frame", "behavior", "speed_px_frame", "speed_cm_s"]

    def test_the_run_records_that_it_had_no_model(self, tmp_path, pose_csv):
        from glider.analysis.behavior.classify import read_run_manifest

        out = tmp_path / "out"
        classify(
            video="clip.mp4",
            model_path=None,
            yolo_path=None,
            keypoint_names=KP,
            fps_override=30.0,
            output_dir=out,
            pose_csv_in=pose_csv,
            freeze_threshold=1.0,
            dart_threshold=50.0,
        )
        manifest = read_run_manifest(out)
        assert manifest["speed_only"] is True
        assert manifest["model_path"] is None

    def test_a_disabled_side_records_as_null_not_infinity(self, tmp_path, pose_csv):
        """run.json stays valid JSON for anything that reads it."""
        import json

        from glider.analysis.behavior.classify import read_run_manifest

        out = tmp_path / "out"
        classify(
            video="clip.mp4",
            model_path=None,
            yolo_path=None,
            keypoint_names=KP,
            fps_override=30.0,
            output_dir=out,
            pose_csv_in=pose_csv,
            freeze_threshold=1.0,
            score_darting=False,
        )
        assert read_run_manifest(out)["dart_threshold"] is None
        json.loads((out / "run.json").read_text())  # no bare Infinity token

    def test_without_a_threshold_there_is_nothing_to_score(self, tmp_path, pose_csv):
        with pytest.raises(ValueError, match="freezing or a darting threshold"):
            classify(
                video="clip.mp4",
                model_path=None,
                yolo_path=None,
                keypoint_names=KP,
                fps_override=30.0,
                output_dir=tmp_path / "out",
                pose_csv_in=pose_csv,
            )

    def test_an_annotated_video_is_refused_rather_than_written_blank(self, tmp_path, pose_csv):
        with pytest.raises(ValueError, match="annotated video"):
            classify(
                video="clip.mp4",
                model_path=None,
                yolo_path=None,
                keypoint_names=KP,
                fps_override=30.0,
                output_dir=tmp_path / "out",
                pose_csv_in=pose_csv,
                write_annotated=True,
                freeze_threshold=1.0,
                dart_threshold=50.0,
            )

    def test_poses_are_tracked_when_there_are_none_and_weights_exist(self, tmp_path, monkeypatch):
        """The streaming path cannot serve this mode, so tracking is its own pass."""
        import glider.analysis.behavior.classify as mod

        calls = []

        def fake_track(config, pose_csv_out):
            calls.append((config.yolo_model_path, pose_csv_out))
            return _pose()

        monkeypatch.setattr(mod, "_track_poses", fake_track)

        out = tmp_path / "out"
        classify(
            video="clip.mp4",
            model_path=None,
            yolo_path="yolo.pt",
            keypoint_names=KP,
            fps_override=30.0,
            output_dir=out,
            predict_every=1,
            freeze_threshold=1.0,
            dart_threshold=50.0,
            freeze_min_frames=5,
        )
        assert len(calls) == 1
        assert calls[0][0] == "yolo.pt"
        df = pd.read_csv(out / "ethogram_raw.csv", keep_default_na=False)
        assert set(df["behavior"]) == {"freezing"}

    def test_tracking_writes_the_poses_it_derived(self, tmp_path, monkeypatch):
        """An hour of GPU work is not thrown away once it has been scored."""
        monkeypatch.setattr("glider.vision.pose.core.infer_video", lambda *a, **k: _pose())
        out = tmp_path / "out"
        classify(
            video="clip.mp4",
            model_path=None,
            yolo_path="yolo.pt",
            keypoint_names=KP,
            fps_override=30.0,
            output_dir=out,
            freeze_threshold=1.0,
            dart_threshold=50.0,
        )
        assert (out / "clipDLC_yolo.csv").exists()


class TestPoseWeightsAreOptional:
    def test_a_run_with_poses_needs_no_weights(self, tmp_path, pose_csv, monkeypatch):
        """The classified path too, not just speed-only."""
        from glider.analysis.behavior.classify import batch as batch_mod

        seen = {}

        def fake_batch(config, ethogram_csv, model, frame_range=None, pose=None):
            seen["yolo"] = config.yolo_model_path
            seen["pose_out"] = config.pose_csv_out
            ethogram_csv.write_text("frame,behavior\n0,rear\n1,rear\n")
            return True

        monkeypatch.setattr(batch_mod, "batch_apply", fake_batch)
        monkeypatch.setattr(
            "glider.analysis.behavior.classify.pipeline._load_behavior_model",
            lambda path: object(),
        )

        classify(
            video="clip.mp4",
            model_path="model.pkl",
            yolo_path=None,
            keypoint_names=KP,
            fps_override=30.0,
            output_dir=tmp_path / "out",
            pose_csv_in=pose_csv,
        )
        assert seen["yolo"] is None
        # Nothing was tracked, so there is nothing to name after the weights.
        assert seen["pose_out"] is None

    def test_a_video_with_neither_poses_nor_weights_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no poses"):
            classify(
                video="clip.mp4",
                model_path="model.pkl",
                yolo_path=None,
                keypoint_names=KP,
                fps_override=30.0,
                output_dir=tmp_path / "out",
            )
