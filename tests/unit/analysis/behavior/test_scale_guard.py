"""Catching a cohort that is not at the scale the model was trained on.

Both checks exist because the failure is silent: the labels look plausible,
nothing raises, and no downstream number says the scale changed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from glider.analysis.behavior.features import FeatureSpec  # noqa: E402
from glider.analysis.behavior.scale_guard import (  # noqa: E402
    body_length_splits,
    calibration_spread_warning,
    scale_warning,
)
from glider.vision.pose.core import PoseData  # noqa: E402

KP = ["nose", "left_ear", "right_ear", "body_center", "tail_base"]


def _pose(n=200, scale=1.0, seed=0):
    """An animal roughly `70 * scale` px nose-to-tail.

    The body stretches and compresses over time. A rigid animal would give a
    single body length, so the model's splits would all land in a 1 px band
    and the check would fire on the very data it was trained on.
    """
    rng = np.random.default_rng(seed)
    centre = np.stack([200 + 40 * np.sin(np.arange(n) / 9), 150 + np.zeros(n)], axis=1)
    stretch = 1.0 + 0.35 * np.sin(np.arange(n) / 5.0)  # posture, frame by frame
    offsets = np.array([[0, -35], [-8, -25], [8, -25], [0, 0], [0, 35]], dtype=float) * scale
    xy = centre[:, None, :] + offsets[None] * stretch[:, None, None]
    xy = xy + rng.normal(0, 0.4, size=(n, len(KP), 2))
    return PoseData(xy=xy, confidence=np.ones((n, len(KP))), keypoint_names=KP, fps=30.0)


@pytest.fixture(scope="module")
def model():
    """A real LightGBM bundle trained on unit-scale animals."""
    lgb = pytest.importorskip("lightgbm")

    from glider.analysis.behavior.features import compute_features
    from glider.analysis.behavior.model import BehaviorModel
    from glider.analysis.behavior.windowing import apply_rolling

    spec = FeatureSpec()
    pose = _pose(n=600)
    feats = compute_features(pose, spec)
    rolled = apply_rolling(feats, window=4, stats=("mean", "std", "max"), min_periods=1).dropna()
    # Labels that genuinely depend on body length, so the trees split on it
    # across the observed range. With random labels LightGBM has no reason to
    # split on the feature at all, and the recovered range is one arbitrary
    # threshold — which is a property of the fixture, not of the check.
    lengths = rolled["body_length__mean"].to_numpy()
    edges = np.quantile(lengths, [0.33, 0.66])
    y = np.where(lengths < edges[0], "a", np.where(lengths < edges[1], "b", "c"))
    clf = lgb.LGBMClassifier(n_estimators=60, num_leaves=7, verbose=-1).fit(rolled, y)
    return BehaviorModel(
        clf, list(rolled.columns), spec, 4, ("mean", "std", "max"), 30.0, ["a", "b", "c"]
    )


class TestSplits:
    def test_it_recovers_plausible_thresholds(self, model):
        splits = body_length_splits(model)
        assert splits is not None and splits.size > 0
        # The training animal is ~70 px nose-to-tail and stretches around that.
        assert 20 < np.median(splits) < 200

    def test_a_classifier_without_a_booster_gives_nothing(self):
        class Plain:
            classifier = object()

        assert body_length_splits(Plain()) is None

    def test_the_std_statistic_is_excluded(self, model):
        """body_length__std is a couple of pixels; pooling it with the means
        would drag the thresholds toward zero and the check would never fire."""
        assert float(np.min(body_length_splits(model))) > 10


class TestScaleWarning:
    def test_the_training_scale_passes_silently(self, model):
        assert scale_warning(model, _pose(scale=1.0, seed=3)) is None

    def test_a_much_smaller_animal_is_flagged(self, model):
        message = scale_warning(model, _pose(scale=0.3, seed=4))
        assert message is not None
        assert "body_length" in message
        assert "smaller" in message

    def test_a_much_larger_animal_is_flagged(self, model):
        message = scale_warning(model, _pose(scale=3.0, seed=5))
        assert message is not None
        assert "larger" in message

    def test_it_names_both_the_observed_and_expected_size(self, model):
        message = scale_warning(model, _pose(scale=0.3, seed=6))
        assert "px nose-to-tail" in message
        assert "retrain" in message  # says what to do, not just what is wrong

    def test_an_empty_pose_is_not_an_error(self, model):
        empty = PoseData(
            xy=np.full((5, len(KP), 2), np.nan),
            confidence=np.zeros((5, len(KP))),
            keypoint_names=KP,
            fps=30.0,
        )
        assert scale_warning(model, empty) is None


def _calibration_master(path, scales, resolution=(640, 480)):
    """A master file shaped like the one Batch Pose Tracking writes."""
    videos = []
    for i, scale in enumerate(scales):
        # One 100 px line standing for 100/scale mm.
        videos.append(
            {
                "video": f"C:/rig/v{i}.mp4",
                "resolution": list(resolution),
                "px_per_mm": scale,
                "mm_per_px": 1.0 / scale,
                "calibration": {
                    "calibration_width": resolution[0],
                    "calibration_height": resolution[1],
                    "lines": [
                        {
                            "start_x": 0.0,
                            "start_y": 0.0,
                            "end_x": 100.0 / resolution[0],
                            "end_y": 0.0,
                            "length": 100.0 / scale,
                            "unit": "mm",
                        }
                    ],
                },
            }
        )
    path.write_text(json.dumps({"schema_version": 1, "videos": videos}))
    return path


class TestCalibrationSpread:
    def test_consistent_calibrations_pass(self, tmp_path):
        path = _calibration_master(tmp_path / "cal.json", [1.30, 1.31, 1.29, 1.30])
        assert calibration_spread_warning(path) is None

    def test_a_wide_spread_at_one_resolution_is_flagged(self, tmp_path):
        path = _calibration_master(tmp_path / "cal.json", [1.26, 1.35, 1.48, 1.30])
        message = calibration_spread_warning(path)
        assert message is not None
        assert "640" in message and "px/mm" in message

    def test_two_videos_are_too_few_to_judge(self, tmp_path):
        """Two disagreeing measurements are not evidence of a systematic problem."""
        path = _calibration_master(tmp_path / "cal.json", [1.20, 1.60])
        assert calibration_spread_warning(path) is None

    def test_an_unreadable_file_is_not_an_error(self, tmp_path):
        bad = tmp_path / "nope.json"
        bad.write_text("{not json")
        assert calibration_spread_warning(bad) is None

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert calibration_spread_warning(tmp_path / "absent.json") is None
