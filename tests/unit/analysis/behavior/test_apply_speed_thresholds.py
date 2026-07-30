"""Apply-time freeze/dart thresholds, expressed in mm/s.

The live detector works in raw pixels per frame, so the millimetre conversion
here is exact -- unlike the hybrid prior's body-length-normalized thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior.classify import resolve_speed_thresholds
from glider.analysis.behavior.units import mm_per_s_to_px_per_frame


class TestMmPerSecondToPixelsPerFrame:
    def test_exact_inverse_of_the_forward_conversion(self):
        # 100 mm/s at 4 px/mm and 30 fps = 400 px/s = 13.333 px/frame.
        assert mm_per_s_to_px_per_frame(100.0, px_per_mm=4.0, fps=30.0) == pytest.approx(
            400.0 / 30.0
        )

    def test_round_trips_against_speed_scale(self):
        from glider.analysis.behavior.units import SpeedScale

        # The live detector is unnormalized: px/frame in, px/frame out.
        scale = SpeedScale(fps=30.0, px_per_mm=4.0, normalized=False)
        px = mm_per_s_to_px_per_frame(75.0, px_per_mm=4.0, fps=30.0)
        assert scale.to_mm_per_s(px) == pytest.approx(75.0)

    def test_missing_or_degenerate_inputs_yield_none(self):
        assert mm_per_s_to_px_per_frame(100.0, px_per_mm=None, fps=30.0) is None
        assert mm_per_s_to_px_per_frame(100.0, px_per_mm=4.0, fps=0.0) is None
        assert mm_per_s_to_px_per_frame(None, px_per_mm=4.0, fps=30.0) is None
        assert mm_per_s_to_px_per_frame(float("nan"), px_per_mm=4.0, fps=30.0) is None


class TestResolveSpeedThresholds:
    def test_no_thresholds_requested_leaves_the_speed_axis_alone(self, tmp_path):
        assert resolve_speed_thresholds(tmp_path / "v.mp4") == {}

    def test_native_px_per_frame_passes_straight_through(self, tmp_path):
        out = resolve_speed_thresholds(
            tmp_path / "v.mp4", freeze_threshold=2.0, dart_threshold=18.0
        )
        assert out == {"freeze_threshold": 2.0, "dart_threshold": 18.0}

    def test_mm_per_second_converts_with_an_explicit_scale(self, tmp_path):
        out = resolve_speed_thresholds(
            tmp_path / "v.mp4",
            freeze_mm_s=15.0,
            dart_mm_s=150.0,
            px_per_mm=4.0,
            fps=30.0,
        )
        assert out["freeze_threshold"] == pytest.approx(2.0)
        assert out["dart_threshold"] == pytest.approx(20.0)

    def test_scale_is_read_from_the_master_calibration_file(self, tmp_path):
        from glider.vision.calibration import CameraCalibration, LengthUnit
        from glider.vision.calibration_set import CalibrationSet

        video = tmp_path / "session01.mp4"
        video.write_bytes(b"")
        cal = CameraCalibration()
        # 640 px across 160 mm = 4 px/mm.
        cal.add_line((0, 240), (640, 240), 160.0, LengthUnit.MILLIMETERS, "w", (640, 480))
        cal_set = CalibrationSet()
        cal_set.set(video, cal)
        master = tmp_path / "pose_calibration.json"
        cal_set.save(master)

        out = resolve_speed_thresholds(
            video, freeze_mm_s=15.0, dart_mm_s=150.0, calibration_master=master, fps=30.0
        )
        assert out["freeze_threshold"] == pytest.approx(2.0)
        assert out["dart_threshold"] == pytest.approx(20.0)

    def test_explicit_scale_wins_over_the_master_file(self, tmp_path):
        from glider.vision.calibration import CameraCalibration, LengthUnit
        from glider.vision.calibration_set import CalibrationSet

        video = tmp_path / "s.mp4"
        video.write_bytes(b"")
        cal = CameraCalibration()
        cal.add_line((0, 240), (640, 240), 160.0, LengthUnit.MILLIMETERS, "w", (640, 480))
        cs = CalibrationSet()
        cs.set(video, cal)
        master = tmp_path / "m.json"
        cs.save(master)

        out = resolve_speed_thresholds(
            video,
            freeze_mm_s=15.0,
            dart_mm_s=150.0,
            calibration_master=master,
            px_per_mm=8.0,  # overrides the file's 4.0
            fps=30.0,
        )
        assert out["freeze_threshold"] == pytest.approx(4.0)

    def test_fps_is_read_from_the_video_when_not_given(self, synthetic_clip):
        # The fixture clip is 10 fps; 40 mm/s at 4 px/mm = 160 px/s = 16 px/frame.
        out = resolve_speed_thresholds(
            synthetic_clip, freeze_mm_s=4.0, dart_mm_s=40.0, px_per_mm=4.0
        )
        assert out["dart_threshold"] == pytest.approx(16.0)

    def test_mm_without_any_scale_fails_loudly(self, tmp_path):
        with pytest.raises(ValueError, match="px_per_mm"):
            resolve_speed_thresholds(
                tmp_path / "v.mp4", freeze_mm_s=15.0, dart_mm_s=150.0, fps=30.0
            )

    def test_mm_without_a_readable_fps_fails_loudly(self, tmp_path):
        # A path that is not a decodable video: fps cannot be discovered.
        bogus = tmp_path / "not_a_video.mp4"
        bogus.write_bytes(b"nope")
        with pytest.raises(ValueError, match="frame rate"):
            resolve_speed_thresholds(bogus, freeze_mm_s=15.0, dart_mm_s=150.0, px_per_mm=4.0)

    def test_one_axis_alone_fails_loudly(self, tmp_path):
        # The detector needs both; one alone would silently disable the axis.
        with pytest.raises(ValueError, match="both"):
            resolve_speed_thresholds(tmp_path / "v.mp4", dart_mm_s=150.0, px_per_mm=4.0, fps=30.0)

    def test_mixing_units_on_one_axis_fails_loudly(self, tmp_path):
        with pytest.raises(ValueError, match="mm/s"):
            resolve_speed_thresholds(
                tmp_path / "v.mp4",
                freeze_mm_s=15.0,
                dart_mm_s=150.0,
                freeze_threshold=2.0,
                px_per_mm=4.0,
                fps=30.0,
            )

    def test_freeze_must_stay_below_dart(self, tmp_path):
        # FreezeDartDetector rejects this too, but failing here names the mm/s
        # values the operator actually typed.
        with pytest.raises(ValueError, match="below"):
            resolve_speed_thresholds(
                tmp_path / "v.mp4",
                freeze_mm_s=200.0,
                dart_mm_s=150.0,
                px_per_mm=4.0,
                fps=30.0,
            )


class TestHybridBundleIsRejectedClearly:
    def _hybrid(self, tmp_path):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        from glider.analysis.behavior.features import FeatureSpec
        from glider.analysis.behavior.hybrid import HybridModel
        from glider.analysis.behavior.model import BehaviorModel
        from glider.analysis.behavior.prior import KinematicPrior

        cols = ["speed_a__mean", "speed_b__mean"]
        x = pd.DataFrame({c: [0, 0, 9, 9, 0, 9] for c in cols})
        clf = RandomForestClassifier(n_estimators=4, random_state=0).fit(
            x, ["rest", "rest", "go", "go", "rest", "go"]
        )
        base = BehaviorModel(clf, cols, FeatureSpec(), 1, ("mean",), 30.0, ["go", "rest"])
        tag_map = {"rest": frozenset({"stationary"}), "go": frozenset({"locomotory"})}
        prior = KinematicPrior(tag_map=tag_map)
        calib = pd.DataFrame({c: np.linspace(0, 0.9, 20) for c in cols})
        calib["body_length__mean"] = 60.0
        prior.calibrate(calib)
        path = tmp_path / "hybrid.pkl"
        HybridModel(base, prior, 0.5, tag_map).save(path)
        return path

    def test_hybrid_bundle_names_the_real_problem(self, tmp_path):
        from glider.analysis.behavior.classify.pipeline import _load_behavior_model

        path = self._hybrid(tmp_path)
        with pytest.raises(ValueError, match="hybrid"):
            _load_behavior_model(path)

    def test_the_message_says_the_prior_will_not_run(self, tmp_path):
        from glider.analysis.behavior.classify.pipeline import _load_behavior_model

        path = self._hybrid(tmp_path)
        with pytest.raises(ValueError) as excinfo:
            _load_behavior_model(path)
        # The operator must not think the prior is participating.
        assert "prior" in str(excinfo.value).lower()

    def test_a_plain_bundle_still_loads(self, tmp_path):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        from glider.analysis.behavior.classify.pipeline import _load_behavior_model
        from glider.analysis.behavior.features import FeatureSpec
        from glider.analysis.behavior.model import BehaviorModel

        cols = ["speed_a__mean"]
        x = pd.DataFrame({c: [0, 0, 9, 9] for c in cols})
        clf = RandomForestClassifier(n_estimators=4, random_state=0).fit(
            x, ["rest", "rest", "go", "go"]
        )
        path = tmp_path / "plain.pkl"
        BehaviorModel(clf, cols, FeatureSpec(), 1, ("mean",), 30.0, ["go", "rest"]).save(path)
        assert _load_behavior_model(path).__class__.__name__ == "BehaviorModel"
