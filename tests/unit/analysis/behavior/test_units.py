"""Real-world unit conversion for the kinematic speed thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior.units import (
    SpeedScale,
    describe_speed_threshold,
    load_px_per_mm,
    median_body_length_px,
)


class TestSpeedScale:
    def test_normalized_speed_becomes_pixels_via_body_length(self):
        # 0.5 body-lengths/frame at a 20 px body = 10 px/frame.
        scale = SpeedScale(fps=30.0, body_length_px=20.0)
        assert scale.to_px_per_frame(0.5) == pytest.approx(10.0)

    def test_unnormalized_speed_is_already_pixels(self):
        # normalize_by_body_length=False: no body length involved at all.
        scale = SpeedScale(fps=30.0, normalized=False)
        assert scale.to_px_per_frame(10.0) == pytest.approx(10.0)

    def test_normalized_without_a_body_length_cannot_convert(self):
        scale = SpeedScale(fps=30.0, body_length_px=None)
        assert scale.to_px_per_frame(0.5) is None

    def test_millimetres_per_second(self):
        # 0.5 bl/frame * 20 px = 10 px/frame; * 30 fps = 300 px/s; / 4 px/mm = 75 mm/s.
        scale = SpeedScale(fps=30.0, body_length_px=20.0, px_per_mm=4.0)
        assert scale.to_mm_per_s(0.5) == pytest.approx(75.0)

    def test_millimetres_need_a_calibration(self):
        scale = SpeedScale(fps=30.0, body_length_px=20.0, px_per_mm=None)
        assert scale.to_mm_per_s(0.5) is None

    def test_per_second_needs_no_calibration_at_all(self):
        # The honest unit when the rig was never measured.
        scale = SpeedScale(fps=30.0, body_length_px=20.0)
        assert scale.to_per_second(0.5) == pytest.approx(15.0)

    def test_native_unit_names_itself(self):
        assert SpeedScale(fps=30.0, body_length_px=20.0).native_unit == "bl/frame"
        assert SpeedScale(fps=30.0, normalized=False).native_unit == "px/frame"

    def test_degenerate_inputs_do_not_divide_by_zero(self):
        assert SpeedScale(fps=30.0, body_length_px=20.0, px_per_mm=0.0).to_mm_per_s(0.5) is None
        assert SpeedScale(fps=0.0, body_length_px=20.0).to_per_second(0.5) is None
        assert SpeedScale(fps=30.0, body_length_px=0.0).to_px_per_frame(0.5) is None

    def test_nan_speed_yields_none(self):
        scale = SpeedScale(fps=30.0, body_length_px=20.0, px_per_mm=4.0)
        assert scale.to_mm_per_s(float("nan")) is None


class TestMedianBodyLength:
    def test_reads_the_raw_feature_column(self):
        df = pd.DataFrame({"body_length": [10.0, 20.0, 30.0]})
        assert median_body_length_px(df) == pytest.approx(20.0)

    def test_falls_back_to_the_windowed_column(self):
        # After apply_rolling the raw column is gone, suffixed per stat.
        df = pd.DataFrame({"body_length__mean": [10.0, 20.0, 30.0], "speed_a__mean": [1.0, 2, 3]})
        assert median_body_length_px(df) == pytest.approx(20.0)

    def test_prefers_the_raw_column_when_both_exist(self):
        df = pd.DataFrame({"body_length": [4.0, 4.0], "body_length__mean": [99.0, 99.0]})
        assert median_body_length_px(df) == pytest.approx(4.0)

    def test_ignores_nan_rows(self):
        df = pd.DataFrame({"body_length": [np.nan, 10.0, 30.0, np.nan]})
        assert median_body_length_px(df) == pytest.approx(20.0)

    def test_absent_column_returns_none(self):
        # include_body_length=False drops it entirely.
        assert median_body_length_px(pd.DataFrame({"speed_a": [1.0]})) is None

    def test_all_nan_returns_none(self):
        assert median_body_length_px(pd.DataFrame({"body_length": [np.nan, np.nan]})) is None


class TestDescribe:
    def test_reports_every_available_unit(self):
        scale = SpeedScale(fps=30.0, body_length_px=20.0, px_per_mm=4.0)
        out = describe_speed_threshold("dart", 0.5, scale)
        assert out["name"] == "dart"
        assert out["native"] == pytest.approx(0.5)
        assert out["native_unit"] == "bl/frame"
        assert out["per_second"] == pytest.approx(15.0)
        assert out["px_per_frame"] == pytest.approx(10.0)
        assert out["mm_per_s"] == pytest.approx(75.0)

    def test_records_the_references_so_the_number_is_auditable(self):
        scale = SpeedScale(fps=30.0, body_length_px=20.0, px_per_mm=4.0)
        out = describe_speed_threshold("dart", 0.5, scale)
        assert out["body_length_px"] == pytest.approx(20.0)
        assert out["px_per_mm"] == pytest.approx(4.0)
        assert out["fps"] == pytest.approx(30.0)

    def test_uncalibrated_omits_millimetres_but_keeps_the_rest(self):
        scale = SpeedScale(fps=30.0, body_length_px=20.0)
        out = describe_speed_threshold("freeze", 0.1, scale)
        assert out["mm_per_s"] is None
        assert out["per_second"] == pytest.approx(3.0)
        assert out["px_per_frame"] == pytest.approx(2.0)

    def test_an_uncalibrated_threshold_is_none(self):
        # calibrate() never ran, so there is no threshold to describe.
        out = describe_speed_threshold("dart", None, SpeedScale(fps=30.0, body_length_px=20.0))
        assert out["native"] is None
        assert out["mm_per_s"] is None


class TestLoadPxPerMm:
    def test_reads_the_batch_master_calibration_file(self, tmp_path):
        from glider.vision.calibration import CameraCalibration, LengthUnit
        from glider.vision.calibration_set import CalibrationSet

        video = tmp_path / "session01.mp4"
        video.write_bytes(b"")
        cal = CameraCalibration()
        cal.add_line((0, 240), (640, 240), 100.0, LengthUnit.MILLIMETERS, "w", (640, 480))
        cal_set = CalibrationSet()
        cal_set.set(video, cal)
        master = tmp_path / "pose_calibration.json"
        cal_set.save(master)

        assert load_px_per_mm(master, video) == pytest.approx(6.4, abs=0.1)

    def test_missing_file_is_not_an_error(self, tmp_path):
        # Analysis must still run for an uncalibrated session.
        assert load_px_per_mm(tmp_path / "nope.json", tmp_path / "a.mp4") is None

    def test_corrupt_file_is_not_an_error(self, tmp_path):
        master = tmp_path / "m.json"
        master.write_text("{not json")
        assert load_px_per_mm(master, tmp_path / "a.mp4") is None

    def test_video_absent_from_the_master_returns_none(self, tmp_path):
        from glider.vision.calibration_set import CalibrationSet

        master = tmp_path / "m.json"
        CalibrationSet().save(master)
        assert load_px_per_mm(master, tmp_path / "unlisted.mp4") is None

    def test_none_path_returns_none(self, tmp_path):
        assert load_px_per_mm(None, tmp_path / "a.mp4") is None


class TestPriorCarriesItsReference:
    """The bundle must describe its own thresholds without the training data."""

    def _windowed(self, body_length=20.0, n=50):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "speed_a__mean": rng.uniform(0.0, 1.0, n),
                "speed_b__mean": rng.uniform(0.0, 1.0, n),
                "body_length__mean": np.full(n, body_length),
            }
        )

    def test_thresholds_are_readable_without_private_access(self):
        from glider.analysis.behavior.prior import KinematicPrior

        prior = KinematicPrior({"rest": frozenset({"stationary"})})
        assert prior.freeze_threshold is None
        assert prior.dart_threshold is None
        prior.calibrate(self._windowed())
        assert prior.freeze_threshold is not None
        assert prior.dart_threshold > prior.freeze_threshold

    def test_calibrate_records_the_reference_body_length(self):
        from glider.analysis.behavior.prior import KinematicPrior

        prior = KinematicPrior({"rest": frozenset({"stationary"})})
        prior.calibrate(self._windowed(body_length=17.5))
        assert prior.body_length_px == pytest.approx(17.5)

    def test_reference_survives_the_bundle_round_trip(self):
        from glider.analysis.behavior.prior import KinematicPrior

        prior = KinematicPrior({"rest": frozenset({"stationary"})})
        prior.calibrate(self._windowed(body_length=17.5))
        restored = KinematicPrior.from_dict(prior.to_dict())
        assert restored.body_length_px == pytest.approx(17.5)
        assert restored.dart_threshold == pytest.approx(prior.dart_threshold)

    def test_legacy_bundle_without_a_reference_still_loads(self):
        from glider.analysis.behavior.prior import KinematicPrior

        prior = KinematicPrior({"rest": frozenset({"stationary"})})
        prior.calibrate(self._windowed())
        legacy = prior.to_dict()
        del legacy["body_length_px"]  # saved before the field existed
        restored = KinematicPrior.from_dict(legacy)
        assert restored.body_length_px is None
        assert restored.dart_threshold is not None

    def test_features_without_body_length_leave_the_reference_unset(self):
        from glider.analysis.behavior.prior import KinematicPrior

        # include_body_length=False drops the column entirely.
        frame = self._windowed().drop(columns=["body_length__mean"])
        prior = KinematicPrior({"rest": frozenset({"stationary"})})
        prior.calibrate(frame)
        assert prior.body_length_px is None
        assert prior.dart_threshold is not None  # thresholds still calibrate


@pytest.fixture
def tiny_hybrid_model():
    """A HybridModel whose prior was calibrated on a frame carrying body_length."""
    from sklearn.ensemble import RandomForestClassifier

    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.hybrid import HybridModel
    from glider.analysis.behavior.model import BehaviorModel
    from glider.analysis.behavior.prior import KinematicPrior

    cols = ["speed_a__mean", "speed_b__mean"]
    x = pd.DataFrame({c: [0, 0, 9, 9, 0, 9] for c in cols})
    y = ["rest", "rest", "locomote", "locomote", "rest", "locomote"]
    clf = RandomForestClassifier(n_estimators=8, random_state=0).fit(x, y)
    base = BehaviorModel(clf, cols, FeatureSpec(), 1, ("mean",), 30.0, ["locomote", "rest"])

    tag_map = {"rest": frozenset({"stationary"}), "locomote": frozenset({"locomotory"})}
    prior = KinematicPrior(tag_map=tag_map)
    calib = pd.DataFrame({c: np.linspace(0, 9, 20) for c in cols})
    calib["body_length__mean"] = 20.0
    prior.calibrate(calib)
    return HybridModel(base, prior, lam=0.5, tag_map=tag_map)


class TestHybridDescribeThresholds:
    def test_reports_both_thresholds_in_real_units(self, tiny_hybrid_model):
        rows = tiny_hybrid_model.describe_thresholds(px_per_mm=4.0)
        assert [r["name"] for r in rows] == ["freeze", "dart"]
        for row in rows:
            assert row["mm_per_s"] is not None
            assert row["native_unit"] == "bl/frame"

    def test_without_a_calibration_it_omits_millimetres(self, tiny_hybrid_model):
        rows = tiny_hybrid_model.describe_thresholds()
        assert all(r["mm_per_s"] is None for r in rows)
        assert all(r["per_second"] is not None for r in rows)

    def test_body_length_can_be_overridden_per_session(self, tiny_hybrid_model):
        # A different animal than the training median.
        rows = tiny_hybrid_model.describe_thresholds(px_per_mm=4.0, body_length_px=40.0)
        assert all(r["body_length_px"] == pytest.approx(40.0) for r in rows)
