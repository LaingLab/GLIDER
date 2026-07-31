"""Cohort-wide freeze/dart thresholds.

Per-video percentiles are circular in a treatment study: a drugged animal that
moves less gets a lower darting threshold, normalising away the effect. These
tests pin that one pooled set of cut-offs is produced instead.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from glider.analysis.behavior.cohort_speed import (
    CM_PER_S,
    PX_PER_FRAME,
    SCHEMA_VERSION,
    CohortSpeedError,
    CohortSpeedThresholds,
    compute_cohort_thresholds,
    session_speeds,
    video_for_pose_csv,
)


def _pose_csv(path, *, step=2.0, n=400, seed=0, fps=30.0):
    """A DLC CSV whose animal drifts at a controllable speed."""
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    rng = np.random.default_rng(seed)
    xy = np.cumsum(rng.normal(0, step, size=(n, 3, 2)), axis=0) + 100.0
    to_dlc_csv(
        PoseData(
            xy=xy,
            confidence=np.ones((n, 3)),
            keypoint_names=["a", "b", "c"],
            fps=fps,
        ),
        path,
    )
    return path


class TestVideoLookup:
    def test_finds_the_video_beside_the_csv(self, tmp_path):
        video = tmp_path / "T7_5.mp4"
        video.write_bytes(b"")
        csv = _pose_csv(tmp_path / "T7_5DLC_exp-6.csv")
        assert video_for_pose_csv(csv) == video

    def test_a_missing_video_is_not_an_error(self, tmp_path):
        csv = _pose_csv(tmp_path / "T7_5DLC_exp-6.csv")
        assert video_for_pose_csv(csv) is None

    def test_a_csv_without_the_dlc_marker_yields_none(self, tmp_path):
        assert video_for_pose_csv(_pose_csv(tmp_path / "plain.csv")) is None


class TestSessionSpeeds:
    def test_unscaled_speeds_are_pixels_per_frame(self, tmp_path):
        speeds, unit = session_speeds(_pose_csv(tmp_path / "aDLC_m.csv"))
        assert unit == PX_PER_FRAME
        assert speeds.size > 0 and np.all(np.isfinite(speeds))

    def test_a_scale_and_rate_give_centimetres_per_second(self, tmp_path):
        csv = _pose_csv(tmp_path / "aDLC_m.csv")
        px, _ = session_speeds(csv)
        cm, unit = session_speeds(csv, px_per_mm=4.0, fps=30.0)
        assert unit == CM_PER_S
        # 1 px/frame at 4 px/mm and 30 fps = 30/4/10 = 0.75 cm/s.
        assert cm[0] == pytest.approx(px[0] * 0.75)

    def test_frame_zero_is_excluded(self, tmp_path):
        # CausalSpeed returns exactly 0.0 for the first frame by construction;
        # keeping it would drag the freeze percentile toward zero.
        speeds, _ = session_speeds(_pose_csv(tmp_path / "aDLC_m.csv", n=50))
        assert speeds.size == 49


class TestPooling:
    def test_pools_across_sessions_rather_than_averaging_thresholds(self, tmp_path):
        slow = _pose_csv(tmp_path / "slowDLC_m.csv", step=0.5, seed=1)
        fast = _pose_csv(tmp_path / "fastDLC_m.csv", step=8.0, seed=2)

        cohort = compute_cohort_thresholds([slow, fast])
        slow_only = compute_cohort_thresholds([slow])
        fast_only = compute_cohort_thresholds([fast])

        # The cohort cut-off sits between the two per-video ones -- the whole
        # point: neither animal is judged against only itself.
        assert slow_only.dart < cohort.dart < fast_only.dart
        assert cohort.n_sessions == 2

    def test_records_its_provenance(self, tmp_path):
        a = _pose_csv(tmp_path / "aDLC_m.csv")
        b = _pose_csv(tmp_path / "bDLC_m.csv", seed=3)
        cohort = compute_cohort_thresholds([a, b], freeze_pct=5.0, dart_pct=95.0)
        assert cohort.freeze_pct == 5.0 and cohort.dart_pct == 95.0
        assert cohort.n_samples > 0
        assert sorted(cohort.sources) == ["aDLC_m.csv", "bDLC_m.csv"]

    def test_pools_in_centimetres_when_every_session_is_calibrated(self, tmp_path):
        a = _pose_csv(tmp_path / "aDLC_m.csv")
        b = _pose_csv(tmp_path / "bDLC_m.csv", seed=3)
        cohort = compute_cohort_thresholds([a, b], px_per_mm=4.0, fps=30.0)
        assert cohort.unit == CM_PER_S
        assert cohort.is_calibrated is True

    def test_falls_back_to_pixels_when_any_session_lacks_a_scale(self, tmp_path):
        """A pool of mixed units would be meaningless, so it is never built."""
        a = _pose_csv(tmp_path / "aDLC_m.csv")
        cohort = compute_cohort_thresholds([a], calibration_master=None)
        assert cohort.unit == PX_PER_FRAME
        assert cohort.is_calibrated is False

    def test_percentile_ordering_is_checked(self, tmp_path):
        a = _pose_csv(tmp_path / "aDLC_m.csv")
        with pytest.raises(CohortSpeedError, match="below"):
            compute_cohort_thresholds([a], freeze_pct=99.0, dart_pct=10.0)

    def test_an_empty_cohort_is_refused(self):
        with pytest.raises(CohortSpeedError, match="no pose CSVs"):
            compute_cohort_thresholds([])

    def test_freeze_sits_below_dart(self, tmp_path):
        a = _pose_csv(tmp_path / "aDLC_m.csv")
        cohort = compute_cohort_thresholds([a])
        assert cohort.freeze < cohort.dart


class TestPersistence:
    def _built(self, tmp_path):
        return compute_cohort_thresholds([_pose_csv(tmp_path / "aDLC_m.csv")])

    def test_round_trip(self, tmp_path):
        original = self._built(tmp_path)
        path = tmp_path / "cohort.json"
        original.save(path)
        loaded = CohortSpeedThresholds.load(path)
        assert loaded.freeze == pytest.approx(original.freeze)
        assert loaded.dart == pytest.approx(original.dart)
        assert loaded.unit == original.unit
        assert loaded.n_samples == original.n_samples

    def test_the_file_records_the_unit_and_percentiles(self, tmp_path):
        path = tmp_path / "cohort.json"
        self._built(tmp_path).save(path)
        data = json.loads(path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["unit"] in (CM_PER_S, PX_PER_FRAME)
        assert "freeze_pct" in data and "n_sessions" in data

    def test_unknown_version_is_refused(self, tmp_path):
        path = tmp_path / "cohort.json"
        path.write_text(json.dumps({"schema_version": 99}))
        with pytest.raises(CohortSpeedError, match="schema_version"):
            CohortSpeedThresholds.load(path)

    def test_malformed_file_is_refused(self, tmp_path):
        path = tmp_path / "cohort.json"
        path.write_text("{not json")
        with pytest.raises(CohortSpeedError):
            CohortSpeedThresholds.load(path)


class TestApplyingToOneVideo:
    def _cm(self, freeze=1.0, dart=15.0):
        return CohortSpeedThresholds(
            freeze=freeze,
            dart=dart,
            unit=CM_PER_S,
            freeze_pct=10.0,
            dart_pct=99.5,
            n_sessions=3,
            n_samples=100,
        )

    def test_centimetre_thresholds_convert_through_this_videos_geometry(self):
        # 1 cm/s at 3 px/mm and 30 fps = 10 mm/s * 3 / 30 = 1 px/frame.
        freeze, dart = self._cm().to_px_per_frame(px_per_mm=3.0, fps=30.0)
        assert freeze == pytest.approx(1.0)
        assert dart == pytest.approx(15.0)

    def test_a_different_video_geometry_gives_different_pixels(self):
        """One physical cut-off, applied to each session on its own terms."""
        a = self._cm().to_px_per_frame(px_per_mm=3.0, fps=30.0)
        b = self._cm().to_px_per_frame(px_per_mm=6.0, fps=30.0)
        assert b[1] == pytest.approx(a[1] * 2)

    def test_centimetre_thresholds_without_a_scale_fail_loudly(self):
        with pytest.raises(CohortSpeedError, match="pixel scale"):
            self._cm().to_px_per_frame(px_per_mm=None, fps=30.0)

    def test_pixel_thresholds_pass_straight_through(self):
        t = CohortSpeedThresholds(
            freeze=2.0,
            dart=18.0,
            unit=PX_PER_FRAME,
            freeze_pct=10.0,
            dart_pct=99.5,
            n_sessions=1,
            n_samples=10,
        )
        assert t.to_px_per_frame() == (2.0, 18.0)
