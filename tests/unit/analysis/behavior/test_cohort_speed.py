"""Cohort-wide freeze/dart thresholds.

Per-video percentiles are circular in a treatment study: a drugged animal that
moves less gets a lower darting threshold, normalising away the effect. These
tests pin that one pooled set of cut-offs is produced instead.
"""

from __future__ import annotations

import json
from pathlib import Path

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


class TestCostAndProgress:
    """Pooling a real cohort is minutes of work, so waste and silence matter."""

    def test_each_session_is_read_exactly_once(self, tmp_path, monkeypatch):
        """The uncalibrated fallback used to re-read and recompute everything."""
        import glider.vision.pose.dlc as dlc_mod
        from glider.analysis.behavior import cohort_speed

        a = _pose_csv(tmp_path / "aDLC_m.csv", n=60)
        b = _pose_csv(tmp_path / "bDLC_m.csv", n=60, seed=2)

        reads: list[str] = []
        real = dlc_mod.from_dlc_csv

        def counting(path, *args, **kwargs):
            reads.append(Path(path).name)
            return real(path, *args, **kwargs)

        monkeypatch.setattr(dlc_mod, "from_dlc_csv", counting)
        # No calibration, so this takes the fallback path.
        cohort_speed.compute_cohort_thresholds([a, b])
        assert reads.count("aDLC_m.csv") == 1, reads
        assert reads.count("bDLC_m.csv") == 1, reads

    def test_progress_is_reported_per_session(self, tmp_path):
        a = _pose_csv(tmp_path / "aDLC_m.csv", n=40)
        b = _pose_csv(tmp_path / "bDLC_m.csv", n=40, seed=2)
        seen = []
        compute_cohort_thresholds([a, b], progress=lambda d, t, name: seen.append((d, t, name)))
        assert [(d, t) for d, t, _ in seen] == [(1, 2), (2, 2)]
        assert seen[0][2] == "aDLC_m.csv"

    def test_calibrated_pooling_still_lands_in_centimetres(self, tmp_path):
        """The single-pass rewrite scales after reading rather than during."""
        a = _pose_csv(tmp_path / "aDLC_m.csv", n=60)
        px = compute_cohort_thresholds([a])
        cm = compute_cohort_thresholds([a], px_per_mm=4.0, fps=30.0)
        assert px.unit == PX_PER_FRAME and cm.unit == CM_PER_S
        # 1 px/frame at 4 px/mm and 30 fps = 0.75 cm/s.
        assert cm.dart == pytest.approx(px.dart * 0.75, rel=1e-6)


class TestPoolingOnlyAWindow:
    """Thresholds describe the behaviour they are applied to.

    A run that scores minutes 2-7 thresholded against the whole recording is
    not a rounding difference: on a real 30-animal cohort the freezing cut-off
    moved 34% between the two, because the settling-in period the ethogram
    never covers is where the stillest frames are.
    """

    def _session(self, tmp_path, name="s1", n=600, fps=30.0):
        """A session that is still for its first third, then moves."""
        from glider.vision.pose.core import PoseData
        from glider.vision.pose.dlc import to_dlc_csv

        xy = np.zeros((n, 3, 2))
        moving = np.arange(n) >= n // 3
        xy[:, :, 0] = np.where(moving, np.arange(n) * 4.0, 0.0)[:, None]
        path = tmp_path / f"{name}DLC_m.csv"
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

    def test_a_window_excludes_the_still_opening(self, tmp_path):
        path = self._session(tmp_path)
        whole, _ = session_speeds(path)
        # The first third is motionless; skipping it must raise the floor.
        windowed, _ = session_speeds(path, start_s=10.0)
        assert windowed.min() > whole.min()

    def test_the_window_is_in_seconds_at_each_session_rate(self, tmp_path):
        fast = self._session(tmp_path, name="fast", n=600, fps=60.0)
        # 2 s at 60 fps is 120 frames; at 30 fps it would be 60.
        speeds, _ = session_speeds(fast, start_s=2.0, end_s=4.0)
        assert 110 <= speeds.size <= 121

    def test_the_causal_filter_is_not_restarted_at_the_window(self, tmp_path):
        """Its value inside the window depends on the frames before it, and
        the apply run windows the same way."""
        from glider.analysis.behavior.classify.speed_state import CausalSpeed
        from glider.vision.pose.dlc import from_dlc_csv

        path = self._session(tmp_path)
        pose = from_dlc_csv(path)
        causal = CausalSpeed()
        full = np.array([causal.push(f) for f in pose.xy])

        windowed, _ = session_speeds(path, start_s=5.0, end_s=6.0)
        first, last = int(round(5.0 * 30)), int(round(6.0 * 30)) - 1
        expected = full[first : last + 1]
        expected = expected[np.isfinite(expected)]
        assert windowed == pytest.approx(expected)

    def test_the_window_is_recorded_in_the_file(self, tmp_path):
        paths = [self._session(tmp_path, name=f"s{i}") for i in range(3)]
        thresholds = compute_cohort_thresholds(paths, start_s=2.0, end_s=8.0)
        assert thresholds.start_s == pytest.approx(2.0)
        assert thresholds.end_s == pytest.approx(8.0)
        assert thresholds.window == (2.0, 8.0)
        assert "min" in thresholds.describe_window()

    def test_the_window_survives_a_round_trip(self, tmp_path):
        paths = [self._session(tmp_path, name=f"s{i}") for i in range(3)]
        out = tmp_path / "cohort.json"
        compute_cohort_thresholds(paths, start_s=2.0, end_s=8.0).save(out)
        assert CohortSpeedThresholds.load(out).window == (2.0, 8.0)

    def test_no_window_means_the_whole_recording(self, tmp_path):
        paths = [self._session(tmp_path, name=f"s{i}") for i in range(3)]
        thresholds = compute_cohort_thresholds(paths)
        assert thresholds.window is None
        assert thresholds.describe_window() == "the whole recording"

    def test_a_file_written_before_windowing_reads_as_whole(self, tmp_path):
        """Absent fields mean exactly what they say."""
        import json

        path = tmp_path / "old.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "unit": "cm/s",
                    "freeze": 0.4,
                    "dart": 27.0,
                    "freeze_pct": 10.0,
                    "dart_pct": 99.5,
                    "n_sessions": 30,
                    "n_samples": 1000,
                    "sources": [],
                }
            )
        )
        assert CohortSpeedThresholds.load(path).window is None

    def test_a_backwards_window_is_refused(self, tmp_path):
        with pytest.raises(CohortSpeedError, match="ends before it starts"):
            compute_cohort_thresholds([self._session(tmp_path)], start_s=8.0, end_s=2.0)
