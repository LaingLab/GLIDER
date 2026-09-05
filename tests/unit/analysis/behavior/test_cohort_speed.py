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


class TestSayingWhatTheFileContains:
    """The numbers live in a JSON nobody opens; the file has to describe itself.

    "Which cut-offs is this run using" is not answered by a path, and the unit
    matters most of all: a cohort pooled without a calibration is in px/frame
    and looks nothing like the cm/s the operator typed everywhere else.
    """

    def _thresholds(self, **kw):
        base = {
            "freeze": 0.55,
            "dart": 27.68,
            "unit": CM_PER_S,
            "freeze_pct": 10.0,
            "dart_pct": 99.5,
            "n_sessions": 30,
            "n_samples": 269_964,
            "start_s": 120.0,
            "end_s": 420.0,
        }
        return CohortSpeedThresholds(**{**base, **kw})

    def test_it_states_the_cut_offs_their_unit_and_their_window(self):
        text = self._thresholds().describe()
        assert "0.55 cm/s" in text
        assert "27.7 cm/s" in text
        assert "p10/p99.5" in text
        assert "30 session(s)" in text
        assert "2–7 min" in text

    def test_a_pixel_pool_says_so_in_the_same_place(self):
        assert "px/frame" in self._thresholds(unit=PX_PER_FRAME).describe()

    def test_a_whole_recording_pool_says_that(self):
        text = self._thresholds(start_s=None, end_s=None).describe()
        assert "the whole recording" in text


class TestWhyAPoolFellBackToPixels:
    """One session without a scale costs the whole pool its units.

    That is correct — a pool of mixed units is meaningless — but it used to be
    reported only to the log, which is invisible from the GUI. The operator saw
    a file in the wrong unit and no reason for it.
    """

    def test_the_count_of_uncalibrated_sessions_is_recorded(self, tmp_path):
        for name in ("a", "b"):
            _pose_csv(tmp_path / f"{name}DLC_x.csv")
            (tmp_path / f"{name}.mp4").write_bytes(b"")
        result = compute_cohort_thresholds(sorted(tmp_path.glob("*DLC_x.csv")))
        assert result.unit == PX_PER_FRAME
        assert result.n_uncalibrated == 2

    def test_a_calibrated_pool_records_none(self, tmp_path):
        for name in ("a", "b"):
            _pose_csv(tmp_path / f"{name}DLC_x.csv")
            (tmp_path / f"{name}.mp4").write_bytes(b"")
        result = compute_cohort_thresholds(
            sorted(tmp_path.glob("*DLC_x.csv")), px_per_mm=4.0, fps=30.0
        )
        assert result.unit == CM_PER_S
        assert result.n_uncalibrated == 0

    def test_it_survives_a_round_trip_through_the_file(self, tmp_path):
        path = tmp_path / "cohort.json"
        CohortSpeedThresholds(
            freeze=1.0,
            dart=9.0,
            unit=PX_PER_FRAME,
            freeze_pct=10.0,
            dart_pct=99.5,
            n_sessions=30,
            n_samples=100,
            n_uncalibrated=7,
        ).save(path)
        assert CohortSpeedThresholds.load(path).n_uncalibrated == 7

    def test_a_file_written_before_this_existed_still_loads(self, tmp_path):
        path = tmp_path / "cohort.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "freeze": 1.0,
                    "dart": 9.0,
                    "unit": PX_PER_FRAME,
                    "n_sessions": 30,
                }
            )
        )
        assert CohortSpeedThresholds.load(path).n_uncalibrated == 0


def _gate_block(gated: bool) -> dict:
    """A gate block as a *session sidecar* carries it: corners, counts and all.

    The cohort keeps only part of this, so the helper hands out the whole
    thing — a test that fed the trimmed shape in could not notice the trim.
    """
    from dataclasses import asdict

    from glider.vision.arena_gate import ArenaGateSettings

    return {
        "frames_total": 9000,
        "frames_considered": 8800,
        "frames_blanked": 41,
        "keypoints_masked": 260,
        "masked_by_keypoint": {"a": 260, "b": 0, "c": 0},
        "settings": asdict(ArenaGateSettings(margin_cm=7.5)),
        "arena_corners": [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]],
        "gated": gated,
    }


def _thresholds(*, gated: bool) -> CohortSpeedThresholds:
    return CohortSpeedThresholds(
        freeze=0.55,
        dart=27.68,
        unit=CM_PER_S,
        freeze_pct=10.0,
        dart_pct=99.5,
        n_sessions=31,
        n_samples=269_964,
        gate_provenance=_gate_block(gated),
    )


def _session_csv(tmp_path, name, *, gated, blanked=41, settings=None):
    """A session pose CSV whose sidecar says whether, and how, it was gated."""
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    rng = np.random.default_rng(len(name))
    xy = np.cumsum(rng.normal(0, 2.0, size=(200, 3, 2)), axis=0) + 100.0
    block = {**_gate_block(True), "frames_blanked": blanked} if gated else None
    if block is not None and settings is not None:
        block["settings"] = settings
    pose = PoseData(
        xy=xy,
        confidence=np.ones((200, 3)),
        keypoint_names=["a", "b", "c"],
        fps=30.0,
        metadata={"arena_gate": block} if block else {},
    )
    return to_dlc_csv(pose, tmp_path / f"{name}DLC_m.csv")


class TestWhichGateTheseWereDerivedUnder:
    """Thresholds are percentiles of a pooled distribution, and gating lowers
    that distribution. Applying an ungated cut-off to a gated track produces a
    plausible number with no error, so the file has to say which it is.
    """

    def test_the_cohort_block_omits_arena_corners(self):
        """Corners are per-video: 31 sessions have 31 perimeters, so no single
        fingerprint could match them all and comparing would raise every time."""
        block = _thresholds(gated=True).to_dict()["arena_gate"]
        assert "arena_corners" not in block
        assert block["gated"] is True
        assert "settings" in block

    def test_a_v1_file_still_loads(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "freeze": 0.5,
                    "dart": 30.0,
                    "unit": CM_PER_S,
                    "n_sessions": 4,
                }
            )
        )
        assert CohortSpeedThresholds.load(path) is not None

    def test_a_v1_file_reads_as_ungated(self, tmp_path):
        """Gating did not exist when v1 was written, so absent means ungated --
        the only reading that does not silently defeat the guard on stale files."""
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "freeze": 0.5,
                    "dart": 30.0,
                    "unit": CM_PER_S,
                    "n_sessions": 4,
                }
            )
        )
        assert CohortSpeedThresholds.load(path).gate_provenance["gated"] is False

    def test_the_block_survives_a_round_trip(self, tmp_path):
        path = tmp_path / "c.json"
        _thresholds(gated=True).save(path)
        loaded = CohortSpeedThresholds.load(path)
        assert loaded.gate_provenance["gated"] is True
        assert loaded.gate_provenance["settings"]["margin_cm"] == 7.5

    def test_a_gated_pool_records_that_it_was_gated(self, tmp_path):
        a = _session_csv(tmp_path, "a", gated=True)
        b = _session_csv(tmp_path, "b", gated=True)
        block = compute_cohort_thresholds([a, b]).gate_provenance
        assert block["gated"] is True
        assert block["settings"]["margin_cm"] == 7.5
        assert "arena_corners" not in block

    def test_an_ungated_pool_records_that_it_was_not(self, tmp_path):
        a = _session_csv(tmp_path, "a", gated=False)
        assert compute_cohort_thresholds([a]).gate_provenance["gated"] is False

    def test_a_mixed_pool_is_refused(self, tmp_path):
        """One boolean cannot describe a half-gated cohort: whichever value it
        took, the other half would hard-raise at scoring time."""
        gated = _session_csv(tmp_path, "a", gated=True)
        ungated = _session_csv(tmp_path, "b", gated=False)
        with pytest.raises(CohortSpeedError, match="mix of gated and ungated"):
            compute_cohort_thresholds([gated, ungated], px_per_mm=1.3, fps=30.0)

    def test_the_mixed_pool_is_refused_before_any_session_is_read(self, tmp_path, monkeypatch):
        """Reading a real cohort is minutes of CSV; the refusal is arithmetic."""
        import glider.vision.pose.dlc as dlc_mod

        gated = _session_csv(tmp_path, "a", gated=True)
        ungated = _session_csv(tmp_path, "b", gated=False)
        monkeypatch.setattr(
            dlc_mod,
            "from_dlc_csv",
            lambda *a, **k: pytest.fail("a session was read before the pool was checked"),
        )
        with pytest.raises(CohortSpeedError, match="mix of gated and ungated"):
            compute_cohort_thresholds([gated, ungated])

    def test_a_pool_gated_under_different_settings_is_refused(self, tmp_path):
        """The block carries one settings dict, so a pool that mixes gates is
        misdescribed by whichever one it keeps -- and the sessions on the other
        side then hard-raise at scoring time, one video at a time, after the
        pooling has been paid for. The documented escalation workflow (defaults
        first, then min_detected_fraction=1.0 for the known-bad sessions) is
        how a folder ends up like this."""
        from dataclasses import asdict

        from glider.vision.arena_gate import ArenaGateSettings

        lenient = _session_csv(tmp_path, "a", gated=True)
        strict = _session_csv(
            tmp_path,
            "b",
            gated=True,
            settings=asdict(ArenaGateSettings(margin_cm=7.5, min_detected_fraction=1.0)),
        )
        with pytest.raises(CohortSpeedError, match="(?i)settings"):
            compute_cohort_thresholds([lenient, strict], px_per_mm=1.3, fps=30.0)

    def test_the_settings_refusal_names_both_sessions(self, tmp_path):
        from dataclasses import asdict

        from glider.vision.arena_gate import ArenaGateSettings

        lenient = _session_csv(tmp_path, "a", gated=True)
        strict = _session_csv(
            tmp_path, "b", gated=True, settings=asdict(ArenaGateSettings(margin_cm=1.0))
        )
        with pytest.raises(CohortSpeedError) as excinfo:
            compute_cohort_thresholds([lenient, strict], px_per_mm=1.3, fps=30.0)
        assert "aDLC_m.csv" in str(excinfo.value)
        assert "bDLC_m.csv" in str(excinfo.value)

    def test_settings_that_only_omit_the_defaults_pool_together(self, tmp_path):
        """Same gate, written two ways. Refusing that pair would refuse a
        cohort that is in fact consistent."""
        spelled_out = _session_csv(tmp_path, "a", gated=True)
        terse = _session_csv(tmp_path, "b", gated=True, settings={"margin_cm": 7.5})
        block = compute_cohort_thresholds([spelled_out, terse], px_per_mm=1.3, fps=30.0)
        assert block.gate_provenance["gated"] is True

    def test_a_heavily_blanked_session_is_named(self, tmp_path, caplog):
        """The cohort percentile is derived from whatever survived the gate."""
        heavy = _session_csv(tmp_path, "heavy", gated=True, blanked=4000)
        with caplog.at_level("WARNING"):
            # Calibrated, so the only thing that could name it is the gate.
            compute_cohort_thresholds([heavy], px_per_mm=4.0, fps=30.0)
        assert "heavyDLC_m.csv" in caplog.text

    def test_a_lightly_blanked_session_is_not(self, tmp_path, caplog):
        light = _session_csv(tmp_path, "light", gated=True, blanked=4)
        with caplog.at_level("WARNING"):
            compute_cohort_thresholds([light], px_per_mm=4.0, fps=30.0)
        assert "lightDLC_m.csv" not in caplog.text
