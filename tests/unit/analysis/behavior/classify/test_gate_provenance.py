"""Scoring a gated track against ungated cut-offs is refused.

Cohort thresholds are percentiles of a pooled speed distribution. Gating
removes out-of-arena detections, which lowers that distribution, so a cut-off
derived from ungated speed and applied to gated speed is the same class of
mistake as thresholding one time window against another -- and it produces a
plausible number with no error at all. This lab has already lost a cohort to a
related artifact, so the mismatch raises.

The check sits *before* the batch/stream fork, not inside ``batch_apply``.
``batch_apply`` declines two runs that nonetheless still read the same CSV --
an annotated output video, and a CNN sequence model -- and the caller then
falls through to :class:`LiveInferencePipeline`, whose ``_make_tracker`` reads
``config.pose_csv_in`` through ``PoseReplay``. Two of the tests below exist
only to pin that placement.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from glider.analysis.behavior.classify import classify
from glider.analysis.behavior.cohort_speed import PX_PER_FRAME, CohortSpeedThresholds
from glider.vision.arena_gate import ArenaGateSettings
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import meta_path, to_dlc_csv

KP = ["nose", "body_center", "tail_base"]


@pytest.fixture(autouse=True)
def no_streaming(monkeypatch):
    """Reaching the streaming pipeline is itself the failure.

    A refusal that fires only after the fork has already been taken would let
    the annotated-video and CNN runs score the mismatch, which is the whole
    thing being guarded against.
    """
    import glider.analysis.behavior.classify as mod

    def _explode(*_args, **_kwargs):
        raise AssertionError("the streaming pipeline should not have been reached")

    monkeypatch.setattr(mod, "LiveInferencePipeline", _explode)


def _gate_block() -> dict:
    """A gate block as a session sidecar carries it."""
    return {
        "frames_total": 200,
        "frames_considered": 198,
        "frames_blanked": 3,
        "keypoints_masked": 11,
        "masked_by_keypoint": dict.fromkeys(KP, 0),
        "settings": asdict(ArenaGateSettings(margin_cm=7.5)),
        "arena_corners": [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]],
        "gated": True,
    }


def _pose(n=200, seed=0):
    """A wandering animal, so percentiles of its speed are not degenerate."""
    rng = np.random.default_rng(seed)
    xy = np.cumsum(rng.normal(0, 1.5, size=(n, len(KP), 2)), axis=0) + 100.0
    return PoseData(xy=xy, confidence=np.ones((n, len(KP))), keypoint_names=KP, fps=30.0)


def _csv(tmp_path, *, gated):
    pose = _pose()
    if gated:
        pose.metadata["arena_gate"] = _gate_block()
    return to_dlc_csv(pose, tmp_path / "clipDLC_yolo.csv")


def _gated_csv(tmp_path):
    return _csv(tmp_path, gated=True)


def _ungated_csv(tmp_path):
    return _csv(tmp_path, gated=False)


def _cohort(tmp_path, *, gated):
    """Cohort cut-offs already in px/frame, so no calibration is involved."""
    path = tmp_path / "cohort_speed.json"
    CohortSpeedThresholds(
        freeze=0.5,
        dart=12.0,
        unit=PX_PER_FRAME,
        freeze_pct=10.0,
        dart_pct=99.5,
        n_sessions=31,
        n_samples=100_000,
        gate_provenance={"gated": gated, "settings": {}},
    ).save(path)
    return path


class _CnnSequenceModel:
    """Stands in for a sequence bundle: none of the tabular-model attributes
    ``batch_apply`` looks for, which is exactly how it declines one."""


def _run(tmp_path, **kwargs):
    """A speed-only apply run over poses already on disk, unless overridden."""
    defaults = {
        "video": "clip.mp4",
        "model_path": None,
        "yolo_path": None,
        "keypoint_names": KP,
        "output_dir": tmp_path / "out",
        "fps_override": 30.0,
        "predict_every": 1,
    }
    return classify(**{**defaults, **kwargs})


class TestRefusingAMismatch:
    def test_gated_csv_against_ungated_thresholds_raises(self, tmp_path):
        with pytest.raises(ValueError, match="(?i)re-derive"):
            _run(
                tmp_path,
                pose_csv_in=_gated_csv(tmp_path),
                cohort_thresholds=_cohort(tmp_path, gated=False),
            )

    def test_an_ungated_csv_against_gated_thresholds_raises_too(self, tmp_path):
        """The comparison is symmetric: either way the two describe different
        distributions, and neither direction is the safe one."""
        with pytest.raises(ValueError, match="(?i)re-derive"):
            _run(
                tmp_path,
                pose_csv_in=_ungated_csv(tmp_path),
                cohort_thresholds=_cohort(tmp_path, gated=True),
            )

    def test_the_message_names_the_file_and_both_sides(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            _run(
                tmp_path,
                pose_csv_in=_gated_csv(tmp_path),
                cohort_thresholds=_cohort(tmp_path, gated=False),
            )
        message = str(excinfo.value)
        assert "clipDLC_yolo.csv" in message
        assert "is gated" in message and "ungated poses" in message

    def test_it_raises_on_the_annotated_video_path_too(self, tmp_path):
        """batch_apply declines this one and falls through to the streaming
        pipeline, which reads the same CSV."""
        with pytest.raises(ValueError, match="(?i)re-derive"):
            _run(
                tmp_path,
                model_path="model.pkl",
                pose_csv_in=_gated_csv(tmp_path),
                cohort_thresholds=_cohort(tmp_path, gated=False),
                write_annotated=True,
            )

    def test_it_raises_for_a_cnn_sequence_model(self, tmp_path):
        """The other batch_apply decline path. Same fall-through, same exposure."""
        with pytest.raises(ValueError, match="(?i)re-derive"):
            _run(
                tmp_path,
                model=_CnnSequenceModel(),
                pose_csv_in=_gated_csv(tmp_path),
                cohort_thresholds=_cohort(tmp_path, gated=False),
            )

    def test_it_raises_before_either_branch_of_the_fork_runs(self, tmp_path, monkeypatch):
        """Neither ``batch_apply`` nor the streaming pipeline may be entered.

        The two tests above are satisfied by a check at the top of
        ``batch_apply`` as well as by one before the fork, because that path
        does at least call it. This one is not: it is what makes "before the
        fork" the only placement that passes.
        """
        from glider.analysis.behavior.classify import batch as batch_mod

        def _explode(*_args, **_kwargs):
            raise AssertionError("the fork was taken before the mismatch was refused")

        monkeypatch.setattr(batch_mod, "batch_apply", _explode)
        with pytest.raises(ValueError, match="(?i)re-derive"):
            _run(
                tmp_path,
                pose_csv_in=_gated_csv(tmp_path),
                cohort_thresholds=_cohort(tmp_path, gated=False),
            )


class TestWhatIsNotChecked:
    def test_matching_provenance_scores_normally(self, tmp_path):
        _run(
            tmp_path,
            pose_csv_in=_gated_csv(tmp_path),
            cohort_thresholds=_cohort(tmp_path, gated=True),
        )
        assert (tmp_path / "out" / "ethogram_raw.csv").exists()

    def test_two_ungated_sides_also_score_normally(self, tmp_path):
        _run(
            tmp_path,
            pose_csv_in=_ungated_csv(tmp_path),
            cohort_thresholds=_cohort(tmp_path, gated=False),
        )
        assert (tmp_path / "out" / "ethogram_raw.csv").exists()

    def test_absolute_thresholds_are_not_checked(self, tmp_path):
        """cm/s cut-offs are not derived from ungated poses at all. Since the
        post-hoc pass rewrites every tracked session, checking here would break
        every non-cohort scoring run."""
        _run(
            tmp_path,
            pose_csv_in=_gated_csv(tmp_path),
            freeze_cm_s=0.5,
            dart_cm_s=30.0,
            px_per_mm=4.0,
        )
        assert (tmp_path / "out" / "ethogram_raw.csv").exists()

    def test_percentile_thresholds_are_not_checked(self, tmp_path):
        """They are derived from the very CSV being scored, so they cannot
        disagree with it."""
        gated = _gated_csv(tmp_path)
        _run(tmp_path, pose_csv_in=gated, pose_csv=gated, freeze_pct=1.0, dart_pct=99.5)
        assert (tmp_path / "out" / "ethogram_raw.csv").exists()

    def test_a_run_that_uses_no_speed_thresholds_is_not_checked(self, tmp_path, monkeypatch):
        """With both sides of the speed axis off, ``resolve_speed_thresholds``
        returns ``{}`` before it opens the cohort file at all -- so no cut-off
        derived under any gate is applied to this CSV, and there is nothing to
        disagree with. The check saw the missing provenance as "ungated" and
        refused a gated CSV over thresholds that were never used.
        """
        from glider.analysis.behavior.classify import batch as batch_mod

        def _apply(config, ethogram_csv, model, **kwargs):
            Path(ethogram_csv).write_text("frame,behavior\n0,walking\n")
            return True

        monkeypatch.setattr(batch_mod, "batch_apply", _apply)
        _run(
            tmp_path,
            model=object(),
            pose_csv_in=_gated_csv(tmp_path),
            cohort_thresholds=_cohort(tmp_path, gated=False),
            score_freezing=False,
            score_darting=False,
        )
        assert (tmp_path / "out" / "ethogram_raw.csv").exists()

    def test_an_explicit_null_gate_block_reads_as_ungated(self, tmp_path):
        """``.get("arena_gate", {})`` hands back the ``None`` that is in the
        file, not the default, and ``None`` has no ``.get`` -- so a sidecar
        spelling the absent block out as null crashed instead of scoring.
        The sibling reads in cohort_speed.py already use the ``or {}`` form.
        """
        csv = _ungated_csv(tmp_path)
        meta = json.loads(meta_path(csv).read_text())
        meta["arena_gate"] = None
        meta_path(csv).write_text(json.dumps(meta))
        _run(tmp_path, pose_csv_in=csv, cohort_thresholds=_cohort(tmp_path, gated=False))
        assert (tmp_path / "out" / "ethogram_raw.csv").exists()

    def test_a_speed_only_run_with_no_csv_is_not_checked(self, tmp_path, monkeypatch):
        """No CSV, no sidecar, no CSV-side provenance to compare."""
        import glider.analysis.behavior.classify as mod

        monkeypatch.setattr(mod, "_track_poses", lambda config, pose_csv_out: _pose())
        _run(
            tmp_path,
            yolo_path="yolo.pt",
            cohort_thresholds=_cohort(tmp_path, gated=False),
        )
        assert (tmp_path / "out" / "ethogram_raw.csv").exists()
