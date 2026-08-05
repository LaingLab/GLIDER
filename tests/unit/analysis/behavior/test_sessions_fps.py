"""Frame-rate resolution across a set of training sessions.

Every windowed feature is specified in seconds, so reading 60 fps pose data
at the old hardcoded 30.0 silently computed each rolling window over half its
intended span — with nothing downstream able to detect it had happened.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior.pipeline import resolve_sessions_fps
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import DEFAULT_FPS, to_dlc_csv

NAMES = ["snout", "neck", "tail_base"]


def _session(tmp_path, stem, fps, *, write_meta=True):
    """A (pose_csv, annotations_csv) pair whose pose CSV records *fps*."""
    pose = PoseData(
        xy=np.zeros((20, len(NAMES), 2)),
        confidence=np.ones((20, len(NAMES))),
        keypoint_names=NAMES,
        fps=fps,
    )
    pose_csv = tmp_path / f"{stem}.csv"
    to_dlc_csv(pose, pose_csv, write_meta=write_meta)
    return (pose_csv, tmp_path / f"{stem}_annotations.csv")


def test_takes_the_rate_the_pose_csvs_recorded(tmp_path):
    sessions = [_session(tmp_path, "a", 60.0), _session(tmp_path, "b", 60.0)]

    assert resolve_sessions_fps(sessions, None) == pytest.approx(60.0)


def test_an_explicit_rate_wins(tmp_path):
    sessions = [_session(tmp_path, "a", 60.0)]

    assert resolve_sessions_fps(sessions, 25.0) == pytest.approx(25.0)


def test_mixed_rates_are_refused(tmp_path):
    """One rolling window can't mean two different durations in one model."""
    sessions = [_session(tmp_path, "fast", 60.0), _session(tmp_path, "slow", 30.0)]

    with pytest.raises(ValueError, match="different frame rates"):
        resolve_sessions_fps(sessions, None)


def test_mixed_rates_can_be_forced(tmp_path):
    sessions = [_session(tmp_path, "fast", 60.0), _session(tmp_path, "slow", 30.0)]

    assert resolve_sessions_fps(sessions, 30.0) == pytest.approx(30.0)


def test_encoder_jitter_is_one_rate(tmp_path):
    """Container timestamps wobble; a cohort filmed on one rig is one rate.

    Real sidecars off the same camera read 29.979620 / 30.000000 / 30.000727 --
    a 0.07% spread, which over an 8-frame window is 0.19 ms. Refusing that
    blocks training on a perfectly uniform cohort, and the difference is far
    below anything a rolling window can express.
    """
    sessions = [
        _session(tmp_path, "a", 29.979620),
        _session(tmp_path, "b", 30.000000),
        _session(tmp_path, "c", 30.000727),
    ]

    assert resolve_sessions_fps(sessions, None) == pytest.approx(30.0, abs=0.05)


def test_ntsc_and_round_thirty_are_one_rate(tmp_path):
    """29.97 vs 30.0 is a 0.1% difference -- a naming convention, not a rate."""
    sessions = [_session(tmp_path, "ntsc", 29.97), _session(tmp_path, "round", 30.0)]

    assert resolve_sessions_fps(sessions, None) == pytest.approx(30.0, abs=0.05)


def test_a_real_rate_difference_is_still_refused(tmp_path):
    """The guard must keep catching what it exists for."""
    sessions = [_session(tmp_path, "pal", 25.0), _session(tmp_path, "ntsc", 30.0)]

    with pytest.raises(ValueError, match="different frame rates"):
        resolve_sessions_fps(sessions, None)


def test_the_refusal_names_the_rates_and_their_sessions(tmp_path):
    """The message has to say which files to split out, not just that it failed."""
    sessions = [_session(tmp_path, "fast", 60.0), _session(tmp_path, "slow", 30.0)]

    with pytest.raises(ValueError) as excinfo:
        resolve_sessions_fps(sessions, None)
    message = str(excinfo.value)
    assert "fast.csv" in message and "slow.csv" in message
    assert "fps=" in message


def test_holdout_sessions_are_checked_too(tmp_path):
    """A holdout at another rate would be scored against mismatched windows."""
    sessions = [_session(tmp_path, "train", 30.0)]
    holdout = [_session(tmp_path, "held", 60.0)]

    with pytest.raises(ValueError, match="different frame rates"):
        resolve_sessions_fps(sessions, None, holdout_sessions=holdout)


def test_unrecorded_rate_warns_and_falls_back(tmp_path):
    """Old CSVs keep working, but the assumption is stated out loud."""
    sessions = [_session(tmp_path, "legacy", 30.0, write_meta=False)]

    with pytest.warns(UserWarning, match="no frame rate recorded"):
        assert resolve_sessions_fps(sessions, None) == pytest.approx(DEFAULT_FPS)


def test_partial_sidecars_use_what_is_known(tmp_path):
    """One session missing its sidecar shouldn't discard the others' rate."""
    sessions = [
        _session(tmp_path, "known", 60.0),
        _session(tmp_path, "legacy", 60.0, write_meta=False),
    ]

    assert resolve_sessions_fps(sessions, None) == pytest.approx(60.0)


def test_train_model_records_the_recorded_rate_in_the_bundle(tmp_path):
    """The rate reaches the bundle, so live inference matches training."""
    pytest.importorskip("sklearn")
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.pipeline import train_model

    rng = np.random.default_rng(0)
    pose = PoseData(
        xy=rng.normal(300, 30, size=(400, len(NAMES), 2)),
        confidence=np.ones((400, len(NAMES))),
        keypoint_names=NAMES,
        fps=60.0,
    )
    pose_csv = tmp_path / "s1.csv"
    to_dlc_csv(pose, pose_csv)

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="walk", start_frame=0, end_frame=200))
    store.add(BehaviorZone(behavior="rear", start_frame=200, end_frame=400))
    ann_csv = tmp_path / "s1_annotations.csv"
    store.save_csv(ann_csv)

    result = train_model([(pose_csv, ann_csv)], window=10, classifier_type="rf")

    assert result.summary["fps"] == pytest.approx(60.0)
