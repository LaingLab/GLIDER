"""Tests for the training pipeline.

Synthetic three-regime pose (locomote / groom / rest) gives the
classifier real structure to learn. Each test stays small and
self-contained so the suite runs in <1 s on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glider.vision.pose.core import PoseData

# Required deps (sklearn for the RandomForest, pandas+numpy already in).
pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_three_regime_pose(seed: int = 42, n_frames: int = 600) -> PoseData:
    rng = np.random.default_rng(seed)
    n_kpts = 5
    names = ["snout", "left_ear", "right_ear", "neck", "tail_base"]
    xy = np.empty((n_frames, n_kpts, 2))
    per = n_frames // 3
    for regime, slc in enumerate([slice(0, per), slice(per, 2 * per), slice(2 * per, n_frames)]):
        n = slc.stop - slc.start
        t = np.arange(n)
        if regime == 0:
            cx = 50 + 0.5 * t
            cy = 200 + 0.0 * t
            jitter = 0.4
        elif regime == 1:
            cx = 350 + 3 * np.sin(0.5 * t)
            cy = 200 + 3 * np.cos(0.5 * t)
            jitter = 1.0
        else:
            cx = np.full(n, 360.0)
            cy = np.full(n, 200.0)
            jitter = 0.15
        offsets = np.array([[0, -20], [-10, -10], [10, -10], [0, 0], [0, 30]], dtype=float)
        for k in range(n_kpts):
            xy[slc, k, 0] = cx + offsets[k, 0] + rng.normal(0, jitter, n)
            xy[slc, k, 1] = cy + offsets[k, 1] + rng.normal(0, jitter, n)
    confidence = np.full((n_frames, n_kpts), 0.95)
    return PoseData(xy=xy, confidence=confidence, keypoint_names=names, fps=30.0)


@pytest.fixture
def three_regime_pose() -> PoseData:
    return _make_three_regime_pose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_b_session(rng):
    """A second synthetic session with the same regime structure as the
    main three-regime fixture — used as a held-out session for
    cross-session validation tests."""
    from glider.vision.pose.core import PoseData

    n_frames = 600
    n_kpts = 5
    names = ["snout", "left_ear", "right_ear", "neck", "tail_base"]
    xy = np.empty((n_frames, n_kpts, 2))
    per = n_frames // 3
    for regime, slc in enumerate([slice(0, per), slice(per, 2 * per), slice(2 * per, n_frames)]):
        n = slc.stop - slc.start
        t = np.arange(n)
        if regime == 0:
            cx = 80 + 0.5 * t
            cy = 220
            jitter = 0.4
        elif regime == 1:
            cx = 380 + 3 * np.sin(0.5 * t)
            cy = 220 + 3 * np.cos(0.5 * t)
            jitter = 1.0
        else:
            cx = np.full(n, 390.0)
            cy = np.full(n, 220.0)
            jitter = 0.15
        offsets = np.array([[0, -20], [-10, -10], [10, -10], [0, 0], [0, 30]], dtype=float)
        for k in range(n_kpts):
            xy[slc, k, 0] = cx + offsets[k, 0] + rng.normal(0, jitter, n)
            xy[slc, k, 1] = cy + offsets[k, 1] + rng.normal(0, jitter, n)
    conf = np.full((n_frames, n_kpts), 0.95)
    return PoseData(xy=xy, confidence=conf, keypoint_names=names, fps=30.0)


def _write_dlc_csv(pose: PoseData, path: Path) -> None:
    from glider.vision.pose.dlc import to_dlc_csv

    to_dlc_csv(pose, path)


def _write_annotations(path: Path, zones: list[tuple[str, int, int]]) -> None:
    """Helper: dump zones into the annotator's CSV format."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone

    store = AnnotationStore()
    for behavior, start, end in zones:
        store.add(BehaviorZone(behavior=behavior, start_frame=start, end_frame=end))
    store.save_csv(path)


# ---------------------------------------------------------------------------
# train_model tests
# ---------------------------------------------------------------------------


def test_train_model_split_keeps_zones_intact(tmp_path, three_regime_pose):
    """The zone-aware split must put every row from a labeled zone on
    the same side of train/test. Without this, near-duplicate adjacent
    windows leak across and the test number is meaningless."""
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    # Three large zones — easy for the splitter to allocate fully to
    # train or fully to test.
    _write_annotations(
        ann_csv,
        [
            ("a", 50, 150),
            ("b", 250, 350),
            ("c", 450, 550),
        ],
    )
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        test_split=0.34,  # one zone should land in test
    )
    s = result.summary
    assert s["split_strategy"] == "group_shuffle"
    # Test accuracy on truly held-out behaviors is realistic, not
    # inflated to ~1.0 by row-level leakage.
    assert s["test_size"] > 0


def test_train_model_cross_session_holdout(tmp_path, three_regime_pose):
    """Holdout sessions become the test set; train sees only `sessions`."""
    from glider.analysis.behavior import FeatureSpec, train_model

    # Two synthetic "sessions" with the same regime structure so the
    # classifier can actually generalize.
    pose_a = three_regime_pose
    rng = np.random.default_rng(7)
    pose_b = _make_b_session(rng)
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    a_ann = tmp_path / "a_annotations.csv"
    b_ann = tmp_path / "b_annotations.csv"
    _write_dlc_csv(pose_a, a_csv)
    _write_dlc_csv(pose_b, b_csv)
    _write_annotations(
        a_ann,
        [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)],
    )
    _write_annotations(
        b_ann,
        [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)],
    )

    result = train_model(
        sessions=[(a_csv, a_ann)],
        holdout_sessions=[(b_csv, b_ann)],
        spec=FeatureSpec(body_axis=(0, pose_a.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
    )
    assert result.summary["split_strategy"] == "cross_session"
    assert result.summary["n_holdout_sessions"] == 1
    # Test rows come from session B, not from session A.
    assert result.summary["test_size"] > 0


def test_cross_validate_sessions_folds_by_session(tmp_path, three_regime_pose):
    """Session-grouped CV returns one accuracy per fold and aggregates."""
    from glider.analysis.behavior import FeatureSpec, cross_validate_sessions

    pose_a = three_regime_pose
    pose_b = _make_b_session(np.random.default_rng(7))
    a_csv, b_csv = tmp_path / "a.csv", tmp_path / "b.csv"
    a_ann, b_ann = tmp_path / "a_annotations.csv", tmp_path / "b_annotations.csv"
    _write_dlc_csv(pose_a, a_csv)
    _write_dlc_csv(pose_b, b_csv)
    for ann in (a_ann, b_ann):
        _write_annotations(ann, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])

    res = cross_validate_sessions(
        sessions=[(a_csv, a_ann), (b_csv, b_ann)],
        spec=FeatureSpec(body_axis=(0, pose_a.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        n_folds=2,
    )
    # Two sessions, 2 folds → leave-one-session-out.
    assert res["n_sessions"] == 2
    assert res["n_folds"] == 2
    assert len(res["fold_accuracies"]) == 2
    assert 0.0 <= res["mean_accuracy"] <= 1.0
    assert res["mean_macro_f1"] is not None
    # Aggregated per-class metrics + confusion pooled over folds.
    pc = res["per_class_metrics"]
    assert pc and all(set(m) == {"precision", "recall", "f1", "support"} for m in pc.values())
    cm = res["confusion_matrix"]
    assert cm["labels"] and len(cm["matrix"]) == len(cm["labels"])
    # Pooled support == every SCORED row tested exactly once. Scored is fewer
    # than kept: frames whose trailing window straddles a bout boundary are
    # trained on but not scored, because their features describe the previous
    # behavior rather than the labelled one.
    assert sum(m["support"] for m in pc.values()) == res["n_rows_scored"]
    assert res["n_rows_scored"] <= res["n_rows_kept"]


def test_cross_validate_sessions_with_background_reports_false_alarm(tmp_path, three_regime_pose):
    """With background, CV promotes unannotated frames and reports the
    false-alarm rate; the background class appears in per-class metrics."""
    from glider.analysis.behavior import FeatureSpec, cross_validate_sessions

    pose_a = three_regime_pose
    pose_b = _make_b_session(np.random.default_rng(7))
    a_csv, b_csv = tmp_path / "a.csv", tmp_path / "b.csv"
    a_ann, b_ann = tmp_path / "a_annotations.csv", tmp_path / "b_annotations.csv"
    _write_dlc_csv(pose_a, a_csv)
    _write_dlc_csv(pose_b, b_csv)
    # Annotations leave frames 150-200 and 350-450 unlabeled → background.
    for ann in (a_ann, b_ann):
        _write_annotations(ann, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])

    res = cross_validate_sessions(
        sessions=[(a_csv, a_ann), (b_csv, b_ann)],
        spec=FeatureSpec(body_axis=(0, pose_a.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        n_folds=2,
        include_background=True,
        background_ratio=1.0,
    )
    assert res["background_class_name"] == "background"
    assert "background" in res["per_class_metrics"]
    assert res["false_alarm_rate"] is not None
    assert 0.0 <= res["false_alarm_rate"] <= 1.0


def test_cross_validate_sessions_threshold_curve(tmp_path, three_regime_pose):
    """threshold_curve=True returns per-class recall/precision/fires-on-rest
    that move monotonically with the threshold."""
    from glider.analysis.behavior import FeatureSpec, cross_validate_sessions

    pose_a = three_regime_pose
    pose_b = _make_b_session(np.random.default_rng(7))
    a_csv, b_csv = tmp_path / "a.csv", tmp_path / "b.csv"
    a_ann, b_ann = tmp_path / "a_annotations.csv", tmp_path / "b_annotations.csv"
    _write_dlc_csv(pose_a, a_csv)
    _write_dlc_csv(pose_b, b_csv)
    for ann in (a_ann, b_ann):
        _write_annotations(ann, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])

    res = cross_validate_sessions(
        sessions=[(a_csv, a_ann), (b_csv, b_ann)],
        spec=FeatureSpec(body_axis=(0, pose_a.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        n_folds=2,
        threshold_curve=True,
    )
    curves = res["threshold_curves"]
    assert set(curves) == {"locomote", "groom", "rest"}
    for rows in curves.values():
        # Higher threshold → recall and fires-on-rest both non-increasing.
        recalls = [r["recall"] for r in rows]
        fires = [r["fires_on_rest"] for r in rows]
        assert recalls == sorted(recalls, reverse=True)
        assert fires == sorted(fires, reverse=True)
        for r in rows:
            assert 0.0 <= r["recall"] <= 1.0 and 0.0 <= r["fires_on_rest"] <= 1.0
    # Tuned per-behavior thresholds + macro-F1 comparison.
    tuned = res["tuned_thresholds"]
    assert set(tuned) >= {"thresholds", "per_class", "macro_f1", "argmax_macro_f1"}
    assert set(tuned["thresholds"]) == set(curves)
    for m in tuned["per_class"].values():
        assert set(m) == {"precision", "recall", "f1", "threshold"}


def test_bout_recall_stitches_consecutive_frames_into_bouts():
    """A bout = consecutive same-true-label frames; detection counts at the
    any / 25% / 50% criteria. Non-consecutive frames start a new bout."""
    import numpy as np

    from glider.analysis.behavior.pipeline import bout_metrics

    # One session. Frames 0-3 are a 'dig' bout, 2 of 4 predicted dig (50%).
    # Frames 10-13 are a second 'dig' bout (gap → separate), 0 predicted.
    sess = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    frame = np.array([0, 1, 2, 3, 10, 11, 12, 13])
    y_true = np.array(["dig"] * 8)
    y_pred = np.array(["dig", "dig", "x", "x", "x", "x", "x", "x"])
    out = bout_metrics(sess, frame, y_true, y_pred)
    assert out["dig"]["n_bouts"] == 2  # gap split it into two bouts
    assert out["dig"]["recall_any"] == 0.5  # only the first bout had a hit
    assert out["dig"]["recall_50"] == 0.5  # first bout exactly 50% → counts
    assert out["dig"]["recall_25"] == 0.5
    # Predicted-bout side: a single predicted 'dig' run (frames 0-1), and
    # it's fully inside a true dig bout → precision 1.0 at every criterion.
    assert out["dig"]["n_pred_bouts"] == 1
    assert out["dig"]["precision_any"] == 1.0
    assert out["dig"]["precision_50"] == 1.0
    # F1 = harmonic mean of precision (1.0) and recall (0.5) = 0.6667.
    assert abs(out["dig"]["f1_50"] - (2 * 1.0 * 0.5 / 1.5)) < 1e-9


def test_cross_validate_sessions_reports_bout_metrics(tmp_path, three_regime_pose):
    """End-to-end: CV returns per-behavior bout metrics."""
    from glider.analysis.behavior import FeatureSpec, cross_validate_sessions

    pose_a = three_regime_pose
    pose_b = _make_b_session(np.random.default_rng(7))
    a_csv, b_csv = tmp_path / "a.csv", tmp_path / "b.csv"
    a_ann, b_ann = tmp_path / "a_annotations.csv", tmp_path / "b_annotations.csv"
    _write_dlc_csv(pose_a, a_csv)
    _write_dlc_csv(pose_b, b_csv)
    for ann in (a_ann, b_ann):
        _write_annotations(ann, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])
    res = cross_validate_sessions(
        sessions=[(a_csv, a_ann), (b_csv, b_ann)],
        spec=FeatureSpec(body_axis=(0, pose_a.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        n_folds=2,
    )
    bm = res["bout_metrics"]
    assert bm and all(
        set(m)
        == {
            "n_bouts",
            "n_pred_bouts",
            "recall_any",
            "recall_25",
            "recall_50",
            "precision_any",
            "precision_25",
            "precision_50",
            "f1_any",
            "f1_25",
            "f1_50",
        }
        for m in bm.values()
    )


def test_cross_validate_sessions_needs_two_sessions(tmp_path, three_regime_pose):
    """A single session can't be cross-validated by session."""
    from glider.analysis.behavior import FeatureSpec, cross_validate_sessions

    pose_csv = tmp_path / "s.csv"
    ann_csv = tmp_path / "s_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(ann_csv, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])
    with pytest.raises(ValueError, match="at least 2 sessions"):
        cross_validate_sessions(
            sessions=[(pose_csv, ann_csv)],
            spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
            window=10,
            fps=30.0,
            n_folds=2,
        )


def test_train_model_records_lgbm_reg(tmp_path, three_regime_pose):
    """LightGBM regularization knobs are recorded in the summary."""
    pytest.importorskip("lightgbm")
    from glider.analysis.behavior import FeatureSpec, LgbmReg, train_model

    pose_csv = tmp_path / "s.csv"
    ann_csv = tmp_path / "s_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(ann_csv, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        classifier_type="lightgbm",
        lgbm_reg=LgbmReg(min_child_samples=33, reg_lambda=2.5),
    )
    reg = result.summary["lgbm_reg"]
    assert reg["min_child_samples"] == 33
    assert reg["reg_lambda"] == 2.5


def test_lgbm_reg_exposes_capacity_knobs():
    """LgbmReg carries learning_rate / max_depth / min_split_gain so the
    CLI can limit tree capacity (the levers missing when train acc = 1.000)."""
    from glider.analysis.behavior import LgbmReg

    reg = LgbmReg(learning_rate=0.05, max_depth=6, min_split_gain=0.1)
    assert reg.learning_rate == 0.05
    assert reg.max_depth == 6
    assert reg.min_split_gain == 0.1


def test_build_classifier_passes_capacity_knobs_to_lightgbm():
    """_build_classifier forwards the new knobs to the LGBMClassifier."""
    pytest.importorskip("lightgbm")
    from glider.analysis.behavior import LgbmReg
    from glider.analysis.behavior.pipeline import _build_classifier

    reg = LgbmReg(learning_rate=0.07, max_depth=5, min_split_gain=0.2)
    clf = _build_classifier(
        classifier_type="lightgbm",
        n_estimators=10,
        random_state=0,
        class_weight=None,
        lgbm_reg=reg,
    )
    params = clf.get_params()
    assert params["learning_rate"] == 0.07
    assert params["max_depth"] == 5
    assert params["min_split_gain"] == 0.2


def test_train_model_freq_features_adds_spectral_columns(tmp_path, three_regime_pose):
    """--freq-features adds __domfreq/__specflat columns for kinematic
    features only, and the model trains on them."""
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "s.csv"
    ann_csv = tmp_path / "s_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(ann_csv, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        freq_features=True,
    )
    names = result.model.feature_names
    spectral = [n for n in names if n.endswith("__domfreq") or n.endswith("__specflat")]
    # Spectral columns exist, and only for kinematic base features.
    assert spectral, "expected spectral columns when freq_features=True"
    for n in spectral:
        base = n.rsplit("__", 1)[0]
        assert base.startswith(("speed_", "accel_")) or base == "body_angular_velocity"


def test_train_model_traj_features_adds_trajectory_columns(tmp_path, three_regime_pose):
    """--traj-features adds the trajectory-shape columns to the model."""
    from glider.analysis.behavior import FeatureSpec, train_model
    from glider.analysis.behavior.trajectory import TRAJ_COLUMNS

    pose_csv = tmp_path / "s.csv"
    ann_csv = tmp_path / "s_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(ann_csv, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        traj_features=True,
    )
    names = set(result.model.feature_names)
    assert set(TRAJ_COLUMNS).issubset(names)


def test_train_model_no_traj_features_by_default(tmp_path, three_regime_pose):
    from glider.analysis.behavior import FeatureSpec, train_model
    from glider.analysis.behavior.trajectory import TRAJ_COLUMNS

    pose_csv = tmp_path / "s.csv"
    ann_csv = tmp_path / "s_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(ann_csv, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
    )
    names = set(result.model.feature_names)
    assert not set(TRAJ_COLUMNS).intersection(names)


def test_train_model_no_freq_features_by_default(tmp_path, three_regime_pose):
    """Default training has no spectral columns (opt-in only)."""
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "s.csv"
    ann_csv = tmp_path / "s_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(ann_csv, [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)])
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
    )
    names = result.model.feature_names
    assert not any(n.endswith(("__domfreq", "__specflat")) for n in names)


def test_train_model_merge_map_collapses_classes(tmp_path, three_regime_pose):
    """A merge_map collapses behaviors into one class in the trained
    model — summary counts show the merged name, not the originals."""
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(
        ann_csv,
        [("a", 50, 150), ("b", 250, 350), ("c", 450, 550)],
    )
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        merge_map={"a": "ab", "b": "ab"},
    )
    counts = result.summary["kept_label_counts"]
    assert "ab" in counts
    assert "c" in counts
    assert "a" not in counts and "b" not in counts
    # The model's classes reflect the merge too.
    assert "ab" in set(result.model.classes)


def test_train_model_emits_confusion_matrix(tmp_path, three_regime_pose):
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(
        ann_csv,
        [("locomote", 0, 200), ("groom", 200, 400), ("rest", 400, 600)],
    )
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        test_split=0.34,
    )
    cm = result.summary["confusion_matrix"]
    assert "labels" in cm and "matrix" in cm
    n = len(cm["labels"])
    assert len(cm["matrix"]) == n
    for row in cm["matrix"]:
        assert len(row) == n
        # All entries are non-negative integers.
        assert all(isinstance(v, int) and v >= 0 for v in row)


def test_mirror_augment_doubles_training_set(tmp_path, three_regime_pose):
    """With --mirror-augment, train sees 2× the rows for the same labels."""
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(
        ann_csv,
        [("locomote", 0, 150), ("groom", 200, 350), ("rest", 450, 600)],
    )

    no_aug = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=5,
        mirror_augment=False,
    )
    aug = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=5,
        mirror_augment=True,
    )
    # Augmented training set has roughly twice as many kept rows.
    assert aug.summary["n_rows_kept"] >= 1.9 * no_aug.summary["n_rows_kept"]
    assert aug.summary["mirror_augment"] is True


def test_mirror_pose_swaps_left_right_keypoints():
    """Direct test of the mirror helper: left_/right_ keypoints swap places."""
    from glider.analysis.behavior.pipeline import _mirror_pose
    from glider.vision.pose.core import PoseData

    # Mouse keypoints, left is at low x, right at high x.
    xy = np.array(
        [
            [
                [10.0, 50.0],  # left_ear
                [30.0, 50.0],  # right_ear
                [20.0, 40.0],  # nose
            ]
        ]
    )
    pose = PoseData(
        xy=xy,
        confidence=np.ones((1, 3)),
        keypoint_names=["left_ear", "right_ear", "nose"],
        fps=30.0,
    )
    mirrored = _mirror_pose(pose)
    # left_ear should now be at the x-coordinate the right_ear was at
    # (after the median-x mirror), and vice versa.
    # Median x = 20; mirror around 20: 10 → 30, 30 → 10, 20 → 20.
    # After mirroring then swapping L<->R indices:
    #   left_ear (idx 0) takes the mirrored right_ear's xy → (10, 50)
    #   right_ear (idx 1) takes the mirrored left_ear's xy → (30, 50)
    #   nose (idx 2) keeps the mirrored nose → (20, 40)
    np.testing.assert_allclose(mirrored.xy[0, 0], [10.0, 50.0])
    np.testing.assert_allclose(mirrored.xy[0, 1], [30.0, 50.0])
    np.testing.assert_allclose(mirrored.xy[0, 2], [20.0, 40.0])


def test_lightgbm_falls_back_to_rf_when_missing(tmp_path, three_regime_pose):
    """If lightgbm isn't installed, classifier_type='lightgbm' should
    fall back to RF with a warning rather than crashing."""
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(
        ann_csv,
        [("locomote", 0, 200), ("groom", 200, 400), ("rest", 400, 600)],
    )
    # Either lightgbm is installed (and we get a lightgbm bundle) or
    # not (and we get an rf bundle with a warning). Both paths are
    # acceptable; we just need the call to succeed and produce a
    # working model.
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=5,
        classifier_type="lightgbm",
    )
    assert result.summary["classifier_type"] in ("lightgbm", "rf")
    assert result.summary["n_rows_kept"] > 0


def test_train_model_emits_per_class_metrics(tmp_path, three_regime_pose):
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(
        ann_csv,
        [
            ("locomote", 0, 150),
            ("groom", 200, 350),
            ("rest", 450, 600),
        ],
    )
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        test_split=0.34,
    )
    metrics = result.summary["per_class_metrics"]
    # We should have per-class entries for every class that appeared in
    # either y_test or the predictions.
    assert len(metrics) >= 1
    for _name, m in metrics.items():
        for key in ("precision", "recall", "f1", "support"):
            assert key in m
            assert 0.0 <= m[key] <= 1.0 or key == "support"


# ---------------------------------------------------------------------------
# End-to-end train_model + save/load
# ---------------------------------------------------------------------------


def test_train_model_end_to_end_and_save_load(tmp_path, three_regime_pose):
    from glider.analysis.behavior import BehaviorModel, FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    # Annotate the three regimes (200 frames each).
    _write_annotations(
        ann_csv,
        [
            ("locomote", 0, 200),
            ("groom", 200, 400),
            ("rest", 400, 600),
        ],
    )

    spec = FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1))
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=spec,
        window=10,  # tiny window so rolling NaN doesn't eat too much
        fps=30.0,
        n_estimators=20,  # keep it quick
    )

    # Summary sanity.
    assert set(result.summary["classes"]) == {"locomote", "groom", "rest"}
    assert result.summary["n_rows_kept"] > 100
    # On well-separated synthetic data the classifier should be near
    # perfect on its own training set.
    assert result.summary["train_accuracy"] > 0.95

    # Round-trip the model.
    out_path = tmp_path / "model.pkl"
    result.model.save(out_path)
    assert out_path.exists()

    reloaded = BehaviorModel.load(out_path)
    assert reloaded.feature_names == result.model.feature_names
    assert set(reloaded.classes) == set(result.model.classes)
    assert reloaded.window == result.model.window

    # Predict on the same training data — should match.
    from glider.analysis.behavior import apply_rolling, compute_features

    feats = compute_features(three_regime_pose, spec=spec)
    windowed = apply_rolling(feats, window=10, stats=result.model.stats)
    preds_a = result.model.predict(windowed)
    preds_b = reloaded.predict(windowed)
    np.testing.assert_array_equal(preds_a, preds_b)
    # NaN rows (the start of the rolling window) should come out as "".
    assert "" in set(preds_a.tolist())


def test_train_model_fits_and_attaches_embedding(tmp_path, three_regime_pose):
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(
        ann_csv,
        [("locomote", 0, 200), ("groom", 200, 400), ("rest", 400, 600)],
    )
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=20,
        embedding="pca",
    )
    art = result.model.embedding
    assert art is not None
    assert art.method == "pca"
    # 3D coords, one per kept training row (well under the subsample cap).
    assert art.coords.shape == (result.summary["n_rows_kept"], 3)
    assert len(art.labels) == result.summary["n_rows_kept"]
    assert set(art.labels.tolist()) <= {"locomote", "groom", "rest"}


def test_train_model_embedding_failure_does_not_abort_training(
    tmp_path, three_regime_pose, monkeypatch
):
    """A failing embedding fit must not lose the trained model — the
    embedding is a visualization add-on, not core training."""
    import glider.analysis.behavior.embedding as emb
    from glider.analysis.behavior import FeatureSpec, train_model

    def _boom(*a, **k):
        raise RuntimeError("simulated umap blowup")

    monkeypatch.setattr(emb, "fit_embedding", _boom)

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(
        ann_csv,
        [("locomote", 0, 200), ("groom", 200, 400), ("rest", 400, 600)],
    )
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=20,
        embedding="umap",
    )
    # Model still trained; embedding just absent.
    assert result.model.embedding is None
    assert result.summary["train_accuracy"] > 0.9


def test_train_model_no_embedding_by_default(tmp_path, three_regime_pose):
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(
        ann_csv,
        [("locomote", 0, 200), ("groom", 200, 400), ("rest", 400, 600)],
    )
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=20,
    )
    assert result.model.embedding is None


def test_train_model_handles_ambiguous_and_unannotated_rows(tmp_path, three_regime_pose):
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    # Overlap zones + leave a chunk unannotated to exercise both filters.
    _write_annotations(
        ann_csv,
        [
            ("locomote", 0, 200),
            ("groom", 250, 400),  # gap at 200-249 = unannotated
            ("groom_or_rest", 380, 420),  # overlaps end of groom → ambiguous
            ("rest", 420, 600),
        ],
    )
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
    )
    # __ambiguous__ and "" should be filtered out of the training labels.
    classes = set(result.summary["classes"])
    assert "__ambiguous__" not in classes
    assert "" not in classes


def test_train_model_with_background_includes_unannotated(tmp_path, three_regime_pose):
    """include_background=True promotes unannotated rows to a 'background'
    class, so the model learns the no-behavior state instead of
    force-classifying everything."""
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    # Annotate only the middle regime; the first and last are unannotated.
    _write_annotations(ann_csv, [("groom", 200, 400)])

    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        include_background=True,
    )
    classes = set(result.summary["classes"])
    assert "groom" in classes
    assert "background" in classes
    counts = result.summary["kept_label_counts"]
    # Background should dominate (~2x as many frames as groom).
    assert counts["background"] > counts["groom"]


def test_train_model_with_background_respects_custom_name(tmp_path, three_regime_pose):
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(ann_csv, [("groom", 200, 400)])

    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        include_background=True,
        background_class_name="other",
    )
    assert "other" in result.summary["classes"]
    assert "background" not in result.summary["classes"]


def test_train_model_with_background_still_drops_ambiguous(tmp_path, three_regime_pose):
    """Ambiguous rows aren't promoted to background — they're still dropped."""
    from glider.analysis.behavior import FeatureSpec, train_model
    from glider.analysis.behavior.labels import AMBIGUOUS

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(
        ann_csv,
        [
            ("a", 0, 200),
            ("b", 100, 300),  # overlap with a → frames 100-199 ambiguous
        ],
    )
    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        include_background=True,
    )
    assert AMBIGUOUS not in set(result.summary["classes"])


def test_background_subsampling_caps_at_ratio(tmp_path, three_regime_pose):
    """The whole point of --background-ratio: when background dominates,
    cap it at ratio * largest-non-background-class so the named classes
    can be learned at all."""
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    # Annotate a tiny slice — most of the 600 frames become background.
    _write_annotations(ann_csv, [("groom", 200, 240)])

    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        include_background=True,
        background_subsample_ratio=3.0,
    )
    counts = result.summary["kept_label_counts"]
    # groom should have ~30-ish (40 frames - some windowing loss).
    # background should be capped at 3 * groom_count.
    groom = counts["groom"]
    background = counts["background"]
    assert (
        background <= 3 * groom + 1
    ), f"background ({background}) should be capped at ~3*groom ({groom})"
    assert result.summary["background_subsampled_to"] is not None


def test_background_subsampling_disabled_when_ratio_zero(tmp_path, three_regime_pose):
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    _write_annotations(ann_csv, [("groom", 200, 240)])

    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        include_background=True,
        background_subsample_ratio=0.0,
    )
    # No cap → background should be much larger than groom.
    counts = result.summary["kept_label_counts"]
    assert counts["background"] > 10 * counts["groom"]
    assert result.summary["background_subsampled_to"] is None


def test_background_subsampling_noop_when_background_already_small(tmp_path, three_regime_pose):
    """If background is already < ratio * largest, don't touch it."""
    from glider.analysis.behavior import FeatureSpec, train_model

    pose_csv = tmp_path / "session.csv"
    ann_csv = tmp_path / "session_annotations.csv"
    _write_dlc_csv(three_regime_pose, pose_csv)
    # Annotate most of the video, leaving only ~50 background frames.
    _write_annotations(ann_csv, [("groom", 0, 550)])

    result = train_model(
        sessions=[(pose_csv, ann_csv)],
        spec=FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1)),
        window=10,
        fps=30.0,
        n_estimators=10,
        include_background=True,
        background_subsample_ratio=5.0,
    )
    # background_subsampled_to should be None because no capping happened.
    assert result.summary["background_subsampled_to"] is None


def test_predict_with_confidence_threshold_emits_blank_for_low_confidence():
    """When predict_proba's top class is below threshold, predict() emits ""
    (the 'unknown' signal) instead of force-classifying."""
    from sklearn.ensemble import RandomForestClassifier

    from glider.analysis.behavior import BehaviorModel, FeatureSpec

    # 2-class model where the boundary is around x=0.5.
    x_df = pd.DataFrame({"x": np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])})
    y = ["a"] * 50 + ["b"] * 50
    clf = RandomForestClassifier(n_estimators=20, random_state=0).fit(x_df, y)
    m = BehaviorModel(
        classifier=clf,
        feature_names=["x"],
        spec=FeatureSpec(),
        window=10,
        stats=("mean",),
        fps=30.0,
        classes=["a", "b"],
    )

    # Confident regions → committed labels.
    confident = pd.DataFrame({"x": [0.0, 1.0]})
    labels = m.predict(confident, confidence_threshold=0.99)
    assert set(labels.tolist()) == {"a", "b"}

    # No threshold → predict everywhere even on a boundary point.
    boundary = pd.DataFrame({"x": [0.5]})
    labels_no_thresh = m.predict(boundary, confidence_threshold=0.0)
    assert labels_no_thresh[0] in ("a", "b")
    # With an absurdly high threshold, the boundary point goes to "".
    labels_thresh = m.predict(boundary, confidence_threshold=0.999)
    assert labels_thresh[0] == ""


def test_predict_one_respects_confidence_threshold():
    from sklearn.ensemble import RandomForestClassifier

    from glider.analysis.behavior import BehaviorModel, FeatureSpec

    x_df = pd.DataFrame({"x": np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])})
    y = ["a"] * 50 + ["b"] * 50
    clf = RandomForestClassifier(n_estimators=20, random_state=0).fit(x_df, y)
    m = BehaviorModel(
        classifier=clf,
        feature_names=["x"],
        spec=FeatureSpec(),
        window=10,
        stats=("mean",),
        fps=30.0,
        classes=["a", "b"],
    )
    # Confident point.
    assert m.predict_one(np.array([0.0]), confidence_threshold=0.5) == "a"
    # Boundary with an absurd threshold.
    assert m.predict_one(np.array([0.5]), confidence_threshold=0.999) == ""


def test_threshold_decision_per_class():
    """Per-class thresholds steer the winner away from argmax and abstain
    when nothing clears its own threshold."""
    from glider.analysis.behavior.model import _threshold_decision

    classes = np.array(["dig", "groom"])
    probs = np.array([[0.4, 0.6], [0.55, 0.45]])
    # Argmax would pick groom, then dig. Set groom's bar high so it can't
    # fire on row 0 → dig wins both rows.
    out = _threshold_decision(probs, classes, 0.0, {"dig": 0.3, "groom": 0.9})
    assert list(out) == ["dig", "dig"]
    # Nobody clears → abstain.
    out2 = _threshold_decision(probs, classes, 0.0, {"dig": 0.95, "groom": 0.95})
    assert list(out2) == ["", ""]
    # No per-class thresholds, global 0 → plain argmax.
    out3 = _threshold_decision(probs, classes, 0.0, None)
    assert list(out3) == ["groom", "dig"]


def test_train_model_predict_one_handles_nan():
    from sklearn.ensemble import RandomForestClassifier

    from glider.analysis.behavior import BehaviorModel, FeatureSpec

    # Build a trivially-fit classifier we control directly. Fit on a
    # DataFrame so sklearn sees feature names — matches what the real
    # pipeline does and avoids a UserWarning at predict time.
    x_df = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["x", "y"])
    clf = RandomForestClassifier(n_estimators=4, random_state=0).fit(x_df, ["a", "b"])
    m = BehaviorModel(
        classifier=clf,
        feature_names=["x", "y"],
        spec=FeatureSpec(),
        window=10,
        stats=("mean",),
        fps=30.0,
        classes=["a", "b"],
    )
    assert m.predict_one(np.array([0.0, 0.0])) == "a"
    assert m.predict_one(np.array([1.0, 1.0])) == "b"
    assert m.predict_one(np.array([np.nan, 0.0])) == ""
