"""cross_validate_and_train: one feature pass, an estimate AND a model.

Cross-validation cannot return a model — it fits n_folds of them, each on a
fraction of the sessions. The standard answer is to measure with CV then refit
on everything, which as two separate calls assembles the feature matrix twice.
That is the dominant cost of both, and with motion features it decodes every
source video twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior import FeatureSpec
from glider.analysis.behavior.pipeline import (
    cross_validate_and_train,
    cross_validate_sessions,
)


def _pose(seed: int, n_frames: int = 300):
    """Three movement regimes, so the labels are separable but not trivially.

    Self-contained rather than reusing test_training's fixture: that one is
    module-local and already duplicated across two files, and these tests need
    only "enough structure to fold", not that exact shape.
    """
    from glider.vision.pose.core import PoseData

    rng = np.random.default_rng(seed)
    names = ["snout", "left_ear", "right_ear", "neck", "tail_base"]
    offsets = np.array([[0, -20], [-10, -10], [10, -10], [0, 0], [0, 30]], dtype=float)
    xy = np.empty((n_frames, len(names), 2))
    per = n_frames // 3
    for regime, slc in enumerate([slice(0, per), slice(per, 2 * per), slice(2 * per, n_frames)]):
        n = slc.stop - slc.start
        t = np.arange(n)
        if regime == 0:  # travelling
            cx, cy, jitter = 50 + 0.5 * t, np.full(n, 200.0), 0.4
        elif regime == 1:  # milling on the spot
            cx, cy, jitter = 350 + 3 * np.sin(0.5 * t), 200 + 3 * np.cos(0.5 * t), 1.0
        else:  # still
            cx, cy, jitter = np.full(n, 360.0), np.full(n, 200.0), 0.15
        for k in range(len(names)):
            xy[slc, k, 0] = cx + offsets[k, 0] + rng.normal(0, jitter, n)
            xy[slc, k, 1] = cy + offsets[k, 1] + rng.normal(0, jitter, n)
    return PoseData(
        xy=xy,
        confidence=np.full((n_frames, len(names)), 0.95),
        keypoint_names=names,
        fps=30.0,
    )


@pytest.fixture
def sessions(tmp_path):
    """Four sessions, so 3-fold CV has whole sessions to split."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.vision.pose.dlc import to_dlc_csv

    pairs = []
    for i in range(4):
        pose = _pose(seed=i)
        pose_csv = tmp_path / f"s{i}.csv"
        to_dlc_csv(pose, pose_csv)
        store = AnnotationStore()
        third = pose.n_frames // 3
        store.add(BehaviorZone("walk", 0, third))
        store.add(BehaviorZone("mill", third, 2 * third))
        store.add(BehaviorZone("still", 2 * third, pose.n_frames))
        ann = tmp_path / f"s{i}_annotations.csv"
        store.save_csv(ann)
        pairs.append((pose_csv, ann))
    return pairs


COMMON = {"window": 8, "n_estimators": 20, "random_state": 42, "n_folds": 3, "fps": 30.0}
SPEC = FeatureSpec(body_axis=(0, 4))


def test_it_returns_both_a_measurement_and_a_model(sessions):
    cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    assert cv["mean_macro_f1"] is not None
    assert trained.model is not None
    assert trained.model.classifier is not None


def test_the_cv_half_matches_cross_validate_sessions_exactly(sessions):
    """Same assembly, same folds, same seed — the numbers must not drift."""
    alone = cross_validate_sessions(sessions, spec=SPEC, **COMMON)
    together, _ = cross_validate_and_train(sessions, spec=SPEC, **COMMON)

    assert together["fold_macro_f1"] == alone["fold_macro_f1"]
    assert together["mean_accuracy"] == alone["mean_accuracy"]
    assert together["per_class_metrics"] == alone["per_class_metrics"]
    assert together["confusion_matrix"] == alone["confusion_matrix"]


def test_the_model_is_fitted_on_every_row_not_one_fold(sessions):
    """A fold model saw (k-1)/k of the data; the returned one must see all."""
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    assert trained.summary["train_size"] == trained.summary["n_rows_kept"]


def test_the_summary_records_the_cv_estimate_not_a_self_score(sessions):
    """A plain fit stores 1.000 train accuracy and no evaluation at all."""
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    s = trained.summary
    assert s["split_strategy"] == "cross_validated"
    assert s["test_accuracy"] == pytest.approx(_cv_mean(sessions))
    assert s["per_class_metrics"], "per-class metrics should come from the folds"


def _cv_mean(sessions):
    return cross_validate_sessions(sessions, spec=SPEC, **COMMON)["mean_accuracy"]


def test_the_model_predicts(sessions):
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    model = trained.model
    n_features = len(model.feature_names)
    row = np.zeros((1, n_features))
    assert model.classifier.predict(row).shape == (1,)


def test_the_bundle_round_trips(sessions, tmp_path):
    """It has to be loadable by the Apply tab like any other model."""
    from glider.analysis.behavior.model import BehaviorModel

    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    path = tmp_path / "cv_model.pkl"
    trained.model.save(path)
    loaded = BehaviorModel.load(path)
    assert loaded.classes == trained.model.classes
    assert loaded.training_summary["split_strategy"] == "cross_validated"


def test_features_are_assembled_once_not_twice(sessions, monkeypatch):
    """The whole point. Two calls would featurise every session twice."""
    from glider.analysis.behavior import pipeline

    calls = {"n": 0}
    real = pipeline.compute_features

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(pipeline, "compute_features", counting)
    cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    combined = calls["n"]

    calls["n"] = 0
    cross_validate_sessions(sessions, spec=SPEC, **COMMON)
    cv_only = calls["n"]

    assert combined == cv_only, (
        f"combined run featurised {combined} times, CV alone {cv_only} — "
        "the refit must reuse the assembled matrix"
    )


def test_motion_and_mirror_are_refused_here_too(sessions):
    with pytest.raises(ValueError, match="mirror"):
        cross_validate_and_train(
            sessions,
            spec=SPEC,
            mirror_augment=True,
            motion_features=True,
            **COMMON,
        )


def test_one_session_is_refused(sessions):
    with pytest.raises(ValueError):
        cross_validate_and_train(sessions[:1], spec=SPEC, **COMMON)
