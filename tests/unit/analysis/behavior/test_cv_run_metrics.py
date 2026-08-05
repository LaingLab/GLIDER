"""A cross-validated run must report its score to the Review tab.

cross_validate_and_train stored its fold scores under cv_-prefixed keys while
every reader — run_report, summary_text, the Review tab — looks for the names
cross_validate_sessions returns. The run therefore knew it was
cross-validated and had no macro F1 to show for it, so the tab rendered a
dash where the headline number belongs.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior import FeatureSpec
from glider.analysis.behavior.pipeline import cross_validate_and_train
from glider.analysis.behavior.run_report import TrainingRun

COMMON = {"window": 8, "n_estimators": 20, "random_state": 42, "n_folds": 3, "fps": 30.0}
SPEC = FeatureSpec(body_axis=(0, 4))


def _pose(seed: int, n_frames: int = 300):
    from glider.vision.pose.core import PoseData

    rng = np.random.default_rng(seed)
    names = ["snout", "left_ear", "right_ear", "neck", "tail_base"]
    offsets = np.array([[0, -20], [-10, -10], [10, -10], [0, 0], [0, 30]], dtype=float)
    xy = np.empty((n_frames, len(names), 2))
    per = n_frames // 3
    for regime, slc in enumerate([slice(0, per), slice(per, 2 * per), slice(2 * per, n_frames)]):
        n = slc.stop - slc.start
        t = np.arange(n)
        if regime == 0:
            cx, cy, jitter = 50 + 0.5 * t, np.full(n, 200.0), 0.4
        elif regime == 1:
            cx, cy, jitter = 350 + 3 * np.sin(0.5 * t), 200 + 3 * np.cos(0.5 * t), 1.0
        else:
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


# ---------------------------------------------------------------------------
# The keys every reader actually looks for
# ---------------------------------------------------------------------------


def test_the_summary_uses_the_canonical_fold_score_names(sessions):
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    assert trained.summary["mean_macro_f1"] is not None
    assert trained.summary["fold_macro_f1"]


def test_the_review_tab_can_read_the_macro_f1(sessions):
    """The exact path the Review tab takes: summary -> TrainingRun.macro_f1."""
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    run = TrainingRun.from_summary(trained.summary)
    assert run.is_cross_validated is True
    assert run.macro_f1 is not None, "Review tab would render a dash here"


def test_the_headline_carries_the_fold_spread(sessions):
    """A cross-validated headline without its spread invites the same
    single-split mistake the folds exist to avoid."""
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    headline = TrainingRun.from_summary(trained.summary).headline
    assert headline.value is not None
    assert headline.spread is not None


def test_the_macro_f1_pools_the_classes_and_the_headline_keeps_the_folds(sessions):
    """Both cross-validated figures survive, each in its own place.

    This used to assert ``macro_f1 == mean_macro_f1``, which made the Review
    tab print one number twice. They are different statistics: the headline's
    mean-of-folds, and the unweighted mean of the pooled per-class F1s shown
    beside it. They diverge whenever a class is missing from a fold's test
    set, so pinning them together hid the more useful of the two.
    """
    cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    run = TrainingRun.from_summary(trained.summary)

    per_class = [m["f1"] for m in cv["per_class_metrics"].values()]
    assert run.macro_f1 == pytest.approx(sum(per_class) / len(per_class))

    # The fold statistic is not lost -- it still leads the page, with spread.
    assert run.headline.value == pytest.approx(cv["mean_macro_f1"])


def test_the_confusion_matrix_reaches_the_review_tab(sessions):
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    run = TrainingRun.from_summary(trained.summary)
    assert run.confusion is not None


def test_per_class_rows_are_present(sessions):
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    run = TrainingRun.from_summary(trained.summary)
    assert run.per_class
    assert all(c.f1 is not None for c in run.per_class)


# ---------------------------------------------------------------------------
# The 3D embedding
# ---------------------------------------------------------------------------


def test_no_embedding_is_fitted_by_default(sessions):
    """Unchanged default: fitting one costs real time on a large cohort."""
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, **COMMON)
    assert trained.model.embedding is None


def test_an_embedding_can_be_fitted_for_the_review_tab(sessions):
    """cross_validate_and_train ignored the option entirely, so a
    cross-validated bundle could never carry one to render."""
    _cv, trained = cross_validate_and_train(sessions, spec=SPEC, embedding="pca", **COMMON)
    assert trained.model.embedding is not None


def test_an_embedding_failure_does_not_lose_the_model(sessions, monkeypatch):
    """It is a visualisation add-on; a fit that took ten minutes must survive."""
    from glider.analysis.behavior import embedding as embedding_mod

    def boom(*a, **k):
        raise RuntimeError("simulated umap blowup")

    monkeypatch.setattr(embedding_mod, "fit_embedding", boom)
    with pytest.warns(UserWarning, match="embedding"):
        _cv, trained = cross_validate_and_train(sessions, spec=SPEC, embedding="pca", **COMMON)
    assert trained.model is not None
    assert trained.model.embedding is None
