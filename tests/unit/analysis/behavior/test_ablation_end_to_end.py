"""Supervised-vs-hybrid ablation, end-to-end through ``benchmarks.metrics``.

This locks in the whole-pipeline invariants: BOTH the supervised base and the
tuned hybrid must train, predict, and score through the same benchmark metrics,
and the two core contracts of the hybrid design must hold end-to-end:

* λ=0 reproduces the supervised base EXACTLY (the blend is a strict superset).
* the tuned λ never scores worse than the base on the validation split.

Eval-frame approach
-------------------
We split the two-session fixture into a train session (A) and a held-out eval
session (B), then build the aligned ``(x_eval, y_eval)`` for session B with the
pipeline's own :func:`_assemble_and_filter` — the exact feature+label assembly
the models consume, with unknown/ambiguous/NaN rows already dropped. That gives
an honest cross-session eval frame whose rows line up with their ground-truth
labels, without reaching into ``train_hybrid_model``'s private train/val split.
Because NaN rows are filtered out, every prediction is a real class (no ``""``
passthrough), so gt and pred stay length-matched with nothing to exclude.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from glider.analysis.behavior import train_hybrid_model, train_model
from glider.analysis.behavior.benchmarks import metrics as m
from glider.analysis.behavior.features import FeatureSpec
from glider.analysis.behavior.hybrid import HybridModel
from glider.analysis.behavior.pipeline import _assemble_and_filter
from glider.analysis.behavior.windowing import DEFAULT_STATS

FPS = 30.0  # matches the hybrid_sessions fixture
CLASSES = ["locomote", "rest"]  # behavior classes only — no "" / background


def _eval_frame(session):
    """Build the aligned windowed ``(x_eval, y_eval)`` for one held-out session.

    Uses the same assemble+filter pass the trainers use, so the feature columns
    and the drop logic (unknown / ambiguous / NaN rows removed) match exactly.
    """
    assembled = _assemble_and_filter(
        [session],
        spec=FeatureSpec(),
        window=30,
        stats=DEFAULT_STATS,
        fps=FPS,
        mirror_augment=False,
        merge_map=None,
        exclude=None,
        freq_features=False,
        traj_features=False,
        motion_features=False,
        include_background=False,
        background_class_name="background",
        background_subsample_ratio=0.0,
        random_state=42,
    )
    return assembled.x_kept, assembled.y_kept


def test_supervised_and_hybrid_ablate_through_metrics(hybrid_sessions):
    sessions, tag_map = hybrid_sessions
    train_sessions = [sessions[0]]  # session A
    eval_session = sessions[1]  # held-out session B

    x_eval, y_eval = _eval_frame(eval_session)
    assert len(x_eval) > 0
    y_eval_list = y_eval.tolist()
    assert "" not in y_eval_list  # filter left only real labels

    # ---- Train BOTH models on the same train session ----
    sup = train_model(
        train_sessions,
        classifier_type="lightgbm",
        fps=FPS,
        n_estimators=40,
        random_state=0,
    )
    hybrid_result = train_hybrid_model(
        train_sessions,
        tag_map=tag_map,
        fps=FPS,
        n_estimators=40,
        random_state=0,
    )

    # ---- 1. Both evaluate through metrics.py with the full field set ----
    sup_pred = sup.model.predict(x_eval)
    hyb_pred = hybrid_result.model.predict(x_eval)
    assert "" not in set(sup_pred.tolist())  # no NaN passthrough on a filtered frame
    assert "" not in set(hyb_pred.tolist())

    for pred in (sup_pred, hyb_pred):
        result = m.evaluate(y_eval_list, pred.tolist(), CLASSES)
        assert isinstance(result, m.SegmentationMetrics)
        assert 0.0 <= result.frame_accuracy <= 1.0
        assert 0.0 <= result.macro_f1 <= 1.0
        assert set(result.f1_at) == {10, 25, 50}  # the IoU thresholds
        assert set(CLASSES) <= set(result.per_class)
        row = result.as_row()
        assert {"frame_acc", "macro_f1", "edit", "f1@10", "f1@25", "f1@50"} <= set(row)

    # ---- 2. Core invariant, end-to-end: λ=0 == the supervised base exactly ----
    # Build a hybrid from the SAME base + calibrated prior as the tuned model,
    # forced to λ=0. It must reproduce the base's predictions on x_eval bit-for-bit.
    base = hybrid_result.model.base
    base_pred = base.predict(x_eval)
    lam0 = HybridModel(base, hybrid_result.model.prior, 0.0, tag_map)
    np.testing.assert_array_equal(lam0.predict(x_eval), base_pred)

    # ---- 2b. Bookend: a forced λ=1 DOES move predictions end-to-end ----
    # The exact counterpart to the λ=0-exact test — proves the prior actually
    # engages (λ=0 ≡ base; λ=1 ≢ base), not just that the selection logic is sound.
    lam1 = HybridModel(base, hybrid_result.model.prior, 1.0, tag_map)
    assert (lam1.predict(x_eval) != base_pred).any()

    # ---- 3. No-regression guarantee on the tuned model ----
    # Selection-logic / no-regression guard: the tuned λ never scores worse than
    # the base (λ=0) on the val split. Proves non-harm, NOT lift — best_lam is a
    # max over a grid containing 0.0, so ≥ holds by construction; the λ=1 bookend
    # above is what demonstrates the prior does something.
    assert hybrid_result.per_lambda_f1[hybrid_result.lam] >= hybrid_result.per_lambda_f1[0.0]
