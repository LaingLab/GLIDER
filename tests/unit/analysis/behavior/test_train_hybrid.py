"""Tests for :func:`train_hybrid_model` — the split-before-fit + λ-tuning
hybrid trainer."""

from __future__ import annotations

import pytest

pytest.importorskip("lightgbm")


def test_train_hybrid_model_tunes_lambda(hybrid_sessions):
    """The hybrid trainer returns a fitted HybridModel whose tuned λ never
    scores worse than the base (λ=0) on the held-out validation split."""
    from glider.analysis.behavior import train_hybrid_model
    from glider.analysis.behavior.hybrid import HybridModel

    sessions, tag_map = hybrid_sessions
    result = train_hybrid_model(
        sessions,
        tag_map=tag_map,
        fps=30.0,
        n_estimators=40,
        random_state=0,
    )

    # A fitted hybrid model over both behaviors.
    assert isinstance(result.model, HybridModel)
    assert {"rest", "locomote"} <= set(result.model.classes)

    # λ came from the grid, and per-λ val F1 was recorded for all of it.
    default_grid = tuple(round(0.1 * i, 1) for i in range(11))
    assert result.lam in default_grid
    assert set(result.per_lambda_f1) == set(default_grid)

    # Tuning never loses to the base: λ* ≥ λ=0 on val, and 0.0 is the base.
    assert result.per_lambda_f1[result.lam] >= result.per_lambda_f1[0.0]
    assert result.base_val_f1 == pytest.approx(result.per_lambda_f1[0.0])
    assert result.n_val > 0


def test_train_hybrid_model_ties_pick_smallest_lambda(hybrid_sessions, monkeypatch):
    """When every λ scores identically, the smallest λ (0.0) wins so the base
    is preferred on no-improvement."""
    from glider.analysis.behavior import pipeline, train_hybrid_model

    # Force a flat F1 surface across λ.
    monkeypatch.setattr(pipeline, "macro_frame_f1", lambda *a, **k: 0.5)

    sessions, tag_map = hybrid_sessions
    result = train_hybrid_model(
        sessions, tag_map=tag_map, fps=30.0, n_estimators=20, random_state=0
    )
    assert result.lam == 0.0
