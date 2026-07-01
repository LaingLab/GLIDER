"""Tests for the 1D-CNN sequence model (analysis/behavior/sequence.py).

The model consumes window-egocentric keypoint sequences: each window is
normalized so the first frame's body center is the origin, its body axis
is canonical, and body length is the unit. That bakes in
translation/rotation/scale invariance while PRESERVING the within-window
trajectory — the shape information mean/std/max discards. The invariances
are the load-bearing property, so they're pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest


def _random_window(w=30, k=7, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(w, k, 2))


def test_egocentric_window_is_translation_invariant():
    from glider.analysis.behavior.sequence import egocentric_window

    xy = _random_window()
    shifted = xy + np.array([12.3, -4.5])
    a = egocentric_window(xy, body_axis=(0, 6))
    b = egocentric_window(shifted, body_axis=(0, 6))
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_egocentric_window_is_rotation_invariant():
    from glider.analysis.behavior.sequence import egocentric_window

    xy = _random_window()
    phi = 0.9
    rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    rotated = xy @ rot_mat.T
    a = egocentric_window(xy, body_axis=(0, 6))
    b = egocentric_window(rotated, body_axis=(0, 6))
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_egocentric_window_is_scale_invariant():
    from glider.analysis.behavior.sequence import egocentric_window

    xy = _random_window()
    a = egocentric_window(xy, body_axis=(0, 6))
    b = egocentric_window(xy * 3.7, body_axis=(0, 6))
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_egocentric_window_preserves_within_window_motion():
    """Translation across the window must survive normalization — that's
    the trajectory signal we're trying to keep. A moving window and a
    static one must NOT normalize to the same thing."""
    from glider.analysis.behavior.sequence import egocentric_window

    w, k = 30, 7
    static = np.tile(np.linspace(-1, 1, k)[:, None] * np.array([1.0, 0.0]), (w, 1, 1))
    moving = static.copy()
    moving[:, :, 0] += np.linspace(0, 5, w)[:, None]  # body drifts +x over window

    a = egocentric_window(static, body_axis=(0, 6))
    b = egocentric_window(moving, body_axis=(0, 6))
    assert not np.allclose(a, b, atol=1e-6)


def test_egocentric_batch_matches_per_window():
    from glider.analysis.behavior.sequence import egocentric_batch, egocentric_window

    wins = np.stack([_random_window(seed=i) for i in range(5)])  # (5, w, k, 2)
    batch = egocentric_batch(wins, body_axis=(0, 6))
    for i in range(len(wins)):
        np.testing.assert_allclose(
            batch[i], egocentric_window(wins[i], body_axis=(0, 6)), atol=1e-9
        )


def test_behavior_cnn_forward_shape():
    pytest.importorskip("torch")
    import torch

    from glider.analysis.behavior.sequence import BehaviorCNN

    net = BehaviorCNN(n_channels=14, n_classes=4)
    x = torch.zeros(8, 14, 30)  # (batch, channels=K*2, time=w)
    out = net(x)
    assert out.shape == (8, 4)


def test_train_cnn_is_reproducible_with_same_seed():
    """Same seed → identical model. Without pinned CUDA/cudnn determinism
    two runs diverge (which is what made the CV numbers unreproducible)."""
    pytest.importorskip("torch")

    from glider.analysis.behavior.sequence import train_cnn

    rng = np.random.default_rng(0)
    n, c, w = 96, 14, 30
    x = rng.normal(size=(n, c, w)).astype("float32")
    y = np.array([0, 1, 2] * (n // 3))

    m1 = train_cnn(x, y, n_classes=3, max_epochs=10, seed=123)
    m2 = train_cnn(x, y, n_classes=3, max_epochs=10, seed=123)
    np.testing.assert_array_equal(m1.predict(x), m2.predict(x))


def test_seed_ensemble_predicts_and_averages():
    """An ensemble of N nets exposes the same predict/predict_proba API and
    averages member probabilities."""
    pytest.importorskip("torch")

    from glider.analysis.behavior.sequence import train_cnn_ensemble

    rng = np.random.default_rng(0)
    n, c, w = 96, 14, 30
    x = rng.normal(0, 0.01, size=(n, c, w)).astype("float32")
    y = np.array([0, 1] * (n // 2))
    x[y == 0, 0, :] = np.linspace(0, 1, w)
    x[y == 1, 0, :] = np.linspace(1, 0, w)

    ens = train_cnn_ensemble(x, y, n_models=3, n_classes=2, max_epochs=40, seed=0)
    proba = ens.predict_proba(x)
    assert proba.shape == (n, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (ens.predict(x) == y).mean() > 0.9


def test_train_cnn_overfits_a_tiny_separable_set():
    """Sanity: the net + training loop can drive train accuracy high on a
    trivially separable toy set. If it can't overfit, the plumbing is broken."""
    pytest.importorskip("torch")

    from glider.analysis.behavior.sequence import train_cnn

    rng = np.random.default_rng(0)
    # Two classes: channel-0 sequences ramp up vs ramp down — trivially separable.
    n, c, w = 64, 14, 30
    x = rng.normal(0, 0.01, size=(n, c, w))
    y = np.array([0, 1] * (n // 2))
    x[y == 0, 0, :] = np.linspace(0, 1, w)
    x[y == 1, 0, :] = np.linspace(1, 0, w)

    model = train_cnn(x, y, n_classes=2, max_epochs=60, val_fraction=0.0, seed=0)
    preds = model.predict(x)
    acc = (preds == y).mean()
    assert acc > 0.9
