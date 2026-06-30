"""Tests for trajectory-shape features (analysis/behavior/trajectory.py).

These window-level features describe the *shape of the body-center path*
— straight run vs jitter-in-place vs scribble — the temporal information
the order-invariant mean/std/max stats throw away. All are
translation/rotation/scale invariant. Pinned here once; the live path
will reuse the same functions.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_straight_path_is_maximally_straight_with_no_turning():
    from glider.analysis.behavior.trajectory import TRAJ_COLUMNS, trajectory_features_batch

    w = 10
    centers = np.stack([np.arange(w), np.zeros(w)], axis=1).astype(float)  # along +x
    out = trajectory_features_batch(centers[None], np.array([2.0]))[0]
    col = dict(zip(TRAJ_COLUMNS, out, strict=False))

    assert col["traj_straightness"] == pytest.approx(1.0)
    assert col["traj_total_turning"] == pytest.approx(0.0, abs=1e-9)
    # path length 9 px / body-length 2 = 4.5; straight ⇒ net == path.
    assert col["traj_path_length"] == pytest.approx(4.5)
    assert col["traj_net_displacement"] == pytest.approx(4.5)


def test_closed_loop_has_zero_straightness_and_high_turning():
    from glider.analysis.behavior.trajectory import TRAJ_COLUMNS, trajectory_features_batch

    # Unit square returning to start: net displacement 0, three right turns.
    centers = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    out = trajectory_features_batch(centers[None], np.array([1.0]))[0]
    col = dict(zip(TRAJ_COLUMNS, out, strict=False))

    assert col["traj_straightness"] == pytest.approx(0.0, abs=1e-9)
    assert col["traj_net_displacement"] == pytest.approx(0.0, abs=1e-9)
    # Three 90° turns = 3·π/2.
    assert col["traj_total_turning"] == pytest.approx(3 * np.pi / 2)


def test_in_place_jitter_has_small_radius_of_gyration():
    """Jittering in place (sniffing) stays spatially compact; a long run
    does not. Radius of gyration captures that."""
    from glider.analysis.behavior.trajectory import TRAJ_COLUMNS, trajectory_features_batch

    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.05, size=(30, 2))  # tight cloud
    run = np.stack([np.arange(30), np.zeros(30)], axis=1).astype(float)  # long run

    l_scale = np.array([1.0])
    jit = dict(zip(TRAJ_COLUMNS, trajectory_features_batch(jitter[None], l_scale)[0], strict=False))
    rn = dict(zip(TRAJ_COLUMNS, trajectory_features_batch(run[None], l_scale)[0], strict=False))

    assert jit["traj_radius_gyration"] < rn["traj_radius_gyration"]
    assert jit["traj_straightness"] < rn["traj_straightness"]


def test_features_are_rotation_invariant():
    from glider.analysis.behavior.trajectory import trajectory_features_batch

    rng = np.random.default_rng(2)
    centers = rng.normal(size=(20, 2))
    theta = 0.7
    r_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = centers @ r_mat.T + np.array([5.0, -3.0])  # rotate + translate

    a = trajectory_features_batch(centers[None], np.array([1.0]))[0]
    b = trajectory_features_batch(rotated[None], np.array([1.0]))[0]
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_apply_trajectory_rolling_aligns_and_pads_partial_windows():
    from glider.analysis.behavior.trajectory import TRAJ_COLUMNS, apply_trajectory_rolling

    # 3-keypoint pose: head, mid, tail. body_axis = (0, 2).
    n, window = 50, 30
    xy = np.zeros((n, 3, 2))
    t = np.arange(n)
    # Body center (midpoint of head & tail) runs straight along +x.
    xy[:, 0, 0] = t + 1.0  # head
    xy[:, 2, 0] = t - 1.0  # tail  → midpoint x = t, length = 2
    out = apply_trajectory_rolling(xy, body_axis=(0, 2), window=window)

    assert list(out.columns) == TRAJ_COLUMNS
    assert len(out) == n
    # Partial windows are NaN; full windows resolve to a straight path.
    assert out["traj_straightness"].iloc[: window - 1].isna().all()
    assert out["traj_straightness"].iloc[-1] == pytest.approx(1.0)
