"""Tests for per-frame feature extraction and rolling-window helpers.

Covers: FeatureSpec, compute_features, _safe_unwrap, and apply_rolling.
Chunk-3 tests (train_model, cross_validate_sessions, label series,
annotations) are intentionally excluded — their dependencies don't exist yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glider.vision.pose.core import PoseData

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
# Features
# ---------------------------------------------------------------------------


def test_compute_features_shape_and_columns(three_regime_pose):
    from glider.analysis.behavior import FeatureSpec, compute_features

    spec = FeatureSpec(
        body_axis=(0, three_regime_pose.n_keypoints - 1),
        angle_triplets=(("body_curl", (0, 3, 4)),),
    )
    df = compute_features(three_regime_pose, spec=spec)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == three_regime_pose.n_frames
    # Required columns exist.
    assert "body_length" in df.columns
    assert "body_angular_velocity" in df.columns
    assert "angle_body_curl" in df.columns
    # Distance columns: K*(K-1)/2 = 5*4/2 = 10.
    dist_cols = [c for c in df.columns if c.startswith("dist_")]
    assert len(dist_cols) == 10
    # Per-keypoint kinematics.
    assert any(c.startswith("speed_") for c in df.columns)
    assert any(c.startswith("accel_") for c in df.columns)


def test_features_are_scale_invariant_when_normalized():
    """Doubling all coordinates should not change normalized distances."""
    from glider.analysis.behavior import FeatureSpec, compute_features

    p1 = _make_three_regime_pose(seed=7)
    p2 = PoseData(
        xy=p1.xy * 2.0,
        confidence=p1.confidence.copy(),
        keypoint_names=list(p1.keypoint_names),
        fps=p1.fps,
    )
    spec = FeatureSpec(body_axis=(0, p1.n_keypoints - 1))
    a = compute_features(p1, spec=spec)
    b = compute_features(p2, spec=spec)
    # Normalized distances should match (within float tolerance).
    dist_cols = [c for c in a.columns if c.startswith("dist_")]
    np.testing.assert_allclose(a[dist_cols].values, b[dist_cols].values, atol=1e-9)
    # body_length itself, which IS the absolute scale, should differ.
    assert not np.allclose(a["body_length"], b["body_length"])


def test_include_body_length_false_drops_only_that_column():
    """Dropping body_length removes the one absolute feature and leaves the
    rest (now fully scale-invariant) untouched."""
    from glider.analysis.behavior import FeatureSpec, compute_features

    pose = _make_three_regime_pose(seed=3)
    kept = compute_features(pose, spec=FeatureSpec(body_axis=(0, pose.n_keypoints - 1)))
    dropped = compute_features(
        pose,
        spec=FeatureSpec(body_axis=(0, pose.n_keypoints - 1), include_body_length=False),
    )
    assert "body_length" in kept.columns
    assert "body_length" not in dropped.columns
    # Every other column is identical — body_length is still the internal
    # normalizer, so dropping the column doesn't perturb the rest.
    assert list(dropped.columns) == [c for c in kept.columns if c != "body_length"]
    np.testing.assert_allclose(
        dropped.values, kept.drop(columns="body_length").values, equal_nan=True
    )


def test_feature_spec_include_body_length_round_trips():
    """The flag survives to_dict/from_dict so saved bundles stay consistent."""
    from glider.analysis.behavior import FeatureSpec

    spec = FeatureSpec(body_axis=(0, 4), include_body_length=False)
    assert FeatureSpec.from_dict(spec.to_dict()).include_body_length is False
    # Legacy dicts without the key default to True (old behaviour).
    legacy = {"body_axis": [0, 4], "normalize_by_body_length": True}
    assert FeatureSpec.from_dict(legacy).include_body_length is True


def test_safe_unwrap_preserves_nan_runs():
    """Verify the NaN-safe unwrap doesn't poison a full feature stream."""
    from glider.analysis.behavior.features import _safe_unwrap

    angle = np.array([0.0, 0.1, 0.2, np.nan, np.nan, 0.5, 0.6])
    out = _safe_unwrap(angle)
    assert np.isnan(out[3]) and np.isnan(out[4])
    np.testing.assert_allclose(out[0:3], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(out[5:7], [0.5, 0.6])


def test_compute_features_auto_generates_all_triplet_angles(three_regime_pose):
    """auto_angles=True (default) → K * (K-1) * (K-2) / 2 angle columns."""
    from glider.analysis.behavior import FeatureSpec, compute_features

    spec = FeatureSpec(body_axis=(0, three_regime_pose.n_keypoints - 1))
    df = compute_features(three_regime_pose, spec=spec)
    angle_cols = [c for c in df.columns if c.startswith("angle_")]
    k = three_regime_pose.n_keypoints
    expected = k * (k - 1) * (k - 2) // 2  # 5*4*3/2 = 30 for the fixture
    assert len(angle_cols) == expected, f"expected {expected} auto angles, got {len(angle_cols)}"
    # All should be finite (no all-NaN columns on the clean fixture).
    for col in angle_cols:
        assert df[col].notna().any()


def test_compute_features_auto_angles_can_be_disabled():
    """auto_angles=False + empty triplets → no angle columns."""
    from glider.analysis.behavior import FeatureSpec, compute_features

    pose = PoseData(
        xy=np.zeros((10, 4, 2)) + np.arange(10)[:, None, None] * 0.5,
        confidence=np.ones((10, 4)),
        keypoint_names=["a", "b", "c", "d"],
        fps=30.0,
    )
    spec = FeatureSpec(body_axis=(0, 3), auto_angles=False)
    df = compute_features(pose, spec=spec)
    assert not any(c.startswith("angle_") for c in df.columns)


def test_compute_features_user_triplets_suppress_auto_duplicates(three_regime_pose):
    """An explicit triplet shouldn't get a duplicate column under the auto name."""
    from glider.analysis.behavior import FeatureSpec, compute_features

    spec = FeatureSpec(
        body_axis=(0, three_regime_pose.n_keypoints - 1),
        # snout (0) at vertex neck (3), other endpoint tail_base (4).
        angle_triplets=(("body_curl", (0, 3, 4)),),
    )
    df = compute_features(three_regime_pose, spec=spec)
    # User column is present.
    assert "angle_body_curl" in df.columns
    # The auto-generated version with systematic naming for the SAME
    # (vertex, endpoint set) is suppressed.
    auto_dup = (
        f"angle_{three_regime_pose.keypoint_names[0]}_at_"
        f"{three_regime_pose.keypoint_names[3]}_"
        f"{three_regime_pose.keypoint_names[4]}"
    )
    assert auto_dup not in df.columns


def test_compute_features_handles_single_frame_without_crashing():
    """Regression: live inference's first tick has a 1-frame PoseData;
    np.gradient used to raise ValueError. compute_features should now
    produce NaN kinematics instead of crashing."""
    from glider.analysis.behavior import FeatureSpec, compute_features

    n_kpts = 3
    xy = np.array([[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]])  # (1, 3, 2)
    pose = PoseData(
        xy=xy,
        confidence=np.ones((1, n_kpts)),
        keypoint_names=["snout", "neck", "tail"],
        fps=30.0,
    )
    spec = FeatureSpec(body_axis=(0, 2))
    df = compute_features(pose, spec=spec)
    assert len(df) == 1
    # Static features (distances, body_length) should be finite.
    assert df["body_length"].iloc[0] == pytest.approx(20.0)
    # Kinematics on a single frame are NaN by design.
    assert np.isnan(df["speed_snout"].iloc[0])
    assert np.isnan(df["body_angular_velocity"].iloc[0])


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def test_apply_rolling_shape_and_naming():
    from glider.analysis.behavior import apply_rolling

    df = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) ** 0.5})
    out = apply_rolling(df, window=10, stats=("mean", "std", "max"))
    assert out.shape == (50, 6)
    assert set(out.columns) == {"a__mean", "a__std", "a__max", "b__mean", "b__std", "b__max"}


def test_apply_rolling_first_rows_are_nan_by_default():
    from glider.analysis.behavior import apply_rolling

    df = pd.DataFrame({"a": np.arange(20.0)})
    out = apply_rolling(df, window=5, stats=("mean",))
    # min_periods defaults to None → pandas uses window, so the first
    # 4 rows are NaN.
    assert out["a__mean"].iloc[:4].isna().all()
    assert not out["a__mean"].iloc[4:].isna().any()


def test_apply_rolling_invalid_stat_raises():
    from glider.analysis.behavior import apply_rolling

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        apply_rolling(df, window=2, stats=("not_a_real_stat",))
