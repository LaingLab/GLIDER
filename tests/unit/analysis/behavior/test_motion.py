"""Tests for egocentric residual-motion features (analysis/behavior/motion.py).

These recover forepaw/substrate motion the 7-keypoint skeleton can't see,
by differencing body-registered image patches. Pinned here on synthetic
frames: a stationary scene -> ~0 motion, localized change -> the right
region, and (the load-bearing one) a translating scene with a translating
pose -> ~0 motion, proving body motion is compensated away.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_identical_frames_have_zero_motion():
    from glider.analysis.behavior.motion import diff_features, egocentric_patch

    rng = np.random.default_rng(0)
    gray = rng.uniform(0, 255, size=(200, 200)).astype(np.float32)
    center = np.array([100.0, 100.0])
    p = egocentric_patch(gray, center, angle=0.0, body_len=40.0)
    out = diff_features(p, p)  # same patch
    assert out[0] == pytest.approx(0.0, abs=1e-6)  # total
    assert out[3] == pytest.approx(0.0, abs=1e-6)  # spread


def test_anterior_change_lands_in_anterior_region():
    from glider.analysis.behavior.motion import MOTION_COLUMNS, diff_features

    size = 160
    prev = np.zeros((size, size), dtype=np.float32)
    cur = prev.copy()
    cur[:, : size // 2] += 60.0  # brighten the head (left) half only
    col = dict(zip(MOTION_COLUMNS, diff_features(prev, cur, thresh=12.0), strict=False))

    assert col["motion_anterior"] > col["motion_posterior"]
    assert col["motion_posterior"] == pytest.approx(0.0, abs=1e-6)
    assert col["motion_total"] > 0.0
    assert col["motion_spread"] == pytest.approx(0.5, abs=1e-6)  # half the pixels


def test_body_translation_is_compensated_away():
    """A whole-scene shift, tracked by an equal pose-center shift, should
    register as ~no motion after egocentric re-centering — that's the point
    of motion compensation. Compare against the un-compensated diff to show
    the compensation is doing real work."""
    from glider.analysis.behavior.motion import diff_features, egocentric_patch

    rng = np.random.default_rng(1)
    # Smooth texture so sub-pixel interpolation after the shift is stable
    # (sharp noise would alias and inflate the residual).
    big = rng.uniform(0, 255, size=(60, 60)).astype(np.float32)
    scene = np.asarray(np.kron(big, np.ones((6, 6))), dtype=np.float32)  # 360x360, smooth blocks
    shift = 30
    c1 = np.array([180.0, 180.0])
    c2 = c1 + np.array([shift, 0.0])  # roll is axis=1 (x) only

    p1 = egocentric_patch(scene, c1, angle=0.0, body_len=50.0)
    # Shift the whole scene by `shift` in +x and move the pose center with it.
    scene2 = np.roll(scene, shift, axis=1)
    p2 = egocentric_patch(scene2, c2, angle=0.0, body_len=50.0)

    compensated = diff_features(p1, p2)[0]
    uncompensated = diff_features(p1, egocentric_patch(scene2, c1, angle=0.0, body_len=50.0))[
        0
    ]  # same crop, scene moved

    assert compensated < 0.15 * uncompensated  # body motion mostly removed
    assert uncompensated > 1.0  # the scene really did move


def test_load_or_compute_motion_uses_cache(tmp_path, monkeypatch):
    """Second call returns the cached sidecar without touching the video."""
    import glider.analysis.behavior.motion as motion

    n = 40
    df_fake = motion.pd.DataFrame(
        np.arange(n * 4).reshape(n, 4) * 1.0, columns=motion.MOTION_COLUMNS
    )
    calls = {"n": 0}

    def fake_compute(*a, **k):
        calls["n"] += 1
        return df_fake

    monkeypatch.setattr(motion, "compute_motion_for_video", fake_compute)

    xy = np.zeros((n, 7, 2))
    pose_csv = tmp_path / "poses" / "S1.csv"
    video = tmp_path / "videos" / "S1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not really a video")  # just needs to exist

    a = motion.load_or_compute_motion(pose_csv, video, xy, (0, 6), cache_dir=tmp_path / "c")
    b = motion.load_or_compute_motion(pose_csv, video, xy, (0, 6), cache_dir=tmp_path / "c")

    assert calls["n"] == 1  # computed once, then cached
    assert list(a.columns) == motion.MOTION_COLUMNS
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())
