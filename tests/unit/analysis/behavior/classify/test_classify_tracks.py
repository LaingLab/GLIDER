"""classify_pose_tracks must equal classify_pose_data called once per slot.

The property worth pinning is not "the inner function was called twice" --
a mock-based test like that would pass against an implementation that fed
the wrong animal's pose to the wrong slot. So the two animals here are built
from different seeds and different arena positions, which makes their
features -- and therefore their scored rows -- different. A slot swap then
shows up as a row mismatch, which is the failure that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from glider.analysis.behavior.classify.batch import (  # noqa: E402
    classify_pose_data,
    classify_pose_tracks,
)
from glider.analysis.behavior.features import FeatureSpec, compute_features  # noqa: E402
from glider.analysis.behavior.model import BehaviorModel  # noqa: E402
from glider.analysis.behavior.windowing import apply_rolling  # noqa: E402
from glider.vision.pose.core import PoseData  # noqa: E402
from glider.vision.pose.tracks import PoseTracks  # noqa: E402

KP = ["nose", "left_ear", "right_ear", "body_center", "tail_base"]
WINDOW = 4
STATS = ("mean", "std", "max")


def _pose(seed: int, base: tuple[float, float], n: int = 120) -> PoseData:
    """A wandering animal; seed and base position make two animals distinguishable."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    bx, by = base
    centre = np.stack([bx + 40 * np.sin(t / 9.0), by + 30 * np.cos(t / 7.0)], axis=1)
    offsets = np.array([[0, -14], [-8, -8], [8, -8], [0, 0], [0, 18]], dtype=float)
    xy = centre[:, None, :] + offsets[None] + rng.normal(0, 0.6, size=(n, len(KP), 2))
    conf = np.ones((n, len(KP)))
    return PoseData(xy=xy, confidence=conf, keypoint_names=KP, fps=30.0)


def _model(pose: PoseData, seed: int) -> BehaviorModel:
    """A real fitted BehaviorModel over this pose's own feature space."""
    from sklearn.tree import DecisionTreeClassifier

    spec = FeatureSpec()
    feats = compute_features(pose, spec)
    rolled = apply_rolling(feats, window=WINDOW, stats=STATS, min_periods=1)
    rolled = rolled.dropna()
    rng = np.random.default_rng(seed)
    y = rng.choice(["groom", "locomote", "rear"], size=len(rolled))
    clf = DecisionTreeClassifier(random_state=0, max_depth=6).fit(rolled, y)
    return BehaviorModel(
        classifier=clf,
        feature_names=list(rolled.columns),
        spec=spec,
        window=WINDOW,
        stats=STATS,
        fps=30.0,
        classes=sorted(set(y)),
    )


def test_classify_pose_tracks_equals_classify_pose_data_per_slot():
    pose0 = _pose(seed=0, base=(200.0, 150.0))
    pose1 = _pose(seed=1, base=(350.0, 260.0))
    tracks = PoseTracks(tracks={0: pose0, 1: pose1}, fps=30.0)
    model = _model(pose0, seed=3)

    want0 = classify_pose_data(pose0, model, predict_every=2, smooth_window=3)
    want1 = classify_pose_data(pose1, model, predict_every=2, smooth_window=3)
    # Fixture sanity: if the two animals scored identically, a slot swap in
    # the implementation under test would be invisible to this test.
    assert want0 != want1

    got = classify_pose_tracks(tracks, model, predict_every=2, smooth_window=3)

    assert set(got) == {0, 1}
    assert got[0] == want0
    assert got[1] == want1


def test_classify_pose_tracks_scores_every_slot():
    pose0 = _pose(seed=4, base=(100.0, 100.0))
    pose1 = _pose(seed=5, base=(220.0, 180.0))
    pose2 = _pose(seed=6, base=(340.0, 260.0))
    tracks = PoseTracks(tracks={0: pose0, 1: pose1, 2: pose2}, fps=30.0)
    model = _model(pose0, seed=7)

    got = classify_pose_tracks(tracks, model, predict_every=1)

    assert set(got) == {0, 1, 2}
    for slot, pose in ((0, pose0), (1, pose1), (2, pose2)):
        assert got[slot] == classify_pose_data(pose, model, predict_every=1)
