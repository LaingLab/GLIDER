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

import pandas as pd  # noqa: E402

from glider.analysis.behavior.classify.batch import (  # noqa: E402
    classify_pose_data,
    classify_pose_tracks,
    write_animal_ethograms,
)
from glider.analysis.behavior.features import FeatureSpec, compute_features  # noqa: E402
from glider.analysis.behavior.model import BehaviorModel  # noqa: E402
from glider.analysis.behavior.session_view import SessionView  # noqa: E402
from glider.analysis.behavior.windowing import apply_rolling  # noqa: E402
from glider.vision.pose.batch import animals_dir  # noqa: E402
from glider.vision.pose.core import PoseData  # noqa: E402
from glider.vision.pose.dlc import to_dlc_csv  # noqa: E402
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


class TestWriteAnimalEthograms:
    """Per-animal ethograms: D2 spec §5. No ``individual`` column, ever --
    each slot's rows go to its own file beside its own pose CSV."""

    def test_one_row_per_scored_frame_per_animal(self, tmp_path):
        pose0 = _pose(seed=10, base=(150.0, 120.0))
        pose1 = _pose(seed=11, base=(300.0, 240.0))
        tracks = PoseTracks(tracks={0: pose0, 1: pose1}, fps=30.0)
        model = _model(pose0, seed=12)
        pose_csv = tmp_path / "sessionDLC_exp-6.csv"

        want = classify_pose_tracks(tracks, model, predict_every=2)
        paths = write_animal_ethograms(pose_csv, tracks, model, speed_axis=False, predict_every=2)

        assert set(paths) == {0, 1}
        assert paths[0] == animals_dir(pose_csv) / "animal0_ethogram.csv"
        assert paths[1] == animals_dir(pose_csv) / "animal1_ethogram.csv"
        for slot in (0, 1):
            df = pd.read_csv(paths[slot])
            assert list(df.columns) == ["frame", "behavior"]
            # One row per scored frame -- not len(pose) x n_animals, which is
            # the failure D2 §5 exists to rule out.
            assert len(df) == len(want[slot].frames)
            assert df["frame"].tolist() == want[slot].frames

    def test_two_animals_differ_when_their_poses_differ(self, tmp_path):
        # Distinguishable fixtures on purpose: two animals that scored
        # identically could not catch a slot swap.
        pose0 = _pose(seed=20, base=(120.0, 90.0))
        pose1 = _pose(seed=21, base=(400.0, 300.0))
        tracks = PoseTracks(tracks={0: pose0, 1: pose1}, fps=30.0)
        model = _model(pose0, seed=22)
        pose_csv = tmp_path / "sessionDLC_exp-6.csv"

        paths = write_animal_ethograms(pose_csv, tracks, model, speed_axis=False, predict_every=1)

        etho0 = pd.read_csv(paths[0])
        etho1 = pd.read_csv(paths[1])
        assert not etho0["behavior"].equals(etho1["behavior"])

    def test_a_never_filled_slot_still_gets_an_ethogram(self, tmp_path):
        """All-NaN slot: written like every other slot, not skipped.

        classify_pose_data already scores an all-NaN pose the same way it
        scores any frame with a missing keypoint -- one row per scored frame,
        blank behavior -- with no special-casing needed here. Writing it
        keeps every slot's directory entries symmetric, so a reader never has
        to guess whether a missing file meant "not scored" or "nothing to
        score".
        """
        pose0 = _pose(seed=30, base=(150.0, 150.0))
        n = pose0.n_frames
        empty = PoseData(
            xy=np.full((n, len(KP), 2), np.nan),
            confidence=np.zeros((n, len(KP))),
            keypoint_names=KP,
            fps=30.0,
        )
        tracks = PoseTracks(tracks={0: pose0, 1: empty}, fps=30.0)
        model = _model(pose0, seed=31)
        pose_csv = tmp_path / "sessionDLC_exp-6.csv"

        paths = write_animal_ethograms(pose_csv, tracks, model, speed_axis=False, predict_every=1)

        assert paths[1].exists()
        df1 = pd.read_csv(paths[1])
        want1 = classify_pose_data(empty, model, predict_every=1)
        assert len(df1) == len(want1.frames)  # same cadence as a filled slot
        assert df1["behavior"].isna().all()  # every label blank


def test_session_view_loads_a_per_animal_ethogram_and_pose_csv_unchanged(tmp_path):
    """The real regression guard for this design: per-animal files cost
    SessionView nothing -- it opens a three-row pose CSV and a
    one-row-per-frame ethogram exactly as it already did before D2."""
    pose0 = _pose(seed=40, base=(180.0, 140.0))
    pose1 = _pose(seed=41, base=(360.0, 260.0))
    tracks = PoseTracks(tracks={0: pose0, 1: pose1}, fps=30.0)
    model = _model(pose0, seed=42)
    pose_csv = tmp_path / "sessionDLC_exp-6.csv"
    out_dir = animals_dir(pose_csv)
    out_dir.mkdir(parents=True)

    animal0_csv = out_dir / "animal0.csv"
    animal1_csv = out_dir / "animal1.csv"
    to_dlc_csv(pose0, animal0_csv)
    to_dlc_csv(pose1, animal1_csv)

    want = classify_pose_tracks(tracks, model, predict_every=1)
    paths = write_animal_ethograms(pose_csv, tracks, model, speed_axis=False, predict_every=1)

    view0 = SessionView.load(paths[0], pose_csv=animal0_csv)
    assert view0.labels == want[0].labels
    assert view0.frames.tolist() == want[0].frames
    assert view0.xy is not None
    assert view0.xy.shape[0] == pose0.n_frames

    view1 = SessionView.load(paths[1], pose_csv=animal1_csv)
    assert view1.labels == want[1].labels
    assert view1.frames.tolist() == want[1].frames
    assert view1.xy is not None
    assert view1.xy.shape[0] == pose1.n_frames

    # The two animals differ -- this is not two loads of the same file.
    assert view0.labels != view1.labels
