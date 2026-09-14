"""Social features: the subject measured against whoever is nearest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior.features import FeatureSpec, compute_features
from glider.vision.pose.core import PoseData

KEYPOINTS = ["snout", "neck", "tail_base"]


def _pose(xy: np.ndarray) -> PoseData:
    """(F, 3, 2) pixel coordinates -> PoseData with full confidence."""
    xy = np.asarray(xy, dtype=float)
    conf = np.where(np.isnan(xy).any(axis=-1), 0.0, 1.0)
    return PoseData(xy=xy, confidence=conf, keypoint_names=list(KEYPOINTS), fps=30.0)


def _walker(n_frames: int, x0: float, dx: float, y: float = 0.0) -> PoseData:
    """An animal of body length 10 walking along +x at dx px/frame."""
    xs = x0 + dx * np.arange(n_frames, dtype=float)
    xy = np.empty((n_frames, 3, 2), dtype=float)
    xy[:, 0, 0] = xs  # snout
    xy[:, 1, 0] = xs - 5.0  # neck
    xy[:, 2, 0] = xs - 10.0  # tail_base
    xy[:, :, 1] = y
    return _pose(xy)


SOCIAL = FeatureSpec(body_axis=(0, 2), auto_angles=False, include_social=True)
PLAIN = FeatureSpec(body_axis=(0, 2), auto_angles=False)


def test_include_social_off_produces_exactly_todays_columns():
    """The bit-identical guarantee. `others` is ignored entirely when off."""
    subject = _walker(20, x0=100.0, dx=1.0)
    other = _walker(20, x0=200.0, dx=-1.0)

    baseline = compute_features(subject, spec=PLAIN)
    with_others = compute_features(subject, spec=PLAIN, others=[other])

    assert list(baseline.columns) == list(with_others.columns)
    assert not any(c.startswith("social_") for c in baseline.columns)
    pd.testing.assert_frame_equal(baseline, with_others)


def test_social_without_others_raises():
    """The guard that keeps a social model off the live path.

    StreamingFeatureExtractor.push calls compute_features with no `others`
    and no way to obtain one. Returning short or NaN columns there would
    misclassify a whole session in silence.
    """
    subject = _walker(20, x0=100.0, dx=1.0)
    with pytest.raises(ValueError) as e:
        compute_features(subject, spec=SOCIAL)
    assert "others" in str(e.value)


def test_social_with_an_empty_others_list_raises():
    """N=1. A 'social' feature computed against nobody is not a number."""
    subject = _walker(20, x0=100.0, dx=1.0)
    with pytest.raises(ValueError):
        compute_features(subject, spec=SOCIAL, others=[])


def test_social_columns_appear_and_distance_is_in_body_lengths():
    subject = _walker(10, x0=0.0, dx=0.0)  # centroid x = -5, body length 10
    other = _walker(10, x0=100.0, dx=0.0)  # centroid x = 95
    df = compute_features(subject, spec=SOCIAL, others=[other])

    for col in ("social_distance", "social_approach", "social_bearing"):
        assert col in df.columns

    # Centroids are the mean over keypoints: subject at -5, other at 95.
    # 100 px apart, body length 10 -> 10 body lengths.
    assert np.allclose(df["social_distance"].to_numpy(), 10.0)


def test_social_approach_is_negative_while_closing_and_positive_while_separating():
    n = 20
    subject = _walker(n, x0=0.0, dx=0.0)
    # Other animal walks toward the subject for 10 frames, then away.
    xs = np.concatenate(
        [
            np.linspace(200.0, 100.0, 10),
            np.linspace(100.0, 200.0, 10),
        ]
    )
    xy = np.empty((n, 3, 2), dtype=float)
    xy[:, 0, 0] = xs
    xy[:, 1, 0] = xs + 5.0
    xy[:, 2, 0] = xs + 10.0
    xy[:, :, 1] = 0.0
    other = _pose(xy)

    approach = compute_features(subject, spec=SOCIAL, others=[other])["social_approach"]
    # Frame 0 is a derivative with no predecessor -> NaN; check the interiors.
    assert (approach.to_numpy()[2:9] < 0).all(), "closing must read negative"
    assert (approach.to_numpy()[12:19] > 0).all(), "separating must read positive"


def test_social_bearing_is_zero_facing_and_pi_facing_away():
    n = 5
    # Subject's snout at x=0, tail at x=-10 -> body axis points toward +x.
    xy = np.zeros((n, 3, 2), dtype=float)
    xy[:, 0, 0] = 0.0
    xy[:, 1, 0] = -5.0
    xy[:, 2, 0] = -10.0
    facing_subject = _pose(xy)

    ahead = _walker(n, x0=100.0, dx=0.0)  # centroid at +95, in front
    behind = _walker(n, x0=-100.0, dx=0.0)  # centroid at -105, behind

    b_ahead = compute_features(facing_subject, spec=SOCIAL, others=[ahead])["social_bearing"]
    b_behind = compute_features(facing_subject, spec=SOCIAL, others=[behind])["social_bearing"]

    assert np.allclose(b_ahead.to_numpy(), 0.0, atol=1e-6)
    assert np.allclose(b_behind.to_numpy(), np.pi, atol=1e-6)


def test_social_columns_are_nan_where_the_other_animal_has_no_pose():
    n = 10
    subject = _walker(n, x0=0.0, dx=0.0)
    other_xy = _walker(n, x0=100.0, dx=0.0).xy.copy()
    other_xy[4:7] = np.nan  # a real dropout, not a forced mask
    other = _pose(other_xy)

    df = compute_features(subject, spec=SOCIAL, others=[other])
    assert df["social_distance"].isna().to_numpy()[4:7].all()
    assert df["social_distance"].notna().to_numpy()[0:4].all()


def test_nearest_is_recomputed_per_frame():
    """Works for N > 2 and survives one animal dropping out.

    Two others: A is nearest for the first half, B for the second. The
    distance column must follow whichever is actually closer, not whichever
    was closer at frame 0.
    """
    n = 20
    subject = _walker(n, x0=0.0, dx=0.0)  # centroid at -5
    a = _walker(n, x0=50.0, dx=10.0)  # walks away
    b = _walker(n, x0=400.0, dx=-15.0)  # comes closer, overtakes

    df = compute_features(subject, spec=SOCIAL, others=[a, b])
    d = df["social_distance"].to_numpy()
    # Monotonic-increasing would mean it locked onto `a` forever.
    assert d[-1] < d[len(d) // 2], "nearest must be recomputed per frame"


def test_nose_to_nose_appears_only_when_the_keypoints_exist():
    subject = _walker(10, x0=0.0, dx=0.0)
    other = _walker(10, x0=100.0, dx=0.0)
    df = compute_features(subject, spec=SOCIAL, others=[other])
    # "snout" and "tail_base" are both present in KEYPOINTS.
    assert "social_nose_to_nose" in df.columns
    assert "social_nose_to_tail" in df.columns

    bare_names = ["a", "b", "c"]
    bare = PoseData(
        xy=subject.xy.copy(),
        confidence=subject.confidence.copy(),
        keypoint_names=bare_names,
        fps=30.0,
    )
    bare_other = PoseData(
        xy=other.xy.copy(),
        confidence=other.confidence.copy(),
        keypoint_names=bare_names,
        fps=30.0,
    )
    df2 = compute_features(bare, spec=SOCIAL, others=[bare_other])
    assert "social_distance" in df2.columns
    assert "social_nose_to_nose" not in df2.columns


def test_a_dropout_does_not_manufacture_an_approach():
    """The nearest animal changing is not the subject moving.

    A closer animal that stops being tracked is made ineligible, so
    `nearest` silently switches to a farther one -- and a plain
    np.gradient over social_distance then differences one animal's
    distance against another's. On a real two-mouse recording with
    dropouts that is a violent approach and retreat on frames where
    nothing moved: a wrong answer that looks plausible.

    Everything here is stationary, so every defensible approach value is
    exactly 0 and any non-zero number is fiction.
    """
    n = 12
    subject = _walker(n, x0=0.0, dx=0.0)  # centroid at -5
    near_xy = _walker(n, x0=60.0, dx=0.0).xy.copy()
    near_xy[6:] = np.nan  # the NEARER animal drops out halfway
    near = _pose(near_xy)
    far = _walker(n, x0=400.0, dx=0.0)  # still there, much farther

    df = compute_features(subject, spec=SOCIAL, others=[near, far])
    dist = df["social_distance"].to_numpy()
    approach = df["social_approach"].to_numpy()

    # Fixture sanity: the switch really happened and really is a big jump.
    assert dist[5] < dist[6]
    assert dist[6] - dist[5] > 10.0  # body lengths, not pixels

    # Nothing moved, so every reported approach must be 0 -- the frames
    # whose derivative would span the switch report nothing at all.
    reported = approach[~np.isnan(approach)]
    assert np.allclose(reported, 0.0), f"fictional approach: {approach.tolist()}"
    assert np.isnan(approach[5]) and np.isnan(approach[6])
    # And it stays a measurement everywhere else -- blanking the whole
    # column would also pass the assertion above.
    assert not np.isnan(approach[0]) and not np.isnan(approach[-1])
