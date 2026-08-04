"""Translation-invariant trajectory features.

Everything else compute_features emits is relative geometry — pairwise
distances and triplet angles — so the model had no notion of the PATH the
animal took. These features describe that path without ever encoding where in
the arena it happened, which is what keeps them from becoming a
which-recording-is-this leak the way absolute position would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior.features import FeatureSpec, compute_features
from glider.vision.pose.core import PoseData

TRAJECTORY_COLUMNS = ("step_length", "turn_angle", "turn_cos")


def _pose(centre_track, spread=10.0, n_kpts=4):
    """A pose whose centroid follows ``centre_track``, rigid otherwise.

    The keypoints sit at fixed offsets around the centre, so the animal's
    shape never changes and every trajectory feature is attributable to the
    path alone.
    """
    centre = np.asarray(centre_track, dtype=float)  # (F, 2)
    offsets = np.array([[spread, 0.0], [-spread, 0.0], [0.0, spread], [0.0, -spread]])[:n_kpts]
    xy = centre[:, None, :] + offsets[None, :, :]
    return PoseData(
        xy=xy,
        confidence=np.ones((len(centre), n_kpts)),
        keypoint_names=[f"k{i}" for i in range(n_kpts)],
        fps=30.0,
    )


def _straight(n=40, step=3.0):
    return [(i * step, 0.0) for i in range(n)]


def _zigzag(n=40, step=3.0):
    return [(i * step, (step if i % 2 else -step)) for i in range(n)]


def _features(track, spread=10.0, **spec_kwargs):
    spec = FeatureSpec(body_axis=(0, 1), **spec_kwargs)
    return compute_features(_pose(track, spread=spread), spec=spec)


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------


def test_trajectory_columns_are_emitted():
    df = _features(_straight())
    for name in TRAJECTORY_COLUMNS:
        assert name in df.columns


def test_trajectory_columns_can_be_switched_off():
    df = _features(_straight(), include_trajectory=False)
    for name in TRAJECTORY_COLUMNS:
        assert name not in df.columns


def test_a_spec_predating_this_feature_still_works():
    """Old model bundles unpickle a FeatureSpec with no such attribute."""
    spec = FeatureSpec(body_axis=(0, 1))
    del spec.__dict__["include_trajectory"]
    df = compute_features(_pose(_straight()), spec=spec)
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# The property that makes these safe: translation invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [(1000.0, 0.0), (-500.0, 250.0), (0.0, 99.0)])
def test_trajectory_features_are_translation_invariant(offset):
    """Absolute position must never reach the model — that is the leak."""
    track = _straight()
    moved = [(x + offset[0], y + offset[1]) for x, y in track]

    here = _features(track)[list(TRAJECTORY_COLUMNS)]
    there = _features(moved)[list(TRAJECTORY_COLUMNS)]

    pd.testing.assert_frame_equal(here, there)


# ---------------------------------------------------------------------------
# Step length
# ---------------------------------------------------------------------------


def test_step_length_matches_the_centroid_displacement():
    """3-4-5 steps, body length 20 → 5/20 = 0.25 body lengths per frame."""
    track = [(3.0 * i, 4.0 * i) for i in range(10)]
    df = _features(track)
    assert df["step_length"].iloc[5] == pytest.approx(0.25)


def test_a_stationary_animal_has_no_step_length():
    df = _features([(7.0, 7.0)] * 20)
    assert df["step_length"].iloc[10] == pytest.approx(0.0)


def test_step_length_is_scale_invariant():
    """Twice the animal moving twice as far is the same behaviour."""
    small = compute_features(_pose(_straight(step=3.0), spread=10.0), FeatureSpec(body_axis=(0, 1)))
    big = compute_features(_pose(_straight(step=6.0), spread=20.0), FeatureSpec(body_axis=(0, 1)))
    assert small["step_length"].iloc[10] == pytest.approx(big["step_length"].iloc[10])


# ---------------------------------------------------------------------------
# Turning, and the straightness index that falls out of it
# ---------------------------------------------------------------------------


def test_a_straight_path_does_not_turn():
    df = _features(_straight())
    assert df["turn_angle"].iloc[10] == pytest.approx(0.0, abs=1e-9)


def test_a_straight_path_has_maximal_directional_persistence():
    df = _features(_straight())
    assert df["turn_cos"].iloc[10] == pytest.approx(1.0)


def test_a_zigzag_path_turns_sharply():
    df = _features(_zigzag())
    assert df["turn_angle"].iloc[10] > 1.0  # radians


def test_mean_turn_cos_separates_a_straight_path_from_a_milling_one():
    """This is the tortuosity signal: the rolling mean of turn_cos is the
    circular mean resultant length, which is ~1 for travel and ~0 for milling.

    It is what should let the model tell locomoting from investigating,
    which is the largest off-diagonal block in the real confusion matrix.
    """
    straight = _features(_straight())["turn_cos"].iloc[5:-5].mean()

    rng = np.random.default_rng(0)
    milling = [(0.0, 0.0)]
    for _ in range(60):
        angle = rng.uniform(0, 2 * np.pi)
        milling.append((milling[-1][0] + 3 * np.cos(angle), milling[-1][1] + 3 * np.sin(angle)))
    wandering = _features(milling)["turn_cos"].iloc[5:-5].mean()

    assert straight > 0.9
    assert wandering < 0.5
    assert straight - wandering > 0.4


def test_turn_angle_is_bounded_to_half_a_turn():
    """A reversal is a half-turn; unwrapping must not report 3π/2."""
    rng = np.random.default_rng(3)
    track = [(0.0, 0.0)]
    for _ in range(60):
        angle = rng.uniform(0, 2 * np.pi)
        track.append((track[-1][0] + 4 * np.cos(angle), track[-1][1] + 4 * np.sin(angle)))
    turns = _features(track)["turn_angle"].dropna()
    assert turns.min() >= -1e-9
    assert turns.max() <= np.pi + 1e-9


# ---------------------------------------------------------------------------
# Degenerate input must not raise
# ---------------------------------------------------------------------------


def test_a_single_frame_does_not_raise():
    df = _features([(0.0, 0.0)])
    assert len(df) == 1


def test_dropout_frames_propagate_nan_rather_than_poisoning_the_column():
    track = [(3.0 * i, 0.0) for i in range(20)]
    pose = _pose(track)
    pose.xy[10, :, :] = np.nan
    df = compute_features(pose, FeatureSpec(body_axis=(0, 1)))
    step = df["step_length"]
    assert np.isnan(step.iloc[10])
    assert not np.isnan(step.iloc[2])
    assert not np.isnan(step.iloc[18])


def test_a_frozen_animal_does_not_produce_a_spurious_heading():
    """Zero-length steps have no direction; a made-up one would be noise."""
    df = _features([(5.0, 5.0)] * 20)
    turns = df["turn_angle"].iloc[2:-2]
    assert (turns.isna() | (turns.abs() < 1e-9)).all()


# ---------------------------------------------------------------------------
# Tracking jitter must not read as turning
#
# A stationary animal's centroid still wanders a pixel or two per frame in a
# random direction. Taking that at face value turns `turn_angle` into uniform
# noise and `turn_cos` into a zero-mean noise column -- injected into exactly
# the classes that do not translate. In a paired 10-run cross-session CV that
# cost grooming F1 in 10 folds out of 10.
# ---------------------------------------------------------------------------


# The gate is expressed in BODY LENGTHS, so a jitter test only means anything
# at a realistic ratio. JITTER_SPREAD gives a body length of 200 px — about a
# mouse at this rig's scale — against ~2 px of wander, i.e. ~0.01 body
# lengths, comfortably under the 0.02 default.
JITTER_SPREAD = 100.0


def _jittering(n=60, amplitude=1.5, seed=0):
    rng = np.random.default_rng(seed)
    return [(5.0 + rng.normal(0, amplitude), 5.0 + rng.normal(0, amplitude)) for _ in range(n)]


def test_the_jitter_fixture_is_actually_sub_threshold():
    """Guards the guard: if this drifts above the gate, the tests below stop
    testing anything and start passing for the wrong reason."""
    df = _features(_jittering(), spread=JITTER_SPREAD)
    assert df["step_length"].iloc[3:-3].mean() < FeatureSpec().trajectory_min_step


def test_jitter_does_not_manufacture_turns():
    df = _features(_jittering(), spread=JITTER_SPREAD)
    turns = df["turn_angle"].iloc[3:-3].dropna()
    assert turns.mean() < 0.2, f"jitter produced a mean turn of {turns.mean():.3f} rad"


def test_jitter_does_not_produce_a_noisy_straightness_column():
    """turn_cos must sit at 1 (no direction change), not scatter about 0.

    Not *exactly* 1 everywhere: jitter is Gaussian, so its tail crosses the
    gate on a minority of frames and those do produce a real turn. The median
    is the honest assertion — the typical stationary frame reports no turn —
    and the mean is checked only to catch a regression back toward the
    zero-mean noise column this replaced (it measured -0.41 before the gate).
    """
    df = _features(_jittering(), spread=JITTER_SPREAD)
    cos = df["turn_cos"].iloc[3:-3].dropna()
    assert cos.median() == pytest.approx(1.0)
    assert cos.mean() > 0.8


def test_real_movement_still_registers_after_the_gate():
    """The jitter gate must not also swallow genuine locomotion."""
    df = _features(_zigzag())
    assert df["turn_angle"].iloc[10] > 1.0
    assert df["step_length"].iloc[10] > 0.1


def test_the_jitter_gate_is_configurable():
    loose = _features(_jittering(), spread=JITTER_SPREAD, trajectory_min_step=0.0)
    tight = _features(_jittering(), spread=JITTER_SPREAD, trajectory_min_step=0.5)
    assert tight["turn_angle"].iloc[3:-3].mean() < loose["turn_angle"].iloc[3:-3].mean()


def test_the_gate_does_not_introduce_nan_that_would_drop_whole_windows():
    """apply_rolling uses min_periods=window, so one NaN blanks the window and
    the pipeline's keep-mask then drops the row. Gating to NaN would have
    deleted the stationary classes from training entirely."""
    df = _features(_jittering(), spread=JITTER_SPREAD)
    for name in TRAJECTORY_COLUMNS:
        assert not df[name].iloc[3:].isna().any(), f"{name} went NaN under jitter"


def test_a_pause_does_not_invent_a_turn_when_movement_resumes():
    """Heading is carried across a stop, so the turn measured on resuming is
    between the two real directions rather than against jitter."""
    track = [(3.0 * i, 0.0) for i in range(10)]
    track += [track[-1]] * 10
    track += [(track[-1][0] + 3.0 * i, 0.0) for i in range(1, 10)]
    turns = _features(track)["turn_angle"].iloc[3:-3].dropna()
    assert turns.max() < 0.2
