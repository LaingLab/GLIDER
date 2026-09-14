"""Proposing clips for one animal of a multi-animal session."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glider.gui.behavior.annotator.sampler import (
    ProposedClip,
    propose_clips_for_animal,
)


def _two_animal_session(tmp_path: Path, n_frames: int = 600) -> tuple[list[Path], Path]:
    """Two per-animal DLC CSVs in D1's `_animals/` layout, plus a stub video.

    Animal 0 mills about near the origin; animal 1 walks a long diagonal. The
    two must NOT be copies of each other: identical fixtures would make a
    subject mix-up invisible, which is exactly the class of test this branch
    has been bitten by three times.
    """
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    names = ["snout", "neck", "tail_base"]
    rng = np.random.default_rng(7)
    animals = tmp_path / "pair_animals"
    animals.mkdir()

    paths = []
    for slot in range(2):
        xy = np.empty((n_frames, 3, 2), dtype=float)
        t = np.arange(n_frames, dtype=float)
        if slot == 0:
            cx = 50.0 + rng.normal(0.0, 2.0, n_frames)
            cy = 50.0 + rng.normal(0.0, 2.0, n_frames)
        else:
            cx = 100.0 + t * 0.4
            cy = 300.0 - t * 0.3
        for k in range(3):
            xy[:, k, 0] = cx - k * 5.0
            xy[:, k, 1] = cy
        pose = PoseData(
            xy=xy,
            confidence=np.full((n_frames, 3), 0.9),
            keypoint_names=list(names),
            fps=30.0,
        )
        path = animals / f"animal{slot}.csv"
        to_dlc_csv(pose, path)
        paths.append(path)

    video = tmp_path / "pair.mp4"
    video.write_bytes(b"")  # placeholder; the sampler never opens it
    return paths, video


def test_every_proposed_clip_carries_its_subject(tmp_path):
    pose_csvs, video = _two_animal_session(tmp_path)
    clips = propose_clips_for_animal(pose_csvs, video, subject=1, n_clips=5, window=10)
    assert clips
    assert {c.individual for c in clips} == {1}


def test_the_subject_chooses_which_animal_is_sampled(tmp_path):
    """The two animals move completely differently, so the picks must differ.

    If `subject` were ignored (or the list indexed wrong), both calls would
    return the same window indices.
    """
    pose_csvs, video = _two_animal_session(tmp_path)
    a = propose_clips_for_animal(pose_csvs, video, subject=0, n_clips=8, window=10)
    b = propose_clips_for_animal(pose_csvs, video, subject=1, n_clips=8, window=10)
    assert [c.window_index for c in a] != [c.window_index for c in b]


def test_a_subject_outside_the_slot_range_raises(tmp_path):
    """Clamping would silently label the wrong animal."""
    pose_csvs, video = _two_animal_session(tmp_path)
    with pytest.raises(ValueError) as e:
        propose_clips_for_animal(pose_csvs, video, subject=5, n_clips=5, window=10)
    assert "5" in str(e.value)


def test_proposed_clip_defaults_to_animal_zero():
    """Positional construction is used across the test suite; a
    non-defaulted field would break every one of them.

    This pins the DATACLASS default only. It is not a statement about what
    any producer of clips should stamp -- ``zones_to_clips`` carries each
    zone's own individual, see test_zones_to_clips_carries_the_zones_animal.
    """
    clip = ProposedClip(0, 50, 40, 60, 0.7, "x.mp4")
    assert clip.individual == 0


def test_zones_to_clips_carries_the_zones_animal():
    """Review mode and resume both go through this. Flattening to 0 would
    highlight the wrong animal, and would re-bind the clip to animal 0's
    zones when the annotator seeds a resumed session."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.gui.behavior.annotator.sampler import zones_to_clips

    store = AnnotationStore()
    store.add(BehaviorZone("grooming", 10, 20, individual=0))
    store.add(BehaviorZone("grooming", 30, 40, individual=1))
    store.add(BehaviorZone("rearing", 50, 60, individual=2))

    clips = zones_to_clips(store, "x.mp4", fps=30.0)
    assert [c.individual for c in clips] == [0, 1, 2]


def test_render_more_stays_on_the_animal_the_pass_is_labelling(tmp_path):
    """ "Render more" used to route through propose_clips_multi, which always
    returns individual=0: a labeller mid-pass on animal 1 pressed the button
    and was silently labelling animal 0 -- and every zone they then created
    was stamped with the wrong animal.

    The session's own CSV is animal 0's on purpose: that is what window.py's
    Resume path holds for a multi-animal video, so a sampler that trusts it
    samples the wrong animal's pose as well as mis-stamping the clip.
    """
    from glider.gui.behavior.annotator.app import make_more_sampler

    pose_csvs, video = _two_animal_session(tmp_path)
    kwargs = {"tracks": {video: pose_csvs}, "window": 10, "fps": 30.0}

    for_one = make_more_sampler([(video, pose_csvs[0])], subjects={video: 1}, **kwargs)
    for_zero = make_more_sampler([(video, pose_csvs[0])], subjects={video: 0}, **kwargs)
    clips_one = for_one(6)
    clips_zero = for_zero(6)

    assert clips_one
    assert {c.individual for c in clips_one} == {1}
    # Not just the stamp: the two animals move completely differently, so
    # sampling the right one must change which frames come back.
    assert sorted(c.center_frame for c in clips_one) != sorted(c.center_frame for c in clips_zero)


def test_render_more_without_per_animal_tracks_is_unchanged(tmp_path):
    """Single-animal sessions must keep working with neither argument."""
    from glider.gui.behavior.annotator.app import make_more_sampler

    pose_csvs, video = _two_animal_session(tmp_path)
    sampler = make_more_sampler([(video, pose_csvs[0])], window=10, fps=30.0)
    clips = sampler(4)
    assert clips
    assert {c.individual for c in clips} == {0}
