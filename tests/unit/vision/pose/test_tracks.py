"""PoseTracks: the container's job is to refuse a set of tracks that cannot
describe one video. Everything it rejects here is something that would
otherwise surface much later as a silent misalignment."""

import numpy as np
import pytest

from glider.vision.pose.core import PoseData
from glider.vision.pose.tracks import PoseTracks


def _pose(n_frames=10, names=("snout", "tail"), fill=1.0):
    k = len(names)
    return PoseData(
        xy=np.full((n_frames, k, 2), fill),
        confidence=np.ones((n_frames, k)),
        keypoint_names=list(names),
        fps=30.0,
    )


def test_holds_two_animals_and_reports_their_count():
    t = PoseTracks(tracks={0: _pose(), 1: _pose()}, fps=30.0)
    assert t.n_animals == 2
    assert t.n_frames == 10
    assert t.keypoint_names == ["snout", "tail"]


def test_individuals_are_named_by_slot_in_ascending_order():
    t = PoseTracks(tracks={1: _pose(), 0: _pose()}, fps=30.0)
    assert t.individuals == ["animal0", "animal1"]
    assert list(t) == [0, 1]


def test_a_slot_that_does_not_span_the_whole_video_is_refused():
    with pytest.raises(ValueError, match="span the whole video"):
        PoseTracks(tracks={0: _pose(10), 1: _pose(9)}, fps=30.0)


def test_slots_carrying_different_keypoints_are_refused():
    with pytest.raises(ValueError, match="same keypoint_names"):
        PoseTracks(tracks={0: _pose(), 1: _pose(names=("snout", "ear"))}, fps=30.0)


def test_slot_ids_must_be_contiguous_from_zero():
    # A gap means an animal was silently dropped somewhere upstream, and
    # "animal2" in the CSV would then not be the third animal.
    with pytest.raises(ValueError, match="contiguous"):
        PoseTracks(tracks={0: _pose(), 2: _pose()}, fps=30.0)


def test_no_tracks_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        PoseTracks(tracks={}, fps=30.0)


def test_fps_must_be_positive():
    with pytest.raises(ValueError, match="fps"):
        PoseTracks(tracks={0: _pose()}, fps=0.0)


def test_getitem_returns_the_slot_pose():
    p = _pose()
    assert PoseTracks(tracks={0: p}, fps=30.0)[0] is p
