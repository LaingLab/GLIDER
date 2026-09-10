"""Fixed-N consolidation, against synthetic fragments -- no model, no video.

Everything here is arithmetic on frame spans and centroids, which is why it is
in its own module: the algorithm that decides which animal is which should be
testable without a GPU."""

import numpy as np

from glider.vision.pose.consolidate import Fragment, seed_slots

K = 2  # two keypoints is enough to have a centroid


def frag(track_id, start, length, x=0.0, y=0.0):
    """A fragment sitting still at (x, y) for `length` frames from `start`."""
    frames = np.arange(start, start + length)
    xy = np.empty((length, K, 2))
    xy[:, :, 0] = x
    xy[:, :, 1] = y
    return Fragment(
        track_id=track_id,
        frames=frames,
        xy=xy,
        confidence=np.ones((length, K)),
    )


def test_start_and_end_are_inclusive_frame_indices():
    f = frag(1, start=10, length=5)
    assert (f.start, f.end) == (10, 14)
    assert len(f) == 5


def test_centroid_is_the_mean_over_keypoints():
    f = frag(1, 0, 3, x=4.0, y=6.0)
    assert np.allclose(f.centroid_at(0), [4.0, 6.0])


def test_centroid_of_an_all_nan_row_is_nan_not_an_error():
    f = frag(1, 0, 2)
    f.xy[0] = np.nan
    assert np.all(np.isnan(f.centroid_at(0)))


def test_the_n_longest_fragments_become_the_seeds():
    fragments = [frag(1, 0, 100), frag(2, 0, 90), frag(3, 0, 3)]
    seeds, rest = seed_slots(fragments, n_animals=2, min_fragment_frames=5)
    assert [f.track_id for f in seeds] == [1, 2]
    assert rest == []


def test_seeds_are_ordered_by_first_appearance_not_by_duration():
    # Slot ids must be reproducible across re-runs. Duration order would
    # renumber the animals the moment a tuning knob changed.
    late_but_longest = frag(1, start=500, length=100)
    early_but_shorter = frag(2, start=0, length=60)
    seeds, _ = seed_slots([late_but_longest, early_but_shorter], n_animals=2, min_fragment_frames=5)
    assert [f.track_id for f in seeds] == [2, 1]


def test_fragments_shorter_than_the_floor_are_dropped_as_flicker():
    fragments = [frag(1, 0, 100), frag(2, 0, 50), frag(3, 0, 2)]
    seeds, rest = seed_slots(fragments, n_animals=2, min_fragment_frames=5)
    assert 3 not in [f.track_id for f in seeds] + [f.track_id for f in rest]


def test_surplus_fragments_come_back_longest_first():
    fragments = [frag(1, 0, 100), frag(2, 0, 20), frag(3, 0, 50)]
    _, rest = seed_slots(fragments, n_animals=1, min_fragment_frames=5)
    assert [f.track_id for f in rest] == [3, 2]


def test_ties_break_deterministically():
    # Two fragments of identical length must not swap slots between runs.
    a, b = frag(7, 0, 50), frag(3, 0, 50)
    first = [f.track_id for f in seed_slots([a, b], n_animals=2, min_fragment_frames=5)[0]]
    second = [f.track_id for f in seed_slots([b, a], n_animals=2, min_fragment_frames=5)[0]]
    assert first == second


def test_fewer_fragments_than_animals_seeds_what_there_is():
    seeds, rest = seed_slots([frag(1, 0, 100)], n_animals=3, min_fragment_frames=5)
    assert len(seeds) == 1 and rest == []
