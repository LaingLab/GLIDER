"""Fixed-N consolidation, against synthetic fragments -- no model, no video.

Everything here is arithmetic on frame spans and centroids, which is why it is
in its own module: the algorithm that decides which animal is which should be
testable without a GPU."""

import math

import numpy as np

from glider.vision.pose.consolidate import Fragment, assignment_cost, seed_slots

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


def test_a_fragment_overlapping_the_slot_is_refused():
    # One animal cannot be in two places. This is the whole value of knowing N.
    slot = [frag(1, start=0, length=100, x=0, y=0)]
    overlapping = frag(2, start=50, length=10, x=1, y=1)
    assert assignment_cost(slot, overlapping, max_travel_px_per_frame=40.0) == math.inf


def test_a_fragment_that_continues_the_slot_costs_little():
    slot = [frag(1, start=0, length=10, x=100, y=100)]
    # Resumes 2 frames later, 10 px away -> 5 px/frame.
    nearby = frag(2, start=11, length=10, x=110, y=100)
    cost = assignment_cost(slot, nearby, max_travel_px_per_frame=40.0)
    assert math.isclose(cost, 5.0)


def test_a_fragment_beyond_the_travel_limit_is_refused():
    slot = [frag(1, start=0, length=10, x=0, y=0)]
    teleport = frag(2, start=10, length=10, x=5000, y=0)
    assert assignment_cost(slot, teleport, max_travel_px_per_frame=40.0) == math.inf


def test_a_longer_gap_forgives_a_longer_distance():
    # The limit is a speed, not a distance: an animal out of view for a second
    # is legitimately further away than one out of view for a frame.
    slot = [frag(1, start=0, length=10, x=0, y=0)]
    far_but_late = frag(2, start=39, length=5, x=600, y=0)  # 600px / 30 frames = 20
    assert math.isfinite(assignment_cost(slot, far_but_late, max_travel_px_per_frame=40.0))


def test_a_fragment_preceding_everything_in_the_slot_is_measured_forwards():
    # Measuring only backwards would strand a fragment that starts before the
    # seed does, and it would be dropped despite being the same animal.
    slot = [frag(1, start=100, length=10, x=0, y=0)]
    earlier = frag(2, start=88, length=10, x=10, y=0)  # ends 97, gap 3, dist 10
    cost = assignment_cost(slot, earlier, max_travel_px_per_frame=40.0)
    assert math.isfinite(cost)
    assert math.isclose(cost, 10.0 / 3.0)


def test_the_nearest_fragment_in_time_is_the_one_measured_from():
    slot = [
        frag(1, start=0, length=10, x=0, y=0),
        frag(2, start=100, length=10, x=500, y=0),
    ]
    # Starts at 112 -> nearest is the fragment ending at 109, not the one at 9.
    candidate = frag(3, start=112, length=5, x=510, y=0)
    cost = assignment_cost(slot, candidate, max_travel_px_per_frame=40.0)
    assert math.isclose(cost, 10.0 / 3.0)


def test_an_empty_slot_is_refused():
    # Unreachable from consolidate() -- an empty slot only exists when there
    # were fewer usable fragments than animals, in which case there is nothing
    # left to assign. Refusing is the safe answer: a slot with no evidence for
    # it should stay all-NaN rather than be handed someone else's fragment.
    assert assignment_cost([], frag(1, 0, 10), max_travel_px_per_frame=40.0) == math.inf


def test_a_nan_endpoint_refuses_rather_than_scoring_nan():
    # A NaN cost would sort unpredictably against real ones.
    slot = [frag(1, start=0, length=10, x=0, y=0)]
    blind = frag(2, start=11, length=10)
    blind.xy[:] = np.nan
    assert assignment_cost(slot, blind, max_travel_px_per_frame=40.0) == math.inf
