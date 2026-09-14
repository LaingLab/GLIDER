"""Fixed-N consolidation, against synthetic fragments -- no model, no video.

Everything here is arithmetic on frame spans and centroids, which is why it is
in its own module: the algorithm that decides which animal is which should be
testable without a GPU."""

import math

import numpy as np

from glider.vision.pose.consolidate import Fragment, assignment_cost, consolidate, seed_slots

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


NAMES = ["snout", "tail"]


def run(fragments, *, n_animals, n_frames=200, **kw):
    return consolidate(
        fragments,
        n_animals=n_animals,
        n_frames=n_frames,
        keypoint_names=NAMES,
        fps=30.0,
        **kw,
    )


def test_two_clean_animals_become_two_slots():
    res = run([frag(1, 0, 200, x=10, y=10), frag(2, 0, 200, x=300, y=300)], n_animals=2)
    assert res.tracks.n_animals == 2
    assert np.allclose(res.tracks[0].xy[0, :, 0], 10.0)
    assert np.allclose(res.tracks[1].xy[0, :, 0], 300.0)
    assert res.dropped == []


def test_a_continuation_is_stitched_into_the_same_slot():
    seed = frag(1, 0, 100, x=10, y=10)
    tail = frag(2, 102, 98, x=12, y=10)
    other = frag(3, 0, 200, x=400, y=400)
    res = run([seed, tail, other], n_animals=2)
    # The tail belongs to the animal at (10,10), not the one at (400,400).
    assert np.allclose(res.tracks[0].xy[150, :, 0], 12.0)
    assert res.stitched[0] == set(range(102, 200))


def test_a_seed_fragment_is_not_marked_stitched():
    res = run([frag(1, 0, 200, x=10, y=10)], n_animals=1)
    assert res.stitched[0] == set()


def test_an_overlapping_fragment_goes_to_the_free_slot():
    a = frag(1, 0, 200, x=10, y=10)
    b = frag(2, 0, 100, x=300, y=300)
    c = frag(3, 100, 100, x=302, y=300)  # overlaps a, continues b
    res = run([a, b, c], n_animals=2)
    assert np.allclose(res.tracks[1].xy[150, :, 0], 302.0)
    assert np.allclose(res.tracks[0].xy[150, :, 0], 10.0)


def test_a_fragment_too_far_from_every_slot_is_dropped_and_reported():
    a = frag(1, 0, 100, x=10, y=10)
    b = frag(2, 0, 100, x=300, y=300)
    stray = frag(3, 101, 20, x=9000, y=9000)
    res = run([a, b, stray], n_animals=2)
    assert res.dropped == [(3, 101, 120)]


def test_frames_no_fragment_covers_are_nan():
    res = run([frag(1, 0, 50, x=10, y=10)], n_animals=1, n_frames=200)
    assert np.all(np.isnan(res.tracks[0].xy[60]))
    assert res.tracks[0].confidence[60].tolist() == [0.0, 0.0]


def test_fewer_animals_than_slots_leaves_the_surplus_all_nan():
    # Asked for 2, found 1. The empty slot is a real answer -- one animal was
    # never seen -- and is far better than raising after an hour of inference.
    res = run([frag(1, 0, 200, x=10, y=10)], n_animals=2)
    assert res.tracks.n_animals == 2
    assert np.all(np.isnan(res.tracks[1].xy))


def test_the_result_does_not_depend_on_input_order():
    a, b, c = (
        frag(1, 0, 100, x=10, y=10),
        frag(2, 0, 100, x=300, y=300),
        frag(3, 101, 99, x=12, y=10),
    )
    first = run([a, b, c], n_animals=2).tracks
    second = run([c, b, a], n_animals=2).tracks
    assert np.allclose(first[0].xy, second[0].xy, equal_nan=True)
    assert np.allclose(first[1].xy, second[1].xy, equal_nan=True)


def test_no_usable_fragments_still_returns_n_empty_slots():
    res = run([frag(1, 0, 2)], n_animals=2, min_fragment_frames=5)
    assert res.tracks.n_animals == 2
    assert np.all(np.isnan(res.tracks[0].xy))


def test_flicker_below_the_floor_is_counted_but_not_reported_as_dropped():
    """Fragments shorter than min_fragment_frames never reach seed_slots'
    seeds or remaining, so they never become a `dropped` entry either --
    below_floor is the only place that count survives."""
    a = frag(1, 0, 100, x=10, y=10)
    b = frag(2, 0, 100, x=300, y=300)
    flicker = frag(3, 50, 2, x=9000, y=9000)
    res = run([a, b, flicker], n_animals=2, min_fragment_frames=5)
    assert res.below_floor == 1
    assert res.dropped == []


# ---------------------------------------------------------------------------
# Shapes observed on real footage
#
# Both of these came out of running a real YOLO pose model over a two-mouse
# recording (CalMS21 mouse001). Neither is hypothetical, and neither was
# pinned by anything above.
# ---------------------------------------------------------------------------


def test_two_boxes_on_one_animal_fill_both_slots_but_are_flagged_close():
    """A duplicate detection is indistinguishable from a second animal here.

    Observed on a frame where the detector put two overlapping boxes on the
    same mouse. Consolidation has no way to know -- two fragments, two slots,
    and nothing in the geometry says one of them is a ghost. What saves the
    analyst is the identity sidecar: the pair never separates, so EVERY frame
    is flagged ``close`` and the whole session reads as untrustworthy rather
    than as two animals that happened to stay together.
    """
    from collections import Counter

    from glider.vision.pose.identity import identity_flags

    real = frag(1, 0, 200, x=100, y=100)
    ghost = frag(2, 0, 200, x=108, y=100)  # 8 px away: the same mouse, twice
    res = run([real, ghost], n_animals=2)

    assert res.tracks.n_animals == 2
    assert res.dropped == []

    flags = identity_flags(res.tracks, stitched=res.stitched, min_separation_px=60.0)
    counts = Counter(flag for _frame, _slot, flag in flags)
    # Both slots, every frame -- no frame of this session is presented as clean.
    assert counts["close"] == 2 * res.tracks.n_frames
    assert set(counts) == {"close"}


def test_a_persistent_false_positive_outsits_an_intermittent_animal():
    """A fixture the detector likes beats a mouse that keeps being occluded.

    Observed on a frame where the detector fired on the cage's metal water
    spout. Hardware is in EVERY frame, so its fragment is long; a real animal
    that keeps disappearing behind its cage-mate yields many short ones. Seeding
    takes the longest fragments, so the spout wins a slot and the animal's
    fragments are all refused.

    This is pinned rather than fixed on purpose. The only signal separating
    "stationary hardware" from "frozen mouse" is motion, and freezing is a
    behaviour these recordings exist to measure -- a motion heuristic here would
    discard real data. Arena gating upstream is the defence: the spout sits
    outside the bedding, and ``infer_video_tracks`` drops out-of-arena
    candidates before consolidation ever sees them.

    What consolidation owes the analyst is evidence, and it pays: every refused
    fragment lands in ``dropped``, so a session that lost an animal this way
    says so without re-running anything.
    """
    spout = frag(9, 0, 200, x=900, y=500)  # hardware: present every frame
    tracked = frag(1, 0, 200, x=100, y=100)  # one mouse, cleanly followed
    # The other mouse: seen in bursts, never for long.
    occluded = [frag(20 + i, i * 20, 8, x=300, y=300) for i in range(10)]

    res = run([spout, tracked, *occluded], n_animals=2)

    slot_x = [res.tracks[s].xy[0, 0, 0] for s in res.tracks]
    assert 100.0 in slot_x, "the well-tracked animal keeps its slot"
    assert 900.0 in slot_x, "the false positive takes the other one"
    assert 300.0 not in slot_x, "the intermittent animal gets no slot at all"

    # The loss is recorded, not silent: every burst is accounted for.
    assert len(res.dropped) == len(occluded)
    assert {t for t, _s, _e in res.dropped} == {20 + i for i in range(10)}
