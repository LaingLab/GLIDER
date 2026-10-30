"""Turning a tracker's fragments into exactly N lifelong animals.

ByteTrack answers "is this the same blob as last frame". Over twenty minutes of
two mice crossing and occluding each other it answers that dozens of times,
producing far more track ids than there are animals. Nothing downstream can use
that: a per-animal ethogram needs "animal 0" to mean one mouse for the length of
the video, not for the length of a tracking fragment.

Offline we have two things the live path does not -- the whole video at once,
and the animal count. Both are exploited here.

Pure numpy on purpose: no ultralytics, no video, no I/O. The decision about
which animal is which is the part most worth testing, and it should not need a
GPU to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from glider.vision.pose.core import PoseData
from glider.vision.pose.tracks import PoseTracks

__all__ = ["Fragment", "assignment_cost", "seed_slots", "ConsolidationResult", "consolidate"]

#: Frames two ids may share and still be one animal changing id. A duplicate box
#: lets the new id start while the old one is still alive, and ByteTrack keeps a
#: lost id for ``track_buffer`` (30) frames, so a real handoff can overlap by a
#: few frames. Two ids together for longer than this are two things.
MAX_HANDOFF_OVERLAP_FRAMES = 30

#: A fragment this long is an animal rather than a flicker, so when the animal
#: count leaves it only one possible slot, it goes there despite a jump.
MIN_ELIMINATION_SECONDS = 1.0


@dataclass
class Fragment:
    """One tracker id's continuous run of detections.

    ``frames`` is strictly increasing but need not be contiguous -- a tracker
    may hold an id across a frame it did not detect, and ByteTrack can give a
    lost id back after another id has covered the animal in between. So
    ``start``/``end`` bound a fragment without saying it was there throughout.
    """

    track_id: int
    frames: np.ndarray  # (F,) int
    xy: np.ndarray  # (F, K, 2)
    confidence: np.ndarray  # (F, K)

    @property
    def start(self) -> int:
        """First frame, inclusive."""
        return int(self.frames[0])

    @property
    def end(self) -> int:
        """Last frame, inclusive."""
        return int(self.frames[-1])

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def centroid_at(self, i: int) -> np.ndarray:
        """Mean keypoint position at row *i*, or NaN when nothing was localized.

        ``np.nanmean`` of an all-NaN row is the right answer but a noisy one --
        it warns. The explicit guard keeps a video full of dropouts from
        emitting thousands of RuntimeWarnings.
        """
        row = self.xy[i]
        if not np.any(np.isfinite(row)):
            return np.array([np.nan, np.nan])
        return np.nanmean(row, axis=0)

    def first_centroid(self) -> np.ndarray:
        return self.centroid_at(0)

    def last_centroid(self) -> np.ndarray:
        return self.centroid_at(len(self) - 1)


def seed_slots(
    fragments: list[Fragment],
    n_animals: int,
    *,
    min_fragment_frames: int,
) -> tuple[list[Fragment], list[Fragment]]:
    """Pick the N fragments that anchor the slots, and return the rest.

    Returns ``(seeds, remaining)``. Seeds are in **first-appearance** order, so
    slot ids are reproducible: ordering them by duration instead would renumber
    every animal the moment a tuning knob changed, and two runs of the same
    video would disagree about who animal 0 was.

    ``remaining`` is longest-first, which is the order Task 3 assigns in -- a
    long fragment carries more evidence about where it belongs, so it should
    claim its slot before a short one can mis-claim it.

    Seeds have to be seen together. Two long fragments that never coexist may
    be one animal before and after an id change, and seeding both would leave
    the other animal no slot to go to. A seed that coexists with none of the
    others is only taken when nothing better is left to fill the slot.
    """
    usable = [f for f in fragments if len(f) >= min_fragment_frames]
    # Sort key is total: length descending, then first frame, then track id.
    # Without the tie-breakers two equal-length fragments could swap slots
    # between runs, which is the same reproducibility problem as above.
    usable.sort(key=lambda f: (-len(f), f.start, f.track_id))

    seeds: list[Fragment] = []
    for fragment in usable:
        if len(seeds) == n_animals:
            break
        if all(len(_shared_rows(fragment, s)[0]) > MAX_HANDOFF_OVERLAP_FRAMES for s in seeds):
            seeds.append(fragment)
    remaining = [f for f in usable if not any(f is s for s in seeds)]
    while len(seeds) < n_animals and remaining:
        seeds.append(remaining.pop(0))
    seeds.sort(key=lambda f: (f.start, f.track_id))
    return seeds, remaining


def _shared_rows(a: Fragment, b: Fragment) -> tuple[np.ndarray, np.ndarray]:
    """Rows of *a* and of *b* holding the frames both were detected on."""
    _, in_a, in_b = np.intersect1d(a.frames, b.frames, assume_unique=True, return_indices=True)
    return in_a, in_b


def _gap_and_distance(a: Fragment, b: Fragment) -> tuple[int, float]:
    """Temporal gap and spatial jump between two fragments sharing no frames.

    Measured between the closest frames they actually hold, in whichever
    direction those sit: *b* may precede *a*, and measuring only backwards would
    strand a fragment that starts before the seed does -- it would be dropped
    despite being the same animal. *b* may equally sit in a hole inside *a*.
    """
    after = np.searchsorted(a.frames, b.frames)
    gaps_after = np.where(
        after < len(a), a.frames[np.minimum(after, len(a) - 1)] - b.frames, np.inf
    )
    gaps_before = np.where(after > 0, b.frames - a.frames[np.maximum(after - 1, 0)], np.inf)

    row_after, row_before = int(np.argmin(gaps_after)), int(np.argmin(gaps_before))
    if gaps_before[row_before] <= gaps_after[row_after]:
        row_b, row_a = row_before, int(after[row_before]) - 1
    else:
        row_b, row_a = row_after, int(after[row_after])
    gap = abs(int(a.frames[row_a]) - int(b.frames[row_b]))
    return gap, float(np.linalg.norm(b.centroid_at(row_b) - a.centroid_at(row_a)))


def _handoff_cost(
    slot: list[Fragment], fragment: Fragment, *, max_travel_px_per_frame: float
) -> float:
    """Cost of a *fragment* that shares frames with *slot*, or ``math.inf``.

    Accepted only if it looks like one animal changing id: briefly together, in
    the same place, with the fragment mostly adding frames the slot lacks.
    Priced above any clean join, so a slot it fits without overlapping wins.
    """
    pairs = [(other, *_shared_rows(other, fragment)) for other in slot]
    n_shared = sum(len(in_other) for _, in_other, _ in pairs)
    if n_shared > MAX_HANDOFF_OVERLAP_FRAMES or n_shared * 2 >= len(fragment):
        return math.inf

    distances = [
        np.linalg.norm(fragment.centroid_at(j) - other.centroid_at(i))
        for other, in_other, in_fragment in pairs
        for i, j in zip(in_other, in_fragment, strict=True)
    ]
    finite = [d for d in distances if math.isfinite(d)]
    distance = float(np.median(finite)) if finite else math.inf
    if distance > max_travel_px_per_frame:
        return math.inf
    return max_travel_px_per_frame + distance


def assignment_cost(
    slot: list[Fragment],
    fragment: Fragment,
    *,
    max_travel_px_per_frame: float,
) -> float:
    """What it would cost to call *fragment* the same animal as *slot*.

    ``math.inf`` means refused, and there are three ways to earn it:

    * the fragment holds frames the slot already holds -- one animal cannot be
      in two places, and this constraint is the entire value of knowing the
      animal count up front. The exception is the brief overlap of an id
      handoff, see :func:`_handoff_cost`. Actual frames are compared, not
      spans: another id can cover a hole in a fragment the tracker gave back;
    * the jump needed to get there exceeds ``max_travel_px_per_frame``;
    * either endpoint is NaN, so there is no evidence either way. Scoring that
      as NaN instead would sort unpredictably against real costs.

    Otherwise the cost is the **implied speed** in pixels per frame. A speed
    rather than a distance because an animal out of view for a second is
    legitimately further away than one out of view for a single frame.
    """
    if not slot:
        return math.inf

    if any(len(_shared_rows(other, fragment)[0]) for other in slot):
        return _handoff_cost(slot, fragment, max_travel_px_per_frame=max_travel_px_per_frame)

    nearest = min(slot, key=lambda o: _gap_and_distance(o, fragment)[0])
    gap, distance = _gap_and_distance(nearest, fragment)
    if not math.isfinite(distance):
        return math.inf
    speed = distance / max(gap, 1)
    return speed if speed <= max_travel_px_per_frame else math.inf


@dataclass
class ConsolidationResult:
    """What consolidation produced, and how much of it was inference.

    ``stitched`` is per slot the set of frames whose data came from a fragment
    *joined* to the slot rather than the one that seeded it. Those frames are
    where an id is a guess, and Task 5 turns them into a flag an analyst can
    filter on.

    ``dropped`` records fragments no slot would accept, as
    ``(track_id, start, end)``. Kept rather than discarded silently: a video
    that drops a lot of them is one whose tuning is wrong, and that should be
    visible without re-running anything.

    ``below_floor`` is how many fragments ``seed_slots`` filtered out for
    being shorter than ``min_fragment_frames``, before seeding or joining was
    even considered -- they reach neither ``seeds`` nor ``remaining``, so
    without this count a run is indistinguishable from one where nothing was
    ever that short. A count, not the spans: which specific fragments is a
    question for the annotated video, not this metadata.
    """

    tracks: PoseTracks
    stitched: dict[int, set[int]]
    dropped: list[tuple[int, int, int]]
    below_floor: int


def consolidate(
    fragments: list[Fragment],
    *,
    n_animals: int,
    n_frames: int,
    keypoint_names: list[str],
    fps: float,
    source: str = "",
    max_travel_px_per_frame: float = 40.0,
    min_fragment_frames: int = 5,
) -> ConsolidationResult:
    """Stitch tracker fragments into exactly *n_animals* lifelong slots.

    Greedy, longest-first, and not a global optimum.

    ponytail: greedy stitching, with a known ceiling. Assignment here is
    order-dependent -- a slot's position changes as fragments join it -- so one
    fixed cost matrix would be solving a different problem, and a global
    assignment is not simply a better version of this. Upgrade path when the
    manual check in the spec's §9 says this is not good enough: multi-hypothesis
    linking over the whole video, or an appearance embedding once the animals
    are marked.
    """
    if n_animals < 1:
        raise ValueError(f"n_animals must be at least 1; got {n_animals}")

    seeds, remaining = seed_slots(fragments, n_animals, min_fragment_frames=min_fragment_frames)
    below_floor = len(fragments) - len(seeds) - len(remaining)
    slots: list[list[Fragment]] = [[s] for s in seeds]
    slots += [[] for _ in range(n_animals - len(slots))]
    seed_ids = {s.track_id for s in seeds}
    dropped: list[tuple[int, int, int]] = []

    for fragment in remaining:
        costs = [
            assignment_cost(slot, fragment, max_travel_px_per_frame=max_travel_px_per_frame)
            for slot in slots
        ]
        best = int(np.argmin(costs))
        # argmin of an all-inf list is 0, which would silently donate every
        # rejected fragment to slot 0.
        if not math.isfinite(costs[best]):
            # Every other slot was busy on this fragment's frames, so with N
            # known this is the only animal it can be. A jump this size is a
            # reason to flag it (it is marked stitched), not to throw it away.
            free = [
                i
                for i, slot in enumerate(slots)
                if not any(len(_shared_rows(o, fragment)[0]) for o in slot)
            ]
            if len(free) == 1 and len(fragment) >= fps * MIN_ELIMINATION_SECONDS:
                slots[free[0]].append(fragment)
                continue
            dropped.append((fragment.track_id, fragment.start, fragment.end))
            continue
        slots[best].append(fragment)

    n_kpts = len(keypoint_names)
    tracks: dict[int, PoseData] = {}
    stitched: dict[int, set[int]] = {}

    for slot_id, slot in enumerate(slots):
        xy = np.full((n_frames, n_kpts, 2), np.nan)
        confidence = np.zeros((n_frames, n_kpts))
        joined: set[int] = set()
        # Seed last and longest later, so that on a handoff's shared frames the
        # better-evidenced fragment is the one that stands.
        for fragment in reversed(slot):
            rows = fragment.frames
            xy[rows] = fragment.xy
            confidence[rows] = fragment.confidence
            if fragment.track_id not in seed_ids:
                joined.update(int(f) for f in rows)
        tracks[slot_id] = PoseData(
            xy=xy,
            confidence=confidence,
            keypoint_names=list(keypoint_names),
            fps=fps,
            source=source or "consolidated",
        )
        stitched[slot_id] = joined

    return ConsolidationResult(
        tracks=PoseTracks(tracks=tracks, fps=fps),
        stitched=stitched,
        dropped=dropped,
        below_floor=below_floor,
    )
