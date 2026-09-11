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


@dataclass
class Fragment:
    """One tracker id's continuous run of detections.

    ``frames`` is strictly increasing but need not be contiguous -- a tracker
    may hold an id across a frame it did not detect.
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
    """
    usable = [f for f in fragments if len(f) >= min_fragment_frames]
    # Sort key is total: length descending, then first frame, then track id.
    # Without the tie-breakers two equal-length fragments could swap slots
    # between runs, which is the same reproducibility problem as above.
    usable.sort(key=lambda f: (-len(f), f.start, f.track_id))

    seeds = usable[:n_animals]
    remaining = usable[n_animals:]
    seeds.sort(key=lambda f: (f.start, f.track_id))
    return seeds, remaining


def _gap_and_distance(a: Fragment, b: Fragment) -> tuple[int, float]:
    """Temporal gap and spatial jump between two non-overlapping fragments.

    Measured in whichever direction they actually sit: *b* may precede *a*, and
    measuring only backwards would strand a fragment that starts before the
    seed does -- it would be dropped despite being the same animal.
    """
    if a.end < b.start:
        gap = b.start - a.end
        here, there = a.last_centroid(), b.first_centroid()
    else:
        gap = a.start - b.end
        here, there = a.first_centroid(), b.last_centroid()
    return gap, float(np.linalg.norm(there - here))


def assignment_cost(
    slot: list[Fragment],
    fragment: Fragment,
    *,
    max_travel_px_per_frame: float,
) -> float:
    """What it would cost to call *fragment* the same animal as *slot*.

    ``math.inf`` means refused, and there are three ways to earn it:

    * the fragment's frames overlap frames the slot already holds -- one animal
      cannot be in two places, and this constraint is the entire value of
      knowing the animal count up front;
    * the jump needed to get there exceeds ``max_travel_px_per_frame``;
    * either endpoint is NaN, so there is no evidence either way. Scoring that
      as NaN instead would sort unpredictably against real costs.

    Otherwise the cost is the **implied speed** in pixels per frame. A speed
    rather than a distance because an animal out of view for a second is
    legitimately further away than one out of view for a single frame.
    """
    if not slot:
        return math.inf

    for other in slot:
        if fragment.start <= other.end and other.start <= fragment.end:
            return math.inf

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
    """

    tracks: PoseTracks
    stitched: dict[int, set[int]]
    dropped: list[tuple[int, int, int]]


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
        for fragment in slot:
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
    )
