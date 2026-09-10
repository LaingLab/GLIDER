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

from dataclasses import dataclass

import numpy as np

__all__ = ["Fragment", "seed_slots"]


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
