"""Telling the animal from a still object the pose model also calls an animal.

A rat model run over a novel-object arena finds the rat *and* each object: the
objects come back as small, confident skeletons a fifth of the rat's length.
Picking by box confidence alone hands the track to an object whenever it edges
out the rat, and before the rat is placed the objects are all there is, so the
session opens with a minute of "rat" flicking between two dishes.

Size separates them where confidence does not. Offline we have the whole video,
so the animal's size is measured from the run itself -- a high percentile of the
largest skeleton per frame, which holds as long as the animal is in view for a
tenth of the video -- and anything far smaller is not the animal.

Pure numpy: no ultralytics, no video, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from glider.vision.pose.core import DEFAULT_MIN_BODY_FRACTION, PoseData

__all__ = [
    "DEFAULT_MIN_BODY_FRACTION",
    "BodySizeReport",
    "body_span",
    "gate_body_size",
    "pick_by_size",
    "reference_span",
]

# DEFAULT_MIN_BODY_FRACTION (0.4): a candidate shorter than that fraction of
# the animal's typical span is not the animal. Measured on seven rat sessions,
# the objects sit at 0.15-0.35 and the rat, curled and grooming included, above.

#: The percentile of per-frame spans taken as the animal's full length. High
#: enough that frames without the animal (an empty arena, an object) do not set
#: it, low enough that one stretched-out outlier does not either.
_REFERENCE_PERCENTILE = 90


def body_span(xy: np.ndarray) -> np.ndarray:
    """Diagonal of the keypoints' bounding box, in pixels, over the last two axes.

    ``xy`` is ``(..., K, 2)``. Keypoints at exactly (0, 0) are how Ultralytics
    reports one it could not place, so they are left out with the NaNs; fewer
    than two placed keypoints give NaN, which means "unknown", not "small".
    """
    xy = np.asarray(xy, dtype=float)
    placed = np.isfinite(xy).all(axis=-1) & (xy != 0).any(axis=-1)
    lo = np.where(placed[..., None], xy, np.inf).min(axis=-2)
    hi = np.where(placed[..., None], xy, -np.inf).max(axis=-2)
    span = np.hypot(*np.moveaxis(hi - lo, -1, 0))
    return np.where(placed.sum(axis=-1) >= 2, span, np.nan)


def reference_span(spans: np.ndarray) -> float | None:
    """The animal's typical full span from per-frame spans, or None if unknown."""
    spans = np.asarray(spans, dtype=float)
    finite = spans[np.isfinite(spans)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, _REFERENCE_PERCENTILE))


def pick_by_size(
    candidates: Sequence[Sequence[np.ndarray]],
    min_fraction: float = DEFAULT_MIN_BODY_FRACTION,
) -> tuple[list[int], float | None]:
    """For each frame, the first candidate big enough to be the animal.

    ``candidates[f]`` is frame *f*'s keypoint arrays ``(K, 2)``, best first by
    whatever ranking came before (confidence, arena). A re-ranking, never a
    filter: a frame whose candidates are all undersized keeps its first, and
    blanking it is :func:`gate_body_size`'s job. Returns the chosen index per
    frame (0 for a frame with no candidates) and the reference span used.
    """
    spans = [body_span(np.stack(c)) if len(c) else np.empty(0) for c in candidates]
    widest = np.array([np.nanmax(s) if np.isfinite(s).any() else np.nan for s in spans])
    reference = reference_span(widest)
    if reference is None:
        return [0] * len(candidates), None
    floor = min_fraction * reference
    chosen = []
    for s in spans:
        ok = np.flatnonzero(~(s < floor))  # NaN is unknown, so it is kept
        chosen.append(int(ok[0]) if ok.size else 0)
    return chosen, reference


@dataclass(frozen=True)
class BodySizeReport:
    """What :func:`gate_body_size` blanked, for the CSV's metadata."""

    frames_total: int
    frames_blanked: int
    reference_span_px: float | None
    min_fraction: float

    @property
    def blanked_fraction(self) -> float:
        return self.frames_blanked / self.frames_total if self.frames_total else 0.0


def gate_body_size(
    pose: PoseData, min_fraction: float = DEFAULT_MIN_BODY_FRACTION
) -> tuple[PoseData, BodySizeReport]:
    """Blank the frames whose skeleton is far too small to be the animal.

    What is left after :func:`pick_by_size` -- frames with no animal-sized
    candidate at all, such as an arena with only its objects in it. Returns a
    copy; the input is untouched.
    """
    out = pose.copy()
    spans = body_span(out.xy)
    reference = reference_span(spans)
    small = np.zeros(len(spans), dtype=bool)
    if reference is not None:
        small = spans < min_fraction * reference  # NaN compares False: kept
        out.xy[small] = np.nan
        out.confidence[small] = 0.0
    return out, BodySizeReport(
        frames_total=len(spans),
        frames_blanked=int(small.sum()),
        reference_span_px=None if reference is None else round(reference, 1),
        min_fraction=min_fraction,
    )
