"""Streaming speed axis for live inference: freezing / darting / none.

Freezing and darting are exact functions of the keypoint speed trace, so they
don't need the learned classifier -- they're detected here by a causal state
machine with the SAME duration rules the offline labeller uses
(:mod:`detect_behaviors` / :mod:`freeze`):

  * freezing -- speed below ``freeze_threshold`` for >= ``freeze_min_frames``
    contiguous frames (default 30 = 1 s @ 30 fps).
  * darting  -- speed above ``dart_threshold`` for >= ``dart_min_frames``
    contiguous frames (default 3).

Thresholds are **absolute** (px/frame) -- calibrate once per rig against the
overlay, the same workflow as ``freeze.py --abs-threshold``. The offline
per-session percentile can't be computed on a live stream.

Online semantics: the label is *confirmation-delayed*, never back-filled. A
freeze is only reported from its ``freeze_min_frames``-th frame onward (an
inherent ~1 s latency that is part of freezing's definition, not a streaming
artifact); the earlier frames of the run read ``""`` until the run is long
enough. Inter-burst merging of darts (the offline ``merge_gap``) is a
post-hoc grouping and is intentionally NOT applied here.

Two small, independent units:

  * :class:`CausalSpeed`     -- keypoints -> causal mean-keypoint speed.
  * :class:`FreezeDartDetector` -- speed -> confirmed freeze/dart label.

The live wiring composes them; both are pure and testable in isolation.
"""

from __future__ import annotations

import warnings
from collections import deque

import numpy as np

FREEZING = "freezing"
DARTING = "darting"
NONE = ""

# Single source of truth for the calibration percentiles, shared with the
# offline kinematic prior (:mod:`glider.analysis.behavior.prior`).
FREEZE_PCT_DEFAULT = 10.0
DART_PCT_DEFAULT = 99.5


class FreezeDartDetector:
    """Online state machine: per-frame speed -> ``"freezing"`` / ``"darting"`` / ``""``.

    Maintains run-length counters for contiguous below-``freeze_threshold`` and
    above-``dart_threshold`` frames and emits the behavior once its counter
    reaches the minimum bout length. A frame between the thresholds, or a NaN
    (dropout) frame, breaks both runs. Comparisons are strict (``<`` / ``>``),
    matching ``freeze.py`` / ``detect_behaviors``.
    """

    def __init__(
        self,
        freeze_threshold: float,
        dart_threshold: float,
        freeze_min_frames: int = 30,
        dart_min_frames: int = 3,
    ):
        if freeze_threshold >= dart_threshold:
            raise ValueError(
                f"freeze_threshold ({freeze_threshold}) must be < dart_threshold "
                f"({dart_threshold})"
            )
        self.freeze_threshold = float(freeze_threshold)
        self.dart_threshold = float(dart_threshold)
        self.freeze_min_frames = int(freeze_min_frames)
        self.dart_min_frames = int(dart_min_frames)
        self._below = 0  # contiguous frames < freeze_threshold
        self._above = 0  # contiguous frames > dart_threshold

    def push(self, speed: float) -> str:
        """Advance one frame; return the confirmed speed-axis label."""
        if speed is None or np.isnan(speed):
            self._below = self._above = 0
            return NONE
        if speed < self.freeze_threshold:
            self._below += 1
            self._above = 0
            return FREEZING if self._below >= self.freeze_min_frames else NONE
        if speed > self.dart_threshold:
            self._above += 1
            self._below = 0
            return DARTING if self._above >= self.dart_min_frames else NONE
        self._below = self._above = 0
        return NONE


def speed_axis_offline(
    speeds,
    freeze_threshold: float,
    dart_threshold: float,
    *,
    freeze_min_frames: int = 30,
    dart_min_frames: int = 3,
    dart_merge_gap: int = 0,
) -> list[str]:
    """Freeze/dart labels for a whole speed trace, with the runs kept whole.

    :class:`FreezeDartDetector` cannot label a bout until it has watched the
    bout happen, so it reports a run of *n* frames as *n - min_frames + 1*.
    That is unavoidable for a live overlay and simply wrong for a recording,
    where every frame is known before anything is written: a one-second freeze
    came out as a single frame, and across a 30-animal cohort more than half
    of all freezing time went unreported.

    The minimum duration is a filter on which runs count, not a haircut on the
    ones that do — matching the offline detector this axis was ported from,
    which keeps every zone at least ``min_frames`` long in full.

    ``dart_merge_gap`` optionally joins dart bursts separated by fewer than
    that many frames, which the original applied at 24 (0.8 s) to treat a
    stutter of bursts as one dart. Off by default: it changes what counts as
    one dart, which is a scoring decision rather than a correction.
    """
    values = np.asarray(speeds, dtype=np.float64)
    valid = np.isfinite(values)
    below = (values < freeze_threshold) & valid
    above = (values > dart_threshold) & valid

    labels = [NONE] * values.size
    for start, end in _zones(below):
        if end - start >= freeze_min_frames:
            labels[start:end] = [FREEZING] * (end - start)
    darts = [z for z in _zones(above) if z[1] - z[0] >= dart_min_frames]
    if dart_merge_gap > 0:
        darts = _merge_close(darts, dart_merge_gap)
    for start, end in darts:
        labels[start:end] = [DARTING] * (end - start)
    return labels


def _zones(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as half-open ``(start, end)`` pairs."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _merge_close(zones: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Join zones separated by fewer than ``max_gap`` frames."""
    merged: list[tuple[int, int]] = []
    for start, end in zones:
        if merged and start - merged[-1][1] < max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


class CausalSpeed:
    """Streaming causal mean-keypoint speed (px/frame), mirroring ``freeze.py``.

    Each ``push`` takes one frame's keypoints ``(K, 2)`` and returns the
    smoothed mean-keypoint speed for that frame using only past frames:

      1. trailing rolling-**median** of coordinates over ``coord_smooth`` frames
         (rejects single-frame keypoint teleports -- the main jitter fix),
      2. mean over keypoints of the displacement from the previous smoothed
         frame,
      3. trailing rolling-**mean** of that speed over ``speed_smooth`` frames.

    The first frame returns ``0.0`` (no predecessor). A full-dropout frame
    (all keypoints NaN) returns ``NaN`` so the detector breaks its run.
    """

    def __init__(self, coord_smooth: int = 5, speed_smooth: int = 3):
        self.coord_smooth = max(1, int(coord_smooth))
        self.speed_smooth = max(1, int(speed_smooth))
        self._coords: deque[np.ndarray] = deque(maxlen=self.coord_smooth)
        self._speeds: deque[float] = deque(maxlen=self.speed_smooth)
        self._prev_smoothed: np.ndarray | None = None

    def push(self, xy: np.ndarray) -> float:
        xy = np.asarray(xy, dtype=np.float64)
        self._coords.append(xy)
        with warnings.catch_warnings():
            # A full-dropout frame yields an all-NaN slice; the NaN result is
            # the intended signal (handled below), so silence the warning.
            warnings.filterwarnings("ignore", r"All-NaN slice encountered", RuntimeWarning)
            smoothed = np.nanmedian(np.stack(self._coords, axis=0), axis=0)  # (K, 2)

        # Seed (or re-seed) the reference. An all-NaN reference can never
        # produce a finite displacement again, and the dropout branch below
        # deliberately does not advance it — so accepting one would make a
        # single dropped opening frame silently blank the speed axis for the
        # entire session, which is exactly what it did.
        if self._prev_smoothed is None or np.all(np.isnan(self._prev_smoothed)):
            if np.all(np.isnan(smoothed)):
                self._speeds.append(np.nan)
                return float("nan")
            self._prev_smoothed = smoothed
            self._speeds.append(0.0)
            return 0.0

        disp = np.linalg.norm(smoothed - self._prev_smoothed, axis=1)  # (K,)
        if np.all(np.isnan(disp)):
            # Full dropout: don't advance the reference; signal the detector.
            self._speeds.append(np.nan)
            return float("nan")

        self._prev_smoothed = smoothed
        self._speeds.append(float(np.nanmean(disp)))
        valid = [s for s in self._speeds if not np.isnan(s)]
        return float(np.mean(valid)) if valid else float("nan")


def causal_speed_series(
    xy_frames,
    *,
    coord_smooth: int = 5,
    speed_smooth: int = 3,
) -> np.ndarray:
    """The whole causal speed trace: one px/frame value per input frame.

    :class:`CausalSpeed` is a streaming object, so every consumer that wants
    the trace over a recording has written the same three-line loop. Two of
    them then diverged in what they did next -- the cohort thresholds drop
    frame 0 and the dropouts before taking percentiles, while a per-frame
    readout must keep every frame at its own index or it cannot be indexed by
    frame at all.

    This returns the trace *unfiltered*: frame 0 is the ``0.0`` it is by
    construction, and a dropout is a ``NaN`` sitting at its own index. Callers
    that want the pooled-percentile view filter it themselves; that way the
    signal has one definition and the filtering is visible at the point it
    matters.
    """
    causal = CausalSpeed(coord_smooth=coord_smooth, speed_smooth=speed_smooth)
    return np.asarray([causal.push(xy) for xy in xy_frames], dtype=np.float64)


def calibrate_speed_thresholds(
    xy_frames,
    *,
    freeze_pct: float = FREEZE_PCT_DEFAULT,
    dart_pct: float = DART_PCT_DEFAULT,
    coord_smooth: int = 5,
    speed_smooth: int = 3,
) -> tuple[float, float]:
    """Derive absolute (freeze, dart) thresholds from a representative session.

    Runs :class:`CausalSpeed` over ``xy_frames`` (an iterable of ``(K, 2)``
    keypoint arrays) and returns the ``freeze_pct`` / ``dart_pct`` percentiles
    of the resulting causal speed. This is the once-per-rig calibration for the
    live :class:`FreezeDartDetector`: it reproduces the offline percentile
    method on a calibration clip, then the fixed px/frame values are reused for
    every live run. Frame 0 (speed 0) and dropout (NaN) frames are excluded.
    """
    cs = CausalSpeed(coord_smooth=coord_smooth, speed_smooth=speed_smooth)
    speeds = [cs.push(xy) for xy in xy_frames]
    arr = np.asarray(speeds[1:], dtype=np.float64)  # drop frame 0 (always 0.0)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("no valid frames to calibrate from")
    return float(np.percentile(arr, freeze_pct)), float(np.percentile(arr, dart_pct))
