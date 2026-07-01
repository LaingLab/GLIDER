"""Pose smoothing and confidence-based interpolation.

YOLO-pose outputs are pixel-precise but jittery frame-to-frame compared to
DeepLabCut. Most behavior tools (Keypoint-MoSeq, VAME, B-SOID) assume smooth
trajectories, so we ship two cheap fixes:

* ``mask_low_confidence``  — set xy to NaN where likelihood < threshold.
* ``interpolate_gaps``     — linearly fill NaN gaps shorter than ``max_gap``.
* ``median_filter``        — temporal median filter with odd window size.

The default pipeline ``smooth(...)`` chains them in the order recommended for
YOLO output.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

from glider.vision.pose.core import PoseData

__all__ = [
    "mask_low_confidence",
    "interpolate_gaps",
    "median_filter",
    "smooth",
]


def mask_low_confidence(pose: PoseData, *, threshold: float = 0.5) -> PoseData:
    """Set xy to NaN wherever confidence is below ``threshold``."""
    out = pose.copy()
    bad = out.confidence < threshold
    # broadcast (T, K) mask over the trailing 2-element axis
    out.xy[bad, :] = np.nan
    return out


def interpolate_gaps(pose: PoseData, *, max_gap: int = 5) -> PoseData:
    """Linearly interpolate NaN gaps up to ``max_gap`` frames long.

    Confidence in interpolated frames is set to the minimum of the bracketing
    real frames (so downstream filters can still flag them).
    """
    if max_gap < 1:
        return pose.copy()

    out = pose.copy()
    n_frames = out.n_frames

    for k in range(out.n_keypoints):
        for axis in range(2):
            series = out.xy[:, k, axis]
            isnan = np.isnan(series)
            if not isnan.any():
                continue

            # Identify contiguous NaN runs.
            edges = np.diff(np.concatenate(([0], isnan.view(np.int8), [0])))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]  # exclusive

            for s, e in zip(starts, ends, strict=False):
                gap_len = e - s
                if gap_len > max_gap:
                    continue
                if s == 0 or e == n_frames:
                    # No bracketing real frame on one side — can't interpolate.
                    continue
                left_x = series[s - 1]
                right_x = series[e]
                if not (np.isfinite(left_x) and np.isfinite(right_x)):
                    continue
                t = np.linspace(0, 1, gap_len + 2)[1:-1]
                series[s:e] = left_x + t * (right_x - left_x)

                # Update confidence too.
                left_c = out.confidence[s - 1, k]
                right_c = out.confidence[e, k]
                out.confidence[s:e, k] = min(left_c, right_c)

            out.xy[:, k, axis] = series

    return out


def median_filter(pose: PoseData, *, window: int = 5) -> PoseData:
    """Apply a temporal median filter to xy (per body part, per axis).

    ``window`` must be odd. NaNs are preserved (scipy's medfilt treats them
    as values, so we mask before/after).
    """
    if window < 3:
        return pose.copy()
    if window % 2 == 0:
        raise ValueError(f"window must be odd; got {window}")

    out = pose.copy()
    nan_mask = np.isnan(out.xy)

    # Replace NaN with 0 for the filter, then re-mask.
    filled = np.where(nan_mask, 0.0, out.xy)
    for k in range(out.n_keypoints):
        for axis in range(2):
            filled[:, k, axis] = medfilt(filled[:, k, axis], kernel_size=window)
    filled[nan_mask] = np.nan
    out.xy = filled
    return out


def smooth(
    pose: PoseData,
    *,
    confidence_threshold: float = 0.5,
    max_gap: int = 5,
    median_window: int = 5,
) -> PoseData:
    """Recommended pipeline: mask -> interpolate -> median filter."""
    p = mask_low_confidence(pose, threshold=confidence_threshold)
    p = interpolate_gaps(p, max_gap=max_gap)
    p = median_filter(p, window=median_window)
    return p
