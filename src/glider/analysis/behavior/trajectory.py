"""Trajectory-shape features: the *form* of the body-center path.

The rolling mean/std/max stats are permutation-invariant over a window —
they can't tell a straight run from a jitter-in-place, because both can
share the same bulk speed statistics. These window-level features
restore that lost "shape of the motion" by describing the path the body
center traces across the window:

* ``traj_straightness`` — net displacement ÷ path length, in ``[0, 1]``.
  ~1 for a straight run (locomotion), ~0 for returning to where it
  started (jitter / in-place behavior). The key locomote/sniff signal.
* ``traj_path_length`` — total distance travelled (in body-lengths).
* ``traj_net_displacement`` — start-to-end distance (in body-lengths).
* ``traj_radius_gyration`` — RMS spread of the path about its centroid
  (in body-lengths). Small = compact/in-place, large = roaming.
* ``traj_total_turning`` — accumulated absolute heading change along the
  path. Small = straight, large = scribbling.

All five are translation-, rotation-, and scale-invariant (distances are
normalized by the per-window median body length; straightness and
turning are unitless), so they generalize across recordings the same way
the rest of the feature set does.

The body center is the midpoint of the two ``body_axis`` keypoints, so
this works for any skeleton without a dedicated center keypoint. The
batch function is shared with the (future) live path so the two can't
drift.
"""

from __future__ import annotations

import numpy as np

TRAJ_COLUMNS: list[str] = [
    "traj_straightness",
    "traj_path_length",
    "traj_net_displacement",
    "traj_radius_gyration",
    "traj_total_turning",
]


def center_and_length(xy: np.ndarray, body_axis: tuple[int, int]):
    """Return ``(centers (F, 2), lengths (F,))`` from a pose ``xy``.

    ``centers`` is the per-frame midpoint of the two body-axis keypoints;
    ``lengths`` is the body-axis length per frame (the scale normalizer).
    """
    head, tail = body_axis
    h = xy[:, head, :]
    t = xy[:, tail, :]
    centers = 0.5 * (h + t)
    lengths = np.linalg.norm(t - h, axis=1)
    return centers, lengths


def trajectory_features_batch(centers, length_scales) -> np.ndarray:
    """Vectorized trajectory features over many windows.

    Parameters
    ----------
    centers
        ``(m, w, 2)`` — ``m`` windows of ``w`` body-center positions.
        Assumed finite (callers fill gaps first).
    length_scales
        ``(m,)`` per-window body-length normalizer. Non-positive scales
        yield NaN for the distance-based columns.

    Returns
    -------
    ndarray
        ``(m, 5)`` in :data:`TRAJ_COLUMNS` order.
    """
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim != 3 or c.shape[2] != 2:
        raise ValueError(f"centers must be (m, w, 2); got {c.shape}")
    m = c.shape[0]

    steps = np.diff(c, axis=1)  # (m, w-1, 2)
    step_len = np.linalg.norm(steps, axis=2)  # (m, w-1)
    path = step_len.sum(axis=1)  # (m,)
    net = np.linalg.norm(c[:, -1, :] - c[:, 0, :], axis=1)  # (m,)
    straightness = np.where(path > 0, net / np.where(path > 0, path, 1.0), 0.0)

    centroid = c.mean(axis=1, keepdims=True)  # (m, 1, 2)
    rg = np.sqrt((np.linalg.norm(c - centroid, axis=2) ** 2).mean(axis=1))  # (m,)

    headings = np.arctan2(steps[:, :, 1], steps[:, :, 0])  # (m, w-1)
    dtheta = np.diff(headings, axis=1)  # (m, w-2)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
    turning = np.abs(dtheta).sum(axis=1)  # (m,)

    scales = np.asarray(length_scales, dtype=np.float64)
    safe = np.where(scales > 0, scales, np.nan)

    out = np.empty((m, 5), dtype=np.float64)
    out[:, 0] = straightness
    out[:, 1] = path / safe
    out[:, 2] = net / safe
    out[:, 3] = rg / safe
    out[:, 4] = turning
    return out


def apply_trajectory_rolling(xy, body_axis: tuple[int, int], window: int = 30):
    """Per-frame rolling trajectory features for a whole session.

    Returns a DataFrame of :data:`TRAJ_COLUMNS` aligned to ``xy``'s frames.
    Each row uses the trailing ``window`` frames; the first ``window-1``
    are NaN (dropped downstream, like the other rolling features). NaN
    gaps in the body-center track (dropped keypoints) are linearly
    interpolated before windowing so a brief dropout doesn't void the
    whole window.
    """
    import pandas as pd

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    xy = np.asarray(xy, dtype=np.float64)
    n = xy.shape[0]
    centers, lengths = center_and_length(xy, body_axis)

    # Interpolate short gaps so windows stay usable; ends are filled by
    # nearest valid value (limit_direction="both").
    cdf = pd.DataFrame(centers, columns=["cx", "cy"]).interpolate(limit_direction="both")
    ls = pd.Series(lengths).interpolate(limit_direction="both")
    centers = cdf.to_numpy()
    lengths = ls.to_numpy()

    out = np.full((n, len(TRAJ_COLUMNS)), np.nan)
    if n >= window and np.isfinite(centers).all():
        # (n-w+1, w, 2) trailing windows landing on row j+window-1.
        cviews = np.moveaxis(
            np.lib.stride_tricks.sliding_window_view(centers, window, axis=0),
            -1,
            1,
        )
        lviews = np.lib.stride_tricks.sliding_window_view(lengths, window)
        scales = np.nanmedian(lviews, axis=1)
        out[window - 1 :] = trajectory_features_batch(cviews, scales)

    return pd.DataFrame(out, columns=TRAJ_COLUMNS)
