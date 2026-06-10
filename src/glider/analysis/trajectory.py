"""
Trajectory + spatial analysis primitives.

Operates on the per-frame position columns the CV pipeline already wrote
into the tracking CSV: ``center_x, center_y`` for the path itself, and
``zone_ids`` for arena-region membership. None of the calculations here
are particularly clever — the heavy lifting is upstream in the tracker.
This module just shapes the columns into the forms an analyst plots
(path, occupancy heatmap, zone dwell, zone transition matrix).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from glider.analysis.ethogram import compute_intervals

_EMPTY_TRAJECTORY = pd.DataFrame(columns=["frame", "flow_elapsed_ms", "center_x", "center_y"])
_EMPTY_ZONE_DWELL = pd.DataFrame(columns=["zone", "total_ms", "n_entries", "mean_bout_ms"])
_EMPTY_ZONE_TRANSITIONS = pd.DataFrame(columns=["from_zone", "to_zone", "count"])


def compute_trajectory(
    tracking: pd.DataFrame,
    *,
    object_id: int = 0,
    include_pre_flow: bool = False,
) -> pd.DataFrame:
    """Extract the position trace for one tracked object.

    Args:
        tracking: ``Session.tracking`` DataFrame.
        object_id: Which object to extract. Single-object recordings use 0.
        include_pre_flow: Pass True to include camera-warmup frames that
            were captured before the flow anchor was set.

    Returns:
        DataFrame: ``frame, flow_elapsed_ms, center_x, center_y``. Empty
        if the object isn't present.
    """
    if tracking.empty:
        return _EMPTY_TRAJECTORY.copy()

    obj_df = tracking[tracking["object_id"] == object_id]
    if not include_pre_flow and "flow_elapsed_ms" in obj_df.columns:
        obj_df = obj_df.dropna(subset=["flow_elapsed_ms"])
    if obj_df.empty:
        return _EMPTY_TRAJECTORY.copy()

    # `frame` is convenient when present but not load-bearing for
    # trajectory itself; hand-crafted DataFrames in tests may omit it.
    cols = [c for c in ["frame", "flow_elapsed_ms", "center_x", "center_y"] if c in obj_df.columns]
    return obj_df[cols].reset_index(drop=True)


def compute_occupancy(
    tracking: pd.DataFrame,
    *,
    object_id: int = 0,
    bins: int | tuple[int, int] = 50,
    frame_size: tuple[int, int] | None = None,
    include_pre_flow: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D histogram of how many frames the subject spent at each pixel.

    Returns the histogram + bin edges in the standard
    ``np.histogram2d`` format so plot helpers can pass them straight to
    ``imshow`` with the right extent.

    Args:
        tracking: ``Session.tracking`` DataFrame.
        object_id: Which object to bin.
        bins: Bin count (single int → square grid) or ``(nx, ny)`` tuple.
        frame_size: ``(width, height)`` to anchor the histogram extent
            to the full frame even if the subject only visited part of
            it. When None, extent comes from observed min/max coords —
            fine for relative occupancy, misleading for absolute.
        include_pre_flow: See ``compute_trajectory``.

    Returns:
        ``(heatmap, x_edges, y_edges)``. Heatmap is ``(nx, ny)`` —
        i.e., the first axis is x. Plot helpers need to transpose for
        ``imshow``.
    """
    traj = compute_trajectory(tracking, object_id=object_id, include_pre_flow=include_pre_flow)
    traj = traj.dropna(subset=["center_x", "center_y"])

    if traj.empty:
        bins_xy = (bins, bins) if isinstance(bins, int) else bins
        return np.zeros(bins_xy), np.array([]), np.array([])

    if frame_size is not None:
        range_ = [[0, frame_size[0]], [0, frame_size[1]]]
    else:
        range_ = None

    heatmap, x_edges, y_edges = np.histogram2d(
        traj["center_x"].to_numpy(),
        traj["center_y"].to_numpy(),
        bins=bins,
        range=range_,
    )
    return heatmap, x_edges, y_edges


def compute_zone_dwell(
    tracking: pd.DataFrame,
    *,
    object_id: int = 0,
) -> pd.DataFrame:
    """Per-zone dwell time, entry count, and mean bout duration.

    The tracking CSV's ``zone_ids`` column is a comma-separated string
    of all zones the subject is in at a given frame (or empty for
    none). This function treats each distinct ``zone_ids`` string as
    its own categorical state — so multi-zone overlaps appear as a
    combined "zone1,zone2" row rather than being double-counted. For
    non-overlapping zones (the common case) the result reads naturally.

    Returns:
        DataFrame: ``zone, total_ms, n_entries, mean_bout_ms``. One row
        per distinct ``zone_ids`` value observed.
    """
    if tracking.empty:
        return _EMPTY_ZONE_DWELL.copy()

    intervals = compute_intervals(
        tracking,
        object_id=object_id,
        state_col="zone_ids",
    )
    if intervals.empty:
        return _EMPTY_ZONE_DWELL.copy()

    dwell = (
        intervals.groupby("state")
        .agg(
            total_ms=("duration_ms", "sum"),
            n_entries=("duration_ms", "count"),
            mean_bout_ms=("duration_ms", "mean"),
        )
        .reset_index()
        .rename(columns={"state": "zone"})
    )
    return dwell


def compute_zone_transitions(
    tracking: pd.DataFrame,
    *,
    object_id: int = 0,
) -> pd.DataFrame:
    """Count transitions between consecutive zone states.

    Pairs adjacent ``zone_ids`` intervals into ``(from_zone, to_zone)``
    and aggregates duplicates into a count. The result is a sparse
    representation of a transition matrix — call ``.pivot_table()`` on
    it if you want the full N×N matrix for plotting.

    Returns:
        DataFrame: ``from_zone, to_zone, count``. Empty for recordings
        with fewer than two zone intervals.
    """
    if tracking.empty:
        return _EMPTY_ZONE_TRANSITIONS.copy()

    intervals = compute_intervals(
        tracking,
        object_id=object_id,
        state_col="zone_ids",
    )
    if len(intervals) < 2:
        return _EMPTY_ZONE_TRANSITIONS.copy()

    from_zones = intervals["state"].iloc[:-1].reset_index(drop=True)
    to_zones = intervals["state"].iloc[1:].reset_index(drop=True)
    pairs = pd.DataFrame({"from_zone": from_zones, "to_zone": to_zones})
    return pairs.groupby(["from_zone", "to_zone"]).size().reset_index(name="count")
