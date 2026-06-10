"""
Kinematic primitives — velocity, speed distributions, cumulative distance.

The tracker already computes per-frame velocity (``velocity_px_frame``)
and running cumulative distance (``cumulative_mm``). This module
exposes them in the shape analysts want:

  - ``compute_velocity_series`` — per-frame velocity vs flow time, with
    optional conversion to per-second (using the recording's frame rate)
    and to mm (when calibration was set during capture).

  - ``compute_speed_distribution`` — histogram bins + counts ready to
    plot as a speed distribution.

  - ``extract_cumulative_distance`` — the cumulative_mm column reshaped
    against ``flow_elapsed_ms`` for a clean "distance traveled" curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EMPTY_VELOCITY = pd.DataFrame(columns=["flow_elapsed_ms", "velocity"])
_EMPTY_CUMULATIVE = pd.DataFrame(columns=["flow_elapsed_ms", "cumulative_mm"])
_EMPTY_SPEED_DIST = pd.DataFrame(columns=["bin_center", "count"])


def compute_velocity_series(
    tracking: pd.DataFrame,
    *,
    object_id: int = 0,
    frame_rate: float | None = None,
    include_pre_flow: bool = False,
) -> pd.DataFrame:
    """Velocity vs flow time for one object.

    ``velocity_px_frame`` from the tracker is in pixels-per-frame —
    inconvenient for plotting if the frame rate varies. Passing
    ``frame_rate`` converts to pixels-per-second so the y-axis units are
    interpretable across recordings with different camera speeds.

    Args:
        tracking: ``Session.tracking`` DataFrame.
        object_id: Which object's velocity to extract.
        frame_rate: Frames per second. If None, returns the raw
            px/frame values (analyst can rescale later).
        include_pre_flow: See ``compute_trajectory``.

    Returns:
        DataFrame: ``flow_elapsed_ms, velocity``. Velocity units are
        px/s when ``frame_rate`` provided, px/frame otherwise. Empty
        if the object isn't present.
    """
    if tracking.empty or "velocity_px_frame" not in tracking.columns:
        return _EMPTY_VELOCITY.copy()

    obj_df = tracking[tracking["object_id"] == object_id]
    if not include_pre_flow:
        obj_df = obj_df.dropna(subset=["flow_elapsed_ms"])
    if obj_df.empty:
        return _EMPTY_VELOCITY.copy()

    velocity = obj_df["velocity_px_frame"].to_numpy(dtype=float)
    if frame_rate is not None and frame_rate > 0:
        velocity = velocity * frame_rate

    return pd.DataFrame(
        {
            "flow_elapsed_ms": obj_df["flow_elapsed_ms"].to_numpy(),
            "velocity": velocity,
        }
    )


def compute_speed_distribution(
    tracking: pd.DataFrame,
    *,
    object_id: int = 0,
    frame_rate: float | None = None,
    bins: int = 20,
    include_pre_flow: bool = False,
) -> pd.DataFrame:
    """Histogram of speeds for one object.

    Speed is the magnitude of velocity (already non-negative in the
    tracker's output). The histogram surfaces *how often* the subject
    is moving slowly vs. quickly — useful for distinguishing
    "mostly resting with brief bursts" from "constantly moving" without
    eyeballing the time series.

    Args:
        bins: Number of histogram bins.
        Other args mirror ``compute_velocity_series``.

    Returns:
        DataFrame: ``bin_center, count``. ``bin_center`` is the midpoint
        of each bin (in the same units as ``velocity`` above). Empty
        if no velocity data.
    """
    vel = compute_velocity_series(
        tracking,
        object_id=object_id,
        frame_rate=frame_rate,
        include_pre_flow=include_pre_flow,
    )
    if vel.empty:
        return _EMPTY_SPEED_DIST.copy()

    speeds = vel["velocity"].to_numpy()
    speeds = speeds[~np.isnan(speeds)]
    if speeds.size == 0:
        return _EMPTY_SPEED_DIST.copy()

    counts, edges = np.histogram(speeds, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"bin_center": centers, "count": counts})


def extract_cumulative_distance(
    tracking: pd.DataFrame,
    *,
    object_id: int = 0,
    include_pre_flow: bool = False,
) -> pd.DataFrame:
    """Cumulative distance traveled, in mm if calibrated.

    The tracker writes ``cumulative_mm`` per frame; this just slices it
    against ``flow_elapsed_ms`` so the analyst can plot a clean
    "distance vs time" curve. Note: when the recording wasn't
    calibrated, the column actually contains pixel distance (see the
    tracker — it falls back to pixels when calibration is absent and
    keeps writing the same column name).

    Returns:
        DataFrame: ``flow_elapsed_ms, cumulative_mm``. Empty if no
        tracking or object missing.
    """
    if tracking.empty or "cumulative_mm" not in tracking.columns:
        return _EMPTY_CUMULATIVE.copy()

    obj_df = tracking[tracking["object_id"] == object_id]
    if not include_pre_flow:
        obj_df = obj_df.dropna(subset=["flow_elapsed_ms"])
    if obj_df.empty:
        return _EMPTY_CUMULATIVE.copy()

    return pd.DataFrame(
        {
            "flow_elapsed_ms": obj_df["flow_elapsed_ms"].to_numpy(),
            "cumulative_mm": obj_df["cumulative_mm"].to_numpy(dtype=float),
        }
    )
