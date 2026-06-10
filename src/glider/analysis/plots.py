"""
Plotting helpers for analysis primitives.

Matplotlib is a soft dependency — the core ``glider.analysis`` library
(Session loading, ethogram intervals, stats) works without it. Only the
``plot_*`` functions in this module require matplotlib; importing them
when matplotlib is missing raises a friendly install instruction.

Plots are deliberately paper-quality by default (clean colormaps, legible
fonts, no chartjunk). For interactive in-app use, pass ``ax=`` an
existing Axes embedded in a FigureCanvas widget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _require_matplotlib() -> Any:
    """Lazy-import matplotlib; raise a useful message if missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install matplotlib"
        ) from e
    return plt


def plot_ethogram(
    intervals: pd.DataFrame,
    *,
    ax: Axes | None = None,
    state_colors: dict[str, Any] | None = None,
    title: str = "Ethogram",
    time_unit: str = "s",
) -> Axes:
    """Render an ethogram strip chart from interval-form data.

    Each state is drawn as a colored horizontal bar at y = object_id.
    Multiple objects stack on separate rows. State→color mapping is
    deterministic per state name so repeated calls produce consistent
    colors (useful when comparing recordings).

    Args:
        intervals: DataFrame from ``compute_intervals`` (or any frame
            with ``object_id, state, start_ms, end_ms`` columns).
        ax: Matplotlib Axes to draw into. If None, a new figure is
            created sized to fit the number of objects.
        state_colors: Optional explicit state→color mapping. When None,
            colors come from the ``tab10`` qualitative colormap in
            sorted-state order.
        title: Axes title.
        time_unit: ``"s"`` (default) divides times by 1000; ``"ms"``
            plots raw milliseconds.

    Returns:
        The Axes that was drawn into.
    """
    plt = _require_matplotlib()
    from matplotlib.patches import Patch

    if intervals.empty:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 2))
        ax.set_title(f"{title} (no data)")
        ax.set_xlabel(f"Flow time ({time_unit})")
        ax.set_yticks([])
        return ax

    object_ids = sorted(intervals["object_id"].unique())
    if ax is None:
        # Height scales with object count so each object's row stays a
        # readable thickness regardless of how many subjects are stacked.
        _, ax = plt.subplots(figsize=(12, max(1.5, 0.6 * len(object_ids) + 1.0)))

    divisor = 1000.0 if time_unit == "s" else 1.0

    states = sorted(intervals["state"].dropna().unique())
    if state_colors is None:
        cmap = plt.get_cmap("tab10", max(len(states), 1))
        state_colors = {s: cmap(i % cmap.N) for i, s in enumerate(states)}

    for obj_id in object_ids:
        sub = intervals[intervals["object_id"] == obj_id]
        for _, row in sub.iterrows():
            ax.barh(
                y=obj_id,
                width=(row["end_ms"] - row["start_ms"]) / divisor,
                left=row["start_ms"] / divisor,
                height=0.8,
                color=state_colors.get(row["state"], "lightgray"),
                edgecolor="none",
            )

    ax.set_yticks(object_ids)
    ax.set_yticklabels([f"obj {i}" for i in object_ids])
    ax.set_xlabel(f"Flow time ({time_unit})")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Patch(facecolor=c, edgecolor="none", label=s) for s, c in state_colors.items()]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.9,
            fontsize="small",
            ncol=min(len(handles), 4),
        )

    return ax


def plot_trajectory(
    trajectory: pd.DataFrame,
    *,
    ax: Axes | None = None,
    color_by: str = "time",
    cmap: str = "viridis",
    title: str = "Trajectory",
) -> Axes:
    """2D position path, colored by time (default) or any column present.

    Each frame is rendered as a small scatter point; a faint connecting
    line preserves the path order without dominating the plot. Y-axis is
    inverted because tracking coordinates are image-pixel space
    (origin top-left).

    Args:
        trajectory: Output of ``compute_trajectory``.
        ax: Axes to render into.
        color_by: ``"time"`` to color by ``flow_elapsed_ms``, or the
            name of any other column in ``trajectory``. Falls back to
            sequential index if the column isn't present.
        cmap: Matplotlib colormap name.

    Returns:
        The Axes.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if trajectory.empty:
        ax.set_title(f"{title} (no data)")
        return ax

    x = trajectory["center_x"].to_numpy()
    y = trajectory["center_y"].to_numpy()

    if color_by == "time" and "flow_elapsed_ms" in trajectory.columns:
        c = trajectory["flow_elapsed_ms"].to_numpy()
        cbar_label = "Flow time (ms)"
    elif color_by in trajectory.columns:
        c = trajectory[color_by].to_numpy()
        cbar_label = color_by
    else:
        c = np.arange(len(trajectory))
        cbar_label = "frame index"

    ax.plot(x, y, "-", linewidth=0.3, alpha=0.3, color="gray", zorder=1)
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=4, alpha=0.7, zorder=2)
    plt.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_title(title)
    return ax


def plot_occupancy_heatmap(
    heatmap: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    ax: Axes | None = None,
    cmap: str = "hot",
    title: str = "Occupancy",
) -> Axes:
    """Render a 2D occupancy histogram via imshow.

    Args:
        heatmap: ``(nx, ny)`` array from ``compute_occupancy``.
        x_edges, y_edges: Bin edges from ``compute_occupancy``.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if heatmap.size == 0 or heatmap.sum() == 0:
        ax.set_title(f"{title} (no data)")
        return ax

    # imshow's first axis is rows (y), so transpose; extent maps pixel
    # coords to actual x/y range.
    extent = [x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]]
    im = ax.imshow(heatmap.T, extent=extent, cmap=cmap, aspect="auto", origin="upper")
    plt.colorbar(im, ax=ax, label="frames")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(title)
    return ax


def plot_zone_dwell(
    dwell: pd.DataFrame,
    *,
    ax: Axes | None = None,
    title: str = "Zone dwell time",
    time_unit: str = "s",
) -> Axes:
    """Bar chart of dwell time per zone."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1 + 0.6 * max(len(dwell), 1)), 4))

    if dwell.empty:
        ax.set_title(f"{title} (no data)")
        return ax

    divisor = 1000.0 if time_unit == "s" else 1.0
    zones = dwell["zone"].astype(str).fillna("(none)").to_numpy()
    times = (dwell["total_ms"] / divisor).to_numpy()
    ax.bar(zones, times, edgecolor="black", linewidth=0.4, color="C0")
    ax.set_xlabel("Zone")
    ax.set_ylabel(f"Total time ({time_unit})")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_velocity(
    velocity: pd.DataFrame,
    *,
    ax: Axes | None = None,
    title: str = "Velocity over time",
    time_unit: str = "s",
) -> Axes:
    """Line plot of velocity vs flow time."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    if velocity.empty:
        ax.set_title(f"{title} (no data)")
        return ax

    divisor = 1000.0 if time_unit == "s" else 1.0
    ax.plot(
        velocity["flow_elapsed_ms"].to_numpy() / divisor,
        velocity["velocity"].to_numpy(),
        linewidth=0.7,
        color="C0",
    )
    ax.set_xlabel(f"Flow time ({time_unit})")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_bouts(
    bouts: pd.Series,
    *,
    ax: Axes | None = None,
    bins: int = 20,
    title: str = "Bout duration distribution",
    time_unit: str = "s",
) -> Axes:
    """Histogram of bout durations (output of ``compute_bouts``).

    Pass a single state's Series, not the full dict — call
    ``compute_bouts(intervals, state="resting")`` first.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if len(bouts) == 0:
        ax.set_title(f"{title} (no data)")
        return ax

    divisor = 1000.0 if time_unit == "s" else 1.0
    ax.hist(bouts.to_numpy() / divisor, bins=bins, edgecolor="black", linewidth=0.4)
    ax.set_xlabel(f"Bout duration ({time_unit})")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_event_triggered(
    eta: pd.DataFrame,
    *,
    ax: Axes | None = None,
    n_bins: int = 50,
    title: str = "Event-triggered average",
    time_unit: str = "s",
) -> Axes:
    """Mean ± SEM of value across trials, time-locked to event (t=0).

    Bins ``time_offset_ms`` into ``n_bins`` equal-width bins, computes
    per-bin mean and SEM across trials, plots the mean as a line and SEM
    as a shaded band. A dashed vertical line marks event time.

    Args:
        eta: Output of ``event_triggered``. Requires a numeric ``value``
            column — this plot doesn't make sense for categorical state.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if eta.empty:
        ax.set_title(f"{title} (no data)")
        return ax

    divisor = 1000.0 if time_unit == "s" else 1.0
    # Coerce values to float; non-numeric slices (e.g. behavioral_state)
    # become NaN and drop out of the aggregation cleanly.
    values = pd.to_numeric(eta["value"], errors="coerce")
    times = eta["time_offset_ms"].to_numpy()

    bins = np.linspace(times.min(), times.max(), n_bins + 1)
    bin_idx = np.digitize(times, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    mean = np.full(n_bins, np.nan)
    sem = np.full(n_bins, np.nan)
    for i in range(n_bins):
        bucket = values.iloc[bin_idx == i]
        bucket = bucket.dropna()
        if len(bucket) == 0:
            continue
        mean[i] = bucket.mean()
        sem[i] = bucket.std(ddof=1) / np.sqrt(len(bucket)) if len(bucket) > 1 else 0.0

    centers = (bins[:-1] + bins[1:]) / 2 / divisor
    ax.plot(centers, mean, color="C0", linewidth=1.4)
    ax.fill_between(centers, mean - sem, mean + sem, alpha=0.3, color="C0")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_xlabel(f"Time from event ({time_unit})")
    ax.set_ylabel("Mean value ± SEM")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
