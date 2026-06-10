"""
Ethogram primitives — reduce per-frame behavioral_state into intervals.

The tracking CSV already labels each frame with a ``behavioral_state``
string (whatever the CV pipeline emits — typically ``"resting"``,
``"active"``, ``"locomotion"``, ``"unknown"``, etc.). For analysis we
want *intervals*: contiguous runs of the same state collapsed to a
``(start_ms, end_ms, duration_ms)`` tuple, ready for plotting as a
strip chart or computing bout statistics.

This module is intentionally minimal — the heavy lifting is delegated
to pandas/numpy. Higher-level stats (bout distributions, transition
matrices, dwell time per state) are layered on top in subsequent
phases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Empty-interval template so callers don't special-case "no tracking".
_EMPTY_INTERVALS = pd.DataFrame(columns=["object_id", "state", "start_ms", "end_ms", "duration_ms"])


def compute_intervals(
    tracking: pd.DataFrame,
    *,
    object_id: int = 0,
    state_col: str = "behavioral_state",
    time_col: str = "flow_elapsed_ms",
    include_pre_flow: bool = False,
) -> pd.DataFrame:
    """Run-length encode behavioral states into intervals.

    Filters to one object (most experiments have a single subject), drops
    pre-flow frames by default (frames captured during setup, before the
    flow anchor was set, have empty ``flow_elapsed_ms``), and collapses
    consecutive frames sharing the same state into one row.

    Args:
        tracking: Tracking DataFrame (as loaded by ``Session.tracking``).
        object_id: Which tracked object to ethogram. Most experiments are
            single-subject; multi-object support is a v2 feature.
        state_col: Column to RLE. Defaults to ``behavioral_state``.
        time_col: Time axis for ``start_ms``/``end_ms``. Defaults to
            ``flow_elapsed_ms`` (analyst-friendly — t=0 at flow start);
            use ``elapsed_ms`` for session-relative time.
        include_pre_flow: When True, also include frames captured before
            ``set_flow_anchor`` was called. Their ``flow_elapsed_ms`` is
            NaN, so if ``time_col="flow_elapsed_ms"`` you'll get NaN
            start/end. Mostly useful for diagnostics, not analysis.

    Returns:
        DataFrame with columns ``object_id, state, start_ms, end_ms,
        duration_ms``. One row per contiguous run. Empty DataFrame if
        the requested object isn't present.
    """
    if tracking.empty or state_col not in tracking.columns:
        return _EMPTY_INTERVALS.copy()

    obj_df = tracking[tracking["object_id"] == object_id]
    if obj_df.empty:
        return _EMPTY_INTERVALS.copy()

    if not include_pre_flow and time_col == "flow_elapsed_ms":
        obj_df = obj_df.dropna(subset=["flow_elapsed_ms"])
        if obj_df.empty:
            return _EMPTY_INTERVALS.copy()

    # Sort by time so RLE is meaningful even if upstream wrote out-of-order
    # (it doesn't, but be defensive — analysts sometimes merge files).
    obj_df = obj_df.sort_values(time_col).reset_index(drop=True)

    states = obj_df[state_col].astype("string").fillna("unknown").to_numpy()
    times = obj_df[time_col].to_numpy()

    # Run-length encode: find indices where state changes.
    change_points = np.flatnonzero(states[1:] != states[:-1]) + 1
    boundaries = np.concatenate([[0], change_points, [len(states)]])

    rows = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1] - 1
        start_ms = times[start_idx]
        end_ms = times[end_idx]
        rows.append(
            {
                "object_id": object_id,
                "state": states[start_idx],
                "start_ms": float(start_ms),
                "end_ms": float(end_ms),
                "duration_ms": float(end_ms - start_ms),
            }
        )

    return pd.DataFrame(rows, columns=["object_id", "state", "start_ms", "end_ms", "duration_ms"])


def compute_bouts(
    intervals: pd.DataFrame, *, state: str | None = None
) -> pd.Series | dict[str, pd.Series]:
    """Extract bout durations from interval-form data.

    A "bout" is one contiguous run of a single state — i.e., exactly one
    interval row. The histogram of bout durations per state is a standard
    ethology summary (gives you the mean/SD/distribution of how long the
    subject typically stays in each state before transitioning out).

    Args:
        intervals: Output of ``compute_intervals``.
        state: If specified, return only durations for this state as a
            Series. If None, return a dict ``{state: Series}`` covering
            every state present.

    Returns:
        ``Series`` of durations (ms) for the requested state, or a dict
        of one such Series per state.
    """
    if state is not None:
        sub = intervals[intervals["state"] == state]
        return sub["duration_ms"].reset_index(drop=True)

    return {
        s: group["duration_ms"].reset_index(drop=True) for s, group in intervals.groupby("state")
    }


def compute_state_transitions(intervals: pd.DataFrame) -> pd.DataFrame:
    """Count transitions between consecutive states.

    For an interval sequence ``[resting, active, resting, locomotion]``
    the result is one row each for ``(resting → active)``, ``(active →
    resting)``, ``(resting → locomotion)``. Repeated transitions
    aggregate into the ``count`` column.

    Returns:
        DataFrame with columns ``from_state, to_state, count``. Empty
        for sequences with fewer than two intervals.
    """
    if len(intervals) < 2:
        return pd.DataFrame(columns=["from_state", "to_state", "count"])

    from_states = intervals["state"].iloc[:-1].reset_index(drop=True)
    to_states = intervals["state"].iloc[1:].reset_index(drop=True)
    pairs = pd.DataFrame({"from_state": from_states, "to_state": to_states})
    return pairs.groupby(["from_state", "to_state"]).size().reset_index(name="count")
