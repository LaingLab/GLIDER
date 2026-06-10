"""
Event log queries + event-triggered analysis.

The event log captures three classes of timestamped events with the
same schema: ``input_change`` (pin-edge readings), ``output_write``
(commanded pin writes), and ``flow_marker`` (start/end boundaries).

This module provides two layers on top:

  - ``find_events``: filter the events frame by source and/or value
    (``find_events(s.events, source="output_write", value="1")`` etc.).

  - ``event_triggered``: time-lock a per-frame tracking column to a set
    of event times. The classic "average velocity ±2 s around each LED
    onset" plot reduces to one ``event_triggered`` call plus the
    ``plot_event_triggered`` helper.

Event times come in as ``flow_elapsed_ms`` (flow-relative), not raw
Unix timestamps — the analyst should already be reasoning in flow time
(t=0 = StartExperiment), and the conversion from event_log timestamps
to flow time is done by ``Session.events_with_flow_ms``.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

_EMPTY_EVENT_TRIGGERED = pd.DataFrame(
    columns=["trial_id", "event_time_ms", "time_offset_ms", "value"]
)


def find_events(
    events: pd.DataFrame,
    *,
    source: str | None = None,
    value: str | None = None,
) -> pd.DataFrame:
    """Filter the events frame by source and/or value.

    Args:
        events: ``Session.events`` (or ``Session.events_with_flow_ms``).
        source: ``"input_change"`` / ``"output_write"`` / ``"flow_marker"``;
            None matches all.
        value: Exact match on the ``value`` column. The events frame stores
            values as strings — ``"1"`` for a digital-high, ``"start"`` for
            a flow_marker start, etc. None matches all.

    Returns:
        Filtered DataFrame, index reset for downstream iteration.
    """
    if events.empty:
        return events.copy()

    out = events
    if source is not None:
        out = out[out["source"] == source]
    if value is not None:
        out = out[out["value"].astype(str) == str(value)]
    return out.reset_index(drop=True)


def event_triggered(
    tracking: pd.DataFrame,
    *,
    event_times_ms: Iterable[float],
    window_ms: tuple[float, float] = (-1000.0, 5000.0),
    value_col: str = "velocity_px_frame",
    object_id: int = 0,
    include_pre_flow: bool = False,
) -> pd.DataFrame:
    """Slice a tracking column around each event time, time-locked.

    For every event, returns the tracking rows whose ``flow_elapsed_ms``
    falls within ``[event_t + window_ms[0], event_t + window_ms[1]]``,
    re-expressed as ``time_offset_ms = row_time - event_t`` so a downstream
    average aligns trials at t=0.

    Args:
        tracking: ``Session.tracking`` DataFrame.
        event_times_ms: Iterable of event times (in flow_elapsed_ms
            units). Typically derived from
            ``Session.events_with_flow_ms`` filtered to the events of
            interest.
        window_ms: ``(pre, post)`` window relative to each event. Default
            is 1 s before, 5 s after — adjust for your paradigm.
        value_col: Tracking column to extract per row. Default is
            ``velocity_px_frame``; common alternatives are ``center_x``,
            ``center_y``, ``cumulative_mm``, ``behavioral_state``.
        object_id: Which tracked object to use.
        include_pre_flow: See ``compute_trajectory``.

    Returns:
        Long-form DataFrame: one row per (trial, frame-in-window):

            ``trial_id, event_time_ms, time_offset_ms, value``

        Empty when there are no events or no tracking rows fall in
        any window.
    """
    if tracking.empty or value_col not in tracking.columns:
        return _EMPTY_EVENT_TRIGGERED.copy()

    obj_df = tracking[tracking["object_id"] == object_id]
    if not include_pre_flow:
        obj_df = obj_df.dropna(subset=["flow_elapsed_ms"])
    if obj_df.empty:
        return _EMPTY_EVENT_TRIGGERED.copy()

    pre, post = window_ms
    flow_ms = obj_df["flow_elapsed_ms"].to_numpy()
    values = obj_df[value_col].to_numpy()

    rows = []
    for trial_id, event_t in enumerate(event_times_ms):
        lo = event_t + pre
        hi = event_t + post
        mask = (flow_ms >= lo) & (flow_ms <= hi)
        if not mask.any():
            continue
        # Vectorize the per-trial extraction so this stays cheap even
        # for hundreds of events × thousands of frames.
        sliced_times = flow_ms[mask]
        sliced_values = values[mask]
        for t, v in zip(sliced_times, sliced_values, strict=True):
            rows.append(
                {
                    "trial_id": trial_id,
                    "event_time_ms": float(event_t),
                    "time_offset_ms": float(t - event_t),
                    "value": v,
                }
            )

    if not rows:
        return _EMPTY_EVENT_TRIGGERED.copy()

    return pd.DataFrame(rows, columns=["trial_id", "event_time_ms", "time_offset_ms", "value"])
