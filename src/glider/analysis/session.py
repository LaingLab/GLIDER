"""
Session — the central object for post-analysis of a GLIDER recording.

Loads the tracking / data / events CSVs from a directory, exposes them
as pandas DataFrames, and derives a few cross-cutting timing properties
(flow_start_wall, flow_end_wall, flow_duration_s, frame_rate) on demand.

Designed for both interactive notebook use and in-app embedding:

    >>> from glider.analysis import Session
    >>> s = Session.load("/path/to/recording_dir")
    >>> s.flow_duration_s
    142.04
    >>> s.tracking.head()
       frame                  timestamp  elapsed_ms  flow_elapsed_ms  ...
    0      1  2026-05-25T14:00:30.033       33.4              NaN  ...
    >>> s.ethogram()  # delegates to glider.analysis.ethogram
       object_id    state  start_ms   end_ms  duration_ms

Missing artifacts are tolerated — properties return empty DataFrames or
None as appropriate rather than raising. The analysis library should
work even on partial recordings (e.g., headless sessions with no
tracking, or runs where the operator stopped before flow markers fired).
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import pandas as pd

from glider.analysis._io import Artifacts, discover, parse_csv

# Empty-frame templates so callers don't have to special-case missing
# artifacts. Column lists mirror the writers in event_logger /
# tracking_logger / data_recorder.
_EMPTY_TRACKING = pd.DataFrame(
    columns=[
        "frame",
        "timestamp",
        "elapsed_ms",
        "flow_elapsed_ms",
        "object_id",
        "class",
        "x",
        "y",
        "w",
        "h",
        "confidence",
        "center_x",
        "center_y",
        "distance_px",
        "distance_mm",
        "cumulative_mm",
        "zone_ids",
        "behavioral_state",
        "velocity_px_frame",
    ]
)

_EMPTY_EVENTS = pd.DataFrame(
    columns=[
        "frame",
        "timestamp",
        "elapsed_ms",
        "source",
        "board_id",
        "device_id",
        "device_type",
        "pin",
        "pin_type",
        "value",
    ]
)

_EMPTY_DATA = pd.DataFrame(columns=["frame", "timestamp", "elapsed_ms", "flow_elapsed_ms"])


class Session:
    """A loaded GLIDER recording, exposing aligned CSVs + derived timing.

    Use ``Session.load(directory)`` to construct one from a recording
    directory. CSVs are parsed eagerly on load (typical recording sizes
    are small enough that this is fine); derived properties like
    ``flow_duration_s`` are cached on first access.
    """

    def __init__(
        self,
        artifacts: Artifacts,
        tracking: pd.DataFrame,
        data: pd.DataFrame,
        events: pd.DataFrame,
        metadata: dict[str, str],
    ):
        self._artifacts = artifacts
        self.tracking = tracking
        self.data = data
        self.events = events
        self.metadata = metadata

    @classmethod
    def load(cls, directory: str | Path) -> Session:
        """Load a recording directory into a Session.

        Discovers tracking/data/events CSVs by their ``# GLIDER ...``
        header marker, parses each, and merges metadata headers into
        a single dict (later files overwrite earlier ones if a key
        repeats — order is tracking, then data, then events, so events
        wins by convention since it's the canonical source for flow
        boundaries).

        Args:
            directory: Path to a recording directory.

        Returns:
            Session instance.
        """
        artifacts = discover(Path(directory))

        tracking = _EMPTY_TRACKING.copy()
        data = _EMPTY_DATA.copy()
        events = _EMPTY_EVENTS.copy()
        metadata: dict[str, str] = {}

        if artifacts.tracking is not None:
            tracking_meta, tracking = parse_csv(artifacts.tracking)
            metadata.update(tracking_meta)
        if artifacts.data is not None:
            data_meta, data = parse_csv(artifacts.data)
            metadata.update(data_meta)
        if artifacts.events is not None:
            events_meta, events = parse_csv(artifacts.events)
            metadata.update(events_meta)
            # Normalize timestamps to pandas datetime for downstream
            # lookups. Each writer uses isoformat(timespec="milliseconds"),
            # which pd.to_datetime parses without explicit format.
            if not events.empty:
                events["timestamp"] = pd.to_datetime(events["timestamp"])

        return cls(
            artifacts=artifacts,
            tracking=tracking,
            data=data,
            events=events,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Artifact paths
    # ------------------------------------------------------------------

    @property
    def directory(self) -> Path:
        return self._artifacts.directory

    @property
    def video_path(self) -> Path | None:
        """Annotated video if available, else raw. Use ``raw_video_path``
        or ``annotated_video_path`` to disambiguate when both exist."""
        return self._artifacts.annotated_video or self._artifacts.video

    @property
    def raw_video_path(self) -> Path | None:
        return self._artifacts.video

    @property
    def annotated_video_path(self) -> Path | None:
        return self._artifacts.annotated_video

    # ------------------------------------------------------------------
    # Flow boundaries (from event log)
    # ------------------------------------------------------------------

    @cached_property
    def flow_start_wall(self) -> float | None:
        """Unix timestamp of flow start, from the ``flow_marker[start]``
        row in the event log. None if the event log is missing or no
        start marker was written (operator stopped before flow began,
        legacy recording from before flow markers existed).
        """
        ts = self._flow_marker_timestamp("start")
        return ts.timestamp() if ts is not None else None

    @cached_property
    def flow_end_wall(self) -> float | None:
        """Unix timestamp of flow end. None for the same reasons as
        flow_start_wall, or if the experiment is still running."""
        ts = self._flow_marker_timestamp("end")
        return ts.timestamp() if ts is not None else None

    @cached_property
    def flow_duration_s(self) -> float | None:
        """Flow duration in seconds, derived from the event-log markers.
        This is the authoritative duration — same value the runner timer
        reports and the recorder footer records."""
        if self.flow_start_wall is None or self.flow_end_wall is None:
            return None
        return self.flow_end_wall - self.flow_start_wall

    def _flow_marker_timestamp(self, marker: str) -> pd.Timestamp | None:
        if self.events.empty:
            return None
        rows = self.events[
            (self.events["source"] == "flow_marker") & (self.events["value"] == marker)
        ]
        if rows.empty:
            return None
        # First matching row wins. A well-formed recording has exactly one
        # of each, but we'd rather return *a* boundary than nothing.
        return rows["timestamp"].iloc[0]

    # ------------------------------------------------------------------
    # Frame rate (probed from tracking)
    # ------------------------------------------------------------------

    @cached_property
    def frame_rate(self) -> float | None:
        """Estimated frame rate (Hz), from the median tracking-frame
        timestamp interval. None if there are fewer than 2 frames.

        v2 will probe the video container directly when present; the
        tracking-derived estimate matches the actual capture rate to
        within a frame or two for any non-degenerate recording.
        """
        if self.tracking.empty or len(self.tracking) < 2:
            return None
        ts = pd.to_datetime(self.tracking["timestamp"])
        # Use unique frame timestamps — multiple objects in the same
        # frame share a timestamp and would bias the median to zero.
        unique_ts = ts.drop_duplicates().sort_values()
        if len(unique_ts) < 2:
            return None
        intervals_s = unique_ts.diff().dt.total_seconds().dropna()
        median_interval = intervals_s.median()
        if median_interval <= 0:
            return None
        return 1.0 / median_interval

    # ------------------------------------------------------------------
    # Event log with flow-relative time
    # ------------------------------------------------------------------

    @cached_property
    def events_with_flow_ms(self) -> pd.DataFrame:
        """Events frame with an added ``flow_elapsed_ms`` column.

        Each event's timestamp is converted to flow-relative
        milliseconds using ``flow_start_wall`` as t=0. Events whose
        timestamp predates ``flow_start_wall`` (e.g., setup-time
        ``output_write`` rows from device initialization) get negative
        values. When no flow boundary is known the column is NaN.

        Cached so callers can take the frame and slice it cheaply
        across multiple analyses.
        """
        if self.events.empty:
            return self.events.assign(flow_elapsed_ms=pd.Series(dtype="float64"))
        out = self.events.copy()
        flow_start_ts = self._flow_marker_timestamp("start")
        if flow_start_ts is None:
            out["flow_elapsed_ms"] = float("nan")
            return out
        # Subtract two Timestamps directly — gives a tz-safe timedelta
        # regardless of system timezone. Converting via .astype("int64")
        # on a datetime64[ns] Series silently drifts in pandas 2.x
        # depending on whether the Series went through pd.to_datetime
        # or stayed as object-of-Timestamp.
        out["flow_elapsed_ms"] = (out["timestamp"] - flow_start_ts).dt.total_seconds() * 1000
        return out

    # ------------------------------------------------------------------
    # Convenience accessors (delegate to submodules)
    # ------------------------------------------------------------------

    def ethogram(self, object_id: int = 0, **kwargs) -> pd.DataFrame:
        """Run-length encode behavioral_state into intervals."""
        from glider.analysis.ethogram import compute_intervals

        return compute_intervals(self.tracking, object_id=object_id, **kwargs)

    def bouts(self, state: str | None = None, object_id: int = 0):
        """Bout durations from this session's ethogram."""
        from glider.analysis.ethogram import compute_bouts

        return compute_bouts(self.ethogram(object_id=object_id), state=state)

    def state_transitions(self, object_id: int = 0) -> pd.DataFrame:
        """Transition counts between consecutive behavioral states."""
        from glider.analysis.ethogram import compute_state_transitions

        return compute_state_transitions(self.ethogram(object_id=object_id))

    def trajectory(self, object_id: int = 0, **kwargs) -> pd.DataFrame:
        """Position trace for one object."""
        from glider.analysis.trajectory import compute_trajectory

        return compute_trajectory(self.tracking, object_id=object_id, **kwargs)

    def occupancy(self, object_id: int = 0, **kwargs):
        """2D occupancy histogram. Returns ``(heatmap, x_edges, y_edges)``."""
        from glider.analysis.trajectory import compute_occupancy

        return compute_occupancy(self.tracking, object_id=object_id, **kwargs)

    def zone_dwell(self, object_id: int = 0) -> pd.DataFrame:
        """Per-zone dwell time + entry count + mean bout duration."""
        from glider.analysis.trajectory import compute_zone_dwell

        return compute_zone_dwell(self.tracking, object_id=object_id)

    def zone_transitions(self, object_id: int = 0) -> pd.DataFrame:
        """Transition counts between consecutive zone states."""
        from glider.analysis.trajectory import compute_zone_transitions

        return compute_zone_transitions(self.tracking, object_id=object_id)

    def velocity(self, object_id: int = 0, use_frame_rate: bool = True, **kwargs) -> pd.DataFrame:
        """Velocity vs flow time. With ``use_frame_rate=True`` (default)
        scales the raw px/frame values to px/s using ``self.frame_rate``."""
        from glider.analysis.kinematics import compute_velocity_series

        frame_rate = self.frame_rate if use_frame_rate else None
        return compute_velocity_series(
            self.tracking, object_id=object_id, frame_rate=frame_rate, **kwargs
        )

    def speed_distribution(
        self, object_id: int = 0, use_frame_rate: bool = True, **kwargs
    ) -> pd.DataFrame:
        """Speed histogram (bin_center, count)."""
        from glider.analysis.kinematics import compute_speed_distribution

        frame_rate = self.frame_rate if use_frame_rate else None
        return compute_speed_distribution(
            self.tracking, object_id=object_id, frame_rate=frame_rate, **kwargs
        )

    def cumulative_distance(self, object_id: int = 0, **kwargs) -> pd.DataFrame:
        """Cumulative distance traveled (mm if calibrated, px otherwise)."""
        from glider.analysis.kinematics import extract_cumulative_distance

        return extract_cumulative_distance(self.tracking, object_id=object_id, **kwargs)

    def find_events(self, source: str | None = None, value: str | None = None) -> pd.DataFrame:
        """Filter ``events_with_flow_ms`` by source and/or value."""
        from glider.analysis.events import find_events

        return find_events(self.events_with_flow_ms, source=source, value=value)

    def event_triggered(
        self,
        *,
        source: str,
        value: str | None = None,
        window_ms: tuple[float, float] = (-1000.0, 5000.0),
        value_col: str = "velocity_px_frame",
        object_id: int = 0,
    ) -> pd.DataFrame:
        """End-to-end event-triggered slice: filter events by
        source/value, then extract the per-trial window from tracking.
        """
        from glider.analysis.events import event_triggered

        matched = self.find_events(source=source, value=value)
        if matched.empty:
            return event_triggered(self.tracking, event_times_ms=[], window_ms=window_ms)
        return event_triggered(
            self.tracking,
            event_times_ms=matched["flow_elapsed_ms"].dropna().tolist(),
            window_ms=window_ms,
            value_col=value_col,
            object_id=object_id,
        )
