"""
GLIDER post-analysis library.

Loads a recording directory and provides ergonomic accessors for the
data products an ethology paper typically needs: ethograms, trajectory,
zone occupancy, kinematics, and event-triggered slicing. Designed for
notebook use (``from glider.analysis import Session``) and for embedding
in the in-app Analysis panel.

Quick start:

    >>> from glider.analysis import Session
    >>> s = Session.load("/path/to/recording_dir")
    >>> s.flow_duration_s
    142.04
    >>> intervals = s.ethogram()
    >>> from glider.analysis import plot_ethogram
    >>> plot_ethogram(intervals)

Phase 1 shipped ``Session`` + ethogram. Phase 2 adds trajectory,
occupancy, zone analysis, kinematics, and event-triggered analysis,
plus matching plot helpers. Phase 3+ will layer on the in-app panel
and synchronized video playback.
"""

from glider.analysis.ethogram import (
    compute_bouts,
    compute_intervals,
    compute_state_transitions,
)
from glider.analysis.events import event_triggered, find_events
from glider.analysis.kinematics import (
    compute_speed_distribution,
    compute_velocity_series,
    extract_cumulative_distance,
)
from glider.analysis.plots import (
    plot_bouts,
    plot_ethogram,
    plot_event_triggered,
    plot_occupancy_heatmap,
    plot_trajectory,
    plot_velocity,
    plot_zone_dwell,
)
from glider.analysis.session import Session
from glider.analysis.trajectory import (
    compute_occupancy,
    compute_trajectory,
    compute_zone_dwell,
    compute_zone_transitions,
)

__all__ = [
    "Session",
    # ethogram
    "compute_bouts",
    "compute_intervals",
    "compute_state_transitions",
    # trajectory + zones
    "compute_occupancy",
    "compute_trajectory",
    "compute_zone_dwell",
    "compute_zone_transitions",
    # kinematics
    "compute_speed_distribution",
    "compute_velocity_series",
    "extract_cumulative_distance",
    # events
    "event_triggered",
    "find_events",
    # plots
    "plot_bouts",
    "plot_ethogram",
    "plot_event_triggered",
    "plot_occupancy_heatmap",
    "plot_trajectory",
    "plot_velocity",
    "plot_zone_dwell",
]
