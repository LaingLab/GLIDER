"""Smoke tests for the plot helpers. Skipped if matplotlib isn't installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glider.analysis import (
    Session,
    compute_intervals,
    plot_bouts,
    plot_ethogram,
    plot_event_triggered,
    plot_occupancy_heatmap,
    plot_trajectory,
    plot_velocity,
    plot_zone_dwell,
)

from .conftest import RecordingSpec, write_synthetic_recording

# Matplotlib is a soft dep; skip these tests cleanly when absent rather
# than red-failing on systems that only use the data side of the lib.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend for CI


def test_plot_ethogram_returns_axes_with_data(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    intervals = s.ethogram()

    ax = plot_ethogram(intervals)

    # One bar per interval; expect 3 from the default state schedule.
    assert len(ax.patches) >= 3, "Expected one bar per interval"
    assert ax.get_title() == "Ethogram"


def test_plot_ethogram_handles_empty_intervals():
    empty = pd.DataFrame(columns=["object_id", "state", "start_ms", "end_ms", "duration_ms"])
    ax = plot_ethogram(empty)
    assert "no data" in ax.get_title().lower()


def test_plot_ethogram_renders_into_provided_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    df = pd.DataFrame(
        {
            "object_id": [0, 0],
            "behavioral_state": ["a", "b"],
            "flow_elapsed_ms": [0.0, 100.0],
        }
    )
    returned_ax = plot_ethogram(compute_intervals(df), ax=ax)
    assert returned_ax is ax
    plt.close(fig)


def test_plot_trajectory_smoke(synthetic_recording: Path):
    import matplotlib.pyplot as plt

    s = Session.load(synthetic_recording)
    ax = plot_trajectory(s.trajectory())
    assert ax.get_title() == "Trajectory"
    # At least one PathCollection (scatter) added.
    assert len(ax.collections) >= 1
    plt.close(ax.figure)


def test_plot_occupancy_heatmap_smoke(synthetic_recording: Path):
    import matplotlib.pyplot as plt

    s = Session.load(synthetic_recording)
    heatmap, xe, ye = s.occupancy(bins=10)
    ax = plot_occupancy_heatmap(heatmap, xe, ye)
    assert "Occupancy" in ax.get_title()
    plt.close(ax.figure)


def test_plot_zone_dwell_smoke(synthetic_recording: Path):
    import matplotlib.pyplot as plt

    s = Session.load(synthetic_recording)
    ax = plot_zone_dwell(s.zone_dwell())
    assert "Zone" in ax.get_title()
    plt.close(ax.figure)


def test_plot_velocity_smoke(synthetic_recording: Path):
    import matplotlib.pyplot as plt

    s = Session.load(synthetic_recording)
    ax = plot_velocity(s.velocity())
    assert "Velocity" in ax.get_title()
    plt.close(ax.figure)


def test_plot_bouts_smoke(synthetic_recording: Path):
    import matplotlib.pyplot as plt

    s = Session.load(synthetic_recording)
    ax = plot_bouts(s.bouts(state="resting"))
    assert "Bout" in ax.get_title()
    plt.close(ax.figure)


def test_plot_event_triggered_smoke(tmp_path: Path):
    """Generate a recording with an LED event, slice around it, plot."""
    import matplotlib.pyplot as plt

    spec = RecordingSpec(extra_events=((1500.0, "output_write", "board0", "5", "1"),))
    write_synthetic_recording(tmp_path / "rec", spec)
    s = Session.load(tmp_path / "rec")
    eta = s.event_triggered(source="output_write", window_ms=(-500.0, 1000.0))
    ax = plot_event_triggered(eta)
    assert "Event" in ax.get_title()
    plt.close(ax.figure)


def test_plot_empty_inputs_handled_gracefully():
    """Each plot returns an Axes labeled '(no data)' for empty inputs
    rather than raising."""
    import matplotlib.pyplot as plt
    import numpy as np

    empty_intervals = pd.DataFrame(
        columns=["object_id", "state", "start_ms", "end_ms", "duration_ms"]
    )
    empty_traj = pd.DataFrame(columns=["frame", "flow_elapsed_ms", "center_x", "center_y"])
    empty_dwell = pd.DataFrame(columns=["zone", "total_ms", "n_entries", "mean_bout_ms"])
    empty_vel = pd.DataFrame(columns=["flow_elapsed_ms", "velocity"])
    empty_eta = pd.DataFrame(columns=["trial_id", "event_time_ms", "time_offset_ms", "value"])

    for ax in [
        plot_ethogram(empty_intervals),
        plot_trajectory(empty_traj),
        plot_occupancy_heatmap(np.zeros((10, 10)), np.linspace(0, 1, 11), np.linspace(0, 1, 11)),
        plot_zone_dwell(empty_dwell),
        plot_velocity(empty_vel),
        plot_bouts(pd.Series([], dtype="float64")),
        plot_event_triggered(empty_eta),
    ]:
        assert "no data" in ax.get_title().lower()
        plt.close(ax.figure)


def test_plotting_onto_a_supplied_axes_leaves_no_pyplot_figure():
    """The colorbar used to go through pyplot's gcf(), which conjures a figure
    even when the caller brought its own - leaking one per export and warning
    about it."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from glider.analysis.plots import plot_occupancy_heatmap

    for num in plt.get_fignums():
        plt.close(num)

    fig = Figure(figsize=(4, 3))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    plot_occupancy_heatmap(np.arange(16.0).reshape(4, 4), np.arange(5.0), np.arange(5.0), ax=ax)

    assert plt.get_fignums() == []
    # the colorbar still lands on the caller's figure: image axes + colorbar axes
    assert len(fig.axes) == 2


def test_trajectory_onto_a_supplied_axes_leaves_no_pyplot_figure():
    """Same defect as the occupancy heatmap - the trajectory colorbar went
    through gcf() too, leaking a figure on every analysis-panel render."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from glider.analysis.plots import plot_trajectory

    for num in plt.get_fignums():
        plt.close(num)

    fig = Figure(figsize=(4, 3))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    # Non-empty, and carrying flow_elapsed_ms so this takes the default
    # color_by="time" branch the panel uses. An empty frame returns early,
    # before the colorbar is ever drawn.
    trajectory = pd.DataFrame(
        {
            "center_x": [0.0, 1.0, 2.0, 3.0],
            "center_y": [0.0, 1.0, 0.0, 1.0],
            "flow_elapsed_ms": [0.0, 100.0, 200.0, 300.0],
        }
    )
    plot_trajectory(trajectory, ax=ax)

    assert plt.get_fignums() == []
    # scatter axes + colorbar axes, both on the caller's figure
    assert len(fig.axes) == 2
