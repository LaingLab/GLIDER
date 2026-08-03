"""Where the animal was, for sessions that came from a video.

GLIDER has had the spatial suite since the live rig: zones, occupancy
histograms, dwell times, transitions, heatmaps. All of it reads
``Session.tracking`` — the per-frame CSV a live recording writes, with
``center_x``/``center_y`` in pixels and a ``zone_ids`` column the tracker
filled in as it went.

An apply run over a recorded video produces none of that. It produces poses
and an ethogram, so none of the spatial analysis could be reached from the
behaviour pipeline at all: a cohort scored from video had no time-in-zone, no
entries, no heatmap. Not because the analysis was missing, but because it was
wired to the other input format.

This is that wire. A :class:`~glider.analysis.behavior.session_view.SessionView`
already carries everything needed — the poses, the frame rate, the arena
resolution, the pixel scale — so it can be re-expressed in the shape the
existing functions expect, and every one of them then works unchanged. There
is no second implementation of dwell or occupancy here, deliberately: two
implementations of "time in zone" is how a cohort ends up with two answers.

Which body point counts as the animal's position is a real choice, not a
detail — commercial trackers charge for the distinction between a centre-point
and a nose-point, because a nose crossing into a zone is a different event
from a body crossing into it. ``point`` takes either.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CENTROID",
    "SpatialError",
    "load_zones",
    "occupancy_grid",
    "position_track",
    "tracking_frame",
    "zone_occupancy",
]

#: The default body point: the mean of the tracked keypoints. A dropped nose
#: nudges it rather than teleporting it, which a single keypoint cannot claim.
CENTROID = "centroid"


class SpatialError(ValueError):
    """A spatial analysis could not be set up."""


def load_zones(path: Path | str):
    """A :class:`ZoneConfiguration` from the file the zone editor writes."""
    from glider.vision.zones import ZoneConfiguration

    config = ZoneConfiguration()
    if not config.load(Path(path)):
        raise SpatialError(f"{Path(path).name} is not a readable zone configuration")
    return config


def position_track(view, point: str = CENTROID) -> np.ndarray:
    """``(n_frames, 2)`` pixel positions of *point*, or raise.

    ``point`` is either :data:`CENTROID` or the name of a keypoint.
    """
    if view.xy is None:
        raise SpatialError("this session has no poses, so it has no position track")
    if point == CENTROID:
        track = view.centroid()
        if track is None:
            raise SpatialError("no centroid could be computed")
        return track
    if point not in view.keypoint_names:
        raise SpatialError(
            f"{point!r} is not a keypoint of this session; known: "
            f"{', '.join(view.keypoint_names)}"
        )
    return view.xy[:, view.keypoint_names.index(point), :]


def tracking_frame(view, zones=None, *, point: str = CENTROID) -> pd.DataFrame:
    """A ``Session.tracking``-shaped frame for a video-derived session.

    Columns match what the live tracker logs, because that is what every
    spatial function in :mod:`glider.analysis.trajectory` reads:
    ``object_id, frame, flow_elapsed_ms, center_x, center_y, zone_ids``,
    plus ``behavioral_state`` so behaviour and position can be asked about
    together.

    Positions are pixels in the source video, the same units the live tracker
    logs and the same the occupancy histogram bins in. Zone membership needs
    the arena size to normalise against, so a session whose resolution was
    never recorded gets empty ``zone_ids`` rather than a guess — the sidecar
    repair in the session review window exists for exactly that.
    """
    track = position_track(view, point)
    frames = np.asarray(view.frames)
    n = min(len(frames), len(track))
    frames, track = frames[:n], track[:n]

    data = {
        "object_id": 0,
        "frame": frames,
        "flow_elapsed_ms": frames / view.fps * 1000.0 if view.fps else np.zeros(n),
        "center_x": track[:, 0],
        "center_y": track[:, 1],
        "behavioral_state": [view.labels[i] if i < len(view.labels) else "" for i in range(n)],
    }
    data["zone_ids"] = _zone_ids(track, zones, view.resolution)
    return pd.DataFrame(data)


def _zone_ids(track: np.ndarray, zones, resolution) -> list[str]:
    """Comma-separated zone names per frame, matching the live logger.

    The same joined-string convention on purpose: ``compute_zone_dwell``
    treats each distinct string as its own state, so an animal in two
    overlapping zones reads as one combined row rather than being
    double-counted, and a video-derived session must not disagree with a
    live-recorded one about that.
    """
    if zones is None or not getattr(zones, "zones", None):
        return [""] * len(track)
    if not resolution or not all(resolution):
        logger.info("no arena resolution, so zone membership cannot be resolved")
        return [""] * len(track)

    width, height = resolution
    out: list[str] = []
    for x, y in track:
        if not (np.isfinite(x) and np.isfinite(y)):
            out.append("")  # a dropped frame is not "outside every zone"
            continue
        out.append(",".join(zones.get_zone_names_for_point(x / width, y / height)))
    return out


def zone_occupancy(
    view,
    zones,
    *,
    point: str = CENTROID,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> pd.DataFrame:
    """Time, entries and mean bout per zone, over an optional frame window.

    Adds ``fraction`` and a latency: the seconds from the start of the window
    to the first frame in that zone, which is a standard readout and is
    ``NaN`` when the zone was never entered — distinct from a latency of zero.
    """
    from glider.analysis.trajectory import compute_zone_dwell

    frame = tracking_frame(view, zones, point=point)
    if start_frame is not None or end_frame is not None:
        low = -np.inf if start_frame is None else start_frame
        high = np.inf if end_frame is None else end_frame
        frame = frame[(frame["frame"] >= low) & (frame["frame"] <= high)].reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(
            columns=["zone", "total_s", "fraction", "n_entries", "mean_bout_s", "latency_s"]
        )

    dwell = compute_zone_dwell(frame)
    window_s = len(frame) / view.fps if view.fps else 0.0
    first_ms = float(frame["flow_elapsed_ms"].iloc[0])

    rows = []
    for _, row in dwell.iterrows():
        zone = str(row["zone"])
        entered = frame.loc[frame["zone_ids"] == zone, "flow_elapsed_ms"]
        total_s = float(row["total_ms"]) / 1000.0
        rows.append(
            {
                "zone": zone or "(outside)",
                "total_s": total_s,
                "fraction": total_s / window_s if window_s else 0.0,
                "n_entries": int(row["n_entries"]),
                "mean_bout_s": float(row["mean_bout_ms"]) / 1000.0,
                "latency_s": (
                    (float(entered.iloc[0]) - first_ms) / 1000.0 if not entered.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("total_s", ascending=False, ignore_index=True)


def occupancy_grid(
    view,
    *,
    point: str = CENTROID,
    bins: int | tuple[int, int] = 50,
    start_frame: int | None = None,
    end_frame: int | None = None,
):
    """``(heatmap, x_edges, y_edges)`` over the arena, in pixels.

    Anchored to the recorded resolution rather than to the coordinates
    observed, so two animals' heatmaps are drawn on the same arena and can be
    compared; letting the extent follow the track would rescale the enclosure
    to wherever each animal happened to go.
    """
    from glider.analysis.trajectory import compute_occupancy

    frame = tracking_frame(view, None, point=point)
    if start_frame is not None or end_frame is not None:
        low = -np.inf if start_frame is None else start_frame
        high = np.inf if end_frame is None else end_frame
        frame = frame[(frame["frame"] >= low) & (frame["frame"] <= high)].reset_index(drop=True)
    return compute_occupancy(frame, bins=bins, frame_size=view.resolution)
