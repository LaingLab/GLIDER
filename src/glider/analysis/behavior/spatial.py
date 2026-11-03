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
from a body crossing into it. ``point`` takes either, or a list of keypoints:
a novel-object test scores investigation from the head (nose and ears), not
from the body sitting beside the object. With a list, the position is the
mean of those keypoints and a zone holds the animal on any frame where any
one of them is inside it -- the nose reaching into an object's zone counts,
even when the head's mean point has not yet crossed the line.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CENTROID",
    "SpatialError",
    "describe_point",
    "load_zones",
    "occupancy_grid",
    "position_track",
    "tracking_frame",
    "write_occupancy_export",
    "zone_occupancy",
]

#: The default body point: the mean of the tracked keypoints. A dropped nose
#: nudges it rather than teleporting it, which a single keypoint cannot claim.
CENTROID = "centroid"

#: A body point: :data:`CENTROID`, one keypoint name, or several.
Point = str | Sequence[str]


class SpatialError(ValueError):
    """A spatial analysis could not be set up."""


def load_zones(path: Path | str):
    """A :class:`ZoneConfiguration` from the file the zone editor writes."""
    from glider.vision.zones import ZoneConfiguration

    config = ZoneConfiguration()
    if not config.load(Path(path)):
        raise SpatialError(f"{Path(path).name} is not a readable zone configuration")
    return config


def describe_point(point: Point) -> str:
    """*point* as a reader would name it: ``centroid``, ``nose``, ``nose + left_ear``."""
    if isinstance(point, str):
        return point
    return " + ".join(point) if point else CENTROID


def _keypoint_indices(view, point: Point) -> list[int] | None:
    """The keypoint columns *point* names, or None for the centroid."""
    names = [point] if isinstance(point, str) else list(point)
    if not names or names == [CENTROID]:
        return None
    missing = [n for n in names if n not in view.keypoint_names]
    if missing:
        raise SpatialError(
            f"{', '.join(map(repr, missing))} "
            f"{'is not a keypoint' if len(missing) == 1 else 'are not keypoints'} "
            f"of this session; known: {', '.join(view.keypoint_names)}"
        )
    return [view.keypoint_names.index(n) for n in dict.fromkeys(names)]


def _keypoint_stack(view, point: Point) -> np.ndarray | None:
    """``(n_frames, k, 2)`` for a keypoint list, or None for the centroid."""
    if view.xy is None:
        raise SpatialError("this session has no poses, so it has no position track")
    indices = _keypoint_indices(view, point)
    return None if indices is None else view.xy[:, indices, :]


def position_track(view, point: Point = CENTROID) -> np.ndarray:
    """``(n_frames, 2)`` pixel positions of *point*, or raise.

    ``point`` is :data:`CENTROID`, the name of a keypoint, or a list of them,
    whose mean is the position -- the head from the nose and both ears.
    """
    stack = _keypoint_stack(view, point)
    if stack is None:
        track = view.centroid()
        if track is None:
            raise SpatialError("no centroid could be computed")
        return track
    if stack.shape[1] == 1:
        return stack[:, 0, :]
    import warnings

    with warnings.catch_warnings():
        # A frame with every chosen keypoint dropped is NaN, as it should be.
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        return np.nanmean(stack, axis=1)


def tracking_frame(view, zones=None, *, point: Point = CENTROID) -> pd.DataFrame:
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

    With several keypoints, ``center_x``/``center_y`` is their mean and
    ``zone_ids`` names every zone any one of them is inside.
    """
    track = position_track(view, point)
    stack = _keypoint_stack(view, point)
    frames = np.asarray(view.frames, dtype=int)
    labels = view.labels

    # Look the position up BY frame number. The track has one row per video
    # frame; the ethogram has one row per scored frame, which is neither the
    # same count nor the same origin — a run over minutes 2–7 starts at frame
    # 3600, and a cadence of 3 emits every third frame. Zipping the two by
    # position pairs each label with the wrong place entirely: the whole
    # session slides by its own start frame, so a heatmap of "minutes 2–7"
    # was really minutes 0–5, and every zone number was drawn from it.
    row = np.arange(len(frames))
    keep = (frames >= 0) & (frames < len(track))
    frames, row = frames[keep], row[keep]
    track = track[frames]
    members = track if stack is None else stack[frames]
    n = len(frames)

    data = {
        "object_id": 0,
        "frame": frames,
        "flow_elapsed_ms": frames / view.fps * 1000.0 if view.fps else np.zeros(n),
        "center_x": track[:, 0],
        "center_y": track[:, 1],
        "behavioral_state": [labels[i] if i < len(labels) else "" for i in row],
    }
    data["zone_ids"] = _zone_ids(members, zones, view.resolution)
    return pd.DataFrame(data)


def _zone_ids(track: np.ndarray, zones, resolution) -> list[str]:
    """Comma-separated zone names per frame, matching the live logger.

    The same joined-string convention on purpose: ``compute_zone_dwell``
    treats each distinct string as its own state, so an animal in two
    overlapping zones reads as one combined row rather than being
    double-counted, and a video-derived session must not disagree with a
    live-recorded one about that.

    ``track`` is ``(n, 2)``, or ``(n, k, 2)`` for several keypoints, where a
    frame is in a zone when any of its k points is; a frame is dropped only
    when all k are.
    """
    if zones is None or not getattr(zones, "zones", None):
        return [""] * len(track)
    if not resolution or not all(resolution):
        logger.info("no arena resolution, so zone membership cannot be resolved")
        return [""] * len(track)

    width, height = resolution
    points = track if track.ndim == 3 else track[:, None, :]
    n, k = points.shape[:2]
    xs = points[..., 0].reshape(-1) / width
    ys = points[..., 1].reshape(-1) / height
    finite_point = np.isfinite(xs) & np.isfinite(ys)
    finite = finite_point.reshape(n, k).any(axis=1)

    # One vectorised membership test per zone rather than one scalar call per
    # frame. A cohort is thirty sessions of ~45,000 frames, so the scalar form
    # is over a million Python-level point-in-shape calls every time a window
    # changes — which is what made the table feel like a hang.
    # ``_zone_mask`` is checked against Zone.contains_point in the tests, so
    # the two cannot drift apart.
    aspect = width / height
    per_zone = [
        (zone.name, (_zone_mask(zone, xs, ys, aspect) & finite_point).reshape(n, k).any(axis=1))
        for zone in zones.zones
    ]

    out: list[str] = []
    for i in range(n):
        if not finite[i]:
            out.append("")  # a dropped frame is not "outside every zone"
            continue
        out.append(",".join(name for name, mask in per_zone if mask[i]))
    return out


def _zone_mask(zone, xs: np.ndarray, ys: np.ndarray, aspect: float = 1.0) -> np.ndarray:
    """Boolean membership for every point at once, mirroring Zone geometry.

    ``aspect`` is frame width / height, for circles measured in pixels.
    """
    from glider.vision.zones import ZoneShape

    verts = zone.vertices
    if not verts:
        return np.zeros(xs.shape, dtype=bool)

    if zone.shape == ZoneShape.RECTANGLE:
        if len(verts) < 2:
            return np.zeros(xs.shape, dtype=bool)
        (x1, y1), (x2, y2) = verts[0], verts[1]
        return (xs >= min(x1, x2)) & (xs <= max(x1, x2)) & (ys >= min(y1, y2)) & (ys <= max(y1, y2))

    if zone.shape == ZoneShape.CIRCLE:
        if len(verts) < 2:
            return np.zeros(xs.shape, dtype=bool)
        # A true pixel circle, as the overlay draws it: normalized x is scaled
        # by width/height so a 16:9 frame does not test an ellipse.
        from glider.vision.zones import circle_contains

        (cx, cy), (rx, ry) = verts[0], verts[1]
        return circle_contains(cx, cy, rx, ry, xs, ys, aspect)

    if zone.shape == ZoneShape.POLYGON and len(verts) >= 3:
        # cv2.pointPolygonTest over the same integer-scaled vertices the
        # scalar path uses, so boundary cases land identically.
        import cv2

        scale = 10000
        pts = np.array([(int(vx * scale), int(vy * scale)) for vx, vy in verts], dtype=np.int32)
        mask = np.zeros(xs.shape, dtype=bool)
        for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            mask[i] = cv2.pointPolygonTest(pts, (int(x * scale), int(y * scale)), False) >= 0
        return mask

    return np.zeros(xs.shape, dtype=bool)


def zone_occupancy(
    view,
    zones,
    *,
    point: Point = CENTROID,
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
    # Real time spanned, not the row count: rows arrive every `stride` frames.
    frames = frame["frame"].to_numpy()
    stride = max(1, int(np.median(np.diff(frames)))) if len(frames) > 1 else 1
    window_s = (int(frames[-1]) - int(frames[0]) + stride) / view.fps if view.fps else 0.0
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
    point: Point = CENTROID,
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


def write_occupancy_export(
    grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    base_path: Path | str,
    *,
    title: str | None = None,
) -> tuple[Path | None, Path]:
    """Write ``<base>.png`` and ``<base>.csv``. Returns ``(png_or_None, csv)``.

    Only a trailing ``.png``/``.csv`` is stripped from ``base_path``, and the
    two extensions are then appended rather than substituted: session
    directories are named from the video stem, so dotted names are ordinary,
    and ``with_suffix`` would replace everything after the last dot -- eating
    the frame range out of the caller's own filename.

    ``grid`` is ``(nx, ny)`` as ``compute_occupancy`` returns it. The CSV is
    written the other way round -- rows are y, columns are x -- because that is
    how a reader expects a table of the arena to look, so the data block is
    ``grid.T`` and cell ``(row j, column i)`` holds ``grid[i, j]``. The GUI bins
    a square grid, so a transposed file would raise nothing and be visible only
    by eye against the figure; the orientation is pinned by test.

    ``png`` is None when matplotlib is unavailable -- the CSV is the half worth
    keeping, and losing it because an optional renderer is missing would be the
    wrong trade.
    """
    p = Path(base_path)
    base = p.with_suffix("") if p.suffix.lower() in {".png", ".csv"} else p
    if (
        grid.size == 0
        or x_edges.size == 0
        or y_edges.size == 0
        or not np.isfinite(grid).any()
        or np.nansum(grid) == 0
    ):
        raise ValueError(
            "nothing to export: the occupancy grid is empty. An empty or "
            "all-NaN track yields a full-size grid with no bin edges."
        )
    csv_path = base.with_name(base.name + ".csv")

    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0
    table = pd.DataFrame(grid.T.astype(int), index=y_centres, columns=x_centres)
    table.to_csv(csv_path)

    png_path: Path | None = base.with_name(base.name + ".png")
    try:
        # Our own figure, never pyplot's: this runs inside a live Qt app, and a
        # pyplot figure per export would both leak and risk pulling in the Qt
        # backend. The colorbar fix in plots.py is what makes this hold.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from glider.analysis.plots import plot_occupancy_heatmap

        figure = Figure(figsize=(6.0, 5.0), layout="tight")
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        extra = {} if title is None else {"title": title}
        plot_occupancy_heatmap(grid, x_edges, y_edges, ax=axes, **extra)
        figure.savefig(png_path, dpi=150)
        figure.clf()
    except ImportError:
        logger.info("matplotlib unavailable; wrote %s without a figure", csv_path.name)
        png_path = None
    return png_path, csv_path
