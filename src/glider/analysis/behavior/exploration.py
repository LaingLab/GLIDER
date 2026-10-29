"""How much of the arena the animal covered, on a grid laid over the drawn floor.

Exploration is the share of an N x N grid of floor cells the animal entered.
The grid is laid out in centimetres on the floor and mapped into the image
through the arena's homography, so under a tilted camera the far cells are
drawn smaller -- they are the same patch of floor as the near ones, which is
what makes "7 of 9 cells" mean the same thing in every video. A grid drawn
straight over the pixels would give the near wall more floor per cell than the
far wall, and a coverage figure that depends on where the camera sat.

A round arena keeps only the cells whose centre is on the floor, so a corner
of the bounding box that is empty air never counts against the animal; a plus
maze likewise keeps the cells on its arms.

Qt-free, so the epoch table, an export and a notebook all get the same number.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from glider.analysis.behavior.spatial import CENTROID, tracking_frame

logger = logging.getLogger(__name__)

__all__ = [
    "ARENA_SUFFIX",
    "Exploration",
    "arena_file_for",
    "cell_polygon",
    "explore",
    "find_session_arena",
    "floor_cells",
    "grid_segments",
    "read_arena",
]

#: Saved-arena filename suffix, the spelling the project layout already reads
#: (``ProjectSession.arena``).
ARENA_SUFFIX = "_arena.json"

_MASTER = "pose_calibration.json"


# ----------------------------------------------------------------------
# finding the arena
# ----------------------------------------------------------------------


def read_arena(path: Path | str):
    """The :class:`ArenaCalibration` in an arena file, or None.

    Accepts both the bare calibration and the document ``tools/arena_zones.py``
    writes, which wraps it under ``arena`` beside its derived numbers.
    """
    from glider.vision.arena import ArenaCalibration

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for candidate in (data.get("arena"), data):
        if not isinstance(candidate, dict) or "corners" not in candidate:
            continue
        try:
            return ArenaCalibration.from_dict(candidate)
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _session_folder(session_path: Path) -> Path:
    return session_path if session_path.is_dir() else session_path.parent


def arena_file_for(session_path: Path | str) -> Path:
    """Where an arena drawn in Session Review is saved for this session."""
    folder = _session_folder(Path(session_path))
    return folder / f"{folder.name}{ARENA_SUFFIX}"


def find_session_arena(session_path: Path | str, view=None):
    """``(arena, source)`` for a session, or None when none was drawn.

    Looked for first as a saved arena beside the session (named after its
    folder or its video), then in the master ``pose_calibration.json`` an apply
    run read, matched by the session's video.
    """
    from glider.analysis.behavior.session_view import _find_upward

    folder = _session_folder(Path(session_path))
    video = getattr(view, "video_path", None)
    stems = [folder.name] + ([Path(video).stem] if video is not None else [])
    beside = [folder / f"{stem}{ARENA_SUFFIX}" for stem in dict.fromkeys(stems)]
    if video is not None:
        beside.append(Path(video).with_name(f"{Path(video).stem}{ARENA_SUFFIX}"))
    for candidate in beside:
        if candidate.is_file():
            arena = read_arena(candidate)
            if arena is not None:
                return arena, candidate
            logger.info("ignoring unreadable arena %s", candidate)

    master = _find_upward(folder, _MASTER)
    if master is None:
        return None
    from glider.vision.calibration_set import CalibrationSet, CalibrationSetError

    try:
        loaded = CalibrationSet.load(master, known_videos=[Path(video)] if video else None)
    except (CalibrationSetError, OSError) as e:
        logger.info("no usable calibration in %s: %s", master, e)
        return None
    if video is not None:
        arena = loaded.get_arena(video)
        if arena is not None:
            return arena, master
    # The video was found somewhere other than where the batch saw it; the
    # session folder is named after it, so match by stem -- but only a unique
    # match, never a guess between two.
    matches = [a for v, a in loaded.arenas.items() if v.stem in stems]
    return (matches[0], master) if len(matches) == 1 else None


# ----------------------------------------------------------------------
# grid geometry
# ----------------------------------------------------------------------


def _to_cm(arena, normalised: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Floor centimetres for ``(n, 2)`` normalised image points; NaN where unmappable."""
    pts = np.asarray(normalised, dtype=np.float64).reshape(-1, 2)
    h = arena.homography()
    w = pts[:, 0] * h[2, 0] + pts[:, 1] * h[2, 1] + h[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (pts[:, 0] * h[0, 0] + pts[:, 1] * h[0, 1] + h[0, 2]) / w
        y = (pts[:, 0] * h[1, 0] + pts[:, 1] * h[1, 1] + h[1, 2]) / w
    # Past the horizon a point maps to the floor's far side, flipped; it is
    # not on the floor at all.
    bad = ~(w > 0)
    x[bad] = np.nan
    y[bad] = np.nan
    return x, y


def _cell_size(arena, n: int) -> tuple[float, float]:
    return arena.width_cm / n, arena.height_cm / n


def floor_cells(arena, n: int) -> np.ndarray:
    """``(n, n)`` bool, row = y, column = x: the cells that count as floor."""
    cw, ch = _cell_size(arena, n)
    xs = (np.arange(n) + 0.5) * cw
    ys = (np.arange(n) + 0.5) * ch
    gx, gy = np.meshgrid(xs, ys)
    return ~np.asarray(arena.outside_cm(gx, gy), dtype=bool)


def grid_segments(arena, n: int, samples: int = 48) -> list[list[tuple[float, float]]]:
    """The grid's interior lines as normalised image polylines, clipped to the floor.

    Each line is sampled along its length and broken where it leaves the
    floor, so the same code clips a round arena's chords and a plus maze's
    arms; a homography keeps lines straight, so the samples cost nothing in
    accuracy.
    """
    if n < 2:
        return []
    w, h = arena.width_cm, arena.height_cm
    t = np.linspace(0.0, 1.0, samples)
    lines_cm = [(np.full_like(t, w * k / n), t * h) for k in range(1, n)]
    lines_cm += [(t * w, np.full_like(t, h * k / n)) for k in range(1, n)]
    out: list[list[tuple[float, float]]] = []
    for xs, ys in lines_cm:
        # A whisker of margin, so a line along an arm's edge is not dropped
        # for sitting exactly on the boundary.
        on = ~np.asarray(arena.outside_cm(xs, ys, 1e-6), dtype=bool)
        run: list[tuple[float, float]] = []
        for i in range(len(t)):
            if on[i]:
                run.append((float(xs[i]), float(ys[i])))
            elif run:
                out.append(run)
                run = []
        if run:
            out.append(run)
    return [arena.to_image(run) for run in out if len(run) > 1]


def cell_polygon(arena, n: int, col: int, row: int) -> list[tuple[float, float]]:
    """One cell's four corners, as normalised image points."""
    cw, ch = _cell_size(arena, n)
    x0, y0 = col * cw, row * ch
    return arena.to_image([(x0, y0), (x0 + cw, y0), (x0 + cw, y0 + ch), (x0, y0 + ch)])


# ----------------------------------------------------------------------
# the statistic
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Exploration:
    """Grid coverage over one window.

    ``seconds`` is ``(n, n)``, row = y and column = x, the time in each cell,
    NaN for a cell that is not floor. ``full_coverage_s`` is how long into the
    window the last unvisited cell was first entered, or None if some cell
    never was.
    """

    n: int
    floor: np.ndarray
    seconds: np.ndarray
    full_coverage_s: float | None

    @property
    def cells(self) -> int:
        return int(self.floor.sum())

    @property
    def visited(self) -> np.ndarray:
        return self.floor & (np.nan_to_num(self.seconds) > 0)

    @property
    def cells_visited(self) -> int:
        return int(self.visited.sum())

    @property
    def fraction(self) -> float:
        return self.cells_visited / self.cells if self.cells else 0.0


def explore(
    view,
    arena,
    n: int,
    *,
    point: str = CENTROID,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> Exploration:
    """How many of the arena's ``n x n`` cells the animal entered in a window.

    A cell counts once the body point has been in it for a single scored
    frame. Raises :class:`~glider.analysis.behavior.spatial.SpatialError`
    without poses, and ``ValueError`` without the frame size, since pixels
    cannot be placed on the floor without it.
    """
    if n < 1:
        raise ValueError(f"A grid needs at least one cell a side, got {n}")
    if not view.resolution or not all(view.resolution):
        raise ValueError("this session's frame size is unknown, so the arena cannot be placed")
    frame = tracking_frame(view, None, point=point)
    if start_frame is not None or end_frame is not None:
        low = -np.inf if start_frame is None else start_frame
        high = np.inf if end_frame is None else end_frame
        frame = frame[(frame["frame"] >= low) & (frame["frame"] <= high)].reset_index(drop=True)

    floor = floor_cells(arena, n)
    seconds = np.where(floor, 0.0, np.nan)
    if frame.empty:
        return Exploration(n, floor, seconds, None)

    frames = frame["frame"].to_numpy()
    # Each row stands for `stride` frames: rows arrive every `stride` frames.
    stride = max(1, int(np.median(np.diff(frames)))) if len(frames) > 1 else 1
    row_s = stride / view.fps if view.fps else 0.0

    width, height = view.resolution
    norm = np.column_stack(
        [frame["center_x"].to_numpy() / width, frame["center_y"].to_numpy() / height]
    )
    x, y = _to_cm(arena, norm)
    finite = np.isfinite(x) & np.isfinite(y)
    on = np.zeros(len(x), dtype=bool)
    on[finite] = ~np.asarray(arena.outside_cm(x[finite], y[finite]), dtype=bool)
    cw, ch = _cell_size(arena, n)
    col = np.clip(np.floor(np.where(on, x, 0) / cw).astype(int), 0, n - 1)
    row = np.clip(np.floor(np.where(on, y, 0) / ch).astype(int), 0, n - 1)
    on &= floor[row, col]

    np.add.at(seconds, (row[on], col[on]), row_s)

    full = None
    hits = np.flatnonzero(on)
    cells, first = np.unique(row[hits] * n + col[hits], return_index=True)
    if floor.any() and len(cells) == int(floor.sum()) and view.fps:
        last = hits[first.max()]
        full = (int(frames[last]) - int(frames[0])) / view.fps
    return Exploration(n, floor, seconds, full)
