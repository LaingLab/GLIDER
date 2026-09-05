"""DeepLabCut format writer.

Produces the canonical 3-row-header CSV that B-SOID, VAME, Keypoint-MoSeq,
SimBA, and DLC's own analysis tools expect:

    scorer,my_yolo,my_yolo,my_yolo,...
    bodyparts,snout,snout,snout,left_ear,...
    coords,x,y,likelihood,x,y,likelihood,...
    0,412.3,288.1,0.97,...

The companion ``to_dlc_h5`` writer emits the same data as a pandas-flavor
HDF5 file, which is what DLC saves natively (``*_DLC_resnet50_*.h5``) — some
downstream tools (notably DLC's own ``analyze_videos`` post-processors) prefer
this form.

The DLC header has exactly three rows and no room for a frame rate, but every
downstream feature is windowed in *seconds*, so losing the rate silently
rescales the science: a 60 fps recording read back at the old 30.0 default
computes every rolling window over half the intended span. :func:`to_dlc_csv`
therefore drops a small JSON sidecar (``<stem>.meta.json``) next to the CSV
carrying the rate inference measured, and :func:`from_dlc_csv` reads it back
when the caller does not name a rate explicitly. Tools that only understand
DLC ignore the extra file.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glider.vision.pose.core import PoseData

#: Fallback when a CSV predates the sidecar and the caller named no rate.
#: Matches the historical ``from_dlc_csv`` default, so old data reads exactly
#: as it did before the sidecar existed.
DEFAULT_FPS = 30.0

META_SCHEMA_VERSION = 1


def meta_path(csv_path: str | Path) -> Path:
    """Sidecar path for a pose CSV. Pure string work — no I/O."""
    csv_path = Path(csv_path)
    return csv_path.parent / f"{csv_path.stem}.meta.json"


def write_pose_meta(pose: PoseData, csv_path: str | Path) -> Path:
    """Write the sidecar describing *pose* beside its CSV.

    Best-effort by design: the CSV is the artifact that matters, so a
    read-only directory or a full disk must not fail a finished inference
    run. A missing sidecar degrades to :data:`DEFAULT_FPS` on read.
    """
    path = meta_path(csv_path)
    payload = {
        "schema_version": META_SCHEMA_VERSION,
        "fps": float(pose.fps),
        "source": pose.source,
        "keypoint_names": list(pose.keypoint_names),
        "n_frames": int(pose.n_frames),
    }
    # Frame size, when the producer knew it. Pose coordinates are pixels, so
    # anything drawing them without the video -- the analysis viewer -- needs
    # the canvas they were measured on. Omitted rather than guessed when
    # unknown: inferring it from the coordinate range would silently shrink
    # the arena to whatever the animal happened to visit.
    resolution = pose.metadata.get("resolution") if pose.metadata else None
    if resolution:
        try:
            width, height = (int(v) for v in resolution)
            if width > 0 and height > 0:
                payload["resolution"] = [width, height]
        except (TypeError, ValueError):
            pass
    # Provenance, not decoration: scoring refuses thresholds derived under a
    # different gate, so this block is what makes that check possible. Absent
    # means ungated, which is true of every file written before the gate
    # existed. Optional and additive, so META_SCHEMA_VERSION does not move.
    gate = pose.metadata.get("arena_gate") if pose.metadata else None
    if gate:
        payload["arena_gate"] = gate
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    except OSError as e:  # pragma: no cover - depends on filesystem state
        warnings.warn(
            f"could not write pose metadata beside {Path(csv_path).name}: {e}. "
            f"The CSV is fine, but readers will assume {DEFAULT_FPS} fps.",
            stacklevel=2,
        )
    return path


def read_pose_meta(csv_path: str | Path) -> dict[str, Any] | None:
    """Read the sidecar beside a pose CSV, or ``None`` if absent/unusable.

    Never raises: a corrupt sidecar is a missing sidecar as far as callers
    are concerned, because the CSV alone is still perfectly readable.
    """
    path = meta_path(csv_path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def resolution_for_csv(csv_path: str | Path) -> tuple[int, int] | None:
    """``(width, height)`` recorded beside a pose CSV, or None if unknown.

    None is meaningful: a viewer drawing keypoints without the video must say
    it cannot size the arena rather than invent one.
    """
    data = read_pose_meta(csv_path)
    if not data:
        return None
    value = data.get("resolution")
    try:
        width, height = (int(v) for v in value)
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def backfill_resolution(csv_path: str | Path, resolution) -> bool:
    """Add a resolution to an existing sidecar. True if it was written.

    Sidecars written before the field existed are otherwise unusable by the
    analysis viewer, and re-running inference to recover a number the video
    already knows would be absurd.
    """
    path = meta_path(csv_path)
    data = read_pose_meta(csv_path)
    if data is None:
        return False
    try:
        width, height = (int(v) for v in resolution)
    except (TypeError, ValueError):
        return False
    if width <= 0 or height <= 0:
        return False
    data["resolution"] = [width, height]
    try:
        path.write_text(json.dumps(data, indent=2) + "\n")
    except OSError:
        return False
    return True


def fps_for_csv(csv_path: str | Path) -> float | None:
    """The frame rate recorded beside a pose CSV, if one was recorded.

    ``None`` means "unknown" — deliberately distinct from
    :data:`DEFAULT_FPS`, so callers can tell a real 30 fps recording from an
    assumption and warn accordingly.
    """
    data = read_pose_meta(csv_path)
    if data is None:
        return None
    fps = data.get("fps")
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def _build_dataframe(pose: PoseData) -> pd.DataFrame:
    """Build the DLC multi-index DataFrame (no I/O)."""
    n_frames = pose.n_frames
    n_kpts = pose.n_keypoints

    # Interleave x, y, likelihood per body part.
    flat = np.empty((n_frames, n_kpts * 3), dtype=float)
    flat[:, 0::3] = pose.xy[:, :, 0]
    flat[:, 1::3] = pose.xy[:, :, 1]
    flat[:, 2::3] = pose.confidence

    columns = pd.MultiIndex.from_product(
        [[pose.source], pose.keypoint_names, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(flat, columns=columns)
    # DLC convention: index has no name (so pd.to_csv emits exactly 3 header rows).
    return df


#: Stem suffixes that share a pose CSV's "<stem>DLC_<model>" prefix but are not
#: pose data to analyse. Lives here because two separate discovery paths need
#: it and they drifted apart once already: find_pose_csv excluded _raw while
#: the cohort collector did not, so every session was pooled twice and each
#: animal's weight in the percentiles was silently halved.
NOT_POSE_SUFFIXES = ("_raw", "_annotations", "_ungated")


def to_dlc_csv(pose: PoseData, path: str | Path, *, write_meta: bool = True) -> Path:
    """Write a DeepLabCut-format CSV, plus its frame-rate sidecar.

    Returns the resolved path written. Pass ``write_meta=False`` for a bare
    DLC CSV with no companion file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _build_dataframe(pose)
    df.to_csv(path)
    if write_meta:
        write_pose_meta(pose, path)
    return path


def to_dlc_h5(pose: PoseData, path: str | Path, *, key: str = "df_with_missing") -> Path:
    """Write a DeepLabCut-style HDF5 (pandas pytables format).

    Some downstream tools require this rather than CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _build_dataframe(pose)
    # DLC stores the table under a fixed key with format='table' so it can be
    # appended to / queried; we mirror that.
    df.to_hdf(path, key=key, mode="w", format="table")
    return path


def from_dlc_csv(path: str | Path, *, fps: float | None = None) -> PoseData:
    """Read a DLC-format CSV back into a PoseData.

    Useful for round-trip tests and for chaining with downstream filters.

    ``fps=None`` (the default) takes the rate from the sidecar
    :func:`to_dlc_csv` wrote, falling back to :data:`DEFAULT_FPS` for CSVs
    that predate it. Pass a number to override both.
    """
    if fps is None:
        fps = fps_for_csv(path)
        if fps is None:
            fps = DEFAULT_FPS
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    # Header shape: (scorer, bodyparts, coords)
    scorer_levels = df.columns.get_level_values("scorer").unique().tolist()
    if len(scorer_levels) != 1:
        raise ValueError(f"expected a single scorer column, got: {scorer_levels}")
    scorer = scorer_levels[0]

    bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
    n_frames = len(df)
    n_kpts = len(bodyparts)

    xy = np.empty((n_frames, n_kpts, 2), dtype=float)
    cf = np.empty((n_frames, n_kpts), dtype=float)
    for i, bp in enumerate(bodyparts):
        xy[:, i, 0] = df[(scorer, bp, "x")].to_numpy()
        xy[:, i, 1] = df[(scorer, bp, "y")].to_numpy()
        cf[:, i] = df[(scorer, bp, "likelihood")].to_numpy()

    return PoseData(
        xy=xy,
        confidence=cf,
        keypoint_names=bodyparts,
        fps=fps,
        source=scorer,
    )
