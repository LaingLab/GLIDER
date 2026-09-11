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

Multi-animal tracking adds a fourth header row -- DLC's own convention
inserts ``individuals`` between ``scorer`` and ``bodyparts`` -- so
:func:`to_dlc_csv_multi` and :func:`write_tracks_meta` write that shape from
a :class:`~glider.vision.pose.tracks.PoseTracks` instead of a single
:class:`PoseData`, one column block per animal in the same CSV. Same sidecar,
same rate-and-resolution provenance as the single-animal path below.

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
from glider.vision.pose.tracks import PoseTracks

#: Fallback when a CSV predates the sidecar and the caller named no rate.
#: Matches the historical ``from_dlc_csv`` default, so old data reads exactly
#: as it did before the sidecar existed.
DEFAULT_FPS = 30.0

META_SCHEMA_VERSION = 1


def meta_path(csv_path: str | Path) -> Path:
    """Sidecar path for a pose CSV. Pure string work — no I/O."""
    csv_path = Path(csv_path)
    return csv_path.parent / f"{csv_path.stem}.meta.json"


def _validated_resolution(metadata: dict[str, Any] | None) -> list[int] | None:
    """``[width, height]`` from a metadata dict's ``resolution``, or ``None``.

    Shared by :func:`write_pose_meta` and :func:`write_tracks_meta`: a
    malformed resolution (wrong arity, non-numeric, zero or negative) is
    silently ignored rather than written to a sidecar a reader would trust.
    """
    resolution = metadata.get("resolution") if metadata else None
    if not resolution:
        return None
    try:
        width, height = (int(v) for v in resolution)
    except (TypeError, ValueError):
        return None
    return [width, height] if width > 0 and height > 0 else None


def _write_meta_payload(payload: dict[str, Any], csv_path: str | Path) -> Path:
    """Write *payload* to the sidecar path beside *csv_path*.

    Best-effort by design: the CSV is the artifact that matters, so a
    read-only directory or a full disk must not fail a finished inference
    run. A missing sidecar degrades to :data:`DEFAULT_FPS` on read.
    """
    path = meta_path(csv_path)
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    except OSError as e:  # pragma: no cover - depends on filesystem state
        warnings.warn(
            f"could not write pose metadata beside {Path(csv_path).name}: {e}. "
            f"The CSV is fine, but readers will assume {DEFAULT_FPS} fps.",
            stacklevel=3,
        )
    return path


def write_pose_meta(pose: PoseData, csv_path: str | Path) -> Path:
    """Write the sidecar describing *pose* beside its CSV.

    Best-effort by design: the CSV is the artifact that matters, so a
    read-only directory or a full disk must not fail a finished inference
    run. A missing sidecar degrades to :data:`DEFAULT_FPS` on read.
    """
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
    resolution = _validated_resolution(pose.metadata)
    if resolution:
        payload["resolution"] = resolution
    # Provenance, not decoration: scoring refuses thresholds derived under a
    # different gate, so this block is what makes that check possible. Absent
    # means ungated, which is true of every file written before the gate
    # existed. Optional and additive, so META_SCHEMA_VERSION does not move.
    gate = pose.metadata.get("arena_gate") if pose.metadata else None
    if gate:
        payload["arena_gate"] = gate
    return _write_meta_payload(payload, csv_path)


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


def _interleave_xyc(pose: PoseData) -> np.ndarray:
    """``(n_frames, n_keypoints * 3)`` array of x, y, likelihood interleaved
    per body part -- the column layout both DLC builders share."""
    flat = np.empty((pose.n_frames, pose.n_keypoints * 3), dtype=float)
    flat[:, 0::3] = pose.xy[:, :, 0]
    flat[:, 1::3] = pose.xy[:, :, 1]
    flat[:, 2::3] = pose.confidence
    return flat


def _build_dataframe(pose: PoseData) -> pd.DataFrame:
    """Build the DLC multi-index DataFrame (no I/O)."""
    flat = _interleave_xyc(pose)
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
NOT_POSE_SUFFIXES = ("_raw", "_annotations", "_ungated", "_identity")


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


def header_depth(path: str | Path) -> int:
    """3 for a single-animal DLC CSV, 4 for a multi-animal one.

    Decided from the second line's first cell rather than by counting, because
    the two formats are otherwise indistinguishable without parsing the whole
    file -- and a three-row file misread as four silently shifts every column.
    """
    with open(path, encoding="utf-8") as f:
        second = f.readline() and f.readline()
    return 4 if second.split(",", 1)[0].strip() == "individuals" else 3


def list_individuals(path: str | Path) -> list[str]:
    """Individual names in a multi-animal CSV; ``[]`` for a single-animal one."""
    if header_depth(path) != 4:
        return []
    with open(path, encoding="utf-8") as f:
        f.readline()
        cells = f.readline().rstrip("\n").split(",")[1:]
    seen: list[str] = []
    for cell in cells:
        name = cell.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def from_dlc_csv(
    path: str | Path,
    *,
    fps: float | None = None,
    individual: str | int | None = None,
) -> PoseData:
    """Read a DLC-format CSV back into a PoseData.

    Three-row files read exactly as they always have. For a four-row file,
    ``individual`` names which animal to return -- by name (``"animal1"``) or
    by slot (``1``).

    A multi-animal file with no ``individual`` **raises**. Handing back the
    first animal would be indistinguishable from a single-animal read, and
    silently analysing one arbitrary mouse from a social recording is the exact
    failure this whole feature exists to end.
    """
    if fps is None:
        fps = fps_for_csv(path)
        if fps is None:
            fps = DEFAULT_FPS

    depth = header_depth(path)
    if depth == 3:
        if individual is not None:
            raise ValueError(
                f"{Path(path).name} carries no individuals -- it is a "
                f"single-animal DLC CSV. Drop the individual= argument."
            )
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    else:
        names = list_individuals(path)
        if individual is None:
            raise ValueError(
                f"{Path(path).name} holds {len(names)} animals "
                f"({', '.join(names)}); pass individual= to say which one."
            )
        wanted = f"animal{individual}" if isinstance(individual, int) else str(individual)
        if wanted not in names:
            raise ValueError(
                f"{wanted!r} is not in {Path(path).name}; it holds " f"{', '.join(names)}."
            )
        df = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
        df = df.xs(wanted, axis=1, level="individuals")

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

    # Carry the sidecar's frame size across. Without this a read/write
    # round-trip writes a sidecar with no resolution, and downstream that is
    # not an error -- it is a blank speed axis, because resolution is what
    # converts px to cm and freezing/darting are never scored without it.
    metadata: dict[str, Any] = {}
    resolution = resolution_for_csv(path)
    if resolution is not None:
        metadata["resolution"] = list(resolution)

    return PoseData(
        xy=xy,
        confidence=cf,
        keypoint_names=bodyparts,
        fps=fps,
        source=scorer,
        metadata=metadata,
    )


def _build_dataframe_multi(tracks: PoseTracks) -> pd.DataFrame:
    """The four-row-header DLC frame: scorer / individuals / bodyparts / coords.

    Built per animal and concatenated so column order is animal-major, which is
    what DLC itself writes and what tools reading it expect.
    """
    blocks = []
    for slot in tracks:
        pose = tracks[slot]
        flat = _interleave_xyc(pose)
        columns = pd.MultiIndex.from_product(
            [
                [pose.source],
                [f"animal{slot}"],
                pose.keypoint_names,
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        blocks.append(pd.DataFrame(flat, columns=columns))
    return pd.concat(blocks, axis=1)


def write_tracks_meta(tracks: PoseTracks, csv_path: str | Path) -> Path:
    """Sidecar for a multi-animal CSV.

    Same shape as :func:`write_pose_meta` plus the animal count and names. A
    re-run with different consolidation knobs has to be distinguishable from
    the original after the fact, so whatever the caller recorded in
    ``tracks.metadata['consolidation']`` is carried through -- that one is
    genuinely container-level, set once for the whole video.

    ``arena_gate`` is not: every producer sets it on the first animal's
    ``PoseData.metadata`` (never on ``tracks.metadata``), because one arena
    and one gate config are shared by every animal in the video -- it is read
    from there, exactly like ``resolution`` one block above.
    """
    first = tracks[0]
    payload = {
        "schema_version": META_SCHEMA_VERSION,
        "fps": float(tracks.fps),
        "source": first.source,
        "keypoint_names": list(tracks.keypoint_names),
        "n_frames": int(tracks.n_frames),
        "n_animals": int(tracks.n_animals),
        "individuals": list(tracks.individuals),
    }
    resolution = _validated_resolution(first.metadata)
    if resolution:
        payload["resolution"] = resolution
    consolidation = (tracks.metadata or {}).get("consolidation")
    if consolidation:
        payload["consolidation"] = consolidation
    gate = first.metadata.get("arena_gate") if first.metadata else None
    if gate:
        payload["arena_gate"] = gate
    return _write_meta_payload(payload, csv_path)


def to_dlc_csv_multi(tracks: PoseTracks, path: str | Path, *, write_meta: bool = True) -> Path:
    """Write N animals as one DeepLabCut CSV, plus its sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataframe_multi(tracks).to_csv(path)
    if write_meta:
        write_tracks_meta(tracks, path)
    return path
