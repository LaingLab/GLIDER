"""Sturman Open Field Test (OFT) benchmark adapter.

Converts the Sturman et al. (2020) OFT dataset — the same benchmark DLC2action
reports on — into the ``(pose CSV, annotations CSV)`` session pairs that
:func:`glider.analysis.behavior.pipeline.train_model` consumes, so the GLIDER
behavior classifier can be evaluated head-to-head against the DLC2action
action-segmentation models (C2F-TCN et al.).

Two format gaps are bridged here.

**Pose.** Sturman DLC CSVs track the mouse *and* five static arena markers
(``tl``, ``tr``, ``br``, ``bl``, ``centre``). Those are dropped so features are
computed on the animal only — otherwise every pairwise distance to a fixed
marker leaks the animal's absolute arena position (a cross-video confound) and
the body axis can't be resolved to real anatomy. The cleaned pose is re-emitted
as a standard DLC CSV that ``from_dlc_csv`` reads back.

**Labels.** Sturman manual annotations are an *episode table* — one row per
behavior bout, columns ``from`` / ``to`` (in **seconds**) and ``type``
(``Grooming`` / ``Supported`` / ``Unsupported``), with a video-linking column
(``CSVname`` / ``DLCFile`` / ``file``). Frames in no bout are implicitly
background. Each bout becomes a GLIDER
:class:`~glider.analysis.behavior.annotations.BehaviorZone` half-open
``[start_frame, end_frame)`` interval (``seconds × fps``, rounded). Frames left
unlabeled train as "unknown" and are dropped by the pipeline — matching the
benchmark's implicit-background convention.

Dataset facts (Sturman et al. 2020, *Neuropsychopharmacology*; mirrored by
DLC2action at ``examples/benchmarks/sturman_oft.py``): 25 fps, DeepLabCut pose,
three behaviors above. The raw dataset is not vendored here — point
:func:`build_oft_benchmark` at a local copy.

Column names and the seconds-vs-frames time unit are the two things most likely
to vary between Sturman exports and DLC2action's per-video split; both are
tolerant / parameterised so a different export only needs the right arguments,
not a code change.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
from glider.analysis.behavior.features import FeatureSpec
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import from_dlc_csv, to_dlc_csv

logger = logging.getLogger(__name__)

# --- Sturman OFT constants ----------------------------------------------------

OFT_FPS = 25.0
# Static arena fiducials tracked alongside the mouse; not body parts.
OFT_ARENA_MARKERS: frozenset[str] = frozenset({"tl", "tr", "br", "bl", "centre"})
# The three mutually-exclusive OFT behaviors (canonical casing).
OFT_BEHAVIORS: tuple[str, ...] = ("Grooming", "Supported", "Unsupported")
# (head, tail) keypoint NAMES defining the body axis, resolved to indices after
# arena markers are dropped.
OFT_BODY_AXIS: tuple[str, str] = ("nose", "tailbase")

# Tolerant column aliases for the episode table (matched case-insensitively).
_FROM_ALIASES = ("from", "start", "start_time", "onset", "begin")
_TO_ALIASES = ("to", "stop", "end", "end_time", "offset", "finish")
_TYPE_ALIASES = ("type", "behavior", "behaviour", "label", "class")
# DLCFile before file: DLCFile links to the pose CSV name; file is the
# annotation filename (e.g. "OFT_11_Jin.csv") which does NOT match pose files.
_VIDEO_ALIASES = ("DLCFile", "CSVname", "video", "filename", "name", "file")
_EXPERIMENTER_ALIASES = ("Experimenter", "experimenter", "annotator", "rater", "scorer")

# Strip a DeepLabCut scorer suffix so a pose file and its label row match on the
# bare video stem: "<video>DeepCut_resnet50_...1030000" -> "<video>".
_DLC_SUFFIX = re.compile(r"(deepcut|dlc_?resnet|dlc)\w*.*$", re.IGNORECASE)


# --- Pose ---------------------------------------------------------------------


def drop_keypoints(pose: PoseData, drop: Iterable[str] = OFT_ARENA_MARKERS) -> PoseData:
    """Return ``pose`` with the named keypoints removed (case-insensitive).

    Keypoint order is otherwise preserved, so indices into the returned pose are
    stable and match the CSV re-emitted by :func:`to_dlc_csv`.
    """
    drop_lc = {str(d).strip().lower() for d in drop}
    keep = [i for i, n in enumerate(pose.keypoint_names) if n.strip().lower() not in drop_lc]
    if not keep:
        raise ValueError(
            f"dropping {sorted(drop_lc)} would remove every keypoint from " f"{pose.keypoint_names}"
        )
    return PoseData(
        xy=pose.xy[:, keep, :].copy(),
        confidence=pose.confidence[:, keep].copy(),
        keypoint_names=[pose.keypoint_names[i] for i in keep],
        fps=pose.fps,
        source=pose.source,
        metadata=dict(pose.metadata),
    )


def load_sturman_pose(
    csv_path: str | Path,
    *,
    markers: Iterable[str] = OFT_ARENA_MARKERS,
    fps: float = OFT_FPS,
) -> PoseData:
    """Read a Sturman DLC CSV and drop the arena markers, returning animal pose."""
    pose = from_dlc_csv(csv_path, fps=fps)
    return drop_keypoints(pose, markers)


def _find_kp(pose: PoseData, name: str) -> int:
    """Case-insensitive keypoint lookup (Sturman uses lowercase names)."""
    target = name.strip().lower()
    for i, n in enumerate(pose.keypoint_names):
        if n.strip().lower() == target:
            return i
    raise KeyError(f"keypoint {name!r} not found in {pose.keypoint_names}")


def resolve_body_axis(pose: PoseData, names: tuple[str, str] = OFT_BODY_AXIS) -> tuple[int, int]:
    """Resolve the (head, tail) body-axis keypoint names to indices in ``pose``."""
    head, tail = names
    return (_find_kp(pose, head), _find_kp(pose, tail))


def oft_feature_spec(pose: PoseData, names: tuple[str, str] = OFT_BODY_AXIS) -> FeatureSpec:
    """Build a :class:`FeatureSpec` with the OFT body axis resolved for ``pose``.

    Pass the result to ``train_model(spec=...)`` so pairwise distances normalise
    by the animal's nose→tailbase length rather than a default index guess.
    """
    return FeatureSpec(body_axis=resolve_body_axis(pose, names))


# --- Labels -------------------------------------------------------------------


def _sniff_sep(path: Path) -> str:
    """Guess the delimiter from the header row (Sturman masters use ';')."""
    with path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def read_label_table(path: str | Path, *, sep: str | None = None) -> pd.DataFrame:
    """Read a Sturman/DLC2action episode table, auto-sniffing ';' vs ',' if needed."""
    path = Path(path)
    sep = sep if sep is not None else _sniff_sep(path)
    df = pd.read_csv(path, sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _resolve_col(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _video_key(name: object) -> str:
    """Normalise a filename / link value to a bare video stem for matching."""
    stem = Path(str(name)).name
    stem = re.sub(r"\.csv$", "", stem, flags=re.IGNORECASE)
    stem = _DLC_SUFFIX.sub("", stem)
    return stem.strip("_ .").lower()


def list_experimenters(df: pd.DataFrame) -> list[str]:
    """Sorted unique annotator names, or ``[]`` if the table has no such column.

    Each Sturman OFT video is scored by several human raters; selecting one via
    ``experimenter=`` gives a single clean ground truth instead of a union.
    """
    col = _resolve_col(df, _EXPERIMENTER_ALIASES)
    if col is None:
        return []
    return sorted(str(v) for v in df[col].dropna().unique())


def episodes_for_video(
    df: pd.DataFrame,
    video_key: str | None = None,
    *,
    experimenter: str | None = None,
) -> list[tuple[str, float, float]]:
    """Extract ``(behavior_type, from, to)`` rows, optionally filtered to one video.

    ``video_key`` is a normalised stem (see :func:`_video_key`); rows whose
    video-link column normalises to it are kept. When ``video_key`` is ``None``
    (or the table has no recognisable link column) every row is returned — the
    single-video-file case.

    ``experimenter`` restricts to one annotator (case-insensitive). When a video
    is multiply-annotated and no annotator is chosen, all raters' bouts are
    returned and later coalesced into their union — usually NOT what a benchmark
    wants, so a warning is logged.
    """
    fcol = _resolve_col(df, _FROM_ALIASES)
    tcol = _resolve_col(df, _TO_ALIASES)
    ycol = _resolve_col(df, _TYPE_ALIASES)
    missing = [n for n, c in (("from", fcol), ("to", tcol), ("type", ycol)) if c is None]
    if missing:
        raise ValueError(
            f"label table is missing required column(s) {missing}; "
            f"found columns {list(df.columns)}"
        )
    vcol = _resolve_col(df, _VIDEO_ALIASES)
    ecol = _resolve_col(df, _EXPERIMENTER_ALIASES)

    rows = df
    if video_key is not None and vcol is not None:
        rows = rows[rows[vcol].map(lambda v: _video_key(v) == video_key)]
    if experimenter is not None:
        if ecol is None:
            raise ValueError("experimenter given but the table has no experimenter column")
        want = experimenter.strip().lower()
        rows = rows[rows[ecol].map(lambda v: str(v).strip().lower() == want)]
    elif ecol is not None and rows[ecol].nunique() > 1:
        logger.warning(
            "video %r is annotated by %d experimenters %s and none was selected; "
            "bouts will be unioned across raters. Pass experimenter= for a clean "
            "single-rater ground truth.",
            video_key,
            rows[ecol].nunique(),
            sorted(str(v) for v in rows[ecol].dropna().unique()),
        )

    out: list[tuple[str, float, float]] = []
    for _, row in rows.iterrows():
        try:
            start = float(row[fcol])
            stop = float(row[tcol])
        except (TypeError, ValueError):
            continue  # blank/garbage timing row — skip
        out.append((str(row[ycol]), start, stop))
    return out


def _coalesce(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge strictly-overlapping half-open intervals (same behavior)."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[list[int]] = [list(ordered[0])]
    for start, stop in ordered[1:]:
        if start < merged[-1][1]:  # strict overlap ([s, e) touching is not)
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(s, e) for s, e in merged]


def labels_to_store(
    episodes: Iterable[tuple[str, float, float]],
    *,
    fps: float = OFT_FPS,
    behaviors: Sequence[str] = OFT_BEHAVIORS,
    n_frames: int | None = None,
    time_unit: str = "seconds",
) -> AnnotationStore:
    """Convert episode rows to a GLIDER :class:`AnnotationStore`.

    ``time_unit`` is ``"seconds"`` (Sturman raw export; ``from``/``to`` are
    multiplied by ``fps``) or ``"frames"`` (already frame indices). Episode
    ``type`` values are matched to ``behaviors`` case-insensitively and
    normalised to the canonical casing; unrecognised types are skipped. When
    ``n_frames`` is given, zones are clamped to ``[0, n_frames)`` and zones that
    start past the end are dropped. Same-behavior overlaps are coalesced (the
    store forbids them); cross-behavior overlap can't occur for the mutually
    exclusive OFT set.
    """
    if time_unit not in ("seconds", "frames"):
        raise ValueError(f"time_unit must be 'seconds' or 'frames', got {time_unit!r}")
    scale = float(fps) if time_unit == "seconds" else 1.0
    canon = {b.strip().lower(): b for b in behaviors}

    by_behavior: dict[str, list[tuple[int, int]]] = {}
    for beh, raw_start, raw_stop in episodes:
        key = str(beh).strip().lower()
        if key not in canon:
            continue
        start = max(0, int(round(float(raw_start) * scale)))
        stop = int(round(float(raw_stop) * scale))
        if stop <= start:
            stop = start + 1  # a bout must span >= 1 frame (half-open)
        if n_frames is not None:
            if start >= n_frames:
                continue
            stop = min(stop, n_frames)
            if stop <= start:
                continue
        by_behavior.setdefault(canon[key], []).append((start, stop))

    store = AnnotationStore()
    for beh, intervals in by_behavior.items():
        for start, stop in _coalesce(intervals):
            store.add(
                BehaviorZone(
                    behavior=beh,
                    start_frame=start,
                    end_frame=stop,
                    note="sturman-oft",
                )
            )
    return store


# --- Orchestration ------------------------------------------------------------


@dataclass(frozen=True)
class OFTSession:
    """One converted OFT video: cleaned pose + GLIDER annotations, on disk."""

    video_id: str
    pose_csv: Path
    annotations_csv: Path
    n_frames: int
    n_labeled_frames: int

    @property
    def pair(self) -> tuple[Path, Path]:
        """The ``(pose_csv, annotations_csv)`` tuple ``train_model`` wants."""
        return (self.pose_csv, self.annotations_csv)


def build_oft_benchmark(
    pose_dir: str | Path,
    label_path: str | Path,
    out_dir: str | Path,
    *,
    experimenter: str | None = None,
    require_labels: bool = True,
    fps: float = OFT_FPS,
    markers: Iterable[str] = OFT_ARENA_MARKERS,
    behaviors: Sequence[str] = OFT_BEHAVIORS,
    pose_glob: str = "*.csv",
    time_unit: str = "seconds",
    sep: str | None = None,
) -> list[OFTSession]:
    """Convert a Sturman OFT dataset into GLIDER session pairs under ``out_dir``.

    ``pose_dir`` holds the DLC pose CSVs (Sturman ``Output_DLC/``); ``label_path``
    is the episode table (one master file, or a single video's file). For each
    kept pose CSV a ``<out_dir>/<video_id>/`` folder gets a cleaned ``pose.csv``
    (arena markers dropped) and an ``annotations.csv`` (GLIDER zones). Returns one
    :class:`OFTSession` per kept pose file, in sorted order.

    ``experimenter`` selects a single annotator's labels (the Sturman set is
    multiply-annotated; see :func:`list_experimenters`). ``require_labels``
    (default ``True``) skips pose videos with no matching behavior bouts — so a
    mixed ``Output_DLC/`` of labeled + unlabeled recordings yields exactly the
    benchmark's labeled subset. Set it ``False`` to emit an all-background
    session for every pose file instead.
    """
    pose_dir = Path(pose_dir)
    out_dir = Path(out_dir)
    label_df = read_label_table(label_path, sep=sep)

    pose_files = sorted(pose_dir.glob(pose_glob))
    if not pose_files:
        raise FileNotFoundError(f"no pose CSVs matched {pose_glob!r} under {pose_dir}")

    sessions: list[OFTSession] = []
    skipped = 0
    for pose_csv in pose_files:
        video_key = _video_key(pose_csv.name)
        episodes = episodes_for_video(label_df, video_key, experimenter=experimenter)
        pose = load_sturman_pose(pose_csv, markers=markers, fps=fps)
        store = labels_to_store(
            episodes,
            fps=fps,
            behaviors=behaviors,
            n_frames=pose.n_frames,
            time_unit=time_unit,
        )
        labeled = sum(store.total_frames_by_behavior().values())
        if labeled == 0 and require_labels:
            # No behavior bouts for this video (unlabeled recording, or matched
            # only non-behavior markers like Start/End) — leave it out.
            skipped += 1
            continue

        vid_dir = out_dir / video_key
        pose_out = to_dlc_csv(pose, vid_dir / "pose.csv")
        ann_out = store.save_csv(vid_dir / "annotations.csv")
        sessions.append(
            OFTSession(
                video_id=video_key,
                pose_csv=pose_out,
                annotations_csv=ann_out,
                n_frames=pose.n_frames,
                n_labeled_frames=labeled,
            )
        )
    if skipped:
        logger.info("skipped %d pose video(s) with no matching behavior labels", skipped)
    if not sessions:
        raise ValueError(
            f"no labeled videos produced (require_labels={require_labels}); check the "
            "experimenter name and that DLCFile links match pose filenames"
        )
    return sessions


def sessions_to_pairs(sessions: Iterable[OFTSession]) -> list[tuple[Path, Path]]:
    """Flatten sessions to the ``[(pose_csv, annotations_csv), ...]`` list."""
    return [s.pair for s in sessions]


def leave_one_out_splits(
    sessions: Sequence[OFTSession],
) -> list[tuple[list[OFTSession], OFTSession]]:
    """Leave-one-video-out CV folds: ``(train_sessions, held_out_session)`` each.

    Matches the DLC2action OFT protocol (leave-one-out over videos). Sessions
    with no labeled frames are still valid held-out sets only if something is
    labeled; a caller doing metrics should skip an all-background test video.
    """
    folds: list[tuple[list[OFTSession], OFTSession]] = []
    for i, test in enumerate(sessions):
        train = [s for j, s in enumerate(sessions) if j != i]
        folds.append((train, test))
    return folds
