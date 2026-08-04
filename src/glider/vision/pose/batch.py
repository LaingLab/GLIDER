"""Batch YOLO-pose inference over directories of videos.

Qt-free by design: :mod:`glider.gui.pose_batch` drives this from a QThread, and
the same functions stay usable from a script or a notebook.

Output naming follows DeepLabCut's own convention so existing DLC analysis
tooling finds the files without configuration::

    session01.mp4  +  exp-6.pt  ->  session01DLC_exp-6.csv

Heavy imports (ultralytics, torch, pandas) stay inside :func:`run_batch` so
importing this module — which the GUI does while building menus — stays cheap.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from glider.vision.pose.core import PoseData

logger = logging.getLogger(__name__)

#: Video containers both halves of the pipeline accept. Single-sourced on
#: purpose: this list used to be restated in the behavior tool with ``.webm``
#: in place of ``.wmv``, so a folder could offer a video to one tool and hide
#: it from the other with nothing on screen explaining the difference.
VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"})

#: Qt file-dialog filter over :data:`VIDEO_EXTS`, so pickers can't drift from
#: what discovery actually accepts.
VIDEO_FILTER = "Video files (" + " ".join(f"*{e}" for e in sorted(VIDEO_EXTS)) + ");;All files (*)"

__all__ = [
    "VIDEO_EXTS",
    "VIDEO_FILTER",
    "BatchEvent",
    "BatchResult",
    "EventKind",
    "FilterSettings",
    "discover_videos",
    "dlc_output_path",
    "find_pose_csv",
    "raw_output_path",
    "run_batch",
]


def discover_videos(paths: Iterable[Path], *, recursive: bool = True) -> list[Path]:
    """Collect video files from a mix of directories and individual files.

    Missing paths are ignored rather than raising: the input is a user's
    drag-and-drop selection, which can go stale between the drop and Run.
    """
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            if path.suffix.lower() in VIDEO_EXTS:
                found.add(path.resolve())
        elif path.is_dir():
            children = path.rglob("*") if recursive else path.glob("*")
            for child in children:
                if child.is_file() and child.suffix.lower() in VIDEO_EXTS:
                    found.add(child.resolve())
    return sorted(found)


def _output_stem(video: Path, model: Path) -> str:
    """Shared stem, so the primary and raw names can never drift apart."""
    return f"{Path(video).stem}DLC_{Path(model).stem}"


def dlc_output_path(video: Path, model: Path) -> Path:
    """Primary DLC CSV path, written beside the video."""
    video = Path(video)
    return video.parent / f"{_output_stem(video, model)}.csv"


def raw_output_path(video: Path, model: Path) -> Path:
    """Unfiltered companion CSV. Written only when filtering is enabled."""
    video = Path(video)
    return video.parent / f"{_output_stem(video, model)}_raw.csv"


# Stem suffixes that share the "<stem>DLC_<model>" prefix but are not pose
# data. Anything a tool writes beside a pose CSV using its stem belongs here.
_NOT_POSE_SUFFIXES = ("_raw", "_annotations")


def find_pose_csv(video: Path | str, search_dir: Path | str | None = None) -> Path | None:
    """The pose CSV belonging to *video*, or ``None`` if there isn't one.

    Lives here because this module owns the output naming, and every reader
    downstream has to recognise what :func:`run_batch` wrote. Looks in
    *search_dir* (default: the video's own directory) for either naming in
    use across GLIDER:

    ``<stem>.csv``
        A hand-placed file, or one exported from DeepLabCut proper. This is
        what the annotate and train paths have always expected.
    ``<stem>DLC_<model>.csv``
        What :func:`run_batch` writes. Companions that share this prefix but
        are not pose data are skipped (see :data:`_NOT_POSE_SUFFIXES`):
        ``_raw`` is the unsmoothed inference, never the one to analyze, and
        ``_annotations`` is the annotator's behavior zones. The latter is
        written *after* the pose CSV, so before it was excluded it won the
        most-recent tie-break and every reader downstream — annotate, train,
        apply — was handed a file of labels in place of keypoints.

    The exact-stem form wins when both exist, so a file the operator placed
    deliberately is never shadowed by a batch run. When several models have
    been run over one video the most recently written wins, and the choice is
    logged: alphabetical order would silently prefer ``exp-5`` over ``exp-7``,
    quietly scoring a cohort with a superseded pose model. Naming the file
    explicitly is still the only way to be certain.
    """
    video = Path(video)
    directory = Path(search_dir) if search_dir is not None else video.parent
    if not directory.is_dir():
        return None

    exact = directory / f"{video.stem}.csv"
    if exact.exists():
        return exact

    matches = [
        p
        for p in sorted(directory.glob(f"{video.stem}DLC_*.csv"))
        if not p.stem.endswith(_NOT_POSE_SUFFIXES)
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    # Newest wins. mtime can be unreadable on a share mid-copy; those sort
    # last rather than raising, so a transient stat error cannot pick the file.
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return float("-inf")

    chosen = max(matches, key=_mtime)
    logger.info(
        "%s has %d pose CSVs (%s); using the most recent, %s",
        video.name,
        len(matches),
        ", ".join(p.name for p in matches),
        chosen.name,
    )
    return chosen


class EventKind(StrEnum):
    """What happened to one video in the batch."""

    STARTED = "started"
    WROTE = "wrote"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BatchEvent:
    """One transition, handed to ``on_event`` so callers can log progress."""

    kind: EventKind
    video: Path
    index: int  # 0-based position within the batch
    total: int
    output: Path | None = None
    message: str = ""


@dataclass(frozen=True)
class FilterSettings:
    """Post-processing applied before the primary CSV is written.

    Defaults mirror :func:`glider.vision.pose.filtering.smooth`.
    """

    confidence_threshold: float = 0.5
    max_gap: int = 5
    median_window: int = 5


@dataclass
class BatchResult:
    """Outcome of a whole batch."""

    completed: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    cancelled: bool = False

    @property
    def summary(self) -> str:
        """One-line tally. Skips are reported so they aren't mistaken for work."""
        parts = [f"{len(self.completed)} completed"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.cancelled:
            parts.append("cancelled")
        return ", ".join(parts)


def run_batch(
    videos: Sequence[Path],
    model_path: Path,
    keypoint_names: Sequence[str],
    *,
    conf: float = 0.25,
    device: str | None = None,
    require_gpu: bool = False,
    overwrite: bool = False,
    filtering: FilterSettings | None = None,
    on_event: Callable[[BatchEvent], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    infer: Callable[..., PoseData] | None = None,
) -> BatchResult:
    """Run a pose model over ``videos``, writing a DLC CSV beside each one.

    Videos are processed sequentially: inference is GPU-bound, so concurrent
    videos would contend rather than go faster.

    Parameters
    ----------
    overwrite
        When False (the default) a video whose primary CSV already exists is
        skipped, so an interrupted batch resumes cheaply.
    filtering
        When given, the unfiltered result is written to the ``_raw`` path first
        and the smoothed result becomes the primary CSV — an unhappy filter
        setting can never destroy the inference run.
    infer
        Injection point for tests; defaults to
        :func:`glider.vision.pose.core.infer_video`.

    Raises
    ------
    RuntimeError
        If called while an asyncio event loop is running, or if the device
        cannot be resolved.
    ValueError
        If ``keypoint_names`` is empty or contains duplicates.
    """
    # Blocking work on the qasync loop would freeze the whole UI; mirror the
    # guard in glider.vision.video_tracking_runner.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_batch() blocks and must not run on the asyncio event loop; "
            "drive it from a QThread (see glider.gui.pose_batch.worker)."
        )

    names = list(keypoint_names)
    if not names:
        raise ValueError("keypoint_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("keypoint_names must be unique")

    if infer is None:
        from glider.vision.pose.core import infer_video

        infer = infer_video

    # Fail fast: a misconfigured CUDA install or a missing GPU under
    # require_gpu should abort before video one, not after an hour of work.
    from glider.vision.pose import device as device_mod

    device_mod.resolve_device(device, require_gpu=require_gpu)

    from glider.vision.pose.core import PoseCancelledError
    from glider.vision.pose.dlc import to_dlc_csv

    model_path = Path(model_path)
    result = BatchResult()
    total = len(videos)

    def emit(kind: EventKind, video: Path, index: int, **kwargs) -> None:
        if on_event is not None:
            on_event(BatchEvent(kind=kind, video=video, index=index, total=total, **kwargs))

    for index, raw_video in enumerate(videos):
        video = Path(raw_video).resolve()
        primary = dlc_output_path(video, model_path)

        if primary.exists() and not overwrite:
            result.skipped.append(video)
            emit(EventKind.SKIPPED, video, index, output=primary)
            continue

        if cancel_cb is not None and cancel_cb():
            result.cancelled = True
            emit(EventKind.CANCELLED, video, index)
            break

        emit(EventKind.STARTED, video, index)
        try:
            pose = infer(
                model_path=str(model_path),
                video_path=str(video),
                keypoint_names=names,
                conf=conf,
                device=device,
                require_gpu=require_gpu,
                progress=False,
                echo_device=False,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
            if filtering is not None:
                from glider.vision.pose.filtering import smooth

                to_dlc_csv(pose, raw_output_path(video, model_path))
                pose = smooth(
                    pose,
                    confidence_threshold=filtering.confidence_threshold,
                    max_gap=filtering.max_gap,
                    median_window=filtering.median_window,
                )
            # Written only after inference returns a complete PoseData, so a
            # cancelled or failed video never leaves a partial CSV behind.
            to_dlc_csv(pose, primary)
        except PoseCancelledError:
            result.cancelled = True
            emit(EventKind.CANCELLED, video, index)
            break
        except Exception as e:  # one bad video must not end the batch
            result.failed.append((video, str(e)))
            emit(EventKind.FAILED, video, index, message=str(e))
            continue

        result.completed.append(video)
        emit(EventKind.WROTE, video, index, output=primary)

    return result
