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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from glider.vision.arena import ArenaCalibration
    from glider.vision.arena_gate import ArenaGateSettings
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.tracks import PoseTracks
    from glider.vision.zones import ZoneConfiguration

logger = logging.getLogger(__name__)

#: Video containers both halves of the pipeline accept. Single-sourced on
#: purpose: this list used to be restated in the behavior tool with ``.webm``
#: in place of ``.wmv``, so a folder could offer a video to one tool and hide
#: it from the other with nothing on screen explaining the difference.
VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"})

#: Qt file-dialog filter over :data:`VIDEO_EXTS`, so pickers can't drift from
#: what discovery actually accepts.
VIDEO_FILTER = "Video files (" + " ".join(f"*{e}" for e in sorted(VIDEO_EXTS)) + ");;All files (*)"

#: Blanked share above which a video is called out rather than merely logged.
#: On the cohort this was built for the bad sessions ran 3-34%, so a session
#: past this is a tracking problem to look at, not a result to analyse.
_GATE_WARN = 0.10

__all__ = [
    "VIDEO_EXTS",
    "VIDEO_FILTER",
    "BatchEvent",
    "BatchResult",
    "EventKind",
    "FilterSettings",
    "animals_dir",
    "discover_videos",
    "dlc_output_path",
    "find_pose_csv",
    "find_pose_csvs",
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


def _score_zones(video: Path, pose, zones, keypoint: str) -> str:
    """Write zone CSVs for *video*, returning a warning string or "".

    Runs here rather than in a later pass because the track is already in hand:
    whether a point is inside a polygon needs no pixels, so the alternative is
    decoding every video a second time to learn something already known.

    Never raises. By this point the pose CSV is written and valid, and that is
    the artifact that matters - a zone that cannot be scored is a warning about
    the zone, not a failed video.
    """
    if not zones:
        return ""
    config = zones.get(video) or zones.get(Path(video))
    if config is None:
        return ""
    try:
        from glider.vision.zone_scoring import score_pose, write_zone_csvs, zone_output_dir

        resolution = None
        if pose.metadata:
            raw = pose.metadata.get("resolution")
            if raw:
                resolution = (int(raw[0]), int(raw[1]))
        scoring = score_pose(pose, config, resolution=resolution, keypoint=keypoint)
        write_zone_csvs(scoring, zone_output_dir(video))
    except Exception as e:
        logger.warning("could not score zones for %s: %s", video.name, e)
        return f"zones not scored: {e}"
    return ""


def _score_zones_multi(video: Path, tracks: PoseTracks, zones, keypoint: str) -> str:
    """Score every slot in *tracks* against this video's zones.

    Mirrors :func:`_score_zones` deliberately, one animal at a time: same
    guard clauses, same resolution lookup, same never-raises contract. By the
    time this runs, inference is done and the pose CSV is already written and
    valid -- a zone-config problem here is a warning about the zone, not a
    reason to fail a video that just cost an hour of GPU time.
    """
    if not zones:
        return ""
    config = zones.get(video) or zones.get(Path(video))
    if config is None:
        return ""
    try:
        from glider.vision.zone_scoring import score_pose, write_zone_csvs_multi, zone_output_dir

        resolution = None
        first_meta = tracks[0].metadata
        raw = first_meta.get("resolution") if first_meta else None
        if raw:
            resolution = (int(raw[0]), int(raw[1]))

        scorings = {
            f"animal{slot}": score_pose(
                tracks[slot],
                config,
                resolution=resolution,
                keypoint=keypoint,
                object_id=f"animal{slot}",
            )
            for slot in tracks
        }
        write_zone_csvs_multi(scorings, zone_output_dir(video))
    except Exception as e:
        logger.warning("could not score zones for %s: %s", video.name, e)
        return f"zones not scored: {e}"

    # One animal's tracking being far worse than the rest is worth saying out
    # loud: it usually means a slot that consolidation never really filled.
    worst = min(s.coverage for s in scorings.values())
    if worst < 0.5:
        return f"one animal carried a usable point on only {worst:.0%} of frames"
    return ""


def _drop_stale_ungated(primary: Path) -> None:
    """Remove an ``_ungated`` companion orphaned by the primary just written.

    ``gate_pose_csv`` makes that file by renaming the primary it is about to
    replace, and then always gates *it* rather than the primary, so that
    re-gating with different settings cannot compound one gate on another. The
    rule holds only while the companion is the original of the primary beside
    it. A fresh inference run breaks that: the companion becomes the original of
    a track no longer on disk, and the next post-hoc pass would gate it and
    write the previous run's coordinates over this one.

    Nothing the operator asked to keep is lost. The run this file belongs to is
    the one being overwritten, and the pristine original of the *new* run is the
    primary itself -- or its ``_raw`` companion when gating or filtering is on.

    Never raises: by here the pose CSV is written and valid, which is the
    artifact that matters, and a leftover companion is a warning rather than a
    failed video. ``gate_pose_csv`` refuses one it cannot account for anyway.
    """
    from glider.vision.arena_gate import ungated_path
    from glider.vision.pose.dlc import meta_path

    stale = ungated_path(primary)
    if not stale.exists():
        return
    for path in (stale, meta_path(stale)):
        try:
            path.unlink(missing_ok=True)
        except OSError as e:  # pragma: no cover - depends on filesystem state
            logger.warning("could not remove the superseded %s: %s", path.name, e)
    logger.info("%s was re-tracked, so the superseded %s was removed", primary.name, stale.name)


def _output_stem(video: Path, model: Path) -> str:
    """Shared stem, so the primary and raw names can never drift apart."""
    return f"{Path(video).stem}DLC_{Path(model).stem}"


def dlc_output_path(video: Path, model: Path) -> Path:
    """Primary DLC CSV path, written beside the video."""
    video = Path(video)
    return video.parent / f"{_output_stem(video, model)}.csv"


def animals_dir(pose_csv: Path | str) -> Path:
    """Where a multi-animal session's per-animal CSVs live.

    A subdirectory rather than a suffix, because both discovery paths must miss
    it: ``find_pose_csv`` globs the video's own directory flat, and the cohort
    collector (``gui/behavior/window.py``) rglobs but requires ``DLC_`` in the
    stem, which ``animal0.csv`` does not have. Those are two independent
    conditions in two files, and the tests pin both -- the comment on
    NOT_POSE_SUFFIXES records what happened last time they disagreed.
    """
    pose_csv = Path(pose_csv)
    return pose_csv.with_name(f"{pose_csv.stem}_animals")


def raw_output_path(video: Path, model: Path) -> Path:
    """Untouched companion CSV: what the model actually said.

    Written whenever gating or filtering is enabled, since both discard data
    and neither should be able to do so without a companion to compare
    against. Never written when the primary is already the raw inference.
    """
    video = Path(video)
    return video.parent / f"{_output_stem(video, model)}_raw.csv"


def _drop_stale_single(primary: Path, *, raw_is_current: bool) -> None:
    """Remove a stale single-animal ``primary`` before a multi-animal write.

    A video tracked single-animal and later multi-animal leaves ``primary`` on
    disk once :func:`_process_multi` starts writing :func:`animals_dir`
    instead -- nothing ever removes the old file. ``find_pose_csv`` globs for
    exactly that filename and would keep handing back the superseded
    single-animal track as *the* pose CSV for a video that has since been
    multi-tracked, with current coordinates sitting right next to it, unread.

    Removes ``primary`` and everything keyed off it that a single-animal run
    could have left: its rate sidecar, and its own ``_raw`` (when that run
    gated or filtered). ``raw_output_path`` is keyed by video and model alone,
    so it names the same file regardless of ``n_animals`` -- when *this*
    multi-animal run also gated or filtered, it has already overwritten
    ``_raw`` with its own data by the time this runs, and ``raw_is_current``
    must be True so that fresh file is left alone rather than deleted as if
    it were the old run's leftover.

    Never raises: by here :func:`_process_multi` has already written the new
    animals_dir CSVs, which are the artifact that matters, and a leftover
    primary is a warning rather than a failed video.
    """
    if not primary.exists():
        return
    from glider.vision.pose.dlc import meta_path

    candidates = [primary, meta_path(primary)]
    if not raw_is_current:
        raw = primary.with_name(f"{primary.stem}_raw.csv")
        candidates += [raw, meta_path(raw)]

    removed = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            path.unlink()
            removed.append(path.name)
        except OSError as e:  # pragma: no cover - depends on filesystem state
            logger.warning("could not remove the superseded %s: %s", path.name, e)
    if removed:
        logger.info(
            "%s was re-tracked multi-animal, so the superseded %s was removed",
            primary.name,
            ", ".join(removed),
        )


def _drop_stale_animals_dir(primary: Path) -> None:
    """Remove a stale ``animals_dir`` before a single-animal write of ``primary``.

    The reverse of :func:`_drop_stale_single`: a video tracked multi-animal and
    later single-animal leaves the old ``animals_dir`` sitting beside the fresh
    ``primary``. Nothing reads ``primary`` and ``animals_dir`` together today,
    but code that will (per-animal loading, keyed off ``animals_dir(primary)``
    existing) must not find a stale directory next to current data and treat
    the video as still multi-animal.

    Never raises: by here ``primary`` is written and valid, which is the
    artifact that matters, and a leftover directory is a warning rather than a
    failed video.
    """
    stale = animals_dir(primary)
    if not stale.exists():
        return
    import shutil

    try:
        shutil.rmtree(stale)
    except OSError as e:  # pragma: no cover - depends on filesystem state
        logger.warning("could not remove the superseded %s: %s", stale.name, e)
        return
    logger.info(
        "%s was re-tracked single-animal, so the superseded %s was removed",
        primary.name,
        stale.name,
    )


def _drop_orphaned_animal_slots(out_dir: Path, kept_slots: set[int]) -> None:
    """Remove per-animal files for slots the current run does not produce.

    The per-slot write loop only overwrites slots present in the new tracks,
    so rerunning a video with fewer animals than a previous run (3, then 2)
    leaves the extra slot's CSV behind -- the session then appears to still
    hold an animal that this run never saw. Whole-file overwrite of one
    ``primary`` could not do this; per-animal files can.

    Touches the per-animal pose CSVs, their own rate sidecars, and their own
    ``_ethogram.csv`` -- an orphaned slot's ethogram is behaviour scored from
    pose data that no longer exists, and leaving it behind is indistinguishable
    from a live animal's scores to anything that globs
    ``_animals/*_ethogram.csv``. A *kept* slot's ethogram is left alone even
    though it is now stale too: deleting a user's analysis output on a
    tracking run is a bigger call than deleting the pose file it was scored
    from, so that is on the user, not this cleanup. Never raises, for the
    same reason as every other reconciliation here: by the time this matters
    the new run's own files are about to be written and are the artifact
    that counts.
    """
    if not out_dir.is_dir():
        return
    from glider.vision.pose.dlc import meta_path

    for path in sorted(out_dir.glob("animal*.csv")):
        slot_id = path.stem.removeprefix("animal")
        if not slot_id.isdigit() or int(slot_id) in kept_slots:
            continue
        ethogram = out_dir / f"animal{slot_id}_ethogram.csv"
        for p in (path, meta_path(path), ethogram):
            try:
                p.unlink(missing_ok=True)
            except OSError as e:  # pragma: no cover - depends on filesystem state
                logger.warning("could not remove the orphaned %s: %s", p.name, e)
        logger.info(
            "%s no longer tracks animal%s, so the superseded %s was removed",
            out_dir.name,
            slot_id,
            path.name,
        )


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
        are not pose data are skipped (see
        :data:`~glider.vision.pose.dlc.NOT_POSE_SUFFIXES`): ``_raw`` is the
        ungated, unsmoothed inference, never the one to analyze, ``_ungated``
        is its post-hoc equivalent, and ``_annotations`` is the annotator's
        behavior zones. The last is written *after* the pose CSV, so before it
        was excluded it won the most-recent tie-break and every reader
        downstream — annotate, train, apply — was handed a file of labels in
        place of keypoints.

    The exact-stem form wins when both exist, so a file the operator placed
    deliberately is never shadowed by a batch run. When several models have
    been run over one video the most recently written wins, and the choice is
    logged: alphabetical order would silently prefer ``exp-5`` over ``exp-7``,
    quietly scoring a cohort with a superseded pose model. Naming the file
    explicitly is still the only way to be certain.

    A multi-animal session -- :func:`animals_dir` full of per-animal CSVs
    instead of one primary -- has no single file to hand back, so this
    returns ``None`` for it too, exactly as if nothing had been tracked.
    That is deliberate, not an oversight: raising here would abort every
    scan built on a comprehension over this function (several exist, across
    seven files), for the sake of one session among many. The log line is
    the only place the reason surfaces, so it names the video, the animal
    count, and the directory. When several models have each left their own
    ``_animals`` directory beside one video -- ``dlc_output_path`` is keyed
    on (video, model), so nothing stops that -- the same most-recent rule
    used for flat CSVs below picks between them. Callers that want the
    per-animal set should use :func:`find_pose_csvs`.

    Reconciliation (``_drop_stale_single``, ``_drop_stale_animals_dir``) is
    keyed on (video, model), not on video alone, so an ``_animals`` directory
    from one model and a flat CSV from another can legitimately coexist --
    one does not supersede the other just because it exists. So the
    directory does not win outright: it is compared against the newest flat
    match by the same "newest wins" rule as everything else here, and only
    returns ``None`` when it is actually the newer of the two.
    """
    # Imported here, not at module scope: dlc imports pandas, and this module
    # stays cheap to import because the GUI does so while building menus.
    from glider.vision.pose.dlc import NOT_POSE_SUFFIXES

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
        if not p.stem.endswith(NOT_POSE_SUFFIXES)
    ]

    # A multi-animal directory does not automatically outrank a flat CSV:
    # reconciliation is keyed on (video, model), so a newer single-animal
    # run of a *different* model can legitimately sit beside an older
    # _animals directory. Only the newer of the two wins.
    #
    # The directory's own mtime is not that measure: on APFS a directory's
    # mtime only moves when an entry is added or removed, not when a file
    # already inside it is overwritten in place -- which is exactly what a
    # same-model re-track does to animal0.csv, animal1.csv, etc. Rank by the
    # newest file inside the directory instead.
    animal_dir = _pick_animals_dir(directory, video)
    if animal_dir is not None:
        animal_csvs = _animal_csvs(animal_dir)
        animal_mtime = max(_mtime(p) for p in animal_csvs)
        if not matches or animal_mtime > _mtime(max(matches, key=_mtime)):
            logger.info(
                "%s is a multi-animal session (%d animals) tracked in %s; "
                "find_pose_csv has no single file to return -- use "
                "find_pose_csvs for the per-animal paths",
                video.name,
                len(animal_csvs),
                animal_dir,
            )
            return None

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    chosen = max(matches, key=_mtime)
    logger.info(
        "%s has %d pose CSVs (%s); using the most recent, %s",
        video.name,
        len(matches),
        ", ".join(p.name for p in matches),
        chosen.name,
    )
    return chosen


# Newest wins. mtime can be unreadable on a share mid-copy; those sort last
# rather than raising, so a transient stat error cannot pick the file. Shared
# by the flat-CSV tie-break above and the _animals-directory tie-break below,
# so both branches age off the same clock.
def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return float("-inf")


def _animal_csvs(animal_dir: Path) -> list[Path]:
    """Per-animal CSVs in *animal_dir*, in numeric slot order.

    ``sorted()`` on the filenames would put ``animal10`` before ``animal2``
    -- the slot is an integer, not a string, so it is sorted as one.
    """
    slots = []
    for path in animal_dir.glob("animal*.csv"):
        slot_id = path.stem.removeprefix("animal")
        if slot_id.isdigit():
            slots.append((int(slot_id), path))
    return [path for _, path in sorted(slots)]


def _pick_animals_dir(directory: Path, video: Path) -> Path | None:
    """The ``_animals`` directory for *video* in *directory*, or ``None``.

    Mirrors :func:`find_pose_csv`'s flat-CSV tie-break: several models can
    each leave their own ``_animals`` directory beside the same video (see
    the note on :func:`find_pose_csv`), and picking the alphabetically first
    one would silently prefer an older, superseded model's animals over a
    newer model's -- the same failure the flat-CSV tie-break exists to avoid.
    The most recently modified directory wins, logged the same way, only
    when there is more than one candidate to choose between.
    """
    candidates = [
        d for d in sorted(directory.glob(f"{video.stem}DLC_*_animals")) if _animal_csvs(d)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    chosen = max(candidates, key=_mtime)
    logger.info(
        "%s has %d multi-animal directories (%s); using the most recent, %s",
        video.name,
        len(candidates),
        ", ".join(d.name for d in candidates),
        chosen.name,
    )
    return chosen


def find_pose_csvs(video: Path | str, search_dir: Path | str | None = None) -> list[Path]:
    """Every animal's pose CSV for *video*, in numeric slot order.

    The counterpart to :func:`find_pose_csv` for callers that want the whole
    set rather than one file: empty for a single-animal session (or no
    session at all), since those have no :func:`animals_dir` to list. When
    more than one model has left an ``_animals`` directory beside *video*,
    the same most-recently-modified tie-break :func:`find_pose_csv` uses
    applies here too -- see :func:`_pick_animals_dir`.
    """
    video = Path(video)
    directory = Path(search_dir) if search_dir is not None else video.parent
    if not directory.is_dir():
        return []

    animal_dir = _pick_animals_dir(directory, video)
    return _animal_csvs(animal_dir) if animal_dir is not None else []


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
    zones: Mapping[Path, ZoneConfiguration] | None = None,
    zone_keypoint: str = "body_center",
    arenas: Mapping[Path, ArenaCalibration] | None = None,
    gate: ArenaGateSettings | None = None,
    n_animals: int = 1,
    max_travel_px_per_frame: float = 40.0,
    min_fragment_frames: int = 5,
    identity_min_separation_px: float = 60.0,
    infer_tracks: Callable[..., PoseTracks] | None = None,
) -> BatchResult:
    """Run a pose model over ``videos``, writing a DLC CSV beside each one.

    Videos are processed sequentially: inference is GPU-bound, so concurrent
    videos would contend rather than go faster.

    Parameters
    ----------
    overwrite
        When False (the default) a video whose output already exists is
        skipped, so an interrupted batch resumes cheaply. "Output" means the
        primary CSV for ``n_animals`` 1, or a non-empty ``animals_dir`` above
        that -- so resume, and the skip itself, follow whichever shape this
        call would actually produce.
    filtering
        When given, the unfiltered result is written to the ``_raw`` path first
        and the smoothed result becomes the primary CSV — an unhappy filter
        setting can never destroy the inference run.
    arenas, gate
        Both are needed to gate: an arena for the video *and* gate settings.
        Detections that left that arena are blanked before the primary CSV is
        written and before zones are scored, and the ``_raw`` companion holds
        the ungated inference. A video with no arena is untouched.
    infer
        Injection point for tests; defaults to
        :func:`glider.vision.pose.core.infer_video`.
    n_animals
        1 (the default) takes the original single-animal path unchanged, so
        every project already on disk keeps reading exactly as it did. Above
        1, each video is tracked into N animals and written as one three-row
        DLC CSV *per animal* under :func:`animals_dir`, plus one identity
        sidecar for the session. No four-row file: that is an opt-in export
        now, regenerated from the per-animal files so it cannot drift from
        them.
    max_travel_px_per_frame, min_fragment_frames
        Consolidation knobs, forwarded to
        :func:`glider.vision.pose.core.infer_video_tracks`. Unused when
        ``n_animals`` is 1.
    identity_min_separation_px
        Forwarded to :func:`glider.vision.pose.identity.identity_flags`.
        Unused when ``n_animals`` is 1.
    infer_tracks
        Injection point for tests; defaults to
        :func:`glider.vision.pose.core.infer_video_tracks`. Unused when
        ``n_animals`` is 1.

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

    if infer_tracks is None:
        from glider.vision.pose.core import infer_video_tracks

        infer_tracks = infer_video_tracks

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

        # What *this* run would produce: animals_dir for n_animals > 1,
        # primary otherwise. Testing primary alone for n_animals > 1 never
        # matches -- _process_multi never writes it -- so a multi-animal
        # video always looked unfinished (resume never skipped) and, worse,
        # switching a video from single- to multi-animal with the default
        # overwrite=False looked finished from the first frame (primary
        # already existed) and was skipped outright, reporting success while
        # writing nothing.
        #
        # Multi: `any(...)` is satisfied by one file, so a batch killed
        # between slot writes leaves resume calling a half-written directory
        # done. `== n_animals` via `_animal_csvs` (which already excludes
        # `*_ethogram.csv` and other non-slot names) requires every slot this
        # run wants, and also makes a 3-animal-then-2 rerun re-run rather
        # than resume-skip on the two leftover slots.
        #
        # Single: `primary.exists()` alone is satisfied by the four-row
        # export, which deliberately reclaims that path (see
        # `export_actions.export_target`). `not _animal_csvs(animals_dir(primary))`
        # excludes that case, so re-tracking single-animal after an export
        # still runs rather than treating the export as this run's output.
        # A raw `.exists()` on the directory would also be defeated by a
        # leftover *empty* `_animals/` (failed cleanup, or a folder a user
        # made) -- same reason `_pick_animals_dir` requires actual per-animal
        # CSVs rather than just a directory.
        this_run_output = animals_dir(primary) if n_animals > 1 else primary
        already_done = (
            len(_animal_csvs(this_run_output)) == n_animals
            if n_animals > 1
            else primary.exists() and not _animal_csvs(animals_dir(primary))
        )

        if already_done and not overwrite:
            result.skipped.append(video)
            emit(EventKind.SKIPPED, video, index, output=this_run_output)
            continue

        if cancel_cb is not None and cancel_cb():
            result.cancelled = True
            emit(EventKind.CANCELLED, video, index)
            break

        emit(EventKind.STARTED, video, index)
        # Per video, not before the loop: a warning assigned only inside its
        # own branch below would raise NameError on every clean video, and one
        # carried over from the previous video would label this one with a
        # problem it does not have.
        zone_warning = ""
        gate_warning = ""
        try:
            # Resolved before inference, not after: the arena also re-ranks
            # multi-detection frames inside infer_video, and a detection
            # discarded there can never be recovered by the post-hoc gate.
            arena = (arenas or {}).get(video)
            if n_animals > 1:
                zone_warning = _process_multi(
                    video=video,
                    model_path=model_path,
                    names=names,
                    primary=primary,
                    infer_tracks=infer_tracks,
                    conf=conf,
                    device=device,
                    require_gpu=require_gpu,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                    arena=arena,
                    gate=gate,
                    filtering=filtering,
                    zones=zones,
                    zone_keypoint=zone_keypoint,
                    n_animals=n_animals,
                    max_travel_px_per_frame=max_travel_px_per_frame,
                    min_fragment_frames=min_fragment_frames,
                    identity_min_separation_px=identity_min_separation_px,
                )
            else:
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
                    arena=arena,
                    gate_settings=gate,
                )
                gating = gate is not None and arena is not None

                # _raw is the "what did the model actually say" file, so it
                # must be pre-gate as well as pre-filter — and it must exist
                # whenever either is active, or gating discards data with no
                # companion.
                if gating or filtering is not None:
                    to_dlc_csv(pose, raw_output_path(video, model_path))

                if gating:
                    from glider.vision.arena_gate import gate_to_arena

                    try:
                        pose, report = gate_to_arena(pose, arena, settings=gate)
                    except ValueError as e:
                        # DegenerateArenaError subclasses ValueError, so this
                        # covers both. Mirrors _score_zones: by here the
                        # inference is done and valid, and that is what
                        # matters.
                        logger.warning("could not gate %s: %s", video.name, e)
                    else:
                        pose.metadata["arena_gate"] = {**asdict(report), "gated": True}
                        if report.blanked_fraction > _GATE_WARN:
                            gate_warning = (
                                f"gate blanked {report.blanked_fraction:.1%} of "
                                f"{report.frames_considered} tracked frames"
                            )

                if filtering is not None:
                    from glider.vision.pose.filtering import smooth

                    pose = smooth(
                        pose,
                        confidence_threshold=filtering.confidence_threshold,
                        max_gap=filtering.max_gap,
                        median_window=filtering.median_window,
                    )
                # Written only after inference returns a complete PoseData,
                # so a cancelled or failed video never leaves a partial CSV
                # behind.
                to_dlc_csv(pose, primary)
                _drop_stale_ungated(primary)
                _drop_stale_animals_dir(primary)
                zone_warning = _score_zones(video, pose, zones, zone_keypoint)
        except PoseCancelledError:
            result.cancelled = True
            emit(EventKind.CANCELLED, video, index)
            break
        except Exception as e:  # one bad video must not end the batch
            result.failed.append((video, str(e)))
            emit(EventKind.FAILED, video, index, message=str(e))
            continue

        result.completed.append(video)
        emit(
            EventKind.WROTE,
            video,
            index,
            output=this_run_output,
            message="; ".join(w for w in (zone_warning, gate_warning) if w),
        )

    return result


def _process_multi(
    *,
    video: Path,
    model_path: Path,
    names: list[str],
    primary: Path,
    infer_tracks,
    conf: float,
    device: str | None,
    require_gpu: bool,
    progress_cb,
    cancel_cb,
    arena,
    gate,
    filtering: FilterSettings | None,
    zones,
    zone_keypoint: str,
    n_animals: int,
    max_travel_px_per_frame: float,
    min_fragment_frames: int,
    identity_min_separation_px: float,
) -> str:
    """The multi-animal half of one video. Returns a warning string, or "".

    Every per-animal stage is the *same* function the single-animal path
    calls, once per slot. That is the whole point of keeping ``PoseData``
    single-animal: gating, filtering and zone scoring are already correct for
    one animal and are reused rather than re-derived.
    """
    from glider.vision.pose.dlc import to_dlc_csv, to_dlc_csv_multi
    from glider.vision.pose.identity import (
        identity_flags,
        identity_output_path,
        write_identity_csv,
    )
    from glider.vision.pose.tracks import PoseTracks

    tracks = infer_tracks(
        model_path=str(model_path),
        video_path=str(video),
        keypoint_names=names,
        n_animals=n_animals,
        conf=conf,
        device=device,
        require_gpu=require_gpu,
        progress=False,
        echo_device=False,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        arena=arena,
        gate_settings=gate,
        max_travel_px_per_frame=max_travel_px_per_frame,
        min_fragment_frames=min_fragment_frames,
    )
    gating = gate is not None and arena is not None

    # _raw is the "what did the model actually say" file, so it must be
    # pre-gate as well as pre-filter, mirroring the single-animal path.
    if gating or filtering is not None:
        to_dlc_csv_multi(tracks, raw_output_path(video, model_path))

    warnings_out: list[str] = []
    if gating:
        from glider.vision.arena_gate import gate_to_arena

        gated = {}
        worst = 0.0
        for slot in tracks:
            try:
                pose, report = gate_to_arena(tracks[slot], arena, settings=gate)
            except ValueError as e:
                logger.warning("could not gate %s animal%d: %s", video.name, slot, e)
                gated[slot] = tracks[slot]
                continue
            pose.metadata["arena_gate"] = {**asdict(report), "gated": True}
            gated[slot] = pose
            worst = max(worst, report.blanked_fraction)
        tracks = PoseTracks(tracks=gated, fps=tracks.fps, metadata=tracks.metadata)
        if worst > _GATE_WARN:
            warnings_out.append(f"gate blanked up to {worst:.1%} of one animal's tracked frames")

    if filtering is not None:
        from glider.vision.pose.filtering import smooth

        tracks = PoseTracks(
            tracks={
                slot: smooth(
                    tracks[slot],
                    confidence_threshold=filtering.confidence_threshold,
                    max_gap=filtering.max_gap,
                    median_window=filtering.median_window,
                )
                for slot in tracks
            },
            fps=tracks.fps,
            metadata=tracks.metadata,
        )

    # One three-row file per slot, via the same writer the single-animal path
    # uses -- no new format, and every existing from_dlc_csv caller can open
    # one animal unmodified. `primary` is never written; it survives only as
    # the naming anchor for animals_dir, the _raw companion, and the identity
    # sidecar below.
    out_dir = animals_dir(primary)
    for slot in tracks:
        to_dlc_csv(tracks[slot], out_dir / f"animal{slot}.csv")
    # After the writes, not before, matching _drop_stale_ungated and the two
    # cleanups beside it: write the replacement, then delete what it supersedes.
    # The slots removed here are disjoint from the slots just written, so the
    # order is free -- and cleaning first would mean a write that died partway
    # left the orphan already deleted and the kept slots still holding the
    # previous run's data, which is worse than leaving the video untouched.
    _drop_orphaned_animal_slots(out_dir, set(tracks))
    _drop_stale_single(primary, raw_is_current=gating or filtering is not None)

    stitched = {int(s): set(f) for s, f in (tracks.metadata.get("stitched") or {}).items()}
    write_identity_csv(
        identity_output_path(primary),
        identity_flags(tracks, stitched=stitched, min_separation_px=identity_min_separation_px),
    )

    zone_warning = _score_zones_multi(video, tracks, zones, zone_keypoint)
    if zone_warning:
        warnings_out.append(zone_warning)
    return "; ".join(warnings_out)
