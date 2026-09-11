"""Multi-animal DLC CSV export for the Batch Pose Tracking window.

Qt-free for the same reason as ``arena_actions.py``: this is a session-scoped
walk over a set of videos with no window state, so it is testable without
building a window.

The four-row file is derived, never a second source of truth: it is rebuilt
from the per-animal CSVs on every export, so it cannot drift from them the
way a parallel primary could. See ``docs/superpowers/specs/`` D2 spec, §3.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def exportable(videos: Iterable[Path]) -> list[Path]:
    """Those of *videos* with more than one animal tracked -- something to
    combine into a four-row file.

    A single-animal session (or a video that has not been tracked at all)
    has no ``_animals`` directory, so ``find_pose_csvs`` returns at most one
    path for it and there is nothing to export.
    """
    from glider.vision.pose.batch import find_pose_csvs

    return [v for v in videos if len(find_pose_csvs(v)) > 1]


def export_target(csvs: list[Path]) -> Path:
    """Where the four-row file for these per-animal *csvs* lands.

    ``animals_dir(pose_csv)`` names the directory by appending ``_animals``
    to the primary CSV's stem, and that primary is never written once a
    session goes multi-animal -- it survives only as that naming anchor (see
    ``_process_multi`` in ``vision/pose/batch.py``). So the name is free, and
    reclaiming it is simpler and more robust than requiring the model path
    that produced these files: an archived cohort may not have that model
    selected in this window at all (the re-gate button next to this one
    exists for exactly that reason).
    """
    animal_dir = csvs[0].parent
    primary_name = animal_dir.name.removesuffix("_animals") + ".csv"
    return animal_dir.with_name(primary_name)


def export_session(video: Path, csvs: list[Path]) -> Path:
    """Read *csvs* (one video's per-animal files) into one ``PoseTracks`` and
    write the four-row DLC CSV. Returns the path written.

    Raises ``ValueError`` if the per-animal files disagree -- different frame
    counts or keypoint names -- since ``PoseTracks`` validates exactly that on
    construction. That is not papered over here: files that disagree mean
    something is wrong upstream of this export, not in it.
    """
    from glider.vision.pose.dlc import from_dlc_csv, to_dlc_csv_multi
    from glider.vision.pose.tracks import PoseTracks

    slots = {int(p.stem.removeprefix("animal")): from_dlc_csv(p) for p in csvs}
    # Every per-animal file was written from one shared video fps
    # (`_process_multi` passes the same `tracks.fps` to each slot), so any one
    # of them names it; PoseTracks itself does not compare the per-track fps
    # values, only the container's.
    fps = slots[min(slots)].fps
    tracks = PoseTracks(tracks=slots, fps=fps)
    return to_dlc_csv_multi(tracks, export_target(csvs))


def export_sessions(
    videos: Iterable[Path],
    *,
    on_log=None,
    on_progress=None,
) -> tuple[int, int]:
    """Export every multi-animal session in *videos*. Returns ``(exported, skipped)``.

    Never raises for one video, mirroring ``arena_actions.regate_videos``: a
    single-animal session is a plain skip, and per-animal files that disagree
    (``ValueError`` from ``PoseTracks``) or cannot be read (``OSError``) are
    reported and skipped rather than aborting the rest of the cohort.
    """
    from glider.vision.pose.batch import find_pose_csvs

    videos = list(videos)
    exported = skipped = 0
    for index, video in enumerate(videos):
        csvs = find_pose_csvs(video)
        if len(csvs) < 2:
            skipped += 1
            if on_log:
                on_log(f"{video.name}: single-animal session, nothing to export")
        else:
            try:
                path = export_session(video, csvs)
            except (ValueError, OSError) as e:
                skipped += 1
                if on_log:
                    on_log(f"{video.name}: skipped ({e})")
            else:
                exported += 1
                if on_log:
                    on_log(f"{video.name}: wrote {path.name}")
        if on_progress:
            on_progress(index + 1, len(videos))
    return exported, skipped
