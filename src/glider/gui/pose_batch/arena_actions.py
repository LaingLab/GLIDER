"""Arena actions for the Batch Pose Tracking window.

Split out of ``window.py`` for the same reason ``calibration_table.py`` was:
that file is over 1200 lines and these are self-contained operations on a
:class:`CalibrationSet` that need no window state. Keeping them here also
makes them testable without building a window.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from glider.vision.arena import ArenaCalibration
from glider.vision.calibration_set import CalibrationSet

logger = logging.getLogger(__name__)


def resolution_of(video: Path) -> tuple[int, int] | None:
    """``(width, height)`` of *video*, or None when it will not open.

    Mirrors what ``_retarget_calibration`` does inline for lines. Returning
    None rather than a default is deliberate: guessing a resolution is exactly
    the error this retargeting exists to prevent.
    """
    from glider.vision.video_source import VideoFileSource

    reader = VideoFileSource()
    try:
        if not reader.load(video):
            return None
        width, height = reader.resolution
    except Exception as e:
        # Broad on purpose: a file the decoder chokes on must be skipped, not
        # allowed to abort a copy across the rest of the selection.
        logger.info("cannot read the size of %s: %s", video, e)
        return None
    finally:
        reader.release()
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def copy_arena_to(
    calibrations: CalibrationSet, source: Path, targets: Iterable[Path]
) -> list[Path]:
    """Stamp *source*'s corners onto *targets*, unconfirmed. Returns skips.

    Corners are normalized, so they carry across resolutions -- but
    ``frame_size`` must follow the target or ``px_per_cm_at`` reports the
    source's scale for a video that does not have it, which is the same error
    ``_retarget_calibration`` exists to prevent for lines.

    Unconfirmed on purpose. ``residuals()`` is computed from the corners alone,
    so a copy that does not fit this video's floor produces no warning at all;
    on the TRH cohort the camera height varied per animal, which is precisely
    the error the arena was drawn to eliminate.
    """
    arena = calibrations.get_arena(source)
    if arena is None:
        return []
    skipped: list[Path] = []
    for target in targets:
        resolution = resolution_of(target)
        if resolution is None:
            skipped.append(target)
            continue
        calibrations.set_arena(
            target,
            ArenaCalibration(
                corners=list(arena.corners),
                width_cm=arena.width_cm,
                height_cm=arena.height_cm,
                frame_size=resolution,
            ),
            confirmed=False,
        )
    return skipped


def regatable(videos: Iterable[Path]) -> list[Path]:
    """Those of *videos* that already have a pose CSV to re-gate.

    The re-gate pass is for tracks that exist on disk; a cohort that has not
    been tracked yet wants Run, not this.
    """
    from glider.vision.pose.batch import find_pose_csv

    return [video for video in videos if find_pose_csv(video) is not None]


def regate_videos(
    videos: Iterable[Path],
    calibrations: CalibrationSet,
    *,
    settings=None,
    on_log=None,
    on_progress=None,
) -> tuple[int, int]:
    """Re-gate each video's pose CSV in place. Returns ``(gated, skipped)``.

    Never raises for one video. A refusal (``ValueError``) and an unreadable
    file (``OSError``) are both skips with a logged reason: this is a batch
    maintenance pass over a whole cohort, and stopping at the first awkward
    session would leave the folder half-converted with no record of where.
    """
    from glider.vision.arena_gate import gate_pose_csv
    from glider.vision.pose.batch import find_pose_csv

    videos = list(videos)
    gated = skipped = 0
    for index, video in enumerate(videos):
        arena = calibrations.get_arena(video)
        csv = find_pose_csv(video)
        if arena is None or csv is None:
            skipped += 1
            if on_log:
                reason = "no arena drawn" if arena is None else "no pose CSV"
                on_log(f"{video.name}: skipped ({reason})")
        else:
            try:
                report = gate_pose_csv(csv, arena, settings=settings)
            except (ValueError, OSError) as e:
                skipped += 1
                if on_log:
                    on_log(f"{video.name}: skipped ({e})")
            else:
                gated += 1
                if on_log:
                    on_log(f"{video.name}: blanked {report.blanked_fraction:.1%}")
        if on_progress:
            on_progress(index + 1, len(videos))
    return gated, skipped
