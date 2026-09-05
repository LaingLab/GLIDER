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
