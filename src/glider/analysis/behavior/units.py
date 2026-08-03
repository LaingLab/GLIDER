"""Real-world units for the behavior model's kinematic speed thresholds.

The hybrid model's freeze/dart thresholds are stored in whatever units the
feature pipeline produced. By default that is **body-lengths per frame**:
:mod:`glider.analysis.behavior.features` divides keypoint speed by the animal's
per-frame body length so the same behavior in a small and a large mouse yields
the same feature value. With ``normalize_by_body_length=False`` the thresholds
are instead raw **pixels per frame**. Neither is a number you can put in a paper.

This module converts them. Three units, in increasing order of what they cost
you to obtain:

* ``per_second``   -- the native unit per second. Needs only the frame rate.
* ``px_per_frame`` -- needs a reference body length in pixels (the session
  median), and is a no-op when the features were never normalized.
* ``mm_per_s``     -- additionally needs the pixel-to-millimetre scale, read
  from the master calibration file the Batch Pose Tracking tool writes.

Every conversion returns ``None`` rather than guessing when an input is
missing, so an uncalibrated session still reports the units it can and simply
omits millimetres.

**The millimetre figure is approximate.** Body length varies frame to frame
(posture, foreshortening, tracking noise), so scaling a normalized threshold
back to absolute units uses the session's *median* body length as a single
reference. Report it alongside the value -- :func:`describe_speed_threshold`
returns the references it used for exactly that reason.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SpeedScale",
    "describe_speed_threshold",
    "load_px_per_mm",
    "median_body_length_px",
    "mm_per_s_to_px_per_frame",
]

# The feature column carrying absolute body length, before and after windowing.
_BODY_LENGTH_COLUMNS = ("body_length", "body_length__mean")


def _finite(value: float | None) -> float | None:
    """The value if it is a usable finite number, else None."""
    if value is None:
        return None
    v = float(value)
    return None if math.isnan(v) or math.isinf(v) else v


@dataclass(frozen=True)
class SpeedScale:
    """Converts one model's speed threshold into real-world units.

    Parameters
    ----------
    fps
        Frame rate the model was trained at (``BehaviorModel.fps``).
    body_length_px
        Session median body length in pixels, the reference for undoing the
        body-length normalization. Ignored when ``normalized`` is False.
    px_per_mm
        Pixel-to-millimetre scale for the source video, from the master
        calibration file. None when the session was never calibrated.
    normalized
        Whether the speeds are body-lengths/frame (the feature pipeline's
        default) or already pixels/frame. Mirrors
        ``FeatureSpec.normalize_by_body_length``.
    """

    fps: float
    body_length_px: float | None = None
    px_per_mm: float | None = None
    normalized: bool = True

    @property
    def native_unit(self) -> str:
        """What the stored threshold is measured in."""
        return "bl/frame" if self.normalized else "px/frame"

    def to_per_second(self, speed: float | None) -> float | None:
        """Native unit per second. The only conversion needing no rig measurement."""
        s = _finite(speed)
        fps = _finite(self.fps)
        if s is None or not fps or fps <= 0:
            return None
        return s * fps

    def to_px_per_frame(self, speed: float | None) -> float | None:
        """Pixels per frame, undoing the body-length normalization if present."""
        s = _finite(speed)
        if s is None:
            return None
        if not self.normalized:
            return s  # already pixels; no reference length involved
        length = _finite(self.body_length_px)
        if not length or length <= 0:
            return None
        return s * length

    def to_mm_per_s(self, speed: float | None) -> float | None:
        """Millimetres per second. Approximate — see the module docstring."""
        px = self.to_px_per_frame(speed)
        fps = _finite(self.fps)
        ppm = _finite(self.px_per_mm)
        if px is None or not fps or fps <= 0 or not ppm or ppm <= 0:
            return None
        return px * fps / ppm


def mm_per_s_to_px_per_frame(
    mm_per_s: float | None, *, px_per_mm: float | None, fps: float | None
) -> float | None:
    """Millimetres per second -> pixels per frame, for the live speed detector.

    Exact, unlike the hybrid prior's conversion: the live
    :class:`~glider.analysis.behavior.classify.speed_state.CausalSpeed` measures
    raw pixel displacement with no body-length normalization, so no reference
    body length is involved and nothing is approximated.

    Returns None when any input is missing or degenerate — callers that need a
    hard failure should check and raise with their own context.
    """
    speed = _finite(mm_per_s)
    ppm = _finite(px_per_mm)
    rate = _finite(fps)
    if speed is None or not ppm or ppm <= 0 or not rate or rate <= 0:
        return None
    return speed * ppm / rate


def median_body_length_px(frame: pd.DataFrame) -> float | None:
    """Session median body length in pixels, or None if unavailable.

    Prefers the raw ``body_length`` feature; falls back to the windowed
    ``body_length__mean``. Returns None when the model was trained with
    ``include_body_length=False``, which drops the column entirely — in that
    case there is no reference length recoverable from the features alone.
    """
    for column in _BODY_LENGTH_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        median = float(values.median())
        if median > 0:
            return median
    return None


def describe_speed_threshold(
    name: str, value: float | None, scale: SpeedScale
) -> dict[str, float | str | None]:
    """One threshold expressed in every unit the scale can reach.

    The reference values (``body_length_px``, ``px_per_mm``, ``fps``) are
    included so a reader can audit the millimetre figure rather than take it on
    faith. ``value`` is None when the prior was never calibrated.
    """
    return {
        "name": name,
        "native": _finite(value),
        "native_unit": scale.native_unit,
        "per_second": scale.to_per_second(value),
        "px_per_frame": scale.to_px_per_frame(value),
        "mm_per_s": scale.to_mm_per_s(value),
        "body_length_px": _finite(scale.body_length_px),
        "px_per_mm": _finite(scale.px_per_mm),
        "fps": _finite(scale.fps),
    }


def _folder_name(path: Path | str) -> str:
    """Lowercased name of the folder containing ``path``, separator-agnostic.

    ``Path(...).parent.name`` is not enough. A calibration master written on
    Windows stores keys like ``\\\\host\\share\\Sessions\\a.avi``; parsed on
    Linux that is a single component, because pathlib does not treat a
    backslash as a separator there. ``parent.name`` then returns ``""`` and
    every folder comparison below fails, so a scale that should have been
    borrowed silently never is — on CI and on any Linux analysis box, while
    passing on the Windows machine the file came from.

    Returns ``""`` for a bare filename, which matches nothing.
    """
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    return parts[-2].lower() if len(parts) >= 2 else ""


def _borrow_from_folder_mate(cal_set, video: Path) -> float | None:
    """Scale from another calibrated video sitting in the same folder.

    Videos in one folder are almost always one rig, one session, one camera
    height — so requiring every single one to be drawn individually is a tax
    with no scientific payoff. Borrowing is still gated hard: the folder-mate's
    calibration resolution must equal this video's actual resolution, and any
    disagreement between mates refuses rather than picks. Every borrow is
    logged with the video it came from.

    Folders are matched on name, not full path: the same share is routinely
    addressed as both ``Z:\\...`` and ``\\\\host\\share\\...``, and those never
    compare equal.
    """
    from glider.vision.video_source import video_resolution

    resolution = video_resolution(video)
    if resolution is None:
        # Without the true resolution the scale cannot be validated, and an
        # unvalidated borrow is exactly the wrong-millimetres bug.
        logger.info("cannot borrow a calibration for %s: its resolution is unreadable", video)
        return None

    folder = _folder_name(video)
    candidates = []
    for stored, calibration in cal_set.entries.items():
        if not folder or _folder_name(stored) != folder:
            continue
        if (calibration.calibration_width, calibration.calibration_height) != resolution:
            continue
        ppm = calibration.pixels_per_mm
        if ppm > 0:
            candidates.append((stored, ppm))

    if not candidates:
        return None
    distinct = {round(ppm, 6) for _, ppm in candidates}
    if len(distinct) > 1:
        logger.warning(
            "not borrowing a calibration for %s: folder-mates disagree on scale (%s px/mm)",
            video.name,
            ", ".join(f"{v:.4f}" for v in sorted(distinct)),
        )
        return None

    source, ppm = candidates[0]
    logger.info(
        "%s has no calibration of its own; borrowed %.4f px/mm from %s "
        "(same folder, same %dx%d resolution)",
        video.name,
        ppm,
        source.name,
        resolution[0],
        resolution[1],
    )
    return ppm


def load_px_per_mm(
    master_path: Path | str | None,
    video: Path | str,
    *,
    allow_folder_fallback: bool = True,
) -> float | None:
    """Pixel-to-millimetre scale for *video* from a master calibration file.

    Falls back to a folder-mate's calibration when *video* has none of its own
    and the resolutions match — see :func:`_borrow_from_folder_mate`. Pass
    ``allow_folder_fallback=False`` to require an exact per-video entry.

    Tolerant by design: a missing, unreadable, or malformed file, or a video the
    file does not cover, all yield None. An uncalibrated session must still be
    analysable — it simply cannot report millimetres.
    """
    if master_path is None:
        return None
    path = Path(master_path)
    # Imported lazily: this keeps `units` usable in a notebook that never
    # touches the vision stack.
    from glider.vision.calibration_set import CalibrationSet, CalibrationSetError

    try:
        loaded = CalibrationSet.load(path, known_videos=[Path(video)])
    except (CalibrationSetError, OSError) as e:
        logger.info("no usable calibration in %s: %s", path, e)
        return None

    direct = loaded.px_per_mm(video)
    if direct is not None:
        return direct
    if not allow_folder_fallback:
        return None
    return _borrow_from_folder_mate(loaded, Path(video))
