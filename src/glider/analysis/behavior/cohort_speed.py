"""Freeze/dart thresholds derived once for a whole cohort.

Per-video percentiles are circular for a treatment study: each animal gets
thresholds from its own speed distribution, so an animal that moves less
receives a *lower* darting threshold and the effect being measured is
normalised away. Comparability requires one set of cut-offs, derived from the
pooled cohort and applied unchanged to every session.

Speeds are pooled as raw samples rather than by averaging per-video
thresholds: the percentile of a pooled distribution is the quantity of
interest, and averaging quantiles of unequal-length sessions is not it.

Units matter when pooling. Pixels per frame are only comparable across
sessions that share a resolution and camera height, so sessions are converted
to cm/s wherever a pixel scale is available. If any session lacks one the
whole pool falls back to px/frame and says so — mixing the two silently would
be worse than either.

Works from pose CSVs, so a cohort that has already been tracked never needs
re-running. :class:`CausalSpeed` averages displacement across keypoints, which
makes this step indifferent to keypoint names and their order.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PX_PER_FRAME = "px/frame"
CM_PER_S = "cm/s"

__all__ = [
    "CM_PER_S",
    "PX_PER_FRAME",
    "SCHEMA_VERSION",
    "CohortSpeedError",
    "CohortSpeedThresholds",
    "compute_cohort_thresholds",
    "session_speeds",
    "video_for_pose_csv",
]


class CohortSpeedError(ValueError):
    """A cohort threshold file could not be understood, or cannot be built."""


def video_for_pose_csv(pose_csv: Path | str) -> Path | None:
    """The video a Batch Pose Tracking CSV was written beside, if it is there.

    The tool names outputs ``<video stem>DLC_<model>.csv``, so the video stem
    is recoverable. Used only to find a calibration and a frame rate — a
    missing video is not an error.
    """
    path = Path(pose_csv)
    stem = path.stem
    marker = stem.rfind("DLC_")
    if marker <= 0:
        return None
    base = stem[:marker]
    for candidate in sorted(path.parent.glob(f"{base}.*")):
        if candidate.suffix.lower() != ".csv":
            return candidate
    return None


def session_speeds(
    pose_csv: Path | str,
    *,
    px_per_mm: float | None = None,
    fps: float | None = None,
) -> tuple[np.ndarray, str]:
    """``(speeds, unit)`` for one session, from its pose CSV.

    Returns cm/s when both a pixel scale and a frame rate are known, else the
    raw px/frame. Frame 0 (always 0 by construction) and dropout frames are
    excluded so they cannot drag a percentile.
    """
    from glider.analysis.behavior.classify.speed_state import CausalSpeed
    from glider.vision.pose.dlc import from_dlc_csv

    pose = from_dlc_csv(Path(pose_csv))
    causal = CausalSpeed()
    speeds = np.asarray([causal.push(xy) for xy in pose.xy], dtype=np.float64)
    speeds = speeds[1:]  # frame 0 has no predecessor
    speeds = speeds[np.isfinite(speeds)]

    rate = fps if fps is not None else getattr(pose, "fps", None)
    if px_per_mm and px_per_mm > 0 and rate and rate > 0:
        # px/frame -> px/s -> mm/s -> cm/s
        return speeds * float(rate) / float(px_per_mm) / 10.0, CM_PER_S
    return speeds, PX_PER_FRAME


@dataclass
class CohortSpeedThresholds:
    """Cut-offs derived once from a pooled cohort, with their provenance."""

    freeze: float
    dart: float
    unit: str
    freeze_pct: float
    dart_pct: float
    n_sessions: int
    n_samples: int
    sources: list[str] = field(default_factory=list)
    created: str = ""

    @property
    def is_calibrated(self) -> bool:
        """Whether the thresholds are in real units rather than pixels."""
        return self.unit == CM_PER_S

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "created": self.created or datetime.now().isoformat(timespec="seconds"),
            "unit": self.unit,
            "freeze": float(self.freeze),
            "dart": float(self.dart),
            "freeze_pct": float(self.freeze_pct),
            "dart_pct": float(self.dart_pct),
            "n_sessions": int(self.n_sessions),
            "n_samples": int(self.n_samples),
            "sources": list(self.sources),
        }

    def save(self, path: Path | str) -> None:
        """Write the thresholds. Raises OSError if it cannot."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        logger.info(
            "cohort thresholds: freeze=%.4f dart=%.4f %s from %d session(s) -> %s",
            self.freeze,
            self.dart,
            self.unit,
            self.n_sessions,
            path,
        )

    @classmethod
    def from_dict(cls, data: dict) -> CohortSpeedThresholds:
        if not isinstance(data, dict):
            raise CohortSpeedError("not a cohort threshold file")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise CohortSpeedError(
                f"schema_version {data.get('schema_version')!r}; "
                f"this build understands {SCHEMA_VERSION}"
            )
        try:
            return cls(
                freeze=float(data["freeze"]),
                dart=float(data["dart"]),
                unit=str(data["unit"]),
                freeze_pct=float(data.get("freeze_pct", 0.0)),
                dart_pct=float(data.get("dart_pct", 0.0)),
                n_sessions=int(data.get("n_sessions", 0)),
                n_samples=int(data.get("n_samples", 0)),
                sources=list(data.get("sources", [])),
                created=str(data.get("created", "")),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CohortSpeedError(f"malformed cohort threshold file: {e}") from e

    @classmethod
    def load(cls, path: Path | str) -> CohortSpeedThresholds:
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise CohortSpeedError(f"cannot read {path}: {e}") from e
        return cls.from_dict(data)

    def to_px_per_frame(
        self, *, px_per_mm: float | None = None, fps: float | None = None
    ) -> tuple[float, float]:
        """``(freeze, dart)`` in px/frame for one video's geometry.

        Cohort thresholds in cm/s are converted through *this* video's scale
        and rate, which is the point: one physical cut-off, applied to every
        session on its own terms. Already-pixel thresholds pass through, since
        they were only ever valid for a single shared geometry.
        """
        if not self.is_calibrated:
            return self.freeze, self.dart
        if not px_per_mm or px_per_mm <= 0 or not fps or fps <= 0:
            raise CohortSpeedError(
                "these cohort thresholds are in cm/s and need this video's pixel "
                "scale and frame rate to apply; supply a calibration"
            )
        factor = float(px_per_mm) / float(fps) * 10.0  # cm/s -> px/frame
        return self.freeze * factor, self.dart * factor


def compute_cohort_thresholds(
    pose_csvs,
    *,
    freeze_pct: float = 10.0,
    dart_pct: float = 99.5,
    calibration_master: Path | str | None = None,
    px_per_mm: float | None = None,
    fps: float | None = None,
) -> CohortSpeedThresholds:
    """Pool the speed of every session and take the cohort percentiles.

    A pixel scale is looked up per session (explicit ``px_per_mm`` first, then
    the master calibration file for that session's video). If every session
    resolves one the pool is in cm/s; if any does not, the whole pool falls
    back to px/frame and logs which sessions were missing, because a pool of
    mixed units is meaningless.
    """
    from glider.analysis.behavior.units import load_px_per_mm

    paths = [Path(p) for p in pose_csvs]
    if not paths:
        raise CohortSpeedError("no pose CSVs given")
    if float(freeze_pct) >= float(dart_pct):
        raise CohortSpeedError(f"freeze_pct ({freeze_pct}) must be below dart_pct ({dart_pct})")

    per_session: list[tuple[np.ndarray, str]] = []
    uncalibrated: list[str] = []
    for path in paths:
        video = video_for_pose_csv(path)
        scale = px_per_mm
        if scale is None and video is not None:
            scale = load_px_per_mm(calibration_master, video)
        speeds, unit = session_speeds(path, px_per_mm=scale, fps=fps)
        if speeds.size == 0:
            logger.warning("no usable speed samples in %s; skipping", path)
            continue
        if unit == PX_PER_FRAME:
            uncalibrated.append(path.name)
        per_session.append((speeds, unit))

    if not per_session:
        raise CohortSpeedError("no session produced any usable speed samples")

    if uncalibrated:
        # Recompute everything unscaled rather than pool mixed units.
        logger.warning(
            "pooling in %s: %d of %d session(s) had no pixel scale (%s). "
            "This is only valid if every video shares one rig geometry.",
            PX_PER_FRAME,
            len(uncalibrated),
            len(paths),
            ", ".join(uncalibrated[:3]) + ("…" if len(uncalibrated) > 3 else ""),
        )
        pooled = np.concatenate(
            [session_speeds(p, px_per_mm=None, fps=fps)[0] for p in paths if p.exists()]
        )
        unit = PX_PER_FRAME
    else:
        pooled = np.concatenate([s for s, _ in per_session])
        unit = CM_PER_S

    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        raise CohortSpeedError("no usable speed samples after pooling")

    return CohortSpeedThresholds(
        freeze=float(np.percentile(pooled, freeze_pct)),
        dart=float(np.percentile(pooled, dart_pct)),
        unit=unit,
        freeze_pct=float(freeze_pct),
        dart_pct=float(dart_pct),
        n_sessions=len(per_session),
        n_samples=int(pooled.size),
        sources=[p.name for p in paths],
        created=datetime.now().isoformat(timespec="seconds"),
    )
