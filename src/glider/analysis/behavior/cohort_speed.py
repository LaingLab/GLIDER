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

SCHEMA_VERSION = 2
PX_PER_FRAME = "px/frame"
CM_PER_S = "cm/s"

#: Share of a session's tracked frames the arena gate may blank before pooling
#: it earns a callout. The cohort percentile is derived from whatever survived
#: the gate, so a session that lost a twentieth of its tracked frames has
#: contributed a distribution that is not quite the one it was tracked with —
#: worth naming, not worth refusing, because heavy blanking is sometimes
#: exactly what the gate is for.
_REJECT_WARN = 0.05

__all__ = [
    "CM_PER_S",
    "PX_PER_FRAME",
    "SCHEMA_VERSION",
    "CohortSpeedError",
    "CohortSpeedThresholds",
    "compute_cohort_thresholds",
    "frame_window",
    "session_speeds",
    "video_for_pose_csv",
]


class CohortSpeedError(ValueError):
    """A cohort threshold file could not be understood, or cannot be built."""


def _optional_float(value) -> float | None:
    """A float, or None for a missing or unusable value."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cohort_gate_block(block) -> dict:
    """The part of a session's gate block a *cohort* can honestly carry.

    ``arena_corners`` is deliberately dropped, and so are the per-video frame
    counts. Corners are per-video: a cohort pools 31 sessions with 31 drawn
    perimeters, so no single fingerprint could match them all, and a scoring
    run that compared them would raise on every video. What survives is what
    is true of the whole pool — the gate settings, and whether it was gated at
    all.
    """
    block = block or {}
    return {
        "gated": bool(block.get("gated")),
        "settings": dict(block.get("settings") or {}),
    }


def _blanked_fraction(block) -> float:
    """Share of a session's *considered* frames its gate blanked.

    Recomputed rather than read: ``GateReport.blanked_fraction`` is a property,
    so ``asdict(report)`` never wrote it to the sidecar.
    """
    considered = block.get("frames_considered", 0)
    return block.get("frames_blanked", 0) / considered if considered else 0.0


def _pool_gate_provenance(paths) -> dict:
    """The gate block for a pool, refusing one that mixes gates.

    Up front, before a byte of CSV is read: this is arithmetic on sidecars and
    the pooling it guards is minutes of work per cohort.

    A half-gated pool is refused rather than described, because the cohort
    block is one boolean. Whichever value it took, every session on the other
    side would hard-raise at scoring time — and the operator would learn about
    it one video at a time, after the pooling had already been paid for.

    A pool gated under *different settings* is refused for the same reason and
    with the same consequence: the block carries one settings dict, and the
    settings are what decide how much of each track survived, so keeping either
    one misdescribes the sessions gated under the other. The documented
    escalation workflow — derive with defaults, read the report, re-gate the
    known-bad sessions at ``min_detected_fraction=1.0`` — is exactly how a
    folder comes to hold both.
    """
    from glider.vision.arena_gate import settings_from_block
    from glider.vision.pose.dlc import read_pose_meta

    blocks = [(p, (read_pose_meta(p) or {}).get("arena_gate") or {}) for p in paths]
    gated = [p for p, b in blocks if b.get("gated")]
    ungated = [p for p, b in blocks if not b.get("gated")]
    if gated and ungated:
        raise CohortSpeedError(
            f"this pool is a mix of gated and ungated sessions "
            f"({len(gated)} gated, {len(ungated)} not), and one set of cut-offs "
            f"cannot describe both: gating lowers a session's speed "
            f"distribution. Re-gate the whole cohort, or pool only one kind. "
            f"First of each: {gated[0].name}, {ungated[0].name}"
        )

    # Keyed on the normalised settings, so a block that spells out the defaults
    # and one that omits them are the same gate rather than two.
    by_settings: dict[str, list[Path]] = {}
    for path, block in blocks:
        if block.get("gated"):
            key = json.dumps(settings_from_block(block), sort_keys=True, default=str)
            by_settings.setdefault(key, []).append(path)
    if len(by_settings) > 1:
        named = "; ".join(
            f"{group[0].name} gated with {settings}"
            for settings, group in list(by_settings.items())[:2]
        )
        raise CohortSpeedError(
            f"this pool was gated under {len(by_settings)} different sets of gate "
            f"settings, and the cohort file records one: whichever it kept, the "
            f"sessions gated under the others would be scored against cut-offs "
            f"derived from a distribution they are not part of. Re-gate the whole "
            f"cohort the same way, or pool each set separately. First two: {named}"
        )

    for path, block in blocks:
        share = _blanked_fraction(block)
        if share > _REJECT_WARN:
            logger.warning(
                "%s: the arena gate blanked %.1f%% of its tracked frames, so the "
                "cohort percentiles are derived from what survived",
                path.name,
                share * 100,
            )
    # Mixed pools are already refused, so the first gated block speaks for all
    # of them.
    return _cohort_gate_block(next((b for _, b in blocks if b.get("gated")), None))


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


def _causal_speeds(pose, frame_range: tuple[int, int] | None = None) -> np.ndarray:
    """Causal per-frame speed in px/frame, dropouts and frame 0 removed.

    Frame 0 is 0 by construction (no predecessor) and would drag the freeze
    percentile toward zero.

    ``frame_range`` restricts which frames are *kept*, not which are measured.
    The filter still runs from frame 0, because it is causal: its value at
    frame 3600 depends on the frames before it. Starting it at the window
    instead would give the window's first frames a cold filter, so the
    thresholds would be derived from a signal the apply run never produces —
    and the apply run windows the same way, for the same reason.
    """
    from glider.analysis.behavior.classify.speed_state import causal_speed_series

    speeds = causal_speed_series(pose.xy)
    keep = np.zeros(speeds.size, dtype=bool)
    first, last = frame_range if frame_range is not None else (0, speeds.size - 1)
    first = max(first, 1)  # frame 0 is 0 by construction
    keep[first : last + 1] = True
    return speeds[keep & np.isfinite(speeds)]


def frame_window(
    start_s: float | None, end_s: float | None, fps: float | None
) -> tuple[int, int] | None:
    """``(first, last)`` inclusive frames for a time window, or None.

    Seconds in, frames out, per session — so one window means the same stretch
    of every animal's recording whatever it was filmed at.
    """
    if start_s is None and end_s is None:
        return None
    if not fps or fps <= 0:
        raise CohortSpeedError("a time window is in seconds and needs each session's frame rate")
    first = int(round(float(start_s or 0.0) * fps))
    last = int(round(float(end_s) * fps)) - 1 if end_s is not None else 2**31
    if last < first:
        raise CohortSpeedError(
            f"the window ends before it starts: {float(start_s or 0.0):g} s to {float(end_s):g} s"
        )
    return max(0, first), last


def session_speeds(
    pose_csv: Path | str,
    *,
    px_per_mm: float | None = None,
    fps: float | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> tuple[np.ndarray, str]:
    """``(speeds, unit)`` for one session, from its pose CSV.

    Returns cm/s when both a pixel scale and a frame rate are known, else the
    raw px/frame. Frame 0 (always 0 by construction) and dropout frames are
    excluded so they cannot drag a percentile.

    ``start_s`` / ``end_s`` pool only a stretch of the session. Thresholds are
    meant to describe the behaviour being scored, so a run that scores minutes
    two to seven should be thresholded against minutes two to seven — pooling
    the whole recording mixes in a settling-in period the ethogram never
    covers.
    """
    from glider.vision.pose.dlc import from_dlc_csv

    pose = from_dlc_csv(Path(pose_csv))
    rate_for_window = fps if fps is not None else getattr(pose, "fps", None)
    speeds = _causal_speeds(pose, frame_window(start_s, end_s, rate_for_window))

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
    # The stretch of each session these were pooled from. None = the whole
    # recording. Recorded so an apply run can notice it is thresholding a
    # window against cut-offs derived from a different one.
    start_s: float | None = None
    end_s: float | None = None
    # How many sessions had no pixel scale. Non-zero is why a pool that should
    # have been in cm/s came back in pixels, and it is recorded rather than
    # only logged because a log line is invisible from a GUI — the operator
    # sees a file in the wrong unit and no reason for it.
    n_uncalibrated: int = 0
    # Which arena gate these were derived under. Gating removes out-of-arena
    # detections, which lowers the speed distribution these percentiles are
    # taken from, so a cut-off derived from ungated speed and applied to a
    # gated track is wrong — and silently so. Scoring compares this against
    # the CSV's own sidecar and refuses a mismatch. Default ungated, which is
    # what every file written before the gate existed actually is.
    gate_provenance: dict = field(default_factory=lambda: {"gated": False})

    @property
    def window(self) -> tuple[float | None, float | None] | None:
        """``(start_s, end_s)`` these were pooled over, or None for all of it."""
        if self.start_s is None and self.end_s is None:
            return None
        return self.start_s, self.end_s

    def describe(self) -> str:
        """One line: the cut-offs, their units, and what they came from."""
        return (
            f"freezing < {self.freeze:.3g} {self.unit}, "
            f"darting > {self.dart:.3g} {self.unit} "
            f"(p{self.freeze_pct:g}/p{self.dart_pct:g} of {self.n_sessions} session(s) "
            f"over {self.describe_window()})"
        )

    def describe_window(self) -> str:
        """The pooled stretch, in the words an operator would use."""
        if self.window is None:
            return "the whole recording"
        start = f"{(self.start_s or 0.0) / 60:g}"
        end = f"{self.end_s / 60:g} min" if self.end_s is not None else "the end"
        return f"{start}–{end}"

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
            "n_uncalibrated": int(self.n_uncalibrated),
            "sources": list(self.sources),
            "start_s": self.start_s,
            "end_s": self.end_s,
            # Trimmed on the way out as well as on the way in, so a corner list
            # cannot reach the file by some other route.
            "arena_gate": _cohort_gate_block(self.gate_provenance),
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
        # v1 is still read: it predates the arena gate entirely, and its one
        # missing field means exactly what its absence says.
        if data.get("schema_version") not in (1, SCHEMA_VERSION):
            raise CohortSpeedError(
                f"schema_version {data.get('schema_version')!r}; "
                f"this build understands 1 and {SCHEMA_VERSION}"
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
                # Absent in files written before windowing existed, which
                # means exactly what it says: the whole recording.
                start_s=_optional_float(data.get("start_s")),
                end_s=_optional_float(data.get("end_s")),
                n_uncalibrated=int(data.get("n_uncalibrated", 0)),
                # Absent in v1, which is factually ungated: gating did not
                # exist when those files were written. Reading it as anything
                # else would defeat the scoring guard on exactly the stale
                # files that motivate it.
                gate_provenance=_cohort_gate_block(data.get("arena_gate")),
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
    start_s: float | None = None,
    end_s: float | None = None,
    progress=None,
) -> CohortSpeedThresholds:
    """Pool the speed of every session and take the cohort percentiles.

    ``progress`` is called as ``progress(done, total, name)`` before each
    session. Reading a cohort takes minutes, so a caller with a UI needs to be
    able to report it rather than appear hung.

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

    # Sidecars only, before any CSV is opened: a pool that cannot be described
    # by one gate flag must not cost minutes of reading to find that out.
    gate_provenance = _pool_gate_provenance(paths)

    # Each session is read ONCE. The raw px/frame speeds are kept alongside the
    # scale that would convert them, so falling back to pixels is arithmetic
    # rather than a second pass over hundreds of megabytes of CSV.
    per_session: list[tuple[np.ndarray, float | None]] = []
    uncalibrated: list[str] = []
    for i, path in enumerate(paths, 1):
        if progress is not None:
            progress(i, len(paths), path.name)
        video = video_for_pose_csv(path)
        scale = px_per_mm
        if scale is None and video is not None:
            scale = load_px_per_mm(calibration_master, video)
        # One read per session: the CSVs are ~15 MB each and the per-frame
        # causal speed is the expensive part, so nothing may re-open them.
        from glider.vision.pose.dlc import from_dlc_csv

        pose = from_dlc_csv(path)
        rate = fps if fps is not None else getattr(pose, "fps", None)
        # Windowed per session at that session's own rate, so "minutes two to
        # seven" is the same stretch of every animal's recording.
        speeds = _causal_speeds(pose, frame_window(start_s, end_s, rate))
        if speeds.size == 0:
            logger.warning("no usable speed samples in %s; skipping", path)
            continue
        usable = scale if (scale and scale > 0 and rate and rate > 0) else None
        if usable is None:
            uncalibrated.append(path.name)
        per_session.append((speeds, None if usable is None else float(rate) / usable / 10.0))

    if not per_session:
        raise CohortSpeedError("no session produced any usable speed samples")

    if uncalibrated:
        logger.warning(
            "pooling in %s: %d of %d session(s) had no pixel scale (%s). "
            "This is only valid if every video shares one rig geometry.",
            PX_PER_FRAME,
            len(uncalibrated),
            len(paths),
            ", ".join(uncalibrated[:3]) + ("…" if len(uncalibrated) > 3 else ""),
        )
        pooled = np.concatenate([s for s, _ in per_session])
        unit = PX_PER_FRAME
    else:
        pooled = np.concatenate([s * factor for s, factor in per_session])
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
        start_s=None if start_s is None else float(start_s),
        end_s=None if end_s is None else float(end_s),
        n_sessions=len(per_session),
        n_samples=int(pooled.size),
        n_uncalibrated=len(uncalibrated),
        sources=[p.name for p in paths],
        created=datetime.now().isoformat(timespec="seconds"),
        gate_provenance=gate_provenance,
    )
