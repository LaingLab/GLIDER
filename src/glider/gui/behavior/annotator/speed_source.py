"""Per-frame speed for the annotator's trace, and the cache that loads it.

The annotator has never read pose data: it is handed proposed clips and the
paths to write labels to, and nothing else. Showing a speed trace means
reading the same pose CSVs the sampler already read, so this module owns that
-- and only that. It has no Qt and no OpenCV, which keeps it testable without
an event loop and keeps the threading question (whose answer is "on a worker")
out of the data layer entirely.

Two pieces:

* :class:`SessionSpeed` -- one video's trace, indexable by frame, in whatever
  unit it can honestly report.
* :class:`SpeedCache` -- which videos are loaded, loading, or known bad, so a
  17 MB CSV is parsed at most once per session and a broken one is not
  retried on every clip.

Units. The trace is natively px/frame, which needs nothing but the pose data.
Reporting cm/s -- the unit cohort thresholds are written in -- additionally
needs the video's pixel scale and frame rate. Without both, this reports
pixels and says so rather than guessing, because a trace silently in the wrong
unit compared against a cm/s threshold line is worse than no line at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PX_PER_FRAME = "px/frame"
CM_PER_S = "cm/s"

__all__ = [
    "CM_PER_S",
    "PX_PER_FRAME",
    "SessionSpeed",
    "SpeedCache",
    "load_session_speed",
]


@dataclass
class SessionSpeed:
    """One video's causal speed trace, indexed by frame.

    ``px_per_frame`` holds one value per frame of the recording, exactly as
    :func:`~glider.analysis.behavior.classify.speed_state.causal_speed_series`
    produced it -- frame 0 included, dropouts left as ``NaN`` at their own
    index. Reads go through :meth:`at` / :meth:`window`, which convert to the
    display unit and treat out-of-range frames as missing rather than as an
    error: the trim window is deliberately padded past the clip and routinely
    overhangs the start or end of a recording.
    """

    px_per_frame: np.ndarray
    fps: float
    px_per_mm: float | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether this trace can be reported in real units."""
        return bool(self.px_per_mm and self.px_per_mm > 0 and self.fps and self.fps > 0)

    @property
    def unit(self) -> str:
        return CM_PER_S if self.is_calibrated else PX_PER_FRAME

    @property
    def n_frames(self) -> int:
        return int(self.px_per_frame.size)

    @property
    def _scale(self) -> float:
        """Multiplier from px/frame into the display unit."""
        if not self.is_calibrated:
            return 1.0
        # px/frame -> px/s -> mm/s -> cm/s. Same arithmetic as cohort_speed,
        # so a value here is directly comparable to a cohort threshold.
        return float(self.fps) / float(self.px_per_mm) / 10.0

    def at(self, frame: int) -> float:
        """Speed at ``frame`` in the display unit, or NaN if out of range."""
        i = int(frame)
        if i < 0 or i >= self.n_frames:
            return float("nan")
        return float(self.px_per_frame[i]) * self._scale

    def window(self, start: int, end: int) -> np.ndarray:
        """``[start, end)`` in the display unit, NaN-padded past either end.

        The result always has ``end - start`` entries. Keeping the length
        means the caller's frame-to-pixel mapping holds whether or not the
        window overhangs the recording, so a trace never silently shifts.
        """
        start, end = int(start), int(end)
        if end <= start:
            return np.empty(0, dtype=np.float64)
        out = np.full(end - start, np.nan, dtype=np.float64)
        lo, hi = max(start, 0), min(end, self.n_frames)
        if hi > lo:
            out[lo - start : hi - start] = self.px_per_frame[lo:hi] * self._scale
        return out


def load_session_speed(
    pose_csv: Path | str,
    *,
    px_per_mm: float | None = None,
    fps: float | None = None,
) -> SessionSpeed:
    """Read a pose CSV and compute its causal speed trace.

    This is the expensive call -- the CSVs run to tens of megabytes -- and is
    why :class:`SpeedCache` exists. ``fps`` defaults to the rate recorded
    alongside the pose data.
    """
    from glider.analysis.behavior.classify.speed_state import causal_speed_series
    from glider.vision.pose.dlc import from_dlc_csv

    pose = from_dlc_csv(Path(pose_csv))
    rate = fps if fps is not None else getattr(pose, "fps", None)
    return SessionSpeed(
        px_per_frame=causal_speed_series(pose.xy),
        fps=float(rate) if rate else 0.0,
        px_per_mm=px_per_mm,
    )


@dataclass
class SpeedCache:
    """Which videos have a trace, are getting one, or never will.

    Four states per video: ``absent``, ``loading``, ``ready``, ``failed``.
    :meth:`begin` is the gate -- it returns True exactly once per video, so a
    caller can start a worker without tracking in-flight work itself, and a
    video whose CSV is unreadable is not re-attempted on every clip that
    happens to come from it.
    """

    _ready: dict[Path, SessionSpeed] = field(default_factory=dict)
    _loading: set[Path] = field(default_factory=set)
    _failed: dict[Path, str] = field(default_factory=dict)

    @staticmethod
    def _key(video: Path | str) -> Path:
        # The same video arrives as a str from a ProposedClip and as a Path
        # from videos_meta; those must not become two entries.
        return Path(video)

    def state(self, video: Path | str) -> str:
        key = self._key(video)
        if key in self._ready:
            return "ready"
        if key in self._loading:
            return "loading"
        if key in self._failed:
            return "failed"
        return "absent"

    def begin(self, video: Path | str) -> bool:
        """Claim the load for ``video``. True only for the caller that won."""
        if self.state(video) != "absent":
            return False
        self._loading.add(self._key(video))
        return True

    def store(self, video: Path | str, session: SessionSpeed) -> None:
        key = self._key(video)
        self._loading.discard(key)
        self._failed.pop(key, None)
        self._ready[key] = session

    def fail(self, video: Path | str, reason: str) -> None:
        key = self._key(video)
        self._loading.discard(key)
        self._failed[key] = str(reason)

    def get(self, video: Path | str) -> SessionSpeed | None:
        return self._ready.get(self._key(video))

    def error(self, video: Path | str) -> str | None:
        return self._failed.get(self._key(video))
