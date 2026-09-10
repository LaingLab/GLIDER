"""One session as lanes on a single time axis.

The event log has always held everything a hardware raster needs — every
pin edge and every commanded write, timestamped against a session epoch
the other recorders share. Nothing had drawn it. This builds the lanes;
:mod:`glider.gui.widgets.timeline_bar` draws them.

Qt-free on purpose, in the same way :mod:`glider.analysis.behavior.session_view`
is: the axis arithmetic and the lane building are the parts worth testing,
and neither needs a display.

Everything here is in **flow-relative milliseconds** — t=0 is
StartExperiment, matching what an analyst already reasons in. Events that
predate flow start (device initialisation, which is exactly when a rig is
most likely to be left in the wrong state) carry negative times and are
drawn rather than clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from glider.analysis.session import Session

__all__ = ["FrameMap", "build_frame_map"]


@dataclass(frozen=True)
class FrameMap:
    """Frame index to flow-relative milliseconds, and back.

    Args:
        frames: Frame indices, sorted ascending, one per known sample.
        ms: The flow-relative time of each of those frames.
        source: ``"tracking"`` for the empirical per-frame map,
            ``"frame_rate"`` for the nominal-rate estimate.
    """

    frames: np.ndarray
    ms: np.ndarray
    source: str

    def ms_of(self, frame: int) -> float:
        """When ``frame`` happened. Clamps outside the known range."""
        return float(np.interp(float(frame), self.frames, self.ms))

    def frame_at(self, ms: float) -> int:
        """The frame nearest ``ms``. Clamps outside the known range."""
        if len(self.frames) == 0:
            return 0
        idx = int(np.searchsorted(self.ms, float(ms)))
        idx = max(0, min(idx, len(self.frames) - 1))
        return int(self.frames[idx])


def build_frame_map(session: Session, flow_offset_ms: float = 0.0) -> FrameMap | None:
    """The frame/time mapping for a session, or None if it has neither.

    Resolved in order:

    1. The tracking CSV's own ``frame`` and ``elapsed_ms`` columns. This is
       an empirical map and stays correct across dropped frames, which a
       nominal-rate calculation does not — a dropped frame shifts every
       later frame by one in the nominal version and by nothing here.
    2. :attr:`Session.frame_rate` against the frame index, for a recording
       whose tracking is too thin for (1).
    3. Neither: ``None``. The caller draws no hardware lanes, which is
       correct rather than degraded — an ethogram-only session has no
       hardware data to place on a time axis.

    Args:
        session: The loaded recording.
        flow_offset_ms: Session-elapsed ms of flow start, subtracted from
            every time so the result is flow-relative.
    """
    tracking = session.tracking
    if not tracking.empty and {"frame", "elapsed_ms"}.issubset(tracking.columns):
        pairs = tracking[["frame", "elapsed_ms"]].dropna()
        # Several objects in one frame share a timestamp; one row per frame.
        pairs = pairs.drop_duplicates(subset="frame").sort_values("frame")
        if len(pairs) >= 2:
            return FrameMap(
                frames=pairs["frame"].to_numpy(dtype=float),
                ms=pairs["elapsed_ms"].to_numpy(dtype=float) - flow_offset_ms,
                source="tracking",
            )

    fps = session.frame_rate
    if fps:
        last = 1.0
        if not tracking.empty and "frame" in tracking.columns:
            last = max(1.0, float(tracking["frame"].max()))
        frames = np.array([0.0, last])
        return FrameMap(
            frames=frames,
            ms=frames / fps * 1000.0 - flow_offset_ms,
            source="frame_rate",
        )

    return None
