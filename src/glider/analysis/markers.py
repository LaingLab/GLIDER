"""Review markers: point notes and named ranges, kept beside the data. Qt-free.

A marker's time is seconds on its session's own zero -- the same zero the
review timeline's ruler draws: flow start when the session has one, the first
video frame otherwise. That is what lets one "Stim 1:00-4:20" range mean the
same stretch of protocol in every animal of a cohort, whatever each rig's
frame rate or however long it ran before flow start.

Every conversion between those seconds and a session's frames goes through
:func:`seconds_at` and :func:`frame_at`, so markers, snapping, the current
range and the epoch table cannot disagree about where a second falls.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from glider.analysis.timeline import Timeline

__all__ = [
    "frame_at",
    "frames_in",
    "seconds_at",
    "t0_of",
    "uses_ms",
]

# How far below a whole frame still counts as that frame. Float noise from
# ms -> s -> ms must never push a frame's own time over into the next one.
_EPS = 1e-6


def uses_ms(timeline: Timeline | None) -> bool:
    """Whether the timeline's axis is milliseconds: a frame map and a span.

    The same test :meth:`TimelineView.uses_ms` makes; without it the axis is
    frames and a second is ``fps`` of them.
    """
    return (
        timeline is not None
        and timeline.frame_map is not None
        and timeline.end_ms > timeline.start_ms
    )


def _fps(fps: float) -> float:
    return float(fps) if fps else 30.0


def seconds_at(timeline: Timeline | None, fps: float, frame: float) -> float:
    """Where ``frame`` sits on the ruler, in seconds.

    Flow-relative on a timeline with a flow start, video-relative on one
    without, and ``frame / fps`` with no frame map at all. Past either end of
    the frame map the nominal rate carries on, so the frame after the last
    still has a time and a range can end there.
    """
    fps = _fps(fps)
    if not uses_ms(timeline):
        return float(frame) / fps
    frames, ms = timeline.frame_map.frames, timeline.frame_map.ms
    f = float(frame)
    at = float(np.interp(f, frames, ms))
    at += (max(0.0, f - frames[-1]) - max(0.0, frames[0] - f)) * 1000.0 / fps
    return at / 1000.0


def frame_at(timeline: Timeline | None, fps: float, seconds: float) -> int:
    """The first frame at or after ``seconds`` on the ruler.

    The inverse of :func:`seconds_at`: a frame's own time gives that frame
    back, and a time between two frames gives the later one.
    """
    fps = _fps(fps)
    if not uses_ms(timeline):
        return math.ceil(float(seconds) * fps - _EPS)
    frames, ms = timeline.frame_map.frames, timeline.frame_map.ms
    at = float(seconds) * 1000.0
    f = float(np.interp(at, ms, frames))
    f += (max(0.0, at - ms[-1]) - max(0.0, ms[0] - at)) * fps / 1000.0
    return math.ceil(f - _EPS)


def frames_in(
    timeline: Timeline | None,
    fps: float,
    start_s: float,
    end_s: float,
    bounds: tuple[int, int],
) -> tuple[int, int] | None:
    """The frames whose time lies in ``[start_s, end_s)``, clipped to ``bounds``.

    ``(first, last)``, inclusive, as a selection is. None when no frame of the
    session falls inside -- a Post epoch past the end of a short session.
    """
    lo, hi = bounds
    first = max(int(lo), frame_at(timeline, fps, start_s))
    last = min(int(hi), frame_at(timeline, fps, end_s) - 1)
    return None if first > last else (first, last)


def t0_of(timeline: Timeline | None) -> str:
    """What a session's seconds count from: the ruler's own label."""
    if uses_ms(timeline) and timeline.flow_start_ms is not None:
        return "flow_start"
    return "video_start"
