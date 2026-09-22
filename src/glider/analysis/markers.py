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

import contextlib
import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from glider.analysis.timeline import Timeline

__all__ = [
    "COHORT_FILE",
    "SESSION_FILE",
    "SWATCHES",
    "VERSION",
    "Marker",
    "MarkerFileError",
    "MarkerStore",
    "frame_at",
    "frames_in",
    "load_markers",
    "save_markers",
    "seconds_at",
    "stack_rows",
    "t0_of",
    "uses_ms",
]

#: A session's own markers, beside its data (the recording folder, or the
#: folder an apply run wrote the ethogram into).
SESSION_FILE = "review_markers.json"
#: Ranges shared by every session, in the folder that was opened as a cohort.
COHORT_FILE = "cohort_markers.json"
VERSION = 1
#: The colours a marker can be, by name; the GUI maps names to tokens. A
#: fixed set, as Resolve's markers are: a colour is a category the user chose.
SWATCHES = ("cyan", "blue", "violet", "pink", "red", "orange", "yellow", "green", "slate")
_KINDS = ("point", "range")
_SCOPES = ("session", "cohort")

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


# ---------------------------------------------------------------------------
# the model


@dataclass
class Marker:
    """A note at one time, or a named stretch of time, in seconds on its zero."""

    kind: str  # "point" | "range"
    start_s: float
    end_s: float | None = None
    name: str = ""
    color: str = SWATCHES[0]
    note: str = ""
    scope: str = "session"  # "session" | "cohort"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def is_range(self) -> bool:
        return self.kind == "range"

    @property
    def duration_s(self) -> float:
        return 0.0 if self.end_s is None else self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "name": self.name,
            "color": self.color,
            "note": self.note,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Marker:
        """One saved marker. Raises on anything malformed.

        An unknown colour is not malformed -- a later GLIDER may offer more --
        so it loads as the default rather than refusing the whole file.
        """
        kind = str(data["kind"])
        if kind not in _KINDS:
            raise ValueError(f"unknown marker kind {kind!r}")
        start = float(data["start_s"])
        end = data.get("end_s")
        end = None if end is None or kind == "point" else float(end)
        if not math.isfinite(start) or (end is not None and not math.isfinite(end)):
            raise ValueError("a marker's times must be finite")
        if kind == "range" and (end is None or end <= start):
            raise ValueError("a range marker needs an end after its start")
        scope = str(data.get("scope", "session"))
        if scope not in _SCOPES:
            raise ValueError(f"unknown marker scope {scope!r}")
        color = str(data.get("color", SWATCHES[0]))
        return cls(
            kind=kind,
            start_s=start,
            end_s=end,
            name=str(data.get("name", "")),
            color=color if color in SWATCHES else SWATCHES[0],
            note=str(data.get("note", "")),
            scope=scope,
            id=str(data.get("id") or uuid.uuid4().hex),
        )


# ---------------------------------------------------------------------------
# the files


class MarkerFileError(Exception):
    """A marker file that cannot be read, or that a newer GLIDER wrote."""


def load_markers(path: Path) -> tuple[list[Marker], str | None]:
    """The markers in ``path`` and the zero they were saved on.

    ``([], None)`` when there is no file yet. Raises :class:`MarkerFileError`
    for a file that cannot be read or was written by a newer GLIDER -- which
    the caller must then never overwrite.
    """
    path = Path(path)
    if not path.exists():
        return [], None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise MarkerFileError(f"{path.name} could not be read: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("markers"), list):
        raise MarkerFileError(f"{path.name} is not a GLIDER marker file")
    version = data.get("version")
    if not isinstance(version, int):
        raise MarkerFileError(f"{path.name} has no version")
    if version > VERSION:
        raise MarkerFileError(
            f"{path.name} was written by a newer GLIDER (version {version}); "
            "update GLIDER to edit its markers"
        )
    try:
        markers = [Marker.from_dict(item) for item in data["markers"]]
    except (KeyError, TypeError, ValueError) as e:
        raise MarkerFileError(f"{path.name} has a marker that cannot be read: {e}") from e
    t0 = data.get("t0")
    return markers, (None if t0 is None else str(t0))


def save_markers(path: Path, markers: list[Marker], *, t0: str | None = None) -> None:
    """Write ``markers`` atomically: a temp file beside ``path``, then ``os.replace``.

    A write that fails part way leaves the old file exactly as it was, and no
    temp file behind. Raises :class:`OSError`.
    """
    path = Path(path)
    payload: dict = {"version": VERSION}
    if t0 is not None:
        payload["t0"] = t0
    payload["markers"] = [m.to_dict() for m in markers]
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # mkstemp creates the temp file at 0600, and os.replace carries that
        # mode onto the destination -- which would strip a shared lab
        # folder's group-read bit every time a marker file is saved. Match
        # the existing file's mode, or the mode a normal file create would
        # get (0666 masked by the process umask) when there is no file yet.
        if path.exists():
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class MarkerStore:
    """One marker file: the markers it holds, and whether it may be written.

    A file that could not be read is never written back -- saving over it
    would destroy whatever it held -- so the store turns read-only and says
    why in ``error``. ``t0`` is the zero this session's seconds count from
    today (None for a cohort file, whose sessions each use their own); a
    file saved on a different zero still loads, with ``t0_changed`` set so
    the window can warn that its markers may have moved.
    """

    def __init__(self, path: Path, *, t0: str | None = None):
        self.path = Path(path)
        self.t0 = t0
        self.markers: list[Marker] = []
        self.error: str | None = None
        self.t0_changed = False
        try:
            self.markers, stored = load_markers(self.path)
        except MarkerFileError as e:
            self.error = str(e)
            return
        self.t0_changed = t0 is not None and stored is not None and stored != t0

    @property
    def writable(self) -> bool:
        return self.error is None

    def find(self, marker_id: str) -> Marker | None:
        return next((m for m in self.markers if m.id == marker_id), None)

    def put(self, marker: Marker) -> None:
        """Add ``marker``, or replace the one with its id."""
        for i, existing in enumerate(self.markers):
            if existing.id == marker.id:
                self.markers[i] = marker
                return
        self.markers.append(marker)

    def remove(self, marker_id: str) -> None:
        self.markers = [m for m in self.markers if m.id != marker_id]

    def save(self) -> None:
        """Raises :class:`MarkerFileError` when read-only, :class:`OSError` on a failed write."""
        if self.error is not None:
            raise MarkerFileError(self.error)
        save_markers(self.path, self.markers, t0=self.t0)


# ---------------------------------------------------------------------------
# drawing


def stack_rows(spans: list[tuple[float, float]]) -> list[int]:
    """Which of two sub-rows each range draws in, in the order given.

    In start order, each range goes on the first sub-row it doesn't overlap;
    one that overlaps both goes on the second, drawn on top. Two rows, not
    as many as it takes: a marker row taller than the lanes it labels would
    be the tail wagging the dog.
    """
    rows = [0] * len(spans)
    ends = [-math.inf, -math.inf]
    for i in sorted(range(len(spans)), key=lambda i: spans[i][0]):
        start, end = spans[i]
        row = 0 if start >= ends[0] else 1
        rows[i] = row
        ends[row] = max(ends[row], end)
    return rows
