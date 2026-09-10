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

__all__ = [
    "FrameMap",
    "Lane",
    "Marker",
    "Segment",
    "build_frame_map",
    "hardware_lanes",
]


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


#: Full-scale value per pin type, for turning a written value into a bar
#: height. Normalising by each lane's own observed maximum instead would be
#: marginally cheaper and would draw a PWM that never exceeded 10 as full
#: brightness — same amount of code, wrong picture.
_PIN_FULL_SCALE = {
    "DIGITAL": 1.0,
    "PWM": 255.0,
    "SERVO": 180.0,
    "ANALOG": 1023.0,
}


@dataclass(frozen=True)
class Segment:
    """A device holding one value over a span of time."""

    start_ms: float
    end_ms: float
    value: float
    level: float  # 0.0-1.0, for bar height


@dataclass(frozen=True)
class Marker:
    """An instant with no level — a non-numeric event value."""

    at_ms: float
    label: str


@dataclass(frozen=True)
class Lane:
    """One row of the raster."""

    key: str
    label: str
    board_id: str
    segments: list[Segment]
    markers: list[Marker]


def _cell(value) -> str:
    """A CSV cell as text.

    Two shapes have to be flattened. Empty cells arrive as NaN rather than
    "". And a column that mixes blanks with numbers — `pin` does, because
    flow_marker rows leave it empty — is inferred as float64, so pin 7
    arrives as 7.0 and would otherwise name a lane "board0:pin7.0".
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            return str(int(value))
    text = str(value)
    return "" if text in ("nan", "NaN", "<NA>", "None") else text.strip()


def _full_scale(pin_types: list[str], observed_max: float) -> float:
    """The value that should draw as a full-height bar.

    The pin type's known range when there is one, except where the lane
    actually exceeded it — a 12-bit board reads an ANALOG pin to 4095, and
    clipping that at the 10-bit 1023 would draw the whole session as one
    saturated row.
    """
    for pin_type in pin_types:
        full = _PIN_FULL_SCALE.get(pin_type.upper())
        if full is not None:
            return full if observed_max <= full else observed_max
    return observed_max if observed_max > 0 else 1.0


def hardware_lanes(
    session: Session,
    flow_offset_ms: float = 0.0,
    end_ms: float | None = None,
) -> list[Lane]:
    """One lane per device, from the event log.

    Each event sets its device's value and that value holds until the
    device's next event — a zero-order hold. One rule covers digital, PWM
    and servo: a digital pin gives full-height blocks and a PWM ramp gives
    stepped ones, without a branch per pin type.

    Args:
        session: The loaded recording.
        flow_offset_ms: Session-elapsed ms of flow start, subtracted from
            every event time.
        end_ms: Where the last held value stops. Defaults to the last
            event's own time, which draws it as zero-width.
    """
    events = session.events
    if events.empty:
        return []

    rows = events[events["source"] != "flow_marker"].copy()
    if rows.empty:
        return []

    rows["_ms"] = rows["elapsed_ms"].astype(float) - flow_offset_ms
    rows["_device"] = [_cell(v) for v in rows["device_id"]]
    rows["_board"] = [_cell(v) for v in rows["board_id"]]
    rows["_pin"] = [_cell(v) for v in rows["pin"]]
    rows["_pin_type"] = [_cell(v) for v in rows["pin_type"]]
    # A board-level write with no resolved device still deserves a row.
    rows["_key"] = [
        device or f"{board}:pin{pin}"
        for device, board, pin in zip(rows["_device"], rows["_board"], rows["_pin"], strict=True)
    ]
    rows = rows.sort_values("_ms", kind="stable")

    tail = end_ms if end_ms is not None else float(rows["_ms"].max())

    lanes: list[Lane] = []
    for key, group in rows.groupby("_key", sort=False):
        times = group["_ms"].to_numpy(dtype=float)
        markers: list[Marker] = []
        levels: list[tuple[float, float]] = []  # (ms, numeric value)

        for ms, raw in zip(times, group["value"], strict=True):
            text = _cell(raw)
            if not text:
                # The event logger writes "" for a None value, not a
                # missing cell — neither a level nor a marker for that.
                continue
            try:
                levels.append((float(ms), float(text)))
            except ValueError:
                markers.append(Marker(at_ms=float(ms), label=text))

        observed_max = max((v for _, v in levels), default=0.0)
        full = _full_scale(list(dict.fromkeys(group["_pin_type"])), observed_max)

        segments = [
            Segment(
                start_ms=ms,
                # The last segment ends at `tail` unless that end would
                # precede its own start — a session can legitimately end
                # before its last event (camera stops before the flow
                # tears down), and a negative span draws as an inverted
                # or invisible rect.
                end_ms=(levels[i + 1][0] if i + 1 < len(levels) else max(tail, ms)),
                value=value,
                level=max(0.0, min(1.0, value / full)),
            )
            for i, (ms, value) in enumerate(levels)
        ]

        label = _cell(group["_device"].iloc[0]) or str(key)
        lanes.append(
            Lane(
                key=str(key),
                label=label,
                board_id=_cell(group["_board"].iloc[0]),
                segments=segments,
                markers=markers,
            )
        )

    lanes.sort(key=lambda lane: (lane.board_id, lane.key))
    return lanes
