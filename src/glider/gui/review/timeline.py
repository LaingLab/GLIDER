"""The Session Review timeline: track headers, ruler and lanes.

DaVinci Resolve's Edit-page timeline, where each track is a behaviour source or
a device instead of audio or video. The models are Qt-free
(:mod:`glider.analysis.timeline`, :mod:`glider.gui.review.viewport`); this
module draws them and turns input into frames.

The axis is flow-relative milliseconds when the session has a frame map and
frames otherwise, exactly as P1 drew it. ``scrubbed`` and ``selection_changed``
still speak frames, so every table below the timeline is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.analysis.timeline import (
    BehaviorLane,
    Lane,
    Timeline,
    describe_value,
    lane_role,
)
from glider.gui.review.viewport import Viewport, format_timecode, snap, tick_spacing
from glider.gui.styles import colors
from glider.gui.widgets.tool_ui import data_font, readable_text_on

__all__ = [
    "BEHAVIOR_H",
    "GROUP_H",
    "HEADER_W",
    "HW_H",
    "MARKER_H",
    "RULER_H",
    "TOP_H",
    "Navigator",
    "Row",
    "TimelinePanel",
    "TimelineView",
    "behavior_order",
    "behavior_qcolor",
    "lane_colour",
]

HEADER_W = 212
RULER_H = 26
MARKER_H = 18  # empty in phase 1; phase 2 draws markers here
TOP_H = RULER_H + MARKER_H
GROUP_H = 18
BEHAVIOR_H = 44
HW_H = 22
HW_MIN_H = 10
MIN_SPAN_FRAMES = 30
CLIP_MIN_PX = 2.0
SNAP_PX = 8  # how close an edge pulls
EDGE_PX = 4  # how close counts as grabbing a selection edge
DRAG_PX = 3  # movement that turns a click into a drag


def behavior_qcolor(name: str, order: list[str] | None = None) -> QColor:
    """The colour the annotated video would have drawn this behaviour in.

    Shared with the overlay so a bout looks the same wherever it is shown;
    blank (unscored) frames read as background rather than a colour.

    ``order`` is the behaviours present, which is what makes the colours
    reliably *different*. Without it the palette slot comes from a hash of the
    name, and a hash has no reason to avoid collisions: two behaviours in one
    session could land on the same colour, and neighbouring ones routinely
    landed on adjacent hues. Given the session's own label set, the first N
    palette entries are handed out in order, and N distinct behaviours get N
    distinct colours.
    """
    if not name:
        return QColor(colors.BORDER)
    from glider.analysis.behavior.classify.overlay import color_for_behavior

    b, g, r = color_for_behavior(name, order)
    return QColor(r, g, b)


def behavior_order(labels) -> list[str]:
    """The behaviours present, in a stable order.

    Sorted rather than first-appearance: the same cohort scored twice must
    colour the same behaviour the same way, and first-appearance makes that
    depend on which animal happened to groom first.
    """
    return sorted({label for label in labels if label})


_ROLE_COLOURS = {
    "output": colors.LANE_OUTPUT,
    "motor": colors.LANE_MOTOR,
    "input": colors.LANE_INPUT,
}


def lane_colour(lane: Lane) -> QColor:
    """What a device lane draws in: outputs, motors and inputs kept apart."""
    return QColor(_ROLE_COLOURS[lane_role(lane)])


def _behavior_title(source: str) -> tuple[str, str]:
    """Header name and sub-label for a behaviour lane, from its source."""
    if source == "ethogram":
        return "Classifier", "ethogram"
    if source.startswith("tracking"):
        return f"Live state {source[len('tracking'):]}".strip(), "tracking · behavioral_state"
    return source, ""


def _label_at(lane: BehaviorLane, frame: int) -> str:
    frames = np.asarray(lane.frames)
    index = int(np.searchsorted(frames, frame, side="right")) - 1
    return lane.labels[index] if 0 <= index < len(lane.labels) else ""


def _column_coverage(a: np.ndarray, b: np.ndarray, width: int) -> np.ndarray:
    """How much of each pixel column the spans ``[a, b)`` cover, capped at 1."""
    a, b = np.clip(a, 0, width), np.clip(b, 0, width)
    c0, c1 = a.astype(int), b.astype(int)
    n = width + 2
    one = c0 == c1
    cover = np.bincount(c0[one], (b - a)[one], minlength=n)
    c0, c1, a, b = c0[~one], c1[~one], a[~one], b[~one]
    cover += np.bincount(c0, c0 + 1 - a, minlength=n) + np.bincount(c1, b - c1, minlength=n)
    # Columns wholly inside a span: +1 from the one after its first to its last.
    cover += np.cumsum(np.bincount(c0 + 1, minlength=n) - np.bincount(c1, minlength=n))
    return np.minimum(cover[:width], 1.0)


def _clock(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    whole = int(round(abs(seconds)))
    return f"{sign}{whole // 60}:{whole % 60:02d}"


def _smaller(font: QFont, by: float = 2.0) -> QFont:
    size = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
    small = QFont(font)
    small.setPointSizeF(max(7.0, size - by))
    return small


@dataclass(frozen=True)
class Row:
    """One horizontal band of the timeline, header and lane together."""

    kind: str  # "group" | "behavior" | "hardware"
    key: str  # group key, behaviour source, or device key
    label: str
    sub: str
    lane: BehaviorLane | Lane | None
    top: float
    height: float


class TimelineView(QWidget):
    """Every lane of a session under one playhead, with its track headers.

    Clicking the ruler scrubs; clicking a lane moves the playhead; dragging
    across lanes selects a range (input handling is in the next section).
    Selection and playhead stay separate, so a range survives scrubbing
    around inside it.
    """

    scrubbed = pyqtSignal(int)  # frame
    selection_changed = pyqtSignal(int, int)  # start, end frame (inclusive)
    selection_cleared = pyqtSignal()
    viewport_changed = pyqtSignal()
    playhead_moved = pyqtSignal(int)
    context_menu_requested = pyqtSignal(QPoint, int)  # global position, frame
    hidden_changed = pyqtSignal(list)  # lane keys

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(TOP_H + BEHAVIOR_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._view = None
        self._timeline: Timeline | None = None
        self._order: list[str] = []
        self._codes: dict[str, np.ndarray] = {}
        self._palette: list[QColor] = []
        self._runs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._snap_frames = np.empty(0)
        self._vp: Viewport | None = None
        self._static: QPixmap | None = None
        self._static_key = None
        self._frame = 0
        self._selection: tuple[int, int] | None = None
        self._drag: str | None = None
        self._fixed: int | None = None  # trim's opposite edge, fixed for the drag
        self._press_x = 0.0
        self._snap_on = True
        self._collapsed: set[str] = set()
        self._hidden: set[str] = set()

    # ------------------------------------------------------------------
    # loading

    def set_view(self, view) -> None:
        """Adopt a :class:`SessionView`. Kept for ethogram-only sessions."""
        self._view = view
        self._reload()

    def set_timeline(self, timeline: Timeline | None) -> None:
        self._timeline = timeline
        self._reload()
        self.updateGeometry()

    def set_session(self, view, timeline: Timeline | None) -> None:
        """Both at once, so a session switch rebuilds once, not twice."""
        self._view, self._timeline = view, timeline
        self._reload()
        self.updateGeometry()

    def timeline(self) -> Timeline | None:
        return self._timeline

    def _reload(self) -> None:
        self._rebuild_codes()
        self._rebuild_snap()
        lo, hi = self._axis_bounds()
        span = self._axis_span_of_frames(MIN_SPAN_FRAMES)
        self._vp = Viewport(lo, hi, min_span=span) if hi > lo else None
        self._static = None
        self._frame = self.frame_bounds()[0]
        self._selection = None
        self.update()
        self.viewport_changed.emit()

    def _behavior_lanes(self) -> list[BehaviorLane]:
        """A timeline's own lanes when it has any; otherwise one from the view."""
        if self._timeline is not None and self._timeline.behavior:
            return list(self._timeline.behavior)
        # All-blank labels (a recording without behavioral_state) would draw
        # an empty lane titled "Classifier".
        if self._view is not None and any(self._view.labels):
            return [
                BehaviorLane(
                    source="ethogram",
                    labels=list(self._view.labels),
                    frames=self._view.frames,
                )
            ]
        return []

    def _rebuild_codes(self) -> None:
        """Label-to-colour-slot per lane, against one shared order.

        The order is pooled across lanes so a behaviour is one colour whichever
        source scored it; the per-column majority needs labels as integers.
        """
        lanes = self._behavior_lanes()
        pooled: list[str] = []
        for lane in lanes:
            pooled.extend(lane.labels)
        self._order = behavior_order(pooled) if pooled else []
        slot = {name: i + 1 for i, name in enumerate(self._order)}  # 0 = unscored
        self._codes = {
            lane.source: np.array([slot.get(v, 0) for v in lane.labels], dtype=np.int64)
            for lane in lanes
        }
        self._palette = [behavior_qcolor("", self._order)] + [
            behavior_qcolor(name, self._order) for name in self._order
        ]
        self._runs = {}

    def behavior_order(self) -> list[str]:
        """The pooled order the lanes were coloured from (see ``_rebuild_codes``)."""
        return list(self._order)

    def _lane_runs(self, lane: BehaviorLane):
        """``(first, next_first, code)`` per run of one label: half-open in frames."""
        cached = self._runs.get(lane.source)
        if cached is None:
            codes = self._codes.get(lane.source, np.empty(0, dtype=np.int64))
            frames = np.asarray(lane.frames)
            if len(codes) == 0:
                empty = np.empty(0, dtype=np.int64)
                cached = (empty, empty, empty)
            else:
                change = np.flatnonzero(np.diff(codes)) + 1
                starts = np.concatenate([[0], change])
                following = np.concatenate([frames[starts[1:]], [frames[-1] + 1]])
                cached = (frames[starts], following, codes[starts])
            self._runs[lane.source] = cached
        return cached

    def _rebuild_snap(self) -> None:
        """Frames worth snapping to: bout starts and hardware switches."""
        frames: list[float] = []
        for lane in self._behavior_lanes():
            first, following, _codes = self._lane_runs(lane)
            frames.extend(first.tolist())
            frames.extend(following.tolist())
        if self._timeline is not None and self.uses_ms():
            frame_map = self._timeline.frame_map
            for lane in self._timeline.lanes:
                frames.extend(frame_map.frame_at(s.start_ms) for s in lane.segments)
        self._snap_frames = np.unique(np.asarray(frames, dtype=float))

    # ------------------------------------------------------------------
    # axis

    def uses_ms(self) -> bool:
        t = self._timeline
        return t is not None and t.frame_map is not None and t.end_ms > t.start_ms

    def fps(self) -> float:
        return float(getattr(self._view, "fps", 0) or 30.0)

    def frame_bounds(self) -> tuple[int, int]:
        """``(first, last)`` frame the timeline covers.

        From the frame map when there is one; otherwise the behaviour lanes'
        own range, which is deliberately not zero -- a windowed run scores
        minutes two to seven, so its ethogram starts at frame 3600.
        """
        if self.uses_ms():
            frame_map = self._timeline.frame_map
            return (
                frame_map.frame_at(self._timeline.start_ms),
                frame_map.frame_at(self._timeline.end_ms),
            )
        lanes = [ln for ln in self._behavior_lanes() if len(ln.frames)]
        if lanes:
            return (
                min(int(np.asarray(ln.frames)[0]) for ln in lanes),
                max(int(np.asarray(ln.frames)[-1]) for ln in lanes),
            )
        return 0, 0

    def _axis_bounds(self) -> tuple[float, float]:
        if self.uses_ms():
            return self._timeline.start_ms, self._timeline.end_ms
        first, last = self.frame_bounds()
        # An empty or one-frame session has no span, hence no viewport.
        return float(first), float(last + 1 if last > first else first)

    def axis_of_frame(self, frame: int) -> float:
        if self.uses_ms():
            return self._timeline.frame_map.ms_of(frame)
        return float(frame)

    def _axis_of_frames(self, frames: np.ndarray) -> np.ndarray:
        if self.uses_ms():
            frame_map = self._timeline.frame_map
            return np.interp(np.asarray(frames, dtype=float), frame_map.frames, frame_map.ms)
        return np.asarray(frames, dtype=float)

    def frame_of_axis(self, t: float) -> int:
        first, last = self.frame_bounds()
        if self.uses_ms():
            frame = self._timeline.frame_map.frame_at(t)
        else:
            frame = int(np.floor(t))
        return max(first, min(last, frame))

    def _axis_span_of_frames(self, n: int) -> float:
        return n / self.fps() * 1000.0 if self.uses_ms() else float(n)

    def _axis_of_seconds(self, seconds: float) -> float:
        return seconds * 1000.0 if self.uses_ms() else seconds * self.fps()

    def seconds_of_axis(self, t: float) -> float:
        return t / 1000.0 if self.uses_ms() else t / self.fps()

    # ------------------------------------------------------------------
    # geometry

    def _lane_width(self) -> float:
        return max(0.0, float(self.width() - HEADER_W))

    def lanes_rect(self) -> QRectF:
        return QRectF(HEADER_W, TOP_H, self._lane_width(), max(0.0, self.height() - TOP_H))

    def lane_rect(self, key: str) -> QRectF | None:
        """Where a behaviour lane (by source) or device lane (by key) is drawn."""
        for row in self._rows():
            if row.kind != "group" and row.key == key:
                return QRectF(HEADER_W, row.top, self._lane_width(), row.height)
        return None

    def x_of_axis(self, t: float) -> float:
        return HEADER_W + (self._vp.x_of(t, self._lane_width()) if self._vp else 0.0)

    def _xs(self, ts: np.ndarray) -> np.ndarray:
        vp = self._vp
        if vp is None or vp.span <= 0:
            return np.full(len(ts), float(HEADER_W))
        return HEADER_W + (np.asarray(ts, dtype=float) - vp.start) / vp.span * self._lane_width()

    def x_of_frame(self, frame: int) -> float:
        return self.x_of_axis(self.axis_of_frame(frame))

    def frame_at_x(self, x: float) -> int:
        if self._vp is None:
            return self.frame_bounds()[0]
        return self.frame_of_axis(self._vp.t_at(x - HEADER_W, self._lane_width()))

    def _rows(self) -> list[Row]:
        rows: list[Row] = []
        top = float(TOP_H)
        behavior = self._behavior_lanes()
        if behavior:
            rows.append(Row("group", "behavior", "Behavior", "", None, top, GROUP_H))
            top += GROUP_H
            if "behavior" not in self._collapsed:
                for lane in behavior:
                    title, sub = _behavior_title(lane.source)
                    rows.append(Row("behavior", lane.source, title, sub, lane, top, BEHAVIOR_H))
                    top += BEHAVIOR_H
        hardware = [
            ln
            for ln in (self._timeline.lanes if self._timeline else [])
            if ln.key not in self._hidden
        ]
        boards = list(dict.fromkeys(ln.board_id for ln in hardware))
        shown = [ln for ln in hardware if f"board:{ln.board_id}" not in self._collapsed]
        # Rows compress rather than overflow when a rig has more devices than
        # the panel has pixels -- down to a floor, never to zero.
        room = self.height() - top - GROUP_H * len(boards)
        each = HW_H if not shown else max(HW_MIN_H, min(HW_H, room / len(shown)))
        for board in boards:
            key = f"board:{board}"
            rows.append(Row("group", key, board or "board", "", None, top, GROUP_H))
            top += GROUP_H
            if key in self._collapsed:
                continue
            for lane in (ln for ln in hardware if ln.board_id == board):
                rows.append(Row("hardware", lane.key, lane.label, lane.pin_type, lane, top, each))
                top += each
        return rows

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        behavior = self._behavior_lanes()
        hardware = self._timeline.lanes if self._timeline else []
        boards = len({ln.board_id for ln in hardware})
        height = TOP_H + (GROUP_H + BEHAVIOR_H * len(behavior) if behavior else 0)
        height += GROUP_H * boards + HW_H * len(hardware)
        return QSize(HEADER_W + 400, max(height, TOP_H + BEHAVIOR_H))

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        self._static = None
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # state

    def set_frame(self, frame: int, *, follow: bool = False) -> None:
        self._frame = int(frame)
        if follow and self._vp is not None:
            before = (self._vp.start, self._vp.end)
            self._vp.follow(self.axis_of_frame(self._frame))
            if (self._vp.start, self._vp.end) != before:
                self.viewport_changed.emit()
        self.playhead_moved.emit(self._frame)
        self.update()

    def current_frame(self) -> int:
        return self._frame

    def selection(self) -> tuple[int, int] | None:
        return self._selection

    def set_selection(self, start: int, end: int) -> None:
        self._selection = (int(min(start, end)), int(max(start, end)))
        self.update()
        self.selection_changed.emit(*self._selection)

    def clear_selection(self) -> None:
        if self._selection is None:
            return
        self._selection = None
        self.update()
        self.selection_cleared.emit()

    @property
    def viewport(self) -> Viewport | None:
        return self._vp

    def refresh_viewport(self) -> None:
        """Repaint after the viewport moved (navigator, wheel, zoom buttons)."""
        self.update()
        self.viewport_changed.emit()

    def fit(self) -> None:
        if self._vp is not None:
            self._vp.fit()
            self.refresh_viewport()

    def zoom_to_selection(self) -> None:
        if self._vp is None or self._selection is None:
            return
        start, end = self._selection
        self._vp.show(self.axis_of_frame(start), self.axis_of_frame(end + 1))
        self.refresh_viewport()

    def set_snap(self, enabled: bool) -> None:
        self._snap_on = bool(enabled)

    def set_hidden(self, keys) -> None:
        self._hidden = set(keys)
        self._static = None
        self.update()
        self.hidden_changed.emit(sorted(self._hidden))

    def hidden(self) -> set[str]:
        return set(self._hidden)

    def _toggle_group(self, key: str) -> None:
        self._collapsed ^= {key}
        self.update()
        self.updateGeometry()

    # ------------------------------------------------------------------
    # painting: static layer (cached per size, window and layout)

    def _static_pixmap(self, rows: list[Row]) -> QPixmap:
        vp = self._vp
        key = (
            self.width(),
            self.height(),
            self.devicePixelRatioF(),
            None if vp is None else (vp.start, vp.end),
            frozenset(self._collapsed),
            frozenset(self._hidden),
            id(self._timeline),
            id(self._view),
        )
        if self._static is None or self._static_key != key:
            ratio = self.devicePixelRatioF()
            pixmap = QPixmap(int(self.width() * ratio), int(self.height() * ratio))
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(QColor(colors.BASE))
            painter = QPainter(pixmap)
            try:
                self._paint_static(painter, rows)
            finally:
                painter.end()
            self._static, self._static_key = pixmap, key
        return self._static

    def _paint_static(self, p: QPainter, rows: list[Row]) -> None:
        w, h = float(self.width()), float(self.height())
        p.fillRect(QRectF(0, 0, HEADER_W, h), QColor(colors.SURFACE_1))
        p.fillRect(QRectF(HEADER_W, RULER_H, w - HEADER_W, MARKER_H), QColor(colors.CHROME))
        self._paint_ruler(p)
        for row in rows:
            lane_rect = QRectF(HEADER_W, row.top, w - HEADER_W, row.height)
            if row.kind == "group":
                p.fillRect(QRectF(0, row.top, w, row.height), QColor(colors.CHROME))
                self._paint_group_header(p, row)
            else:
                p.fillRect(lane_rect, QColor(colors.CANVAS))
                if row.kind == "behavior":
                    self._paint_behavior(p, row.lane, lane_rect.adjusted(0, 3, 0, -3))
                else:
                    self._paint_hardware(p, row.lane, lane_rect)
                self._paint_lane_header(p, row)
            p.setPen(QPen(QColor(colors.BORDER), 1))
            y = row.top + row.height - 0.5
            p.drawLine(QPointF(0, y), QPointF(w, y))
        p.setPen(QPen(QColor(colors.BORDER), 1))
        p.drawLine(QPointF(HEADER_W - 0.5, 0), QPointF(HEADER_W - 0.5, h))
        p.drawLine(QPointF(HEADER_W, TOP_H - 0.5), QPointF(w, TOP_H - 0.5))
        self._paint_flow(p)

    def _paint_ruler(self, p: QPainter) -> None:
        if self._vp is None:
            return
        vp = self._vp
        s0, s1 = self.seconds_of_axis(vp.start), self.seconds_of_axis(vp.end)
        major, minor = tick_spacing(s1 - s0)
        p.save()
        p.setClipRect(QRectF(HEADER_W, 0, self._lane_width(), RULER_H))
        p.fillRect(QRectF(HEADER_W, 0, self._lane_width(), RULER_H), QColor(colors.BASE))
        if self.uses_ms() and vp.start < 0:
            shade = colors.qcolor_with_alpha(QColor(colors.TEXT_TERTIARY), 0.07)
            p.fillRect(QRectF(HEADER_W, 0, self.x_of_axis(0.0) - HEADER_W, RULER_H), shade)
        p.setFont(data_font(9))
        for k in range(int(np.ceil(s0 / minor)), int(np.floor(s1 / minor)) + 1):
            second = k * minor
            x = self.x_of_axis(self._axis_of_seconds(second))
            is_major = second % major == 0
            p.setPen(QPen(QColor(colors.TEXT_DISABLED if is_major else colors.BORDER), 1))
            p.drawLine(QPointF(x, RULER_H), QPointF(x, RULER_H - (10 if is_major else 4)))
            if is_major:
                p.setPen(QColor(colors.TEXT_MUTED))
                p.drawText(QPointF(x + 3, 12), _clock(second))
        p.restore()

    def _paint_flow(self, p: QPainter) -> None:
        """Flow boundaries as rules across every lane; pre-flow hatched."""
        t = self._timeline
        if t is None or not self.uses_ms() or self._vp is None:
            return
        lanes = self.lanes_rect()
        p.save()
        p.setClipRect(QRectF(HEADER_W, 0, self._lane_width(), self.height()))
        if t.flow_start_ms is not None:
            x0 = self.x_of_axis(t.flow_start_ms)
            if x0 > HEADER_W:
                hatch = QBrush(
                    colors.qcolor_with_alpha(QColor(colors.TEXT_TERTIARY), 0.10),
                    Qt.BrushStyle.BDiagPattern,
                )
                p.fillRect(QRectF(HEADER_W, lanes.top(), x0 - HEADER_W, lanes.height()), hatch)
        p.setPen(QPen(QColor(colors.TEXT_MUTED), 1, Qt.PenStyle.DashLine))
        for boundary in (t.flow_start_ms, t.flow_end_ms):
            if boundary is not None:
                x = self.x_of_axis(boundary)
                p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.restore()

    def _paint_group_header(self, p: QPainter, row: Row) -> None:
        font = _smaller(self.font(), 3)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        p.setFont(font)
        p.setPen(QColor(colors.TEXT_MUTED))
        caret = "▸" if row.key in self._collapsed else "▾"
        p.drawText(
            QRectF(10, row.top, HEADER_W - 20, row.height),
            Qt.AlignmentFlag.AlignVCenter,
            f"{caret}  {row.label.upper()}",
        )

    def _row_colour(self, row: Row) -> QColor:
        if row.kind == "hardware":
            return lane_colour(row.lane)
        return QColor(colors.TEXT_TERTIARY)

    def _paint_lane_header(self, p: QPainter, row: Row) -> None:
        p.fillRect(QRectF(0, row.top + 1, 3, row.height - 2), self._row_colour(row))
        p.setFont(self.font())
        p.setPen(QColor(colors.TEXT_PRIMARY))
        room = HEADER_W - 96
        if row.kind == "behavior" and row.height >= 26:
            p.drawText(
                QRectF(10, row.top + 2, room, row.height / 2),
                Qt.AlignmentFlag.AlignBottom,
                p.fontMetrics().elidedText(row.label, Qt.TextElideMode.ElideRight, room),
            )
            p.setFont(_smaller(self.font(), 2.5))
            p.setPen(QColor(colors.TEXT_DISABLED))
            p.drawText(
                QRectF(10, row.top + row.height / 2, room, row.height / 2 - 2),
                Qt.AlignmentFlag.AlignTop,
                p.fontMetrics().elidedText(row.sub, Qt.TextElideMode.ElideRight, room),
            )
            return
        p.drawText(
            QRectF(10, row.top, room, row.height),
            Qt.AlignmentFlag.AlignVCenter,
            p.fontMetrics().elidedText(row.label, Qt.TextElideMode.ElideRight, room),
        )

    def _uses_clips(self, lane: BehaviorLane) -> bool:
        """Whether runs are wide enough on screen to draw as labelled clips."""
        first, following, _codes = self._lane_runs(lane)
        if self._vp is None or not len(first):
            return False
        x0 = self._xs(self._axis_of_frames(first))
        x1 = self._xs(self._axis_of_frames(following))
        visible = (x1 > HEADER_W) & (x0 < self.width())
        return bool(visible.any()) and float(np.median((x1 - x0)[visible])) >= CLIP_MIN_PX

    def _paint_behavior(self, p: QPainter, lane: BehaviorLane, rect: QRectF) -> None:
        if self._vp is None or not lane.labels:
            return
        p.save()
        p.setClipRect(rect)
        if self._uses_clips(lane):
            self._paint_clips(p, lane, rect)
        else:
            self._paint_columns(p, lane, rect, self._vp.start, self._vp.end)
        p.restore()

    def _paint_clips(self, p: QPainter, lane: BehaviorLane, rect: QRectF) -> None:
        """Zoomed in: each bout a rounded clip carrying its name, as Resolve names clips."""
        first, following, codes = self._lane_runs(lane)
        x0 = self._xs(self._axis_of_frames(first))
        x1 = self._xs(self._axis_of_frames(following))
        visible = np.flatnonzero((x1 > rect.left()) & (x0 < rect.right()) & (codes > 0))
        font = QFont(self.font())
        font.setBold(True)
        p.setFont(font)
        metrics = p.fontMetrics()
        fps = self.fps()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for i in visible:
            a, b = float(x0[i]), float(x1[i])
            colour = self._palette[int(codes[i])]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(colour)
            p.drawRoundedRect(
                QRectF(a + 0.5, rect.top(), max(1.0, b - a - 1.0), rect.height()), 3, 3
            )
            left = max(a, rect.left()) + 6
            room = b - left - 4
            if room < 36:
                continue
            name = self._order[int(codes[i]) - 1]
            text = f"{name}  {(following[i] - first[i]) / fps:.1f} s" if room >= 92 else name
            p.setPen(QColor(readable_text_on(colour.name())))
            p.drawText(
                QRectF(left, rect.top(), room, rect.height()),
                Qt.AlignmentFlag.AlignVCenter,
                metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(room)),
            )
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def _paint_columns(
        self, p: QPainter, lane: BehaviorLane, rect: QRectF, start: float, end: float
    ) -> None:
        """Zoomed out: one band per pixel column, coloured by what dominates it.

        Not one rect per run: sub-pixel rects blend by coverage, and every colour
        became an average of several behaviours. Resolving whole columns keeps
        each pixel one behaviour's real colour. A column past the lane's own
        scored frames is left as background -- clamping would invent coverage.
        """
        codes = self._codes.get(lane.source)
        frames = np.asarray(lane.frames)
        width = int(rect.width())
        if codes is None or width <= 0 or not len(frames) or end <= start:
            return
        edges_axis = start + np.arange(width + 1, dtype=float) * (end - start) / width
        if self.uses_ms():
            frame_map = self._timeline.frame_map
            index = np.clip(np.searchsorted(frame_map.ms, edges_axis), 0, len(frame_map.frames) - 1)
            edges = frame_map.frames[index].astype(np.int64)
        else:
            edges = np.floor(edges_axis).astype(np.int64)
        starts = np.searchsorted(frames, edges, side="left")
        scored_first, scored_last = int(frames[0]), int(frames[-1])
        n_codes = len(self._palette)
        left = rect.left()
        for x in range(width):
            lo, hi = int(starts[x]), int(starts[x + 1])
            if hi <= lo:
                if not scored_first <= int(edges[x]) <= scored_last:
                    continue
                lo = max(0, min(lo, len(frames) - 1))
                hi = lo + 1
            code = int(np.bincount(codes[lo:hi], minlength=n_codes).argmax())
            p.fillRect(QRectF(left + x, rect.top(), 1.0, rect.height()), self._palette[code])

    def paint_overview(self, painter: QPainter, rect: QRectF) -> None:
        """The first behaviour lane across the whole session, for the navigator."""
        lanes = self._behavior_lanes()
        if lanes and self._vp is not None:
            self._paint_columns(painter, lanes[0], rect, self._vp.lo, self._vp.hi)

    def _paint_hardware(self, p: QPainter, lane: Lane, rect: QRectF) -> None:
        """Held values as bars from the baseline: blocks for switches, stairs for levels."""
        if self._vp is None or not self.uses_ms():
            return
        colour = lane_colour(lane)
        inner = rect.adjusted(0, 3, 0, -2)
        baseline = inner.bottom()
        vp = self._vp
        p.save()
        p.setClipRect(rect)
        if lane.binary:
            self._paint_switches(p, lane, rect, inner, colour)
        else:
            for segment in lane.segments:
                if segment.level <= 0 or segment.end_ms < vp.start or segment.start_ms > vp.end:
                    continue
                x0, x1 = self.x_of_axis(segment.start_ms), self.x_of_axis(segment.end_ms)
                width = max(1.0, x1 - x0)
                height = inner.height() * segment.level
                bar = QRectF(x0, baseline - height, width, height)
                p.fillRect(bar, colors.qcolor_with_alpha(colour, 0.4))
                p.fillRect(QRectF(x0, baseline - height, width, 1.5), colour)
                left = max(x0, rect.left()) + 5
                if x1 - left >= 30:
                    p.setFont(data_font(8))
                    p.setPen(QColor(colors.TEXT_PRIMARY))
                    unit = "°" if lane.pin_type == "SERVO" else ""
                    p.drawText(QPointF(left, baseline - 3), f"{segment.value:g}{unit}")
        p.setPen(QPen(QColor(colors.TEXT_MUTED), 1))
        for marker in lane.markers:
            x = self.x_of_axis(marker.at_ms)
            p.drawLine(QPointF(x, inner.top()), QPointF(x, baseline))
        p.restore()

    def _paint_switches(self, p, lane: Lane, rect: QRectF, inner: QRectF, colour: QColor) -> None:
        """ON spans as blocks; a column holding more than one switch at its ON fraction.

        Solid would draw a 10 Hz, 50 % train exactly like a lamp held on for
        twenty seconds. Vectorised: a whole-session train is 30,000 segments,
        and a fillRect plus an x_of_axis call for each cost ~60 ms a repaint.
        Spans >= 1 px draw individually; narrower pulses only as columns, and a
        column holding one lone pulse draws it solid, 1 px wide.
        """
        width = int(rect.width())
        if width <= 0 or not lane.segments:
            return
        left = rect.left()
        x0 = self._xs(lane.starts_ms) - left
        x1 = self._xs(lane.ends_ms) - left
        on = (x1 >= 0) & (x0 <= width) & (lane.values > 0)
        wide = on & (x1 - x0 >= 1)
        for a, b in zip(x0[wide], x1[wide], strict=True):
            p.fillRect(QRectF(left + a, inner.top(), b - a, inner.height()), colour)
        starts = x0[(x0 >= 0) & (x0 < width)].astype(int)
        switches = np.bincount(starts, minlength=width)
        pulses = np.bincount(np.clip(x0[on & ~wide], 0, width - 1).astype(int), minlength=width)
        lone = (pulses == 1) & (switches <= 2)  # one pulse's own rise and fall
        busy = (switches > 1) & ~lone
        for column in np.flatnonzero((pulses > 0) & ~busy):
            p.fillRect(QRectF(left + column, inner.top(), 1.0, inner.height()), colour)
        if not busy.any():
            return
        coverage = _column_coverage(x0[on], x1[on], width)
        background = QColor(colors.CANVAS)
        for column in np.flatnonzero(busy):
            x = QRectF(left + column, inner.top(), 1.0, inner.height())
            p.fillRect(x, background)
            if coverage[column] > 0:
                p.fillRect(x, colors.qcolor_with_alpha(colour, float(coverage[column])))

    # ------------------------------------------------------------------
    # painting: live layer (every frame of playback)

    def paintEvent(self, _event):  # noqa: N802 - Qt override
        p = QPainter(self)
        rows = self._rows()
        if not rows:
            p.fillRect(self.rect(), QColor(colors.BASE))
            p.setPen(QColor(colors.TEXT_MUTED))
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a session to see its timeline"
            )
            return
        p.drawPixmap(0, 0, self._static_pixmap(rows))
        self._paint_corner(p)
        self._paint_values(p, rows)
        self._paint_selection(p)
        self._paint_playhead(p)

    def _paint_corner(self, p: QPainter) -> None:
        p.fillRect(QRectF(0, 0, HEADER_W, TOP_H), QColor(colors.SURFACE_1))
        seconds = self.seconds_of_axis(self.axis_of_frame(self._frame))
        p.setFont(data_font(12))
        p.setPen(QColor(colors.PLAYHEAD))
        p.drawText(
            QRectF(10, 4, HEADER_W - 20, 22),
            Qt.AlignmentFlag.AlignVCenter,
            format_timecode(seconds, self.fps()),
        )
        flow = self.uses_ms() and self._timeline.flow_start_ms is not None
        p.setFont(_smaller(self.font(), 3))
        p.setPen(QColor(colors.TEXT_DISABLED))
        p.drawText(
            QRectF(10, 25, HEADER_W - 20, 16),
            Qt.AlignmentFlag.AlignVCenter,
            "flow-relative" if flow else "video-relative",
        )

    def _paint_values(self, p: QPainter, rows: list[Row]) -> None:
        """Each track's value at the playhead, like an audio track's meter."""
        ms = self.axis_of_frame(self._frame) if self.uses_ms() else None
        p.setFont(data_font(8))
        metrics = p.fontMetrics()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for row in rows:
            dot = None
            if row.kind == "behavior":
                name = _label_at(row.lane, self._frame)
                text, active = (name or "—"), bool(name)
                if name in self._order:
                    dot = self._palette[self._order.index(name) + 1]
            elif row.kind == "hardware" and ms is not None:
                text, active = describe_value(row.lane, ms)
            else:
                continue
            if row.height < 14:
                continue
            width = metrics.horizontalAdvance(text) + 12 + (10 if dot is not None else 0)
            chip = QRectF(HEADER_W - 8 - width, row.top + (row.height - 16) / 2, width, 16)
            tone = (
                self._row_colour(row)
                if active and row.kind == "hardware"
                else QColor(colors.TEXT_MUTED)
            )
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(colors.qcolor_with_alpha(tone, 0.16 if active else 0.08))
            p.drawRoundedRect(chip, 4, 4)
            x = chip.left() + 6
            if dot is not None:
                p.setBrush(dot)
                p.drawRoundedRect(QRectF(x, chip.center().y() - 3.5, 7, 7), 2, 2)
                x += 10
            p.setPen(tone.lighter(130) if active else QColor(colors.TEXT_MUTED))
            p.drawText(
                QRectF(x, chip.top(), chip.right() - x, chip.height()),
                Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def _paint_selection(self, p: QPainter) -> None:
        """Shade what is EXCLUDED, never tint what is chosen.

        Tinting the selection dragged every behaviour inside it toward the
        accent -- and the usual selection is the whole session.
        """
        if self._selection is None or self._vp is None:
            return
        start, end = self._selection
        x0, x1 = self.x_of_frame(start), self.x_of_frame(end + 1)
        lanes = self.lanes_rect()
        p.save()
        p.setClipRect(QRectF(HEADER_W, 0, self._lane_width(), self.height()))
        scrim = colors.qcolor_with_alpha(QColor(colors.BASE), 0.72)
        p.fillRect(QRectF(HEADER_W, lanes.top(), max(0.0, x0 - HEADER_W), lanes.height()), scrim)
        p.fillRect(QRectF(x1, lanes.top(), max(0.0, self.width() - x1), lanes.height()), scrim)
        accent = QColor(colors.ACCENT)
        p.fillRect(QRectF(x0, 0, x1 - x0, RULER_H), colors.qcolor_with_alpha(accent, 0.16))
        p.fillRect(QRectF(x0, RULER_H - 3, x1 - x0, 3), accent)
        p.setPen(QPen(accent, 2))
        p.drawLine(QPointF(x0, 0), QPointF(x0, self.height()))
        p.drawLine(QPointF(x1, 0), QPointF(x1, self.height()))
        p.restore()

    def _paint_playhead(self, p: QPainter) -> None:
        if self._vp is None:
            return
        x = self.x_of_frame(self._frame)
        if not HEADER_W <= x <= self.width():
            return
        colour = QColor(colors.PLAYHEAD)
        p.setPen(QPen(colour, 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        head = QPainterPath()
        head.moveTo(x - 6, 0)
        head.lineTo(x + 6, 0)
        head.lineTo(x, 9)
        head.closeSubpath()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillPath(head, colour)

    # ------------------------------------------------------------------
    # input

    def _row_at(self, y: float) -> Row | None:
        return next((r for r in self._rows() if r.top <= y < r.top + r.height), None)

    def _edge_at(self, x: float) -> str | None:
        if self._selection is None:
            return None
        if abs(x - self.x_of_frame(self._selection[0])) <= EDGE_PX:
            return "trim-in"
        if abs(x - self.x_of_frame(self._selection[1] + 1)) <= EDGE_PX:
            return "trim-out"
        return None

    def _snapped(self, x: float, *, end: bool = False) -> int:
        """The frame under ``x``, pulled onto a nearby bout edge or hardware switch.

        Edges are first frames; a range's *end* is inclusive, so it snaps to the
        frame before an edge.
        """
        frame = self.frame_at_x(x)
        if not self._snap_on or self._snap_frames.size == 0:
            return frame
        tolerance = max(1, abs(self.frame_at_x(x + SNAP_PX) - frame))
        candidates = self._snap_frames - 1 if end else self._snap_frames
        first, last = self.frame_bounds()
        return max(first, min(last, int(round(snap(frame, candidates, tolerance)))))

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if self._vp is None or event.button() != Qt.MouseButton.LeftButton:
            return
        x, y = event.position().x(), event.position().y()
        if x < HEADER_W:
            row = self._row_at(y)
            if row is not None and row.kind == "group":
                self._toggle_group(row.key)
            return
        if y < TOP_H:
            self._drag = "scrub"
            self.scrubbed.emit(self.frame_at_x(x))
            return
        edge = self._edge_at(x)
        if edge is not None:
            self._fixed = self._selection[1] if edge == "trim-in" else self._selection[0]
            self._drag = "select"
            return
        self._drag = "press"
        self._press_x = x
        self._fixed = None

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        x = event.position().x()
        if self._drag is not None and not (event.buttons() & Qt.MouseButton.LeftButton):
            # The release was missed (e.g. it happened outside the widget).
            self._drag = None
            self._fixed = None
            return
        if self._drag is None:
            grab = event.position().y() >= TOP_H and self._edge_at(x) is not None
            self.setCursor(Qt.CursorShape.SizeHorCursor if grab else Qt.CursorShape.ArrowCursor)
            return
        if self._drag == "scrub":
            self.scrubbed.emit(self.frame_at_x(x))
            return
        if self._drag == "press":
            if abs(x - self._press_x) < DRAG_PX:
                return
            self._drag = "select"
        if self._drag == "select":
            if self._fixed is not None:  # trimming: the other edge stays put
                edge = self._snapped(x, end=self.frame_at_x(x) >= self._fixed)
                self.set_selection(self._fixed, edge)
            else:  # a fresh range: snap each end by the role it ends up playing
                lo, hi = sorted((x, self._press_x))
                self.set_selection(self._snapped(lo), self._snapped(hi, end=True))

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drag == "press":
            # A click, not a drag: move the playhead, keep any selection.
            self.scrubbed.emit(self.frame_at_x(self._press_x))
        self._drag = None
        self._fixed = None

    def wheelEvent(self, event):  # noqa: N802 - Qt override
        if self._vp is None:
            return
        modifiers = event.modifiers()
        dx, dy = event.angleDelta().x(), event.angleDelta().y()
        zooming = modifiers & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        )
        if zooming:
            about = self._vp.t_at(event.position().x() - HEADER_W, self._lane_width())
            self._vp.zoom(0.85 ** (dy / 120.0), about)
        else:
            steps = (dx or dy) / 120.0
            self._vp.pan(-steps * 0.1 * self._vp.span)
        self._static = None
        self.refresh_viewport()
        event.accept()

    def event(self, event):  # noqa: D401 - Qt override
        # Trackpad pinch arrives as a native gesture, not a wheel event.
        if event.type() == QEvent.Type.NativeGesture and self._vp is not None:
            if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                about = self._vp.t_at(event.position().x() - HEADER_W, self._lane_width())
                self._vp.zoom(max(0.1, 1.0 - event.value()), about)
                self.refresh_viewport()
                return True
        return super().event(event)

    def contextMenuEvent(self, event):  # noqa: N802 - Qt override
        pos = event.pos()
        if pos.x() < HEADER_W:
            row = self._row_at(pos.y())
            if row is not None:
                menu = self._header_menu(row)
                if not menu.isEmpty():
                    menu.exec(event.globalPos())
                menu.deleteLater()
            return
        if self._vp is not None and pos.y() >= RULER_H:
            self.context_menu_requested.emit(event.globalPos(), self.frame_at_x(pos.x()))

    def _header_menu(self, row: Row) -> QMenu:
        menu = QMenu(self)
        if row.kind == "hardware":
            menu.addAction(f"Hide {row.label}", lambda: self.set_hidden(self._hidden | {row.key}))
        if row.kind == "group":
            label = "Expand group" if row.key in self._collapsed else "Collapse group"
            menu.addAction(label, lambda: self._toggle_group(row.key))
        if self._hidden:
            menu.addAction(
                f"Show hidden lanes ({len(self._hidden)})", lambda: self.set_hidden(set())
            )
        return menu


def _span_text(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else _clock(seconds)


class Navigator(QWidget):
    """The whole session, always, with a box around what the timeline shows.

    Drag the box to pan, drag its edges to zoom, click outside it to jump there.
    Resolve's Cut page keeps a whole-session timeline above the zoomed one for
    the same reason: at 45,000 frames you have to see where you are.
    """

    HEIGHT = 48
    EDGE_PX = 5

    def __init__(self, view: TimelineView, parent=None):
        super().__init__(parent)
        self._view = view
        self.setFixedHeight(self.HEIGHT)
        self.setMouseTracking(True)
        self._drag: str | None = None
        self._grab = 0.0
        self._strip: QPixmap | None = None
        self._strip_key = None
        view.viewport_changed.connect(self.update)
        view.playhead_moved.connect(lambda _frame: self.update())
        view.selection_changed.connect(lambda *_: self.update())
        view.selection_cleared.connect(self.update)

    def _lane_width(self) -> float:
        return max(1.0, float(self.width() - HEADER_W))

    def _x(self, t: float) -> float:
        vp = self._view.viewport
        return HEADER_W + (t - vp.lo) / (vp.hi - vp.lo) * self._lane_width()

    def _t(self, x: float) -> float:
        vp = self._view.viewport
        return vp.lo + (x - HEADER_W) / self._lane_width() * (vp.hi - vp.lo)

    def box(self) -> tuple[float, float] | None:
        vp = self._view.viewport
        if vp is None or vp.hi <= vp.lo:
            return None
        return self._x(vp.start), self._x(vp.end)

    def _strip_pixmap(self) -> QPixmap:
        width, height = int(self._lane_width()), self.height()
        ratio = self.devicePixelRatioF()
        key = (width, height, ratio, id(self._view.timeline()), id(self._view._view))
        if self._strip is None or self._strip_key != key:
            pixmap = QPixmap(max(1, int(width * ratio)), max(1, int(height * ratio)))
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(QColor(colors.CHROME))
            painter = QPainter(pixmap)
            try:
                self._view.paint_overview(painter, QRectF(0, 7, width, height - 14))
            finally:
                painter.end()
            self._strip, self._strip_key = pixmap, key
        return self._strip

    def paintEvent(self, _event):  # noqa: N802 - Qt override
        p = QPainter(self)
        h = float(self.height())
        p.fillRect(self.rect(), QColor(colors.CHROME))
        p.fillRect(QRectF(0, 0, HEADER_W, h), QColor(colors.SURFACE_1))
        caption = _smaller(self.font(), 3)
        caption.setBold(True)
        p.setFont(caption)
        p.setPen(QColor(colors.TEXT_MUTED))
        p.drawText(
            QRectF(10, h / 2 - 15, HEADER_W - 20, 14), Qt.AlignmentFlag.AlignVCenter, "NAVIGATOR"
        )
        box = self.box()
        if box is not None:
            view, vp = self._view, self._view.viewport
            shown = view.seconds_of_axis(vp.end) - view.seconds_of_axis(vp.start)
            total = view.seconds_of_axis(vp.hi) - view.seconds_of_axis(vp.lo)
            p.setFont(_smaller(self.font(), 2))
            p.setPen(QColor(colors.TEXT_DISABLED))
            p.drawText(
                QRectF(10, h / 2, HEADER_W - 20, 14),
                Qt.AlignmentFlag.AlignVCenter,
                f"{_span_text(shown)} of {_clock(total)} shown",
            )
            p.drawPixmap(HEADER_W, 0, self._strip_pixmap())
            selection = view.selection()
            if selection is not None:
                a = self._x(view.axis_of_frame(selection[0]))
                b = self._x(view.axis_of_frame(selection[1] + 1))
                p.fillRect(
                    QRectF(a, 0, b - a, h), colors.qcolor_with_alpha(QColor(colors.ACCENT), 0.25)
                )
            x = self._x(view.axis_of_frame(view.current_frame()))
            p.fillRect(QRectF(x, 0, 1.5, h), QColor(colors.PLAYHEAD))
            x0, x1 = box
            dim = colors.qcolor_with_alpha(QColor(colors.CANVAS), 0.6)
            p.fillRect(QRectF(HEADER_W, 0, x0 - HEADER_W, h), dim)
            p.fillRect(QRectF(x1, 0, self.width() - x1, h), dim)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(QPen(QColor(colors.TEXT_PRIMARY), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(x0, 1.5, max(2.0, x1 - x0), h - 3), 4, 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(colors.TEXT_PRIMARY))
            for edge in (x0, x1):
                p.drawRoundedRect(QRectF(edge - 2, h / 2 - 7, 4, 14), 2, 2)
        p.setPen(QPen(QColor(colors.BORDER), 1))
        p.drawLine(QPointF(HEADER_W - 0.5, 0), QPointF(HEADER_W - 0.5, h))
        p.drawLine(QPointF(0, h - 0.5), QPointF(self.width(), h - 0.5))

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        box = self.box()
        x = event.position().x()
        if box is None or event.button() != Qt.MouseButton.LeftButton or x < HEADER_W:
            return
        vp = self._view.viewport
        x0, x1 = box
        if abs(x - x0) <= self.EDGE_PX:
            self._drag = "left"
        elif abs(x - x1) <= self.EDGE_PX:
            self._drag = "right"
        elif x0 < x < x1:
            self._drag, self._grab = "move", self._t(x) - vp.start
        else:
            t = self._t(x)
            vp.show(t - vp.span / 2, t + vp.span / 2)
            self._view.refresh_viewport()
            self._drag, self._grab = "move", t - vp.start

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        x = event.position().x()
        if self._drag is not None and not (event.buttons() & Qt.MouseButton.LeftButton):
            # The release was missed (e.g. it happened outside the widget).
            self._drag = None
            return
        if self._drag is None:
            box = self.box()
            near = box is not None and min(abs(x - box[0]), abs(x - box[1])) <= self.EDGE_PX
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if near else Qt.CursorShape.PointingHandCursor
            )
            return
        vp, t = self._view.viewport, self._t(x)
        if self._drag == "move":
            start = t - self._grab
            vp.show(start, start + vp.span)
        elif self._drag == "left":
            vp.show(min(t, vp.end - vp.min_span), vp.end)
        else:
            vp.show(vp.start, max(t, vp.start + vp.min_span))
        self._view.refresh_viewport()

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = None


def _tool(text: str, tip: str, *, checked: bool | None = None) -> QToolButton:
    button = QToolButton()
    button.setObjectName("TimelineTool")
    button.setText(text)
    button.setToolTip(tip)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if checked is not None:
        button.setCheckable(True)
        button.setChecked(checked)
    return button


def _rule() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFixedSize(1, 18)
    line.setObjectName("TimelineRule")
    return line


def _readout(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("TimelineReadout")
    label.setFont(data_font(9))
    return label


class TimelinePanel(QTabWidget):
    """The Timeline tab (navigator above lanes) and whatever tabs follow it.

    The toolbar sits in the tab row, as Resolve's timeline toolbar does, and is
    only shown while the Timeline tab is.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TimelinePanel")
        self.setDocumentMode(True)
        self.view = TimelineView()
        self.navigator = Navigator(self.view)
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.navigator)
        column.addWidget(self.view, 1)
        self.addTab(page, "Timeline")

        tools = QWidget()
        row = QHBoxLayout(tools)
        row.setContentsMargins(8, 2, 8, 2)
        row.setSpacing(4)
        self.snap = _tool(
            "Snap", "Selections snap to bout edges and hardware switches", checked=True
        )
        self.snap.toggled.connect(self.view.set_snap)
        self.loop = _tool("Loop", "Playback loops inside the selected range", checked=False)
        row.addWidget(self.snap)
        row.addWidget(self.loop)
        row.addWidget(_rule())
        self.bout_slot = QHBoxLayout()
        self.bout_slot.setSpacing(4)
        row.addLayout(self.bout_slot)
        row.addWidget(_rule())
        self.in_label, self.out_label, self.dur_label = (
            _readout("In —"),
            _readout("Out —"),
            _readout(""),
        )
        for label in (self.in_label, self.out_label, self.dur_label):
            row.addWidget(label)
        row.addWidget(_rule())
        self.zoom_btn = _tool("Zoom to range", "Fit the selected range to the timeline  (Z)")
        self.fit_btn = _tool("Fit", "Show the whole session  (⇧Z)")
        self.zoom_btn.clicked.connect(self.view.zoom_to_selection)
        self.fit_btn.clicked.connect(self.view.fit)
        row.addWidget(self.zoom_btn)
        row.addWidget(self.fit_btn)
        self.setCornerWidget(tools, Qt.Corner.TopRightCorner)
        self.currentChanged.connect(lambda index: tools.setVisible(index == 0))

        self.view.selection_changed.connect(self._show_range)
        self.view.selection_cleared.connect(self._clear_range)

    def _show_range(self, start: int, end: int) -> None:
        view, fps = self.view, self.view.fps()
        t_in = view.seconds_of_axis(view.axis_of_frame(start))
        t_out = view.seconds_of_axis(view.axis_of_frame(end + 1))
        self.in_label.setText(f"In {format_timecode(t_in, fps)}")
        self.out_label.setText(f"Out {format_timecode(t_out, fps)}")
        self.dur_label.setText(f"{t_out - t_in:.2f} s")

    def _clear_range(self) -> None:
        self.in_label.setText("In —")
        self.out_label.setText("Out —")
        self.dur_label.setText("")
