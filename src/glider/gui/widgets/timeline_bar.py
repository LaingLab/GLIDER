"""The session as a stack of lanes, and the one scrubber for all of them.

Replaces the ethogram bar, which was a timeline of exactly one thing. The
ethogram becomes a lane; the hardware the rig drove becomes the lanes below
it; both share an axis and a playhead.

The axis here is **time**, because the event log is timed and its rows carry
no usable frame index before the first camera frame arrives. But
``selection_changed`` still emits **frames**, because every table in Session
Review consumes a frame range — ``zone_rows(start, end)``,
``cohort_rows(start, end)``, ``segment_stats(start, end)``. Converting at the
signal boundary is what lets the whole panel below stay untouched.

The model is in :mod:`glider.analysis.timeline` and is Qt-free. This only
draws.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget

from glider.analysis.timeline import BehaviorLane, Timeline
from glider.gui.styles import colors

__all__ = ["TimelineBar", "behavior_order", "behavior_qcolor"]

#: Height of a behaviour lane when hardware lanes share the bar. Matches the
#: ethogram bar's old fixed height so a session with no hardware looks
#: exactly as it did.
_BEHAVIOR_LANE_HEIGHT = 46

#: Preferred height of one hardware lane. Rows compress below this rather
#: than overflowing when a rig has more devices than the bar has pixels.
_HW_LANE_HEIGHT = 14

_LANE_GAP = 2


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


class TimelineBar(QWidget):
    """Every lane of a session, and the scrubber across all of them.

    Clicking or dragging with the left button scrubs; dragging with shift
    (or the right button) selects a window. Selection and playhead stay
    separate so a chosen window survives scrubbing around inside it.
    """

    scrubbed = pyqtSignal(int)  # frame
    selection_changed = pyqtSignal(int, int)  # start, end frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(_BEHAVIOR_LANE_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._view = None
        self._timeline: Timeline | None = None
        self._order: list[str] = []
        self._codes: dict[str, np.ndarray] = {}
        self._static: QPixmap | None = None
        self._frame = 0
        self._selection: tuple[int, int] | None = None
        self._drag_anchor: int | None = None

    # ------------------------------------------------------------------
    # loading

    def set_view(self, view) -> None:
        """Adopt a :class:`SessionView`. Kept for ethogram-only sessions."""
        self._view = view
        self._rebuild_codes()
        self._invalidate()

    def set_timeline(self, timeline: Timeline | None) -> None:
        self._timeline = timeline
        self._rebuild_codes()
        self._invalidate()
        self.updateGeometry()

    def _invalidate(self) -> None:
        self._static = None
        self._frame = 0
        self._selection = None
        self.update()

    def _behavior_lanes(self) -> list[BehaviorLane]:
        """The behaviour lanes to draw.

        A timeline's own lanes when it has any; otherwise one synthesised
        from the view, so ``set_view`` alone still draws an ethogram.
        """
        if self._timeline is not None and self._timeline.behavior:
            return list(self._timeline.behavior)
        if self._view is not None and len(self._view.labels):
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

        Computed once per session rather than per band: which colour a
        behaviour gets depends on which behaviours the session contains, and
        the per-column majority in :meth:`_paint_behavior` needs the labels
        as integers. The order is shared across lanes so the same behaviour
        is the same colour whichever source scored it.
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

    # ------------------------------------------------------------------
    # geometry

    def sizeHint(self) -> QSize:
        n_behavior = max(1, len(self._behavior_lanes()))
        n_hardware = len(self._timeline.lanes) if self._timeline else 0
        height = _BEHAVIOR_LANE_HEIGHT * n_behavior + _HW_LANE_HEIGHT * n_hardware
        return QSize(super().sizeHint().width(), height)

    def _rows(self) -> list[tuple[str, object, float, float]]:
        """``(kind, lane, top, height)`` for every row, top to bottom.

        With no hardware the behaviour lanes split the whole bar, which is
        what keeps an ethogram-only session looking exactly as it did before
        there were any other lanes.
        """
        height = float(self.height())
        behavior = self._behavior_lanes()
        hardware = list(self._timeline.lanes) if self._timeline else []

        if not hardware:
            if not behavior:
                return []
            each = height / len(behavior)
            return [("behavior", ln, i * each, each) for i, ln in enumerate(behavior)]

        rows: list[tuple[str, object, float, float]] = []
        top = 0.0
        for lane in behavior:
            rows.append(("behavior", lane, top, float(_BEHAVIOR_LANE_HEIGHT)))
            top += _BEHAVIOR_LANE_HEIGHT + _LANE_GAP
        remaining = max(0.0, height - top)
        each = remaining / len(hardware)
        for lane in hardware:
            rows.append(("hardware", lane, top, each))
            top += each
        return rows

    def _frame_map(self):
        return self._timeline.frame_map if self._timeline else None

    def _ms_bounds(self) -> tuple[float, float] | None:
        if self._timeline is None:
            return None
        if self._timeline.end_ms <= self._timeline.start_ms:
            return None
        return self._timeline.start_ms, self._timeline.end_ms

    def frame_bounds(self) -> tuple[int, int]:
        """``(first, last)`` frame the bar covers.

        From the frame map when there is one. Otherwise the behaviour lanes'
        own range, which is deliberately not zero: a windowed run scores
        minutes two to seven, so its ethogram starts at frame 3600 and a bar
        drawn from zero would spend its first eighth showing nothing.

        The lanes rather than ``self._view``, because an ethogram-only
        ``Timeline`` has no frame map and no ms bounds either — falling back
        to the view alone left ``set_timeline`` without ``set_view``
        reporting ``(0, 0)``, which collapses every column onto frame 0 and
        paints the whole ethogram one flat colour. ``_behavior_lanes``
        already synthesises a lane from the view, so this covers both.
        """
        frame_map = self._frame_map()
        bounds = self._ms_bounds()
        if frame_map is not None and bounds is not None:
            return frame_map.frame_at(bounds[0]), frame_map.frame_at(bounds[1])
        lanes = [ln for ln in self._behavior_lanes() if len(ln.frames)]
        if lanes:
            firsts = [int(np.asarray(ln.frames)[0]) for ln in lanes]
            lasts = [int(np.asarray(ln.frames)[-1]) for ln in lanes]
            return min(firsts), max(lasts)
        return 0, 0

    def _x_of_ms(self, ms: float) -> float:
        bounds = self._ms_bounds()
        if bounds is None or self.width() <= 0:
            return 0.0
        start, end = bounds
        return (float(ms) - start) / (end - start) * self.width()

    def _x_of(self, frame: int) -> float:
        frame_map = self._frame_map()
        if frame_map is not None and self._ms_bounds() is not None:
            return self._x_of_ms(frame_map.ms_of(frame))
        first, last = self.frame_bounds()
        span = max(0, last - first + 1)
        return 0.0 if span == 0 else (frame - first) / span * self.width()

    def _frame_at(self, x: float) -> int:
        first, last = self.frame_bounds()
        if self.width() <= 0:
            return first
        bounds = self._ms_bounds()
        frame_map = self._frame_map()
        if frame_map is not None and bounds is not None:
            start, end = bounds
            ms = start + (x / self.width()) * (end - start)
            return max(first, min(last, frame_map.frame_at(ms)))
        span = max(0, last - first + 1)
        if span == 0:
            return first
        return max(first, min(last, first + int(x / self.width() * span)))

    def resizeEvent(self, event):
        # Bands are resolved per pixel column, so a different width is a
        # different image.
        self._static = None
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # state

    def set_frame(self, frame: int) -> None:
        self._frame = int(frame)
        self.update()

    def selection(self) -> tuple[int, int] | None:
        return self._selection

    def set_selection(self, start: int, end: int) -> None:
        self._selection = (int(min(start, end)), int(max(start, end)))
        self.update()
        self.selection_changed.emit(*self._selection)

    # ------------------------------------------------------------------
    # painting

    def _static_pixmap(self) -> QPixmap:
        """Every lane, drawn once per session and size.

        Cached because the playhead moves every frame during playback and
        the behaviour bands cost a bincount per pixel column to resolve.
        """
        if self._static is None:
            self._static = QPixmap(self.size())
            self._static.fill(QColor(colors.BASE))
            painter = QPainter(self._static)
            try:
                for kind, lane, top, height in self._rows():
                    if kind == "behavior":
                        self._paint_behavior(painter, lane, top, height)
                    else:
                        self._paint_hardware(painter, lane, top, height)
            finally:
                painter.end()
        return self._static

    def _paint_hardware(self, painter, lane, top: float, height: float) -> None:
        """One device's held values as bars growing from the row baseline.

        A digital pin reads as full-height blocks and a PWM ramp as a
        staircase, from one renderer — the value is held until the device's
        next event either way.
        """
        if height <= 0:
            return
        baseline = top + height
        painter.fillRect(QRectF(0, top, float(self.width()), height), QColor(colors.BASE))
        painter.setPen(Qt.PenStyle.NoPen)
        for segment in lane.segments:
            if segment.level <= 0:
                continue
            x0 = self._x_of_ms(segment.start_ms)
            x1 = self._x_of_ms(segment.end_ms)
            bar_height = height * segment.level
            painter.fillRect(
                QRectF(x0, baseline - bar_height, max(1.0, x1 - x0), bar_height),
                QColor(colors.ACCENT),
            )
        painter.setPen(QPen(QColor(colors.TEXT_MUTED), 1))
        for marker in lane.markers:
            x = self._x_of_ms(marker.at_ms)
            painter.drawLine(int(x), int(top), int(x), int(baseline))

    def _paint_behavior(self, painter, lane, top: float, height: float) -> None:
        """One band per *pixel column*, coloured by what dominates it.

        Not one rect per run, which is the obvious thing and was wrong. A
        five-minute session holds around nine thousand scored rows and the
        timeline is at most a couple of thousand pixels wide, so a typical run
        is a fraction of a pixel: Qt drew each as a sub-pixel rectangle and
        blended it with its neighbours by coverage. Every colour on the bar was
        therefore an average of several behaviours — a bright yellow, a green
        and a blue arriving on screen as one flat olive. No palette can survive
        that, and it is why the bar looked washed out however distinct the
        colours themselves were.

        Resolving to whole columns first makes every pixel one behaviour's
        actual colour. It also means a run shorter than a column is not drawn,
        which is honest — the bar shows proportions, and a pixel cannot show a
        three-frame dart without overstating it. The bout stepper is how those
        are reached.
        """
        if height <= 0 or not lane.labels:
            return
        codes = self._codes.get(lane.source)
        if codes is None:
            return
        width = self.width()
        first, last = self.frame_bounds()
        span = max(0, last - first + 1)
        if width <= 0 or span == 0:
            return

        frames = np.asarray(lane.frames)
        bounds = self._ms_bounds()
        frame_map = self._frame_map()
        if frame_map is not None and bounds is not None and len(frame_map.frames):
            # Columns are equal slices of the TIME axis, because that is the
            # axis every other lane and the playhead are drawn on. Slicing the
            # frame axis instead smears the ethogram across any stretch the
            # camera did not cover -- and the axis deliberately spans
            # pre-flow device writes, so that stretch is routinely real.
            start_ms, end_ms = bounds
            edges_ms = start_ms + np.arange(width + 1, dtype=float) * (end_ms - start_ms) / width
            idx = np.clip(np.searchsorted(frame_map.ms, edges_ms), 0, len(frame_map.frames) - 1)
            edges = frame_map.frames[idx].astype(np.int64)
        else:
            # No frame map: the axis IS frames, so equal slices of it are right.
            edges = first + np.arange(width + 1, dtype=np.int64) * span // width
        starts = np.searchsorted(frames, edges, side="left")

        n_codes = len(self._order) + 1  # + the unscored bucket
        for x in range(width):
            lo, hi = int(starts[x]), int(starts[x + 1])
            if hi <= lo:
                # More pixels than rows: this column falls between two rows,
                # so it takes the row to its left rather than a gap.
                lo = max(0, min(lo, len(frames) - 1))
                hi = lo + 1
            counts = np.bincount(codes[lo:hi], minlength=n_codes)
            code = int(counts.argmax())
            painter.fillRect(
                QRectF(x, top, 1.0, height),
                behavior_qcolor(self._order[code - 1] if code else "", self._order),
            )

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(colors.BASE))
        if not self._rows():
            painter.setPen(QPen(QColor(colors.TEXT_MUTED)))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load a session to see its timeline",
            )
            return

        painter.drawPixmap(0, 0, self._static_pixmap())

        # Flow boundaries, across every lane. They are boundaries rather
        # than a device, so they get a rule instead of a lane.
        if self._timeline is not None and self._ms_bounds() is not None:
            pen = QPen(QColor(colors.TEXT_MUTED), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            for boundary in (self._timeline.flow_start_ms, self._timeline.flow_end_ms):
                if boundary is None:
                    continue
                x = int(self._x_of_ms(boundary))
                painter.drawLine(x, 0, x, self.height())

        if self._selection is not None:
            start, end = self._selection
            x0, x1 = self._x_of(start), self._x_of(end + 1)
            # Shade what is EXCLUDED, not what is chosen. Tinting the selection
            # blue meant every behaviour inside it was drawn 28% toward the
            # accent — and since the usual selection is the whole session, that
            # was every colour on the bar, all of them dragged toward the same
            # hue. Shading the outside leaves the data at full strength and
            # says the same thing.
            scrim = QBrush(colors.qcolor_with_alpha(QColor(colors.BASE), 0.72))
            painter.fillRect(QRectF(0, 0, max(0.0, x0), self.height()), scrim)
            painter.fillRect(QRectF(x1, 0, max(0.0, self.width() - x1), self.height()), scrim)
            painter.setPen(QPen(QColor(colors.ACCENT), 2))
            painter.drawLine(QPointF(x0, 0), QPointF(x0, self.height()))
            painter.drawLine(QPointF(x1, 0), QPointF(x1, self.height()))

        x = int(self._x_of(self._frame))
        painter.setPen(QPen(QColor(colors.TEXT_PRIMARY), 2))
        painter.drawLine(x, 0, x, self.height())

    # ------------------------------------------------------------------
    # interaction

    def mousePressEvent(self, event):
        if not self._rows():
            return
        frame = self._frame_at(event.position().x())
        selecting = (
            event.button() == Qt.MouseButton.RightButton
            or event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        if selecting:
            self._drag_anchor = frame
            self.set_selection(frame, frame)
        else:
            self._drag_anchor = None
            self.scrubbed.emit(frame)

    def mouseMoveEvent(self, event):
        if not self._rows():
            return
        frame = self._frame_at(event.position().x())
        if self._drag_anchor is not None:
            self.set_selection(self._drag_anchor, frame)
        elif event.buttons() & Qt.MouseButton.LeftButton:
            self.scrubbed.emit(frame)

    def mouseReleaseEvent(self, _event):
        self._drag_anchor = None
