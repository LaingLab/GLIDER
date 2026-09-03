"""Arena Dialog - click the floor perimeter to calibrate perspective.

The operator clicks the four floor corners of an arena of known size. That
fixes a homography (see :mod:`glider.vision.arena`), which in turn gives a
centre zone meaning the same patch of floor in every video, and a scale that
varies correctly across the floor instead of being one number for all of it.

Two things shape the interaction.

**Corners are often outside the frame.** Cameras get mounted close, and the far
pair of corners frequently sits above the top edge of the picture. The frame is
therefore drawn inset in a larger canvas, and clicks land anywhere in that
canvas - including the margin, where the operator is placing a corner the
sensor never saw. Extending the two visible walls by eye is accurate enough
because the overlay closes the loop immediately: the fitted arena is drawn back
over the bedding, and a corner that is out of place shows up as edges that do
not follow the floor.

**Nothing is committed until it looks right.** Every corner stays draggable,
and the arena outline, the centre zone and the residuals all update live. The
frame can be scrubbed, since bedding kicked into a corner can hide the very
junction the operator needs.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import colors
from glider.vision.arena import CORNER_NAMES, ArenaCalibration, DegenerateArenaError

logger = logging.getLogger(__name__)

#: Fraction of the canvas left as margin around the frame, so that corners
#: outside the picture are still clickable. A fifth is enough for every video in
#: the cohorts seen so far and still leaves the frame big enough to click
#: accurately.
_PAD = 0.20

#: How near a click must land, in canvas pixels, to grab an existing corner
#: rather than place a new one.
_GRAB_RADIUS = 14

_ARENA_COLOUR = QColor("#38bdf8")
_ZONE_COLOUR = QColor("#34d399")
_CORNER_COLOUR = QColor("#fbbf24")
_SUSPECT_COLOUR = QColor("#f87171")


class ArenaCanvas(QWidget):
    """Frame plus a draggable four-corner overlay.

    Reports corners in normalized image coordinates, which may fall outside
    0-1: that is the point of the margin, and clamping them would silently
    move an arena corner onto the frame edge where it does not belong.
    """

    corners_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(720, 560)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._frame: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._corners: list[tuple[float, float]] = []
        self._dragging: int | None = None
        self._zone_cm = 10.0
        self._arena_cm = (30.0, 30.0)

    # -- state ---------------------------------------------------------

    def set_frame(self, frame: np.ndarray | None) -> None:
        self._frame = None if frame is None else frame.copy()
        self._pixmap = None if frame is None else self._to_pixmap(self._frame)
        self.update()

    def set_corners(self, corners) -> None:
        self._corners = [(float(x), float(y)) for x, y in corners][:4]
        self.update()
        self.corners_changed.emit()

    def set_arena_size(self, width_cm: float, height_cm: float) -> None:
        self._arena_cm = (width_cm, height_cm)
        self.update()

    def set_zone_size(self, size_cm: float) -> None:
        self._zone_cm = size_cm
        self.update()

    @property
    def corners(self) -> list[tuple[float, float]]:
        return list(self._corners)

    @property
    def is_complete(self) -> bool:
        return len(self._corners) == 4

    def clear(self) -> None:
        self._corners = []
        self._dragging = None
        self.update()
        self.corners_changed.emit()

    def undo(self) -> None:
        if self._corners:
            self._corners.pop()
            self.update()
            self.corners_changed.emit()

    def calibration(self) -> ArenaCalibration | None:
        """The calibration these corners describe, or None if incomplete."""
        if not self.is_complete or self._frame is None:
            return None
        h, w = self._frame.shape[:2]
        return ArenaCalibration(
            corners=self._corners,
            width_cm=self._arena_cm[0],
            height_cm=self._arena_cm[1],
            frame_size=(w, h),
        )

    # -- coordinates ---------------------------------------------------

    @staticmethod
    def _to_pixmap(frame: np.ndarray) -> QPixmap:
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        rgb = np.ascontiguousarray(rgb)
        return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888))

    def _frame_rect(self) -> QRect | None:
        """Where the frame sits inside the canvas, leaving margin all round."""
        if self._frame is None:
            return None
        fh, fw = self._frame.shape[:2]
        avail_w = self.width() * (1 - 2 * _PAD)
        avail_h = self.height() * (1 - 2 * _PAD)
        scale = min(avail_w / fw, avail_h / fh)
        w, h = int(fw * scale), int(fh * scale)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _to_norm(self, pos: QPoint) -> tuple[float, float] | None:
        rect = self._frame_rect()
        if rect is None or rect.width() == 0 or rect.height() == 0:
            return None
        return (
            (pos.x() - rect.x()) / rect.width(),
            (pos.y() - rect.y()) / rect.height(),
        )

    def _to_canvas(self, point) -> QPoint:
        rect = self._frame_rect()
        x, y = point
        return QPoint(int(rect.x() + x * rect.width()), int(rect.y() + y * rect.height()))

    # -- interaction ---------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._frame is None or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        pos = event.pos()
        for i, corner in enumerate(self._corners):
            if (self._to_canvas(corner) - pos).manhattanLength() <= _GRAB_RADIUS:
                self._dragging = i
                return
        if len(self._corners) < 4:
            norm = self._to_norm(pos)
            if norm is not None:
                self._corners.append(norm)
                self.update()
                self.corners_changed.emit()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging is None:
            return super().mouseMoveEvent(event)
        norm = self._to_norm(event.pos())
        if norm is not None:
            self._corners[self._dragging] = norm
            self.update()
            self.corners_changed.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = None
        super().mouseReleaseEvent(event)

    # -- painting ------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(colors.CANVAS))

        if self._pixmap is None:
            painter.setPen(QColor(colors.TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame")
            return

        rect = self._frame_rect()
        painter.drawPixmap(rect, self._pixmap)
        painter.setPen(QPen(QColor(colors.BORDER), 1))
        painter.drawRect(rect)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_outline(painter)
        self._paint_corners(painter)

    def _paint_outline(self, painter: QPainter) -> None:
        if not self._corners:
            return
        points = [self._to_canvas(c) for c in self._corners]

        suspect = False
        cal = self.calibration()
        if cal is not None:
            try:
                suspect = cal.residuals()["suspect"]
            except DegenerateArenaError:
                suspect = True

        colour = _SUSPECT_COLOUR if suspect else _ARENA_COLOUR
        painter.setPen(QPen(colour, 2))
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])
        if self.is_complete:
            painter.drawLine(points[-1], points[0])

        if cal is None:
            return
        try:
            zone = [self._to_canvas(p) for p in cal.centre_zone_vertices(self._zone_cm)]
        except (DegenerateArenaError, ValueError):
            return
        painter.setPen(QPen(_ZONE_COLOUR, 2))
        for i in range(4):
            painter.drawLine(zone[i], zone[(i + 1) % 4])

    def _paint_corners(self, painter: QPainter) -> None:
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        for i, corner in enumerate(self._corners):
            point = self._to_canvas(corner)
            painter.setPen(QPen(_CORNER_COLOUR, 2))
            painter.drawEllipse(point, 5, 5)
            painter.drawLine(point.x() - 9, point.y(), point.x() + 9, point.y())
            painter.drawLine(point.x(), point.y() - 9, point.x(), point.y() + 9)
            painter.setPen(QColor(colors.TEXT_PRIMARY))
            painter.drawText(point.x() + 10, point.y() - 8, str(i + 1))


class ArenaDialog(QDialog):
    """Click the four floor corners of one video's arena."""

    def __init__(self, frame: np.ndarray, title: str = "", parent=None, **kwargs):
        super().__init__(parent)
        self.setWindowTitle(f"Arena perimeter - {title}" if title else "Arena perimeter")
        self.setModal(True)
        self._on_scrub = kwargs.pop("on_scrub", None)
        self._frame_count = kwargs.pop("frame_count", 0)
        self._build_ui()
        self.canvas.set_frame(frame)
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Click the four floor corners in order: "
            + ", ".join(CORNER_NAMES)
            + ".\nClick where the bedding meets the wall, not the top of the wall. "
            "Corners outside the picture can be clicked in the margin. "
            "Drag any corner to adjust."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(f"color: {colors.TEXT_SECONDARY};")
        layout.addWidget(instructions)

        self.canvas = ArenaCanvas(self)
        self.canvas.corners_changed.connect(self._refresh)
        layout.addWidget(self.canvas, 1)

        if self._frame_count > 1 and self._on_scrub is not None:
            self.scrub = QSlider(Qt.Orientation.Horizontal)
            self.scrub.setRange(0, max(0, self._frame_count - 1))
            self.scrub.sliderReleased.connect(self._scrub_to)
            layout.addWidget(self.scrub)

        controls = QHBoxLayout()
        size_box = QGroupBox("Arena")
        form = QFormLayout(size_box)
        self.arena_spin = QDoubleSpinBox()
        self.arena_spin.setRange(1.0, 500.0)
        self.arena_spin.setValue(30.0)
        self.arena_spin.setSuffix(" cm")
        self.arena_spin.valueChanged.connect(self._size_changed)
        form.addRow("Side", self.arena_spin)
        self.zone_spin = QDoubleSpinBox()
        self.zone_spin.setRange(0.5, 500.0)
        self.zone_spin.setValue(10.0)
        self.zone_spin.setSuffix(" cm")
        self.zone_spin.valueChanged.connect(self._size_changed)
        form.addRow("Centre zone", self.zone_spin)
        controls.addWidget(size_box)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop)
        controls.addWidget(self.status, 1)
        layout.addLayout(controls)

        buttons = QHBoxLayout()
        for text, slot in (("Undo", self.canvas.undo), ("Clear", self.canvas.clear)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.box.accepted.connect(self.accept)
        self.box.rejected.connect(self.reject)
        buttons.addWidget(self.box)
        layout.addLayout(buttons)

    def _size_changed(self) -> None:
        self.canvas.set_arena_size(self.arena_spin.value(), self.arena_spin.value())
        self.canvas.set_zone_size(self.zone_spin.value())
        self._refresh()

    def _scrub_to(self) -> None:
        frame = self._on_scrub(self.scrub.value())
        if frame is not None:
            self.canvas.set_frame(frame)

    def _refresh(self) -> None:
        ok_button = self.box.button(QDialogButtonBox.StandardButton.Ok)
        cal = self.canvas.calibration()
        if cal is None:
            remaining = 4 - len(self.canvas.corners)
            self.status.setText(f"Click {remaining} more corner(s): {CORNER_NAMES[4 - remaining]}")
            self.status.setStyleSheet(f"color: {colors.TEXT_SECONDARY};")
            ok_button.setEnabled(False)
            return
        try:
            residuals = cal.residuals()
            scale = cal.px_per_cm_centre
        except DegenerateArenaError as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet(f"color: {colors.ERROR};")
            ok_button.setEnabled(False)
            return

        lines = [
            f"Scale at centre: {scale:.2f} px/cm",
            f"Opposite edges: {residuals['edge_ratio']:.2f}x",
            f"Scale across floor: {residuals['scale_ratio']:.2f}x",
        ]
        if residuals["clipped"]:
            lines.append("Arena extends beyond the frame (expected on some videos).")
        if residuals["suspect"]:
            lines.append("This quad looks unlikely - check the corners.")
            self.status.setStyleSheet(f"color: {colors.WARNING};")
        else:
            self.status.setStyleSheet(f"color: {colors.SUCCESS};")
        self.status.setText("\n".join(lines))
        ok_button.setEnabled(True)

    def calibration(self) -> ArenaCalibration | None:
        return self.canvas.calibration()

    def zone_size_cm(self) -> float:
        return self.zone_spin.value()
