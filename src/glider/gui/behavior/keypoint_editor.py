"""Lay out keypoints on a mouse figure, in the order a model expects.

The keypoint-names field is a comma-separated string whose *order* is
load-bearing and invisible: get it wrong and every prediction comes back blank
with nothing raising. Typing it from memory is how that happens. Here the
schema is arranged on a figure instead, saved, and reloaded.

The silhouette is drawn in code rather than shipped as an image: it needs no
third-party asset or licence, always renders, and scales cleanly. It is a
diagram, not a photograph — its only job is to make "which one is index 3"
answerable at a glance.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from glider.analysis.behavior.keypoint_schema import (
    Keypoint,
    KeypointSchema,
    KeypointSchemaError,
)

logger = logging.getLogger(__name__)

# The figure occupies this square in scene coordinates; keypoints are stored
# normalised 0-1 and mapped onto it.
_FIG = 520.0
_DOT = 9.0

# Okabe-Ito, so neighbouring points stay distinguishable for colourblind users.
_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def mouse_silhouette() -> QPainterPath:
    """A top-view mouse outline: nose at the top, tail at the bottom.

    Deliberately schematic. Anatomical accuracy would not help anyone decide
    which index is the left hip; a clear head, flanks and tail root will.
    """
    path = QPainterPath()
    # Body: a teardrop, narrow at the snout, widest across the hips.
    path.moveTo(0.50 * _FIG, 0.02 * _FIG)
    path.cubicTo(0.66 * _FIG, 0.08 * _FIG, 0.70 * _FIG, 0.26 * _FIG, 0.66 * _FIG, 0.40 * _FIG)
    path.cubicTo(0.78 * _FIG, 0.52 * _FIG, 0.76 * _FIG, 0.72 * _FIG, 0.60 * _FIG, 0.80 * _FIG)
    path.cubicTo(0.54 * _FIG, 0.83 * _FIG, 0.46 * _FIG, 0.83 * _FIG, 0.40 * _FIG, 0.80 * _FIG)
    path.cubicTo(0.24 * _FIG, 0.72 * _FIG, 0.22 * _FIG, 0.52 * _FIG, 0.34 * _FIG, 0.40 * _FIG)
    path.cubicTo(0.30 * _FIG, 0.26 * _FIG, 0.34 * _FIG, 0.08 * _FIG, 0.50 * _FIG, 0.02 * _FIG)
    # Ears.
    path.addEllipse(QRectF(0.26 * _FIG, 0.11 * _FIG, 0.13 * _FIG, 0.13 * _FIG))
    path.addEllipse(QRectF(0.61 * _FIG, 0.11 * _FIG, 0.13 * _FIG, 0.13 * _FIG))
    # Tail, trailing off the rump.
    tail = QPainterPath()
    tail.moveTo(0.50 * _FIG, 0.81 * _FIG)
    tail.cubicTo(0.56 * _FIG, 0.90 * _FIG, 0.42 * _FIG, 0.94 * _FIG, 0.47 * _FIG, 1.00 * _FIG)
    stroker_width = 0.018 * _FIG
    path.addPath(_thicken(tail, stroker_width))
    return path


def _thicken(path: QPainterPath, width: float) -> QPainterPath:
    from PyQt6.QtGui import QPainterPathStroker

    stroker = QPainterPathStroker()
    stroker.setWidth(width)
    stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
    return stroker.createStroke(path)


class _PointItem(QGraphicsEllipseItem):
    """A draggable keypoint. Reports its normalised position on release."""

    def __init__(self, index: int, editor: KeypointEditorDialog):
        super().__init__(-_DOT, -_DOT, 2 * _DOT, 2 * _DOT)
        self._index = index
        self._editor = editor
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(10)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._editor.point_moved(self._index, self.pos())


class KeypointEditorDialog(QDialog):
    """Arrange, rename, add, remove and reorder keypoints on a mouse figure."""

    def __init__(self, schema: KeypointSchema | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keypoint schema")
        self.resize(880, 640)
        self._schema = schema or KeypointSchema.default_mouse()

        layout = QHBoxLayout(self)

        self._scene = QGraphicsScene(0, 0, _FIG, _FIG, self)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHints(self._view.renderHints())
        layout.addWidget(self._view, 2)

        side = QVBoxLayout()
        blurb = QLabel(
            "Order is what the model matches on, not just the names. Drag a "
            "point to reposition it; use the arrows to change its index."
        )
        blurb.setWordWrap(True)
        side.addWidget(blurb)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(lambda _r: self._redraw())
        side.addWidget(self._list, 1)

        for text, slot in (
            ("Rename…", self._rename),
            ("Add", self._add),
            ("Remove", self._remove),
            ("Move up", lambda: self._move(-1)),
            ("Move down", lambda: self._move(+1)),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            side.addWidget(button)

        side.addSpacing(8)
        for text, slot in (("Load schema…", self._load), ("Save schema…", self._save)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            side.addWidget(button)

        side.addStretch(1)
        self._problem = QLabel("")
        self._problem.setWordWrap(True)
        self._problem.setStyleSheet("color: #c0392b;")
        side.addWidget(self._problem)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self._ok = QPushButton("Use these keypoints")
        self._ok.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self._ok)
        side.addLayout(buttons)

        layout.addLayout(side, 1)
        self._refresh()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def schema(self) -> KeypointSchema:
        return self._schema

    def names(self) -> list[str]:
        return self._schema.names

    def point_moved(self, index: int, pos: QPointF) -> None:
        """Store a dragged point's new position, clamped to the figure."""
        if 0 <= index < len(self._schema.keypoints):
            kp = self._schema.keypoints[index]
            kp.x = min(1.0, max(0.0, pos.x() / _FIG))
            kp.y = min(1.0, max(0.0, pos.y() / _FIG))

    def _current(self) -> int:
        return self._list.currentRow()

    def _refresh(self, keep_row: int | None = None) -> None:
        row = self._current() if keep_row is None else keep_row
        self._list.blockSignals(True)
        self._list.clear()
        for i, kp in enumerate(self._schema.keypoints):
            self._list.addItem(f"{i}:  {kp.name}")
        if 0 <= row < self._list.count():
            self._list.setCurrentRow(row)
        elif self._list.count():
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        self._validate()
        self._redraw()

    def _validate(self) -> None:
        problem = self._schema.problem()
        self._problem.setText(problem or "")
        self._ok.setEnabled(problem is None)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        self._scene.clear()
        self._scene.setBackgroundBrush(QBrush(QColor("#fafafa")))
        self._scene.addPath(
            mouse_silhouette(),
            QPen(QColor("#9e9e9e"), 2),
            QBrush(QColor("#e0e0e0")),
        )
        selected = self._current()
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        for i, kp in enumerate(self._schema.keypoints):
            item = _PointItem(i, self)
            colour = QColor(_PALETTE[i % len(_PALETTE)])
            item.setBrush(QBrush(colour))
            item.setPen(QPen(QColor("#000000"), 3 if i == selected else 1))
            item.setPos(kp.x * _FIG, kp.y * _FIG)
            self._scene.addItem(item)

            label = QGraphicsSimpleTextItem(f"{i}:{kp.name}")
            label.setFont(font)
            label.setBrush(QBrush(QColor("#212121")))
            label.setPos(kp.x * _FIG + _DOT + 3, kp.y * _FIG - _DOT - 4)
            label.setZValue(11)
            self._scene.addItem(label)
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------------
    # editing
    # ------------------------------------------------------------------

    def _rename(self) -> None:
        row = self._current()
        if not 0 <= row < len(self._schema.keypoints):
            return
        name, ok = QInputDialog.getText(
            self, "Rename keypoint", "Name:", text=self._schema.keypoints[row].name
        )
        if ok:
            self._schema.keypoints[row].name = name.strip()
            self._refresh(row)

    def _add(self) -> None:
        name, ok = QInputDialog.getText(self, "Add keypoint", "Name:")
        if not ok:
            return
        # New points land in the middle, where they are easy to find and drag.
        self._schema.keypoints.append(Keypoint(name.strip(), 0.5, 0.5))
        self._refresh(len(self._schema.keypoints) - 1)

    def _remove(self) -> None:
        row = self._current()
        if 0 <= row < len(self._schema.keypoints):
            self._schema.keypoints.pop(row)
            self._refresh(min(row, len(self._schema.keypoints) - 1))

    def _move(self, delta: int) -> None:
        row = self._current()
        self._refresh(self._schema.move(row, delta))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save keypoint schema", "keypoints.json", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self._schema.save(path)
        except OSError as e:
            QMessageBox.critical(self, "Keypoint schema", f"Could not save:\n{e}")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load keypoint schema", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self._schema = KeypointSchema.load(path)
        except KeypointSchemaError as e:
            QMessageBox.warning(self, "Keypoint schema", str(e))
            return
        self._refresh(0)


__all__ = ["KeypointEditorDialog", "mouse_silhouette"]
