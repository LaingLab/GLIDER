"""The segmentation canvas: outline each animal on a frame as a polygon.

Two tools, switched with :meth:`SegmentCanvas.set_tool`:

**Polygon** -- click round the animal to drop vertices; click the first vertex
(or press Enter) to close the outline. Esc abandons it, Backspace takes back
the last vertex. Click inside another outline to select it.

**Auto** -- smart annotation with Segment Anything. Left-click the animal,
right-click background the mask spilled onto, or drag a box round the animal.
The window runs SAM on every change and hands back a proposed outline
(:meth:`SegmentCanvas.set_proposal`), drawn dashed; Enter accepts it. The
canvas itself knows nothing about SAM: it collects prompts and shows proposals.

With either tool, the selected outline's vertices can be dragged; Shift-click
on its edge adds a vertex, and right-click (Polygon) or Delete removes one.
Wheel zooms, middle-drag pans.

Outlines are kept in image pixels while editing and handed out normalised as
:class:`~glider.vision.training.labels.SegInstance`, which is what is saved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from glider.gui.animal_annotator.canvas import KEYPOINT_COLOURS, _to_qimage
from glider.vision.training.labels import SegInstance

__all__ = ["AUTO", "POLYGON", "SegmentCanvas", "class_colour"]

POLYGON = "polygon"
AUTO = "auto"

_VERTEX = 4.0  # vertex handle half-size, screen px
_GRAB = 9.0  # how close a press must be to grab a vertex or edge, screen px
_CLOSE = 11.0  # how close to the first vertex a click closes the polygon
_PROMPT_R = 6.0  # prompt-point radius, screen px
_CLICK_SLOP = 6.0  # a press that moves less than this is a click, not a box drag
_ZOOM_STEP = 1.25
_MAX_ZOOM = 40.0
_ACCENT = QColor(56, 189, 248)
_POSITIVE = QColor(74, 222, 128)
_NEGATIVE = QColor(248, 113, 113)


def class_colour(index: int) -> QColor:
    """The colour class ``index``'s outlines are drawn in."""
    return QColor(*KEYPOINT_COLOURS[index % len(KEYPOINT_COLOURS)])


@dataclass
class _Shape:
    cls: int
    pts: list[list[float]] = field(default_factory=list)  # [x, y] image pixels

    def polygon(self) -> QPolygonF:
        return QPolygonF([QPointF(x, y) for x, y in self.pts])


class SegmentCanvas(QGraphicsView):
    """Draw, refine and select animal outlines on one frame.

    Signals:
        changed: The outlines were edited; read them with :meth:`instances`.
        state_changed: Selection, tool, draft or prompts changed, for the
            window's hint text and lists (also emitted after every edit).
        prompt_changed: The Auto tool's clicks or box changed; read them with
            :meth:`prompts` and answer with :meth:`set_proposal`.
    """

    changed = pyqtSignal()
    state_changed = pyqtSignal()
    prompt_changed = pyqtSignal()

    def __init__(self, class_names: list[str], parent=None):
        super().__init__(parent)
        self._class_names = list(class_names)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(20, 20, 20)))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap: QGraphicsPixmapItem | None = None
        self._image: np.ndarray | None = None
        self._size = (0, 0)
        self._shapes: list[_Shape] = []
        self._current = -1  # selected outline, -1 for none
        #: Whether the user picked the selected outline, rather than it being
        #: selected because it was just made; only a picked one is re-classed.
        self._picked = False
        self._class = 0  # class new outlines get
        self._vertex: int | None = None  # selected vertex of the current outline
        self._dragging = False
        self._pan_from: QPointF | None = None
        self._fitted = True

        self.tool = POLYGON
        self._draft: list[list[float]] = []
        self._hover: QPointF | None = None  # scene pos, for the draft's rubber band
        self._rubber: QGraphicsLineItem | None = None

        # Auto tool
        self._prompt_points: list[tuple[float, float, int]] = []
        self._prompt_box: QRectF | None = None
        self._proposal: list[list[float]] = []
        self._box_from: QPointF | None = None
        self._band: QRectF | None = None
        self.busy = False  # the window is waiting on SAM; shown in the cursor

    # ------------------------------------------------------------------
    # frame and labels in and out
    # ------------------------------------------------------------------

    def set_frame(self, image: np.ndarray | None, instances: list[SegInstance]) -> None:
        """Show ``image`` (BGR) with ``instances`` on it; None clears the canvas."""
        self._scene.clear()
        self._pixmap = self._rubber = None
        self._reset_transient()
        self._shapes = []
        self._current = -1
        self._image = image
        if image is None:
            self._size = (0, 0)
            self.state_changed.emit()
            return
        h, w = image.shape[:2]
        self._size = (w, h)
        self._pixmap = QGraphicsPixmapItem(QPixmap.fromImage(_to_qimage(image)))
        self._scene.addItem(self._pixmap)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self._load(instances)
        self._fitted = True
        self._fit()
        self._redraw()
        self.state_changed.emit()

    def set_instances(self, instances: list[SegInstance]) -> None:
        """Replace the outlines on the frame shown, keeping zoom (for undo/redo)."""
        if not self.has_image:
            return
        current = self._current
        self._reset_transient()
        self._load(instances)
        self._current = min(current, len(self._shapes) - 1)
        self._redraw()
        self.state_changed.emit()

    def instances(self) -> list[SegInstance]:
        """The outlines as drawn, normalised. Unfinished drafts are left out."""
        w, h = self._size
        if not w or not h:
            return []
        return [
            SegInstance([(x / w, y / h) for x, y in s.pts], s.cls)
            for s in self._shapes
            if len(s.pts) >= 3
        ]

    @property
    def image(self) -> np.ndarray | None:
        """The BGR frame shown, for the window to hand to SAM."""
        return self._image

    def _load(self, instances: list[SegInstance]) -> None:
        w, h = self._size
        self._shapes = [
            _Shape(inst.class_id, [[x * w, y * h] for x, y in inst.polygon])
            for inst in instances
            if len(inst.polygon) >= 3
        ]

    def _reset_transient(self) -> None:
        self._picked = False
        self._draft = []
        self._hover = None
        self._vertex = None
        self._dragging = False
        self._prompt_points = []
        self._prompt_box = None
        self._proposal = []
        self._box_from = self._band = None
        self.busy = False

    # ------------------------------------------------------------------
    # state the window shows
    # ------------------------------------------------------------------

    @property
    def has_image(self) -> bool:
        return self._size[0] > 0

    @property
    def shape_count(self) -> int:
        return len(self._shapes)

    @property
    def current_shape(self) -> int:
        return self._current

    @property
    def current_class(self) -> int:
        return self._class

    def shape_class(self, index: int) -> int:
        return self._shapes[index].cls

    def shape_vertex_count(self, index: int) -> int:
        return len(self._shapes[index].pts)

    @property
    def drafting(self) -> bool:
        return bool(self._draft)

    @property
    def draft_length(self) -> int:
        return len(self._draft)

    @property
    def has_prompts(self) -> bool:
        return bool(self._prompt_points) or self._prompt_box is not None

    @property
    def has_proposal(self) -> bool:
        return len(self._proposal) >= 3

    def prompts(self) -> tuple[list[tuple[float, float]], list[int], tuple | None]:
        """``(points, labels, box)`` for SAM, in image pixels; box is x1, y1, x2, y2."""
        points = [(x, y) for x, y, _ in self._prompt_points]
        labels = [label for _, _, label in self._prompt_points]
        box = None
        if self._prompt_box is not None:
            b = self._prompt_box
            box = (b.left(), b.top(), b.right(), b.bottom())
        return points, labels, box

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        """Switch between :data:`POLYGON` and :data:`AUTO`."""
        if tool not in (POLYGON, AUTO):
            raise ValueError(f"unknown tool {tool!r}")
        self.tool = tool
        had_prompts = self.has_prompts or self.has_proposal
        self._reset_transient()
        self._redraw()
        if had_prompts:
            self.prompt_changed.emit()
        self.state_changed.emit()

    def set_class_names(self, names: list[str]) -> None:
        self._class_names = list(names)
        self._class = min(self._class, len(names) - 1)
        self.state_changed.emit()

    def set_class(self, index: int) -> None:
        """The class new outlines get.

        Also re-classes the selected outline, but only one the user picked
        (clicked, chose in the list, or is editing) -- not one just finished:
        outlining a dam and then choosing "pup" for the next animal must not
        turn the dam into a pup.
        """
        if not 0 <= index < len(self._class_names):
            return
        self._class = index
        current = self._shapes[self._current] if 0 <= self._current < len(self._shapes) else None
        if self._picked and current is not None and current.cls != index:
            current.cls = index
            self._redraw()
            self.changed.emit()
        self._redraw()
        self.state_changed.emit()

    def select_shape(self, index: int) -> None:
        """Pick outline ``index`` to edit or re-class; -1 picks none."""
        if not -1 <= index < len(self._shapes):
            return
        self._picked = index >= 0
        if index == self._current:
            return
        self._current = index
        self._vertex = None
        if index >= 0:
            self._class = self._shapes[index].cls
        self._redraw()
        self.state_changed.emit()

    def delete_shape(self) -> None:
        """Remove the selected outline."""
        if not 0 <= self._current < len(self._shapes):
            return
        del self._shapes[self._current]
        self._current = min(self._current, len(self._shapes) - 1)
        self._picked = False
        self._vertex = None
        self._redraw()
        self.changed.emit()
        self.state_changed.emit()

    def add_vertex(self, x: float, y: float) -> bool:
        """Polygon tool: add a vertex to the outline being drawn."""
        w, h = self._size
        if not (0 <= x < w and 0 <= y < h):
            return False
        self._draft.append([x, y])
        self._redraw()
        self.state_changed.emit()
        return True

    def finish_polygon(self) -> bool:
        """Close the outline being drawn; needs three vertices."""
        if len(self._draft) < 3:
            return False
        self._add_shape(self._draft)
        self._draft = []
        return True

    def cancel(self) -> None:
        """Abandon the outline being drawn, or the Auto tool's clicks."""
        had_prompts = self.has_prompts or self.has_proposal
        self._draft = []
        self._prompt_points = []
        self._prompt_box = None
        self._proposal = []
        self._redraw()
        if had_prompts:
            self.prompt_changed.emit()
        self.state_changed.emit()

    def undo_step(self) -> bool:
        """Take back the last draft vertex or Auto click. False if there was none."""
        if self._draft:
            self._draft.pop()
            self._redraw()
            self.state_changed.emit()
            return True
        if self._prompt_points or self._prompt_box is not None:
            if self._prompt_points:
                self._prompt_points.pop()
            else:
                self._prompt_box = None
            if not self.has_prompts:
                self._proposal = []
            self._redraw()
            self.prompt_changed.emit()
            self.state_changed.emit()
            return True
        return False

    def remove(self) -> None:
        """Delete key: a draft vertex or click first, else the selected vertex,
        else the selected outline."""
        if self.undo_step():
            return
        if self._vertex is not None:
            self.remove_selected_vertex()
            return
        self.delete_shape()

    def remove_selected_vertex(self) -> bool:
        """Delete the selected vertex; an outline keeps at least three."""
        if self._vertex is None or not 0 <= self._current < len(self._shapes):
            return False
        pts = self._shapes[self._current].pts
        if len(pts) <= 3:
            return False
        del pts[self._vertex]
        self._vertex = None
        self._redraw()
        self.changed.emit()
        return True

    def insert_vertex(self, view_pos: QPointF) -> bool:
        """Add a vertex to the selected outline where ``view_pos`` meets its edge."""
        edge = self.edge_at(view_pos)
        if edge is None:
            return False
        scene = self.mapToScene(view_pos.toPoint())
        self._shapes[self._current].pts.insert(edge + 1, [scene.x(), scene.y()])
        self._vertex = edge + 1
        self._redraw()
        self.changed.emit()
        return True

    def add_prompt(self, x: float, y: float, positive: bool = True) -> bool:
        """Auto tool: a click on (1) or off (0) the animal."""
        w, h = self._size
        if not (0 <= x < w and 0 <= y < h):
            return False
        self._prompt_points.append((x, y, 1 if positive else 0))
        self._redraw()
        self.prompt_changed.emit()
        self.state_changed.emit()
        return True

    def set_prompt_box(self, box: QRectF | None) -> None:
        """Auto tool: the box round the animal, image pixels."""
        if box is not None:
            w, h = self._size
            box = box.normalized().intersected(QRectF(0, 0, w, h))
            if box.width() < 2 or box.height() < 2:
                box = None
        self._prompt_box = box
        self._redraw()
        self.prompt_changed.emit()
        self.state_changed.emit()

    def set_proposal(self, polygon: list[tuple[float, float]] | None) -> None:
        """Show SAM's outline for the current prompts (image pixels); None clears it."""
        self._proposal = [[float(x), float(y)] for x, y in polygon or []]
        self._redraw()
        self.state_changed.emit()

    def accept_proposal(self) -> bool:
        """Turn the proposed outline into a labelled one and clear the prompts."""
        if not self.has_proposal:
            return False
        pts = self._proposal
        self._prompt_points = []
        self._prompt_box = None
        self._proposal = []
        self._add_shape(pts)
        return True

    def commit(self) -> bool:
        """Enter: close the draft or accept the proposal, whichever is pending."""
        return self.finish_polygon() or self.accept_proposal()

    def _add_shape(self, pts: list[list[float]]) -> None:
        self._shapes.append(_Shape(self._class, [list(p) for p in pts]))
        self._current = len(self._shapes) - 1
        self._picked = False
        self._vertex = None
        self._redraw()
        self.changed.emit()
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # hit testing
    # ------------------------------------------------------------------

    def _view_dist(self, x: float, y: float, view_pos: QPointF) -> float:
        p = self.mapFromScene(QPointF(x, y))
        return math.hypot(p.x() - view_pos.x(), p.y() - view_pos.y())

    def vertex_at(self, view_pos: QPointF) -> int | None:
        """A vertex of the selected outline under a viewport position."""
        if not 0 <= self._current < len(self._shapes):
            return None
        best, best_d = None, _GRAB
        for i, (x, y) in enumerate(self._shapes[self._current].pts):
            d = self._view_dist(x, y, view_pos)
            if d < best_d:
                best, best_d = i, d
        return best

    def edge_at(self, view_pos: QPointF) -> int | None:
        """Index of the selected outline's edge (from vertex i to i+1) under a position."""
        if not 0 <= self._current < len(self._shapes):
            return None
        pts = self._shapes[self._current].pts
        best, best_d = None, _GRAB
        for i in range(len(pts)):
            a = self.mapFromScene(QPointF(*pts[i]))
            b = self.mapFromScene(QPointF(*pts[(i + 1) % len(pts)]))
            d = _segment_distance(view_pos, QPointF(a), QPointF(b))
            if d < best_d:
                best, best_d = i, d
        return best

    def shape_at(self, view_pos: QPointF) -> int | None:
        """The outline containing a viewport position; the smallest if they overlap."""
        scene = self.mapToScene(view_pos.toPoint())
        hits = [
            (abs(_area(s.pts)), i)
            for i, s in enumerate(self._shapes)
            if s.polygon().containsPoint(scene, Qt.FillRule.OddEvenFill)
        ]
        return min(hits)[1] if hits else None

    # ------------------------------------------------------------------
    # mouse and wheel
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.has_image:
            return super().mousePressEvent(event)
        pos = event.position()
        button = event.button()
        if button == Qt.MouseButton.MiddleButton:
            self._pan_from = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        scene = self.mapToScene(pos.toPoint())
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self.tool == AUTO:
            if button == Qt.MouseButton.RightButton:
                self.add_prompt(scene.x(), scene.y(), positive=False)
                return
            if button != Qt.MouseButton.LeftButton:
                return
            if not self.has_prompts and self._grab_vertex_or_edge(pos, shift):
                return
            self._box_from = scene
            return
        # Polygon tool
        if button == Qt.MouseButton.RightButton:
            vertex = self.vertex_at(pos)
            if vertex is not None and not self._draft:
                self._vertex = vertex
                self.remove_selected_vertex()
            return
        if button != Qt.MouseButton.LeftButton:
            return
        if self._draft:
            first = self._draft[0]
            if len(self._draft) >= 3 and self._view_dist(first[0], first[1], pos) <= _CLOSE:
                self.finish_polygon()
            else:
                self.add_vertex(scene.x(), scene.y())
            return
        if self._grab_vertex_or_edge(pos, shift):
            return
        hit = self.shape_at(pos)
        if hit is not None and hit != self._current:
            self.select_shape(hit)
            return
        self.add_vertex(scene.x(), scene.y())

    def _grab_vertex_or_edge(self, pos: QPointF, shift: bool) -> bool:
        """Start dragging a vertex of the selected outline (Shift on an edge adds one)."""
        if shift and self.insert_vertex(pos):
            self._dragging = self._picked = True
            return True
        vertex = self.vertex_at(pos)
        if vertex is None:
            return False
        self._picked = True
        self._vertex = vertex
        self._dragging = True
        self._redraw()
        return True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._pan_from is not None:
            delta = pos - self._pan_from
            self._pan_from = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            return
        scene = self.mapToScene(pos.toPoint())
        if self._dragging and self._vertex is not None and 0 <= self._current < len(self._shapes):
            w, h = self._size
            point = self._shapes[self._current].pts[self._vertex]
            point[0] = min(max(scene.x(), 0.0), w - 1.0)
            point[1] = min(max(scene.y(), 0.0), h - 1.0)
            self._redraw()
            return
        if self._box_from is not None:
            self._band = QRectF(self._box_from, scene).normalized()
            self._redraw()
            return
        if self._draft:
            self._hover = scene
            self._update_rubber()
        self.setCursor(self._cursor_for(pos))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        button = event.button()
        if button == Qt.MouseButton.MiddleButton and self._pan_from is not None:
            self._pan_from = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if button == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.changed.emit()
            return
        if button == Qt.MouseButton.LeftButton and self._box_from is not None:
            start, band = self._box_from, self._band
            self._box_from = self._band = None
            corner = self.mapFromScene(start)
            here = event.position()
            dragged = math.hypot(here.x() - corner.x(), here.y() - corner.y()) > _CLICK_SLOP
            if dragged and band is not None:
                self.set_prompt_box(band)
            else:
                self.add_prompt(start.x(), start.y(), positive=True)
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = None
        self._update_rubber()
        super().leaveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.has_image:
            return
        factor = _ZOOM_STEP if event.angleDelta().y() > 0 else 1 / _ZOOM_STEP
        zoom = self.transform().m11() * factor
        if zoom <= self._fit_scale():
            self._fitted = True
            self._fit()
            return
        if zoom > _MAX_ZOOM:
            return
        self._fitted = False
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fitted:
            self._fit()

    def zoom_to_fit(self) -> None:
        self._fitted = True
        self._fit()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _cursor_for(self, view_pos: QPointF) -> Qt.CursorShape:
        if not self.has_image:
            return Qt.CursorShape.ArrowCursor
        if self.busy:
            return Qt.CursorShape.BusyCursor
        if not self._draft and not self.has_prompts and self.vertex_at(view_pos) is not None:
            return Qt.CursorShape.OpenHandCursor
        if self.tool == AUTO:
            return Qt.CursorShape.PointingHandCursor
        return Qt.CursorShape.CrossCursor

    def _fit_scale(self) -> float:
        w, h = self._size
        if not w or not h:
            return 1.0
        view = self.viewport().rect()
        return min(view.width() / w, view.height() / h)

    def _fit(self) -> None:
        if self.has_image:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _cosmetic(self, colour: QColor, width: float, style=Qt.PenStyle.SolidLine) -> QPen:
        pen = QPen(colour, width, style)
        pen.setCosmetic(True)
        return pen

    def _handle(self, x: float, y: float, colour: QColor, fill: QColor, r: float, round_=False):
        item = (
            QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            if round_
            else QGraphicsRectItem(-r, -r, 2 * r, 2 * r)
        )
        item.setPos(x, y)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        item.setPen(QPen(colour, 1.5))
        item.setBrush(QBrush(fill))
        self._scene.addItem(item)
        return item

    def _update_rubber(self) -> None:
        """The line from the draft's last vertex to the cursor, moved without a full redraw."""
        if self._rubber is not None:
            self._scene.removeItem(self._rubber)
            self._rubber = None
        if not self._draft or self._hover is None:
            return
        last = self._draft[-1]
        self._rubber = QGraphicsLineItem(QLineF(last[0], last[1], self._hover.x(), self._hover.y()))
        self._rubber.setPen(self._cosmetic(class_colour(self._class), 1.2, Qt.PenStyle.DashLine))
        self._scene.addItem(self._rubber)

    def _redraw(self) -> None:
        """Rebuild every overlay item. A few hundred at most: cheap."""
        for item in list(self._scene.items()):
            if item is not self._pixmap:
                self._scene.removeItem(item)
        self._rubber = None
        for i, shape in enumerate(self._shapes):
            current = i == self._current
            colour = class_colour(shape.cls)
            fill = QColor(colour)
            fill.setAlpha(95 if current else 55)
            item = QGraphicsPolygonItem(shape.polygon())
            item.setBrush(QBrush(fill))
            item.setPen(self._cosmetic(colour if not current else QColor(255, 255, 255), 2.0))
            self._scene.addItem(item)
            if current:
                for v, (x, y) in enumerate(shape.pts):
                    selected = v == self._vertex
                    self._handle(
                        x,
                        y,
                        QColor(255, 255, 255),
                        colour if not selected else QColor(255, 255, 255),
                        _VERTEX + (1.5 if selected else 0.0),
                    )
        self._draw_draft()
        self._draw_auto()

    def _draw_draft(self) -> None:
        if not self._draft:
            return
        colour = class_colour(self._class)
        path = QPainterPath(QPointF(*self._draft[0]))
        for x, y in self._draft[1:]:
            path.lineTo(x, y)
        line = QGraphicsPathItem(path)
        line.setPen(self._cosmetic(colour, 2.0))
        self._scene.addItem(line)
        for v, (x, y) in enumerate(self._draft):
            closing = v == 0 and len(self._draft) >= 3
            self._handle(
                x,
                y,
                QColor(255, 255, 255),
                QColor(255, 255, 255) if closing else colour,
                _VERTEX + (2.0 if closing else 0.0),
                round_=closing,
            )
        self._update_rubber()

    def _draw_auto(self) -> None:
        if self._band is not None:
            band = QGraphicsRectItem(self._band)
            band.setPen(self._cosmetic(_ACCENT, 1.5, Qt.PenStyle.DashLine))
            self._scene.addItem(band)
        if self._prompt_box is not None:
            box = QGraphicsRectItem(self._prompt_box)
            box.setPen(self._cosmetic(_ACCENT, 1.5))
            self._scene.addItem(box)
        if len(self._proposal) >= 3:
            colour = class_colour(self._class)
            fill = QColor(colour)
            fill.setAlpha(110)
            item = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in self._proposal]))
            item.setBrush(QBrush(fill))
            item.setPen(self._cosmetic(_ACCENT, 2.0, Qt.PenStyle.DashLine))
            self._scene.addItem(item)
        for x, y, label in self._prompt_points:
            colour = _POSITIVE if label else _NEGATIVE
            self._handle(x, y, QColor(255, 255, 255), colour, _PROMPT_R, round_=True)


def _area(pts: list[list[float]]) -> float:
    """Shoelace area of a polygon."""
    n = len(pts)
    return 0.5 * sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)
    )


def _segment_distance(p: QPointF, a: QPointF, b: QPointF) -> float:
    """Distance from ``p`` to the segment ``a``-``b``."""
    dx, dy = b.x() - a.x(), b.y() - a.y()
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return math.hypot(p.x() - a.x(), p.y() - a.y())
    t = max(0.0, min(1.0, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / length2))
    return math.hypot(p.x() - (a.x() + t * dx), p.y() - (a.y() + t * dy))
