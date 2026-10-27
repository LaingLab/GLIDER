"""The labelling canvas: one frame, the animals on it, and their keypoints.

Two ways to label, switched with :meth:`LabelCanvas.set_mode`:

**Sequential** -- a left click places the current keypoint of the current
animal, then moves on to the next one not yet placed.

**Skeleton** -- a click drops the whole skeleton at once, in the template's
shape, centred on the click (drag out a box instead to size it). The current
animal then gets a box: drag inside it to move the skeleton, drag a corner to
scale it, drag the handle above it to tilt it. Then fix single keypoints.

In both modes:

- **Drag** a point to move it. Pressing a point also makes its animal current.
- **Right click** a point to toggle it between visible and hidden (occluded).
- **Delete** removes the last point pressed; in Sequential mode the next click
  places it again.
- **Wheel** zooms around the cursor; **middle-drag** pans the view.

Only the keypoints are drawn; keypoints are not joined by lines.

Points are kept in image pixels while editing and handed out normalised, as
:class:`~glider.vision.training.labels.PoseInstance`, which is what is saved.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from glider.vision.training.labels import HIDDEN, MISSING, VISIBLE, PoseInstance

__all__ = ["KEYPOINT_COLOURS", "SEQUENTIAL", "SKELETON", "LabelCanvas", "keypoint_colour"]

SEQUENTIAL = "sequential"
SKELETON = "skeleton"

#: Okabe-Ito, the colour-blind-safe palette the keypoint preview already uses,
#: here in RGB. Cycled when a schema has more keypoints than colours.
KEYPOINT_COLOURS = (
    (0, 114, 178),
    (230, 159, 0),
    (0, 158, 115),
    (213, 94, 0),
    (204, 121, 167),
    (86, 180, 233),
    (240, 228, 66),
)

#: Point radius on screen, in pixels, whatever the zoom.
_RADIUS = 6.0
#: How close a press must be to a point to grab it, in screen pixels.
_GRAB = 10.0
_ZOOM_STEP = 1.25
_MAX_ZOOM = 40.0
#: Skeleton mode: handle half-size, box padding and rotate-handle offset, in
#: screen pixels.
_HANDLE = 5.0
_BOX_PAD = 12.0
_ROTATE_GAP = 26.0
#: A press that moves less than this (screen px) is a click, not a box drag.
_CLICK_SLOP = 6.0
#: First skeleton on a frame: its longer side as a fraction of the image's shorter.
_DEFAULT_SIZE = 0.3


def keypoint_colour(index: int) -> QColor:
    return QColor(*KEYPOINT_COLOURS[index % len(KEYPOINT_COLOURS)])


def read_image(path) -> np.ndarray | None:
    """BGR image from ``path``; None if unreadable. Safe on non-ASCII paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _to_qimage(bgr: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class LabelCanvas(QGraphicsView):
    """Place, move and hide keypoints for one or more animals on a frame.

    Signals:
        changed: The labels were edited; read them with :meth:`instances`.
        state_changed: The current animal or keypoint changed (also emitted
            after every edit), for the window's hint text and lists.
    """

    changed = pyqtSignal()
    state_changed = pyqtSignal()

    def __init__(
        self,
        keypoint_names: list[str],
        colours: list[tuple[int, int, int]] | None = None,
        parent=None,
    ):
        """
        Args:
            colours: RGB per keypoint, from the project's skeleton template.
                None: :data:`KEYPOINT_COLOURS`.
        """
        super().__init__(parent)
        self._names = list(keypoint_names)
        self._colours = list(colours or [])
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(20, 20, 20)))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap: QGraphicsPixmapItem | None = None
        self._size = (0, 0)  # image (width, height)
        #: Per animal, per keypoint: [x_px, y_px, visibility].
        self._animals: list[list[list[float]]] = []
        self._current = 0  # current animal
        self._keypoint = 0  # keypoint the next click places
        self._selected: tuple[int, int] | None = None  # last point pressed
        self._dragging: tuple[int, int] | None = None
        self._pan_from: QPointF | None = None
        self._fitted = True  # refit on resize until the user zooms

        self.mode = SEQUENTIAL
        #: Template shape for Skeleton mode: (x, y) per keypoint, 0-1.
        self._layout: list[tuple[float, float]] = []
        #: Longer side of the last skeleton placed or resized, in image pixels,
        #: so the next one comes down at the size the animals actually are.
        self._place_size: float | None = None
        #: An in-progress move/scale/rotate of the current animal.
        self._transform: dict | None = None
        #: Press point (scene) of a click-or-box-drag on empty image.
        self._box_from: QPointF | None = None
        self._band: QRectF | None = None  # the box being dragged out

    # ------------------------------------------------------------------
    # frame and labels in
    # ------------------------------------------------------------------

    def set_frame(self, image: np.ndarray | None, instances: list[PoseInstance]) -> None:
        """Show ``image`` (BGR) with ``instances`` on it; None clears the canvas."""
        self._scene.clear()
        self._pixmap = None
        self._transform = self._box_from = self._band = None
        self._animals = []
        self._selected = self._dragging = None
        if image is None:
            self._size = (0, 0)
            self.state_changed.emit()
            return
        h, w = image.shape[:2]
        self._size = (w, h)
        self._pixmap = QGraphicsPixmapItem(QPixmap.fromImage(_to_qimage(image)))
        self._scene.addItem(self._pixmap)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        k = len(self._names)
        for inst in instances:
            pts = list(inst.keypoints) + [(0.0, 0.0, MISSING)] * (k - len(inst.keypoints))
            self._animals.append([[x * w, y * h, v] for x, y, v in pts[:k]])
        if not self._animals:
            self._animals.append(self._empty())
        self._current = 0
        self._keypoint = self._next_unplaced(0)
        self._fitted = True
        self._fit()
        self._redraw()
        self.state_changed.emit()

    def set_instances(self, instances: list[PoseInstance]) -> None:
        """Replace the labels on the frame already shown, keeping zoom and image.

        For undo and redo: unlike :meth:`set_frame` it does not refit the view,
        and it emits no ``changed`` -- restoring a state is not a new edit.
        """
        w, h = self._size
        if not w or not h:
            return
        k = len(self._names)
        self._animals = []
        for inst in instances:
            pts = list(inst.keypoints) + [(0.0, 0.0, MISSING)] * (k - len(inst.keypoints))
            self._animals.append([[x * w, y * h, v] for x, y, v in pts[:k]])
        if not self._animals:
            self._animals.append(self._empty())
        self._current = min(self._current, len(self._animals) - 1)
        self._keypoint = self._next_unplaced(0)
        self._selected = self._dragging = None
        self._transform = self._box_from = self._band = None
        self._redraw()
        self.state_changed.emit()

    def instances(self) -> list[PoseInstance]:
        """The labels as drawn, normalised. Animals with no points are left out."""
        w, h = self._size
        if not w or not h:
            return []
        out = []
        for animal in self._animals:
            if all(v == MISSING for _x, _y, v in animal):
                continue
            out.append(
                PoseInstance(
                    [
                        (0.0, 0.0, MISSING) if v == MISSING else (x / w, y / h, int(v))
                        for x, y, v in animal
                    ]
                )
            )
        return out

    # ------------------------------------------------------------------
    # state the window shows
    # ------------------------------------------------------------------

    def colour(self, index: int) -> QColor:
        """Keypoint ``index``'s colour: the project's, else the default palette."""
        if index < len(self._colours):
            return QColor(*self._colours[index])
        return keypoint_colour(index)

    @property
    def has_image(self) -> bool:
        return self._size[0] > 0

    @property
    def current_animal(self) -> int:
        return self._current

    @property
    def animal_count(self) -> int:
        return len(self._animals)

    @property
    def current_keypoint(self) -> int:
        """Keypoint the next click places, or -1 when every one is placed."""
        return self._keypoint

    def placed(self) -> list[bool]:
        """Which keypoints of the current animal are placed."""
        if not self._animals:
            return [False] * len(self._names)
        return [v != MISSING for _x, _y, v in self._animals[self._current]]

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch between :data:`SEQUENTIAL` and :data:`SKELETON` labelling."""
        if mode not in (SEQUENTIAL, SKELETON):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self._transform = self._box_from = self._band = None
        self._redraw()
        self.state_changed.emit()

    def set_layout(self, layout: list[tuple[float, float]]) -> None:
        """The template shape Skeleton mode places, one (x, y) per keypoint."""
        self._layout = list(layout)

    def place_skeleton(self, x: float, y: float, box: QRectF | None = None) -> bool:
        """Drop every keypoint at once, in the template's shape.

        Centred on image pixel (x, y) at the last-used size, or stretched to
        fill ``box`` (image pixels) when one was dragged out. Goes into the
        current animal if it is still empty, otherwise starts a new one.
        """
        w, h = self._size
        if not w or not (0 <= x < w and 0 <= y < h):
            return False
        layout = self._layout_or_default()
        xs, ys = [p[0] for p in layout], [p[1] for p in layout]
        lw, lh = max(xs) - min(xs), max(ys) - min(ys)
        lcx, lcy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        if box is not None and box.width() > 2 and box.height() > 2:
            sx = box.width() / (lw or 1.0)
            sy = box.height() / (lh or 1.0)
            cx, cy = box.center().x(), box.center().y()
        else:
            size = self._place_size or _DEFAULT_SIZE * min(w, h)
            sx = sy = size / (max(lw, lh) or 1.0)
            cx, cy = x, y
        points = [
            [
                min(max(cx + (px - lcx) * sx, 0.0), w - 1.0),
                min(max(cy + (py - lcy) * sy, 0.0), h - 1.0),
                VISIBLE,
            ]
            for px, py in layout
        ]
        if any(self.placed()):
            self._animals.append(points)
            self._current = len(self._animals) - 1
        else:
            self._animals[self._current] = points
        self._keypoint = -1
        self._selected = None
        self._remember_size()
        self._redraw()
        self.changed.emit()
        self.state_changed.emit()
        return True

    def rotate_current(self, degrees: float) -> None:
        """Tilt the current animal about its centre."""
        centre = self._centre(self._current)
        if centre is None:
            return
        self._apply(self._current, self._snapshot(), "rotate", centre, math.radians(degrees))
        self.changed.emit()

    def scale_current(self, factor: float) -> None:
        """Grow or shrink the current animal about its centre."""
        centre = self._centre(self._current)
        if centre is None:
            return
        self._apply(self._current, self._snapshot(), "scale", centre, factor)
        self._remember_size()
        self.changed.emit()

    def set_current_keypoint(self, index: int) -> None:
        """Make the next click place (or move) keypoint ``index``."""
        if 0 <= index < len(self._names):
            self._keypoint = index
            self._redraw()
            self.state_changed.emit()

    def new_animal(self) -> None:
        """Start another animal on this frame, unless the current one is still empty."""
        if not self.has_image:
            return
        if any(self.placed()):
            self._animals.append(self._empty())
            self._current = len(self._animals) - 1
        self._keypoint = 0
        self._redraw()
        self.state_changed.emit()

    def delete_animal(self) -> None:
        """Remove the current animal and all its points."""
        if not self._animals:
            return
        had_points = any(self.placed())
        del self._animals[self._current]
        if not self._animals:
            self._animals.append(self._empty())
        self._current = min(self._current, len(self._animals) - 1)
        self._keypoint = self._next_unplaced(0)
        self._selected = None
        self._redraw()
        if had_points:
            self.changed.emit()
        self.state_changed.emit()

    def select_animal(self, index: int) -> None:
        if 0 <= index < len(self._animals) and index != self._current:
            self._current = index
            self._keypoint = self._next_unplaced(0)
            self._redraw()
            self.state_changed.emit()

    def remove_selected_point(self) -> None:
        """Unplace the last point pressed; the next click puts it back."""
        if self._selected is None:
            return
        animal, kp = self._selected
        self._animals[animal][kp] = [0.0, 0.0, MISSING]
        self._current, self._keypoint = animal, kp
        self._selected = None
        self._redraw()
        self.changed.emit()
        self.state_changed.emit()

    def place(self, x: float, y: float) -> bool:
        """Place the current keypoint at image pixel (x, y). Returns whether it did."""
        w, h = self._size
        if self._keypoint < 0 or not (0 <= x < w and 0 <= y < h):
            return False
        animal = self._animals[self._current]
        animal[self._keypoint] = [x, y, VISIBLE]
        self._selected = (self._current, self._keypoint)
        self._keypoint = self._next_unplaced(self._keypoint)
        self._redraw()
        self.changed.emit()
        self.state_changed.emit()
        return True

    def toggle_hidden(self, animal: int, kp: int) -> None:
        point = self._animals[animal][kp]
        if point[2] == MISSING:
            return
        point[2] = HIDDEN if point[2] == VISIBLE else VISIBLE
        self._redraw()
        self.changed.emit()

    def point_at(self, view_pos: QPointF) -> tuple[int, int] | None:
        """The point under a viewport position, current animal first."""
        order = [self._current] + [i for i in range(len(self._animals)) if i != self._current]
        best, best_d = None, _GRAB
        for a in order:
            for k, (x, y, v) in enumerate(self._animals[a]):
                if v == MISSING:
                    continue
                p = self.mapFromScene(QPointF(x, y))
                d = ((p.x() - view_pos.x()) ** 2 + (p.y() - view_pos.y()) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = (a, k), d
        return best

    # ------------------------------------------------------------------
    # mouse and wheel
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.has_image:
            return super().mousePressEvent(event)
        pos = event.position()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_from = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        hit = self.point_at(pos)
        if event.button() == Qt.MouseButton.RightButton:
            if hit is not None:
                self.toggle_hidden(*hit)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        scene = self.mapToScene(pos.toPoint())
        if self.mode == SKELETON:
            handle = self.handle_at(pos)
            if handle is not None:
                self._begin_transform("rotate" if handle == "rotate" else "scale", scene)
                return
        if hit is not None:
            self._selected = self._dragging = hit
            if hit[0] != self._current:
                self._current = hit[0]
                self._keypoint = self._next_unplaced(0)
            self._redraw()
            self.state_changed.emit()
            return
        if self.mode == SKELETON:
            inside = self.animal_box_at(pos)
            if inside is not None:
                if inside != self._current:
                    self.select_animal(inside)
                self._begin_transform("move", scene)
                return
            self._box_from = scene
            return
        self.place(scene.x(), scene.y())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._pan_from is not None:
            delta = pos - self._pan_from
            self._pan_from = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            return
        if self._dragging is not None:
            a, k = self._dragging
            w, h = self._size
            scene = self.mapToScene(pos.toPoint())
            point = self._animals[a][k]
            point[0] = min(max(scene.x(), 0.0), w - 1.0)
            point[1] = min(max(scene.y(), 0.0), h - 1.0)
            self._redraw()
            return
        if self._transform is not None:
            self._update_transform(self.mapToScene(pos.toPoint()))
            return
        if self._box_from is not None:
            scene = self.mapToScene(pos.toPoint())
            self._band = QRectF(self._box_from, scene).normalized()
            self._redraw()
            return
        self.setCursor(self._cursor_for(pos))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_from is not None:
            self._pan_from = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._dragging is not None:
            self._dragging = None
            self.changed.emit()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._transform is not None:
            moved = self._transform.get("moved", False)
            self._transform = None
            if moved:
                self._remember_size()
                self.changed.emit()
            self._redraw()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._box_from is not None:
            start, band = self._box_from, self._band
            self._box_from = self._band = None
            corner = self.mapFromScene(start)
            here = event.position()
            dragged = math.hypot(here.x() - corner.x(), here.y() - corner.y()) > _CLICK_SLOP
            if dragged and band is not None:
                self.place_skeleton(band.center().x(), band.center().y(), band)
            else:
                self.place_skeleton(start.x(), start.y())
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.has_image:
            return
        factor = _ZOOM_STEP if event.angleDelta().y() > 0 else 1 / _ZOOM_STEP
        zoom = self.transform().m11() * factor
        fit = self._fit_scale()
        if zoom <= fit:
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

    def _empty(self) -> list[list[float]]:
        return [[0.0, 0.0, MISSING] for _ in self._names]

    def _layout_or_default(self) -> list[tuple[float, float]]:
        """The template shape, or a ring of keypoints if none was given."""
        k = len(self._names)
        if len(self._layout) == k:
            return self._layout
        return [
            (0.5 + 0.4 * math.cos(2 * math.pi * i / k), 0.5 + 0.4 * math.sin(2 * math.pi * i / k))
            for i in range(k)
        ]

    def _placed_points(self, animal: int) -> list[list[float]]:
        if not 0 <= animal < len(self._animals):
            return []
        return [p for p in self._animals[animal] if p[2] != MISSING]

    def _centre(self, animal: int) -> QPointF | None:
        pts = self._placed_points(animal)
        if len(pts) < 2:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return QPointF((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

    def _zoom(self) -> float:
        return self.transform().m11() or 1.0

    def animal_box(self, animal: int) -> QRectF | None:
        """The box drawn round an animal in Skeleton mode, in image pixels."""
        pts = self._placed_points(animal)
        if len(pts) < 2:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = _BOX_PAD / self._zoom()
        return QRectF(min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad,
                      max(ys) - min(ys) + 2 * pad)  # fmt: skip

    def handles(self) -> dict[str, QPointF]:
        """Corner (scale) and top (rotate) handles of the current animal, image pixels."""
        box = self.animal_box(self._current)
        if box is None or self.mode != SKELETON:
            return {}
        top = QPointF(box.center().x(), box.top() - _ROTATE_GAP / self._zoom())
        return {
            "tl": box.topLeft(),
            "tr": box.topRight(),
            "br": box.bottomRight(),
            "bl": box.bottomLeft(),
            "rotate": top,
        }

    def handle_at(self, view_pos: QPointF) -> str | None:
        for name, point in self.handles().items():
            p = self.mapFromScene(point)
            if math.hypot(p.x() - view_pos.x(), p.y() - view_pos.y()) <= _HANDLE + 4:
                return name
        return None

    def animal_box_at(self, view_pos: QPointF) -> int | None:
        """The animal whose box contains a viewport position, current first."""
        scene = self.mapToScene(view_pos.toPoint())
        order = [self._current] + [i for i in range(len(self._animals)) if i != self._current]
        for a in order:
            box = self.animal_box(a)
            if box is not None and box.contains(scene):
                return a
        return None

    def _cursor_for(self, view_pos: QPointF) -> Qt.CursorShape:
        if not self.has_image:
            return Qt.CursorShape.ArrowCursor
        if self.mode == SKELETON:
            handle = self.handle_at(view_pos)
            if handle == "rotate":
                return Qt.CursorShape.PointingHandCursor
            if handle in ("tl", "br"):
                return Qt.CursorShape.SizeFDiagCursor
            if handle in ("tr", "bl"):
                return Qt.CursorShape.SizeBDiagCursor
        if self.point_at(view_pos) is not None:
            return Qt.CursorShape.OpenHandCursor
        if self.mode == SKELETON and self.animal_box_at(view_pos) is not None:
            return Qt.CursorShape.SizeAllCursor
        return Qt.CursorShape.CrossCursor

    def _snapshot(self) -> list[list[float]]:
        return [list(p) for p in self._animals[self._current]]

    def _begin_transform(self, kind: str, scene: QPointF) -> None:
        centre = self._centre(self._current)
        if centre is None:
            return
        self._transform = {
            "kind": kind,
            "start": scene,
            "centre": centre,
            "original": self._snapshot(),
            "angle": math.atan2(scene.y() - centre.y(), scene.x() - centre.x()),
            "distance": max(1e-6, math.hypot(scene.x() - centre.x(), scene.y() - centre.y())),
        }

    def _update_transform(self, scene: QPointF) -> None:
        t = self._transform
        centre = t["centre"]
        if t["kind"] == "move":
            amount = (scene.x() - t["start"].x(), scene.y() - t["start"].y())
        elif t["kind"] == "rotate":
            angle = math.atan2(scene.y() - centre.y(), scene.x() - centre.x())
            amount = angle - t["angle"]
        else:
            distance = math.hypot(scene.x() - centre.x(), scene.y() - centre.y())
            amount = max(0.05, distance / t["distance"])
        self._apply(self._current, t["original"], t["kind"], centre, amount)
        t["moved"] = True

    def _apply(self, animal, original, kind, centre: QPointF, amount) -> None:
        """Set ``animal``'s placed points to ``original`` moved, scaled or rotated."""
        w, h = self._size
        cx, cy = centre.x(), centre.y()
        out = []
        for x, y, v in original:
            if v != MISSING:
                if kind == "move":
                    x, y = x + amount[0], y + amount[1]
                elif kind == "scale":
                    x, y = cx + (x - cx) * amount, cy + (y - cy) * amount
                else:
                    c, s = math.cos(amount), math.sin(amount)
                    dx, dy = x - cx, y - cy
                    x, y = cx + dx * c - dy * s, cy + dx * s + dy * c
                x, y = min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)
            out.append([x, y, v])
        self._animals[animal] = out
        self._redraw()

    def _remember_size(self) -> None:
        pts = self._placed_points(self._current)
        if len(pts) >= 2:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            self._place_size = max(max(xs) - min(xs), max(ys) - min(ys)) or self._place_size

    def _next_unplaced(self, start: int) -> int:
        """First unplaced keypoint at or after ``start``, wrapping; -1 if none."""
        if not self._animals:
            return 0
        animal = self._animals[self._current]
        k = len(animal)
        for step in range(k):
            i = (start + step) % k
            if animal[i][2] == MISSING:
                return i
        return -1

    def _fit_scale(self) -> float:
        w, h = self._size
        if not w or not h:
            return 1.0
        view = self.viewport().rect()
        return min(view.width() / w, view.height() / h)

    def _fit(self) -> None:
        if self.has_image:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _redraw(self) -> None:
        """Rebuild the point and skeleton items. A few dozen items: cheap."""
        for item in list(self._scene.items()):
            if item is not self._pixmap:
                self._scene.removeItem(item)
        for a, animal in enumerate(self._animals):
            current = a == self._current
            alpha = 255 if current else 120
            for k, (x, y, v) in enumerate(animal):
                if v == MISSING:
                    continue
                colour = self.colour(k)
                colour.setAlpha(alpha)
                dot = QGraphicsEllipseItem(-_RADIUS, -_RADIUS, 2 * _RADIUS, 2 * _RADIUS)
                dot.setPos(x, y)
                dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                outline = QPen(QColor(255, 255, 255, alpha) if current else QColor(0, 0, 0, alpha))
                outline.setWidthF(2.0 if (a, k) == self._selected else 1.0)
                if v == HIDDEN:
                    # Hollow: placed where the part is, but the part is not seen.
                    outline = QPen(colour)
                    outline.setWidthF(2.5)
                    dot.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                else:
                    dot.setBrush(QBrush(colour))
                dot.setPen(outline)
                self._scene.addItem(dot)
        self._draw_skeleton_controls()

    def _draw_skeleton_controls(self) -> None:
        """The current animal's box and handles, and a box being dragged out."""
        if self.mode != SKELETON:
            return
        accent = QColor(56, 189, 248)
        if self._band is not None:
            pen = QPen(accent, 1.5, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            band = QGraphicsRectItem(self._band)
            band.setPen(pen)
            self._scene.addItem(band)
        box = self.animal_box(self._current)
        if box is None:
            return
        pen = QPen(accent, 1.2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        outline = QGraphicsRectItem(box)
        outline.setPen(pen)
        self._scene.addItem(outline)
        handles = self.handles()
        stem = QGraphicsLineItem(
            box.center().x(), box.top(), handles["rotate"].x(), handles["rotate"].y()
        )
        solid = QPen(accent, 1.2)
        solid.setCosmetic(True)
        stem.setPen(solid)
        self._scene.addItem(stem)
        for name, point in handles.items():
            if name == "rotate":
                item = QGraphicsEllipseItem(-_HANDLE, -_HANDLE, 2 * _HANDLE, 2 * _HANDLE)
            else:
                item = QGraphicsRectItem(-_HANDLE, -_HANDLE, 2 * _HANDLE, 2 * _HANDLE)
            item.setPos(point)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setPen(QPen(accent, 1.5))
            item.setBrush(QBrush(QColor(15, 23, 42)))
            self._scene.addItem(item)
