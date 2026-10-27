"""
Zone Dialog - Draw and configure zones on camera view.

Allows users to draw zones of various shapes (rectangle, circle, polygon)
on the camera preview and assign labels for use in the node graph.
"""

import logging
import math
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from glider.gui.styles import colors, radius
from glider.vision.zones import (
    Zone,
    ZoneConfiguration,
    ZoneShape,
    arena_sized_zone_vertices,
    draw_zones,
    sized_zone_vertices,
    translate_zone,
)

if TYPE_CHECKING:
    from glider.vision.camera_manager import CameraManager

logger = logging.getLogger(__name__)


# Default zone colors (BGR for OpenCV)
DEFAULT_COLORS = [
    (0, 255, 0),  # Green
    (255, 165, 0),  # Orange (BGR)
    (255, 0, 0),  # Blue (BGR)
    (0, 255, 255),  # Yellow (BGR)
    (255, 0, 255),  # Magenta
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
    (255, 255, 0),  # Cyan
]

#: Outline drawn around the selected zone (BGR). White reads on every zone colour.
SELECTION_COLOR = (255, 255, 255)

#: Arrow-key nudge in frame pixels; Shift multiplies it.
NUDGE_PX = 1
NUDGE_FAST_PX = 10

#: Where the fixed-size controls remember their state between uses.
SETTINGS_GROUP = "zone_editor"

#: Default fixed sizes per unit, chosen to be a plausible zone on a mouse arena
#: (cm) and on a 640x480 frame (px).
DEFAULT_SIZE = {"cm": (10.0, 10.0), "px": (100.0, 100.0)}

#: A stamp turns a normalized click position and the frame size into the shape
#: and vertices of a fixed-size zone; ValueError means "nothing to place".
Stamp = Callable[[tuple[float, float], tuple[int, int]], tuple[ZoneShape, list]]


def _draw_outline(frame: np.ndarray, zone: Zone, color, thickness: int) -> None:
    """Stroke *zone*'s outline onto *frame* (pixel space), in place."""
    h, w = frame.shape[:2]
    pts = zone.get_pixel_vertices(w, h)
    if zone.shape == ZoneShape.RECTANGLE and len(pts) >= 2:
        cv2.rectangle(frame, pts[0], pts[1], color, thickness)
    elif zone.shape == ZoneShape.CIRCLE and len(pts) >= 2:
        (cx, cy), (rx, ry) = pts[0], pts[1]
        cv2.circle(frame, (cx, cy), int(math.hypot(rx - cx, ry - cy)), color, thickness)
    elif zone.shape == ZoneShape.POLYGON and len(pts) >= 3:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, color, thickness)


class ZonePreviewWidget(QLabel):
    """
    Widget for drawing zones on camera preview.

    Handles mouse events for drawing different zone shapes. Outside drawing
    mode a click selects the zone under the cursor, dragging moves it, and the
    arrow keys nudge it (Shift for bigger steps).
    """

    zone_defined = pyqtSignal(object)  # Emits Zone object
    #: The selected zone's id, or None when the selection is cleared.
    zone_selected = pyqtSignal(object)
    #: A zone's id after a drag or nudge has moved it.
    zone_moved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {colors.CANVAS};
                border: 2px solid {colors.BORDER};
                border-radius: {radius.SMALL}px;
            }}
        """)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._frame: np.ndarray | None = None
        self._zone_config: ZoneConfiguration | None = None
        self._drawing_mode = False
        self._current_shape = ZoneShape.RECTANGLE
        self._current_color = DEFAULT_COLORS[0]

        # Drawing state
        self._points: list[QPoint] = []  # Points collected for current shape
        self._current_point: QPoint | None = None  # Mouse position
        self._image_rect = None  # Actual image area within widget
        self._close_threshold = 15  # Pixels - snap to close polygon

        # Selection / move state
        self._selected_id: str | None = None
        self._drag_press: QPoint | None = None  # widget pos the drag started at
        self._drag_vertices: list[tuple[float, float]] | None = None  # before the drag
        self._dragged = False

        #: Fixed-size placement: when set, rectangles and circles are placed
        #: with one click instead of drawn with two.
        self._stamp: Stamp | None = None

        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        #: Re-fit the frame to the widget whenever it is resized. Off: the
        #: frame is scaled once, to the size the widget has when it is set,
        #: which is what the desktop window always did. The embedded compact
        #: editor turns it on -- it is built before layout gives the preview
        #: its real, much smaller, size, and would otherwise show a crop.
        self.refit_on_resize = False

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.refit_on_resize:
            self._update_display()

    def set_frame(self, frame: np.ndarray) -> None:
        """Set the current frame to display."""
        self._frame = frame.copy()
        self._update_display()

    def set_zone_configuration(self, config: ZoneConfiguration) -> None:
        """Set zone configuration to display existing zones."""
        self._zone_config = config
        self._update_display()

    def set_drawing_mode(self, enabled: bool, shape: ZoneShape = None) -> None:
        """Enable/disable zone drawing mode."""
        self._drawing_mode = enabled
        if shape is not None:
            self._current_shape = shape
        self._points.clear()
        self._current_point = None
        self._drag_press = None
        self._drag_vertices = None

        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._update_display()

    def set_current_color(self, color: tuple[int, int, int]) -> None:
        """Set the color for new zones."""
        self._current_color = color

    def set_stamp(self, stamp: "Stamp | None") -> None:
        """Place fixed-size zones with one click (``None``: draw them freehand)."""
        self._stamp = stamp
        self._update_display()

    def _stamping(self) -> bool:
        return (
            self._drawing_mode
            and self._stamp is not None
            and self._current_shape in (ZoneShape.RECTANGLE, ZoneShape.CIRCLE)
        )

    def selected_zone(self) -> Zone | None:
        """The selected zone, if it is still in the configuration."""
        if self._selected_id is None or self._zone_config is None:
            return None
        return self._zone_config.get_zone(self._selected_id)

    def select_zone(self, zone_id: str | None) -> None:
        """Select a zone by id without emitting :attr:`zone_selected`."""
        self._selected_id = zone_id
        self._update_display()

    def _frame_size(self) -> tuple[int, int]:
        h, w = self._frame.shape[:2]
        return (w, h)

    def _draw_overlays(self, frame: np.ndarray) -> None:
        """Selection outline and fixed-size placement preview, in place."""
        zone = self.selected_zone()
        if zone is not None:
            _draw_outline(frame, zone, SELECTION_COLOR, 3)
        if self._stamping() and self._current_point is not None:
            coords = self._widget_to_image_coords(self._current_point)
            if coords is not None:
                ghost = self._stamp_zone(coords)
                if ghost is not None:
                    _draw_outline(frame, ghost, (255, 0, 255), 2)

    def _stamp_zone(self, coords: tuple[int, int]) -> Zone | None:
        """The fixed-size zone a click at image *coords* would place."""
        w, h = self._frame_size()
        try:
            shape, vertices = self._stamp((coords[0] / w, coords[1] / h), (w, h))
        except ValueError as e:
            logger.debug("Fixed-size zone not placed: %s", e)
            return None
        return Zone(
            id=str(uuid.uuid4()),
            name="",
            shape=shape,
            vertices=vertices,
            color=self._current_color,
        )

    def _update_display(self) -> None:
        """Update the display with frame and zones."""
        if self._frame is None:
            self.setText("No camera frame\nStart camera preview first")
            return

        # Draw on a copy of the frame
        display_frame = self._frame.copy()
        h, w = display_frame.shape[:2]

        # Draw existing zones
        if self._zone_config and self._zone_config.zones:
            display_frame = draw_zones(
                display_frame, self._zone_config, alpha=0.3, show_labels=True
            )
        self._draw_overlays(display_frame)

        # Convert to QImage
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        bytes_per_line = 3 * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

        # Scale to fit widget
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Calculate image rect for mouse coordinate mapping
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2
        self._image_rect = (x_offset, y_offset, scaled.width(), scaled.height())

        self.clear()
        self.setPixmap(scaled)
        self.update()

    def _widget_to_image_coords(self, pos: QPoint) -> tuple[int, int] | None:
        """Convert widget coordinates to image coordinates."""
        if self._image_rect is None or self._frame is None:
            return None

        x_off, y_off, img_w, img_h = self._image_rect
        frame_h, frame_w = self._frame.shape[:2]

        # Check if within image bounds
        rel_x = pos.x() - x_off
        rel_y = pos.y() - y_off

        if rel_x < 0 or rel_y < 0 or rel_x >= img_w or rel_y >= img_h:
            return None

        # Scale to frame coordinates
        img_x = int(rel_x * frame_w / img_w)
        img_y = int(rel_y * frame_h / img_h)

        return (img_x, img_y)

    def _should_close_polygon(self, pos: QPoint) -> bool:
        """Check if current position should close the polygon."""
        if self._current_shape != ZoneShape.POLYGON:
            return False
        if len(self._points) < 3:
            return False

        # Check distance to first point
        coords = self._widget_to_image_coords(pos)
        if coords is None:
            return False

        first_coords = self._widget_to_image_coords(self._points[0])
        if first_coords is None:
            return False

        dist = math.sqrt((coords[0] - first_coords[0]) ** 2 + (coords[1] - first_coords[1]) ** 2)
        return dist <= self._close_threshold

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Draw in drawing mode; otherwise select the zone under the cursor."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        coords = self._widget_to_image_coords(event.pos())
        if coords is None:
            return

        if not self._drawing_mode:
            self._begin_drag(event.pos(), coords)
            return

        if self._stamping():
            zone = self._stamp_zone(coords)
            if zone is not None:
                self._current_point = None
                self.zone_defined.emit(zone)
            return

        if self._current_shape == ZoneShape.RECTANGLE:
            self._handle_rectangle_click(event.pos())
        elif self._current_shape == ZoneShape.CIRCLE:
            self._handle_circle_click(event.pos())
        elif self._current_shape == ZoneShape.POLYGON:
            self._handle_polygon_click(event.pos())

    def _handle_rectangle_click(self, pos: QPoint) -> None:
        """Handle click for rectangle drawing."""
        if len(self._points) == 0:
            # First click - start corner
            self._points.append(pos)
        else:
            # Second click - end corner, create zone
            self._points.append(pos)
            self._create_zone_from_points()

    def _handle_circle_click(self, pos: QPoint) -> None:
        """Handle click for circle drawing."""
        if len(self._points) == 0:
            # First click - center
            self._points.append(pos)
        else:
            # Second click - radius point, create zone
            self._points.append(pos)
            self._create_zone_from_points()

    def _handle_polygon_click(self, pos: QPoint) -> None:
        """Handle click for polygon drawing."""
        # Check if we should close the polygon
        if self._should_close_polygon(pos):
            if len(self._points) >= 3:
                self._create_zone_from_points()
            return

        # Add new vertex
        self._points.append(pos)
        self._draw_preview()

    def _create_zone_from_points(self) -> None:
        """Create a zone from collected points."""
        if self._frame is None or not self._points:
            return

        h, w = self._frame.shape[:2]

        # Convert widget coords to normalized coords
        vertices = []
        for pt in self._points:
            coords = self._widget_to_image_coords(pt)
            if coords:
                norm_x = coords[0] / w
                norm_y = coords[1] / h
                vertices.append((norm_x, norm_y))

        if not vertices:
            return

        # Create zone object
        zone = Zone(
            id=str(uuid.uuid4()),
            name="",  # Will be set by dialog
            shape=self._current_shape,
            vertices=vertices,
            color=self._current_color,
        )

        # Clear drawing state
        self._points.clear()
        self._current_point = None

        # Emit signal
        self.zone_defined.emit(zone)

    def _begin_drag(self, pos: QPoint, coords: tuple[int, int]) -> None:
        """Select the zone under *coords* and remember where a drag would start."""
        zone = None
        if self._zone_config is not None and self._frame is not None:
            w, h = self._frame_size()
            zone = self._zone_config.zone_at(coords[0] / w, coords[1] / h, w / h)
        new_id = zone.id if zone is not None else None
        if new_id != self._selected_id:
            self._selected_id = new_id
            self.zone_selected.emit(new_id)
        if zone is not None:
            self._drag_press = pos
            self._drag_vertices = list(zone.vertices)
            self._dragged = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self._update_display()

    def _drag_to(self, pos: QPoint) -> None:
        """Move the dragged zone so it sits where the cursor has carried it.

        Measured from the press, not the last move event, so rounding never
        accumulates and the zone stays under the same spot of the cursor.
        """
        zone = self.selected_zone()
        if zone is None or self._image_rect is None or self._drag_vertices is None:
            return
        _x, _y, img_w, img_h = self._image_rect
        if img_w <= 0 or img_h <= 0:
            return
        dx = (pos.x() - self._drag_press.x()) / img_w
        dy = (pos.y() - self._drag_press.y()) / img_h
        zone.vertices = list(self._drag_vertices)
        translate_zone(zone, dx, dy)
        self._dragged = zone.vertices != self._drag_vertices
        self._update_display()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for preview."""
        if self._drag_press is not None:
            self._drag_to(event.pos())
        elif self._drawing_mode and self._points:
            self._current_point = event.pos()
            self._draw_preview()
        elif self._stamping():
            self._current_point = event.pos()
            self._update_display()
        elif not self._drawing_mode:
            self._update_hover_cursor(event.pos())
        super().mouseMoveEvent(event)

    def _update_hover_cursor(self, pos: QPoint) -> None:
        """An open hand over a zone says it can be grabbed."""
        coords = self._widget_to_image_coords(pos)
        over = False
        if coords is not None and self._zone_config is not None and self._frame is not None:
            w, h = self._frame_size()
            over = self._zone_config.zone_at(coords[0] / w, coords[1] / h, w / h) is not None
        self.setCursor(Qt.CursorShape.OpenHandCursor if over else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Finish a drag, reporting the move if the zone actually went anywhere."""
        if event.button() == Qt.MouseButton.LeftButton and self._drag_press is not None:
            moved = self._dragged and self._selected_id is not None
            self._drag_press = None
            self._drag_vertices = None
            self._dragged = False
            self._update_hover_cursor(event.pos())
            if moved:
                self.zone_moved.emit(self._selected_id)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Arrow keys nudge the selected zone by a frame pixel (Shift: ten)."""
        steps = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }
        zone = self.selected_zone()
        step = steps.get(event.key())
        if step is None or zone is None or self._frame is None or self._drawing_mode:
            super().keyPressEvent(event)
            return
        px = NUDGE_FAST_PX if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else NUDGE_PX
        w, h = self._frame_size()
        before = list(zone.vertices)
        translate_zone(zone, step[0] * px / w, step[1] * px / h)
        if zone.vertices != before:
            self._update_display()
            self.zone_moved.emit(zone.id)

    def _draw_preview(self) -> None:
        """Draw preview of current shape being drawn."""
        if self._frame is None or not self._points:
            return

        display_frame = self._frame.copy()
        h, w = display_frame.shape[:2]

        # Draw existing zones first
        if self._zone_config and self._zone_config.zones:
            display_frame = draw_zones(
                display_frame, self._zone_config, alpha=0.3, show_labels=True
            )
        self._draw_overlays(display_frame)

        # Get pixel coordinates for current points
        pixel_points = []
        for pt in self._points:
            coords = self._widget_to_image_coords(pt)
            if coords:
                pixel_points.append(coords)

        # Get current mouse position
        current_coords = None
        if self._current_point:
            current_coords = self._widget_to_image_coords(self._current_point)

        color = self._current_color
        preview_color = (255, 0, 255)  # Magenta for preview

        if self._current_shape == ZoneShape.RECTANGLE:
            if len(pixel_points) >= 1 and current_coords:
                p1 = pixel_points[0]
                p2 = current_coords
                cv2.rectangle(display_frame, p1, p2, preview_color, 2)
                cv2.circle(display_frame, p1, 6, color, -1)

        elif self._current_shape == ZoneShape.CIRCLE:
            if len(pixel_points) >= 1 and current_coords:
                center = pixel_points[0]
                radius_pt = current_coords
                radius = int(
                    math.sqrt((radius_pt[0] - center[0]) ** 2 + (radius_pt[1] - center[1]) ** 2)
                )
                cv2.circle(display_frame, center, radius, preview_color, 2)
                cv2.circle(display_frame, center, 6, color, -1)

        elif self._current_shape == ZoneShape.POLYGON:
            # Draw existing vertices
            for i, pt in enumerate(pixel_points):
                cv2.circle(display_frame, pt, 6, color, -1)
                if i > 0:
                    cv2.line(display_frame, pixel_points[i - 1], pt, preview_color, 2)

            # Draw line to current mouse position
            if pixel_points and current_coords:
                cv2.line(display_frame, pixel_points[-1], current_coords, preview_color, 2)

                # Show close indicator if near first point
                if self._should_close_polygon(self._current_point):
                    cv2.circle(display_frame, pixel_points[0], 12, (0, 255, 0), 3)

        # Convert and display
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        bytes_per_line = 3 * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.clear()
        self.setPixmap(scaled)
        self.update()

    def finish_polygon(self) -> None:
        """Finish drawing the current polygon."""
        if self._current_shape == ZoneShape.POLYGON and len(self._points) >= 3:
            self._create_zone_from_points()


class ZoneDialog(QDialog):
    """
    Dialog for creating and managing zones.

    Allows users to:
    - Draw zones of different shapes on camera preview
    - Assign names and colors to zones
    - Save/load zone configurations

    Signals:
        zones_changed: Emitted whenever the zone set is altered -- a zone
            added, removed, cleared or loaded from file. Exists for
            :meth:`embed`: as a modal dialog the caller reads the result once
            on ``accept()``, but embedded in the Experiment tab there is no
            accept, so downstream consumers (the CV processor, the tracking
            logger, the node library's zone list) need telling as it happens.
    """

    zones_changed = pyqtSignal()

    def __init__(
        self,
        camera_manager: "CameraManager | None",
        zone_config: ZoneConfiguration,
        parent=None,
        frame: "np.ndarray | None" = None,
        calibration=None,
        settings: QSettings | None = None,
    ):
        """
        Args:
            calibration: The rig's spatial calibration, if any. An arena
                (anything with ``to_arena_cm``/``to_image``) or a scalar
                calibration (``pixels_per_mm``) lets fixed-size zones be
                given in centimetres; without one they are in frame pixels.
            settings: Where the fixed-size controls are remembered; the
                application's ``QSettings`` by default. Tests pass a throwaway.
        """
        super().__init__(parent)
        self._camera = camera_manager
        self._zone_config = zone_config
        self._calibration = calibration
        self._settings = settings if settings is not None else QSettings()
        self._color_index = len(zone_config.zones) % len(DEFAULT_COLORS)
        self._custom_color: tuple[int, int, int] | None = None
        self._pending_zone: Zone | None = None
        self._compact = False  # set by embed(compact=True)

        self._setup_ui()
        self._load_size_settings()
        self._update_size_controls()
        self._update_table()

        # Seed the canvas. An explicit frame (e.g. a scrubbed video frame) takes
        # precedence over the live camera, so zones can be drawn offline.
        initial = frame
        if initial is None and self._camera is not None and self._camera.is_connected:
            result = self._camera.get_frame()
            if result is not None:
                initial, _timestamp = result
        if initial is not None:
            self._preview.set_frame(initial)
            self._zone_config.config_width = initial.shape[1]
            self._zone_config.config_height = initial.shape[0]

        self._preview.set_zone_configuration(self._zone_config)

    def _setup_ui(self) -> None:
        self.setWindowTitle("Zone Configuration")
        self.setMinimumSize(1000, 700)

        layout = QHBoxLayout(self)
        self._main_layout = layout

        # Left side - preview
        left_layout = QVBoxLayout()

        # Preview widget
        self._preview = ZonePreviewWidget()
        self._preview.zone_defined.connect(self._on_zone_defined)
        self._preview.zone_selected.connect(self._on_preview_selection)
        self._preview.zone_moved.connect(self._on_zone_moved)
        left_layout.addWidget(self._preview, 1)

        self._move_hint = QLabel(
            "Click a zone to select it, then drag it or use the arrow keys "
            "(Shift for bigger steps) to move it."
        )
        self._move_hint.setWordWrap(True)
        left_layout.addWidget(self._move_hint)

        # Shape selector
        shape_group = QGroupBox("Zone Shape")
        shape_layout = self._shape_layout = QHBoxLayout(shape_group)

        self._shape_group = QButtonGroup(self)

        self._rect_radio = QRadioButton("Rectangle")
        self._rect_radio.setChecked(True)
        self._shape_group.addButton(self._rect_radio, 0)
        shape_layout.addWidget(self._rect_radio)

        self._circle_radio = QRadioButton("Circle")
        self._shape_group.addButton(self._circle_radio, 1)
        shape_layout.addWidget(self._circle_radio)

        self._polygon_radio = QRadioButton("Polygon")
        self._shape_group.addButton(self._polygon_radio, 2)
        shape_layout.addWidget(self._polygon_radio)

        shape_layout.addStretch()
        self._shape_group.idToggled.connect(self._on_shape_toggled)
        left_layout.addWidget(shape_group)

        # Fixed size: one click places a rectangle or circle of this size.
        size_layout = self._size_layout = QHBoxLayout()
        self._fixed_size_check = QCheckBox("Fixed size")
        self._fixed_size_check.setToolTip(
            "Place rectangles and circles of a set size with one click, centred where "
            "you click.\nWith an arena calibration the size is in cm on the floor and "
            "the zone follows the camera's perspective (it is stored as a polygon)."
        )
        self._fixed_size_check.toggled.connect(self._on_size_changed)
        size_layout.addWidget(self._fixed_size_check)

        self._width_label = QLabel("Width:")
        size_layout.addWidget(self._width_label)
        self._width_spin = QDoubleSpinBox()
        size_layout.addWidget(self._width_spin)

        self._height_label = QLabel("Height:")
        size_layout.addWidget(self._height_label)
        self._height_spin = QDoubleSpinBox()
        size_layout.addWidget(self._height_spin)

        for spin in (self._width_spin, self._height_spin):
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._on_size_changed)

        self._unit_label = QLabel("px")
        size_layout.addWidget(self._unit_label)
        size_layout.addStretch()
        left_layout.addLayout(size_layout)

        # Drawing controls
        draw_layout = self._draw_layout = QHBoxLayout()

        self._draw_btn = QPushButton("Draw Zone")
        self._draw_btn.setCheckable(True)
        self._draw_btn.toggled.connect(self._on_draw_toggled)
        draw_layout.addWidget(self._draw_btn)

        self._finish_btn = QPushButton("Finish Polygon")
        self._finish_btn.clicked.connect(self._finish_polygon)
        self._finish_btn.setEnabled(False)
        draw_layout.addWidget(self._finish_btn)

        self._capture_btn = QPushButton("Capture Frame")
        self._capture_btn.clicked.connect(self._capture_frame)
        draw_layout.addWidget(self._capture_btn)

        draw_layout.addStretch()
        left_layout.addLayout(draw_layout)

        layout.addLayout(left_layout, 2)

        # Right side - controls and table
        right_layout = QVBoxLayout()

        # New zone group (initially hidden)
        self._new_zone_group = QGroupBox("New Zone")
        new_zone_layout = QFormLayout(self._new_zone_group)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g., 'Feeding Area'")
        new_zone_layout.addRow("Name:", self._name_edit)

        color_layout = QHBoxLayout()
        self._color_preview = QFrame()
        self._color_preview.setFixedSize(40, 24)
        self._update_color_preview()
        color_layout.addWidget(self._color_preview)

        self._color_btn = QPushButton("Choose...")
        self._color_btn.clicked.connect(self._choose_color)
        color_layout.addWidget(self._color_btn)
        color_layout.addStretch()
        new_zone_layout.addRow("Color:", color_layout)

        btn_layout = QHBoxLayout()
        self._add_btn = QPushButton("Add Zone")
        self._add_btn.clicked.connect(self._add_pending_zone)
        self._add_btn.setEnabled(False)
        btn_layout.addWidget(self._add_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_pending_zone)
        btn_layout.addWidget(self._cancel_btn)

        new_zone_layout.addRow(btn_layout)

        self._new_zone_group.hide()
        right_layout.addWidget(self._new_zone_group)

        # Existing zones table
        table_group = QGroupBox("Zones")
        table_layout = QVBoxLayout(table_group)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Name", "Shape", "Color", ""])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(1, 80)
        self._table.setColumnWidth(2, 60)
        self._table.setColumnWidth(3, 60)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        table_layout.addWidget(self._table)

        right_layout.addWidget(table_group, 1)

        # Zone info
        info_group = QGroupBox("Information")
        info_layout = QFormLayout(info_group)

        self._count_label = QLabel("0")
        info_layout.addRow("Total Zones:", self._count_label)

        self._res_label = QLabel("--")
        info_layout.addRow("Resolution:", self._res_label)

        right_layout.addWidget(info_group)

        # File operations
        file_layout = self._file_layout = QHBoxLayout()

        self._save_btn = QPushButton("Save...")
        self._save_btn.clicked.connect(self._save_config)
        file_layout.addWidget(self._save_btn)

        self._load_btn = QPushButton("Load...")
        self._load_btn.clicked.connect(self._load_config)
        file_layout.addWidget(self._load_btn)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_zones)
        file_layout.addWidget(self._clear_btn)

        right_layout.addLayout(file_layout)

        # Dialog buttons
        dialog_btn_layout = QHBoxLayout()
        dialog_btn_layout.addStretch()

        self._ok_btn = QPushButton("OK")
        self._ok_btn.clicked.connect(self.accept)
        dialog_btn_layout.addWidget(self._ok_btn)

        self._cancel_dialog_btn = QPushButton("Cancel")
        self._cancel_dialog_btn.clicked.connect(self.reject)
        dialog_btn_layout.addWidget(self._cancel_dialog_btn)

        right_layout.addLayout(dialog_btn_layout)

        layout.addLayout(right_layout, 1)

    def _get_current_shape(self) -> ZoneShape:
        """Get the currently selected shape."""
        if self._rect_radio.isChecked():
            return ZoneShape.RECTANGLE
        elif self._circle_radio.isChecked():
            return ZoneShape.CIRCLE
        else:
            return ZoneShape.POLYGON

    def _update_color_preview(self) -> None:
        """Update the color preview frame."""
        color = self._get_current_color()
        # Convert BGR to RGB for Qt
        self._color_preview.setStyleSheet(
            f"background-color: rgb({color[2]}, {color[1]}, {color[0]}); "
            f"border: 1px solid {colors.BORDER};"
        )

    # ------------------------------------------------------------------
    # fixed size
    # ------------------------------------------------------------------

    def set_calibration(self, calibration) -> None:
        """Adopt a new rig calibration, which may switch fixed sizes between cm and px."""
        self._calibration = calibration
        self._load_size_settings()
        self._update_size_controls()

    def _arena(self):
        """The calibration, if it can map image points to the floor and back."""
        cal = self._calibration
        if cal is not None and hasattr(cal, "to_arena_cm") and hasattr(cal, "to_image"):
            return cal
        return None

    def _scalar_px_per_cm(self, frame_width: int) -> float | None:
        """Pixels per cm from a scalar calibration on a *frame_width* frame, or None."""
        cal = self._calibration
        if cal is None or not getattr(cal, "is_calibrated", False):
            return None
        ppm = float(getattr(cal, "pixels_per_mm", 0.0) or 0.0)
        if ppm <= 0:
            return None
        cal_width = getattr(cal, "calibration_width", 0) or 0
        if cal_width and frame_width:
            ppm *= frame_width / cal_width
        return ppm * 10.0

    def _size_unit(self) -> str:
        """``"cm"`` when a calibration can turn centimetres into pixels, else ``"px"``."""
        if self._arena() is not None or self._scalar_px_per_cm(0) is not None:
            return "cm"
        return "px"

    def _size_key(self, name: str) -> str:
        return f"{SETTINGS_GROUP}/{name}"

    def _load_size_settings(self) -> None:
        """Restore the fixed-size state; sizes are kept per unit, cm and px apart."""
        unit = self._size_unit()
        default_w, default_h = DEFAULT_SIZE[unit]
        s = self._settings
        try:
            fixed = bool(s.value(self._size_key("fixed_size"), False, type=bool))
            width = float(s.value(self._size_key(f"width_{unit}"), default_w, type=float))
            height = float(s.value(self._size_key(f"height_{unit}"), default_h, type=float))
        except (TypeError, ValueError):
            fixed, width, height = False, default_w, default_h
        widgets = (self._fixed_size_check, self._width_spin, self._height_spin)
        for w in widgets:
            w.blockSignals(True)
        try:
            for spin in (self._width_spin, self._height_spin):
                if unit == "cm":
                    spin.setDecimals(1)
                    spin.setRange(0.1, 1000.0)
                    spin.setSingleStep(0.5)
                else:
                    spin.setDecimals(0)
                    spin.setRange(1.0, 10000.0)
                    spin.setSingleStep(5.0)
            self._fixed_size_check.setChecked(fixed)
            self._width_spin.setValue(width)
            self._height_spin.setValue(height)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _save_size_settings(self) -> None:
        unit = self._size_unit()
        s = self._settings
        s.setValue(self._size_key("fixed_size"), self._fixed_size_check.isChecked())
        s.setValue(self._size_key(f"width_{unit}"), self._width_spin.value())
        s.setValue(self._size_key(f"height_{unit}"), self._height_spin.value())

    def _on_size_changed(self, *_args) -> None:
        self._save_size_settings()
        self._update_size_controls()

    def _on_shape_toggled(self, _id: int, checked: bool) -> None:
        if not checked:
            return
        self._update_size_controls()
        if self._draw_btn.isChecked():
            # Switch the shape being drawn without leaving drawing mode.
            self._on_draw_toggled(True)

    def _update_size_controls(self) -> None:
        """Label, show and enable the size inputs for the current shape and unit."""
        shape = self._get_current_shape()
        sizable = shape in (ZoneShape.RECTANGLE, ZoneShape.CIRCLE)
        circle = shape == ZoneShape.CIRCLE
        self._fixed_size_check.setEnabled(sizable)
        enabled = sizable and self._fixed_size_check.isChecked()
        for w in (self._width_label, self._width_spin, self._height_spin, self._unit_label):
            w.setEnabled(enabled)
        self._height_label.setEnabled(enabled)
        self._width_label.setText("Diameter:" if circle else "Width:")
        self._height_label.setVisible(not circle)
        self._height_spin.setVisible(not circle)
        self._unit_label.setText(self._size_unit())
        self._preview.set_stamp(self._stamp if enabled else None)

    def _stamp(
        self, centre: tuple[float, float], frame_size: tuple[int, int]
    ) -> tuple[ZoneShape, list[tuple[float, float]]]:
        """The fixed-size zone centred on *centre*: its shape and normalized vertices.

        Through an arena the size is laid out on the floor and the result is a
        polygon (see :func:`arena_sized_zone_vertices`); with a scalar
        calibration or none it stays a rectangle or circle.
        """
        shape = self._get_current_shape()
        width = self._width_spin.value()
        height = width if shape == ZoneShape.CIRCLE else self._height_spin.value()
        arena = self._arena()
        if arena is not None:
            return ZoneShape.POLYGON, arena_sized_zone_vertices(arena, shape, centre, width, height)
        scale = self._scalar_px_per_cm(frame_size[0]) or 1.0
        return shape, sized_zone_vertices(shape, centre, width * scale, height * scale, frame_size)

    # ------------------------------------------------------------------
    # selection and moving
    # ------------------------------------------------------------------

    def _on_preview_selection(self, zone_id) -> None:
        """Mirror a click on the canvas in the zones table."""
        self._table.blockSignals(True)
        try:
            self._table.clearSelection()
            for i, zone in enumerate(self._zone_config.zones):
                if zone.id == zone_id:
                    self._table.selectRow(i)
                    break
        finally:
            self._table.blockSignals(False)

    def _on_table_selection(self) -> None:
        """Mirror a table row selection on the canvas, so it can be nudged."""
        rows = self._table.selectionModel().selectedRows()
        zone_id = None
        if rows and 0 <= rows[0].row() < len(self._zone_config.zones):
            zone_id = self._zone_config.zones[rows[0].row()].id
        self._preview.select_zone(zone_id)

    def _on_zone_moved(self, _zone_id: str) -> None:
        self.zones_changed.emit()

    def _on_draw_toggled(self, checked: bool) -> None:
        """Handle draw button toggle."""
        shape = self._get_current_shape()
        self._preview.set_drawing_mode(checked, shape)
        self._preview.set_current_color(self._get_current_color())

        if checked:
            self._draw_btn.setText("Cancel Drawing")
            self._finish_btn.setEnabled(shape == ZoneShape.POLYGON)
        else:
            self._draw_btn.setText("Draw Zone")
            self._finish_btn.setEnabled(False)
            if self._pending_zone is None:
                self._new_zone_group.hide()
                self._add_btn.setEnabled(False)

    def _finish_polygon(self) -> None:
        """Finish the current polygon."""
        self._preview.finish_polygon()

    def _on_zone_defined(self, zone: Zone) -> None:
        """Handle zone definition from preview widget."""
        logger.info(f"Zone defined: {zone.shape.value} with {len(zone.vertices)} vertices")
        self._pending_zone = zone
        self._new_zone_group.show()
        self._add_btn.setEnabled(True)
        self._draw_btn.setChecked(False)

        # Generate default name
        count = len(self._zone_config.zones) + 1
        self._name_edit.setText(f"Zone {count}")
        self._name_edit.selectAll()
        self._name_edit.setFocus()

    def _add_pending_zone(self) -> None:
        """Add the pending zone to configuration."""
        if self._pending_zone is None:
            return

        name = self._name_edit.text().strip()
        if not name:
            name = f"Zone {len(self._zone_config.zones) + 1}"

        # Check for duplicate names
        for zone in self._zone_config.zones:
            if zone.name == name:
                QMessageBox.warning(
                    self,
                    "Duplicate Name",
                    f"A zone named '{name}' already exists. Please choose a different name.",
                )
                return

        self._pending_zone.name = name
        self._pending_zone.color = self._get_current_color()
        self._zone_config.add_zone(self._pending_zone)

        # Increment color index for next zone and reset custom color
        self._color_index += 1
        self._custom_color = None
        self._update_color_preview()

        self._pending_zone = None
        self._new_zone_group.hide()
        self._name_edit.clear()
        self._update_table()
        self._preview.set_zone_configuration(self._zone_config)
        self._preview._update_display()
        self.zones_changed.emit()

    def _cancel_pending_zone(self) -> None:
        """Cancel pending zone addition."""
        self._pending_zone = None
        self._new_zone_group.hide()
        self._name_edit.clear()
        self._preview._update_display()

    def _choose_color(self) -> None:
        """Open color picker."""
        current = self._get_current_color()
        # Convert BGR to QColor (RGB)
        initial = QColor(current[2], current[1], current[0])
        color = QColorDialog.getColor(initial, self, "Choose Zone Color")
        if color.isValid():
            # Store as BGR for OpenCV
            bgr = (color.blue(), color.green(), color.red())
            # Store custom color
            self._custom_color = bgr
            self._update_color_preview()
            self._preview.set_current_color(bgr)

    def _get_current_color(self) -> tuple[int, int, int]:
        """Get the current zone color."""
        if hasattr(self, "_custom_color") and self._custom_color:
            return self._custom_color
        return DEFAULT_COLORS[self._color_index % len(DEFAULT_COLORS)]

    def _capture_frame(self) -> None:
        """Capture a new frame from camera."""
        if self._camera is None:
            # Opened on a still frame — a recorded session being reviewed, say.
            # Re-grabbing is the caller's job there: scrub to the frame you
            # want and open the editor again.
            QMessageBox.information(
                self,
                "No camera",
                "These zones are being drawn on a still frame, so there is "
                "nothing to re-capture from.\n\nScrub to the frame you want "
                "and open the zone editor again.",
            )
            return
        if not self._camera.is_connected:
            QMessageBox.warning(
                self, "No Camera", "Camera is not connected. Start camera preview first."
            )
            return

        result = self._camera.get_frame()
        if result is not None:
            frame, timestamp = result
            self._preview.set_frame(frame)
            self._zone_config.config_width = frame.shape[1]
            self._zone_config.config_height = frame.shape[0]
            self._update_info()

    def _update_table(self) -> None:
        """Update the zones table."""
        # Rebuilding the rows must not clear the canvas selection behind its back.
        self._table.blockSignals(True)
        try:
            self._fill_table()
        finally:
            self._table.blockSignals(False)
        selected = self._preview.selected_zone()
        self._on_preview_selection(selected.id if selected is not None else None)
        self._update_info()

    def _fill_table(self) -> None:
        self._table.setRowCount(len(self._zone_config.zones))

        for i, zone in enumerate(self._zone_config.zones):
            # Name
            name_item = QTableWidgetItem(zone.name)
            self._table.setItem(i, 0, name_item)

            # Shape
            shape_item = QTableWidgetItem(zone.shape.value.capitalize())
            self._table.setItem(i, 1, shape_item)

            # Color preview
            color_widget = QFrame()
            color_widget.setStyleSheet(
                f"background-color: rgb({zone.color[2]}, {zone.color[1]}, {zone.color[0]}); "
                f"border: 1px solid {colors.BORDER};"
            )
            self._table.setCellWidget(i, 2, color_widget)

            # Delete button
            delete_btn = QPushButton("X")
            if not self._compact:
                # Desktop font. At touch padding 40px is narrower than the
                # button's own padding and the X disappears.
                delete_btn.setFixedWidth(40)
            delete_btn.clicked.connect(lambda checked, idx=i: self._remove_zone(idx))
            self._table.setCellWidget(i, 3, delete_btn)

    def _remove_zone(self, index: int) -> None:
        """Remove a zone by index."""
        if 0 <= index < len(self._zone_config.zones):
            zone = self._zone_config.zones[index]
            self._zone_config.remove_zone(zone.id)
            self._update_table()
            self._preview.set_zone_configuration(self._zone_config)
            self.zones_changed.emit()
            self._preview._update_display()

    def _update_info(self) -> None:
        """Update information display."""
        self._count_label.setText(str(len(self._zone_config.zones)))

        if self._zone_config.config_width > 0:
            self._res_label.setText(
                f"{self._zone_config.config_width} x {self._zone_config.config_height}"
            )
        else:
            self._res_label.setText("--")

    def _save_config(self) -> None:
        """Save zone configuration to file."""
        from pathlib import Path

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Zone Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._zone_config.save(Path(path))
            QMessageBox.information(self, "Saved", f"Zone configuration saved to {path}")

    def _load_config(self) -> None:
        """Load zone configuration from file."""
        from pathlib import Path

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Zone Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            if self._zone_config.load(Path(path)):
                self._update_table()
                self._preview.set_zone_configuration(self._zone_config)
                self._preview._update_display()
                self.zones_changed.emit()
                QMessageBox.information(
                    self, "Loaded", f"Loaded {len(self._zone_config.zones)} zones"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to load zone configuration file")

    def _clear_zones(self) -> None:
        """Clear all zones."""
        if not self._zone_config.zones:
            return

        reply = QMessageBox.question(
            self,
            "Clear Zones",
            "Remove all zones?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._zone_config.clear()
            self._update_table()
            self._preview.set_zone_configuration(self._zone_config)
            self._preview._update_display()
            self.zones_changed.emit()

    def embed(self, *, compact: bool = False) -> "ZoneDialog":
        """Turn this editor into a plain widget for the Experiment tab.

        Returns ``self`` so it can be added to a layout in one expression.

        ``compact`` is the Pi's 480px screen: the preview goes above the
        controls instead of beside them, and the 1000x700 / 640x480 minimums
        that make a comfortable desktop window go, so the editor is as wide as
        the screen and the page scrolls to reach the rest.

        ``Qt.WindowType.Widget`` stops Qt giving it a window of its own; the
        OK/Cancel pair goes because neither means anything here. *Cancel* is
        the sharper of the two: this editor mutates the caller's
        :class:`~glider.vision.zones.ZoneConfiguration` in place as you draw,
        so there has never been anything to roll back -- as a modal dialog
        Cancel merely stopped the caller from *propagating* the changes it had
        already made. Embedded, propagation happens on
        :attr:`zones_changed` instead, and a button labelled Cancel that
        silently kept your edits would be a lie.
        """
        self.setWindowFlags(Qt.WindowType.Widget)
        self._ok_btn.hide()
        self._cancel_dialog_btn.hide()
        if compact:
            self._compact = True
            self._main_layout.setDirection(QBoxLayout.Direction.TopToBottom)
            self.setMinimumSize(0, 0)
            self._preview.setMinimumSize(240, 180)
            self._preview.refit_on_resize = True
            # Three buttons (or shapes) abreast are wider than 480px in a wide
            # platform font at touch padding; stacked they never clip.
            for row in (
                self._shape_layout,
                self._size_layout,
                self._draw_layout,
                self._file_layout,
            ):
                row.setDirection(QBoxLayout.Direction.TopToBottom)
            self._update_table()  # rows built in __init__ have the desktop delete button
        return self

    def get_zone_configuration(self) -> ZoneConfiguration:
        """Get the zone configuration."""
        return self._zone_config
