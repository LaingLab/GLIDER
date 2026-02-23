"""
Zone Dialog - Draw and configure zones on camera view.

Allows users to draw zones of various shapes (rectangle, circle, polygon)
on the camera preview and assign labels for use in the node graph.
"""

import logging
import math
import uuid
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QDialog,
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

from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape, draw_zones

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


class ZonePreviewWidget(QLabel):
    """
    Widget for drawing zones on camera preview.

    Handles mouse events for drawing different zone shapes.
    """

    zone_defined = pyqtSignal(object)  # Emits Zone object

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #0d0d1a;
                border: 2px solid #2d2d44;
                border-radius: 4px;
            }
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

        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._update_display()

    def set_current_color(self, color: tuple[int, int, int]) -> None:
        """Set the color for new zones."""
        self._current_color = color

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
        """Handle mouse press for zone drawing."""
        if not self._drawing_mode:
            super().mousePressEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        coords = self._widget_to_image_coords(event.pos())
        if coords is None:
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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for preview."""
        if self._drawing_mode and self._points:
            self._current_point = event.pos()
            self._draw_preview()
        super().mouseMoveEvent(event)

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
    """

    def __init__(
        self, camera_manager: "CameraManager", zone_config: ZoneConfiguration, parent=None
    ):
        super().__init__(parent)
        self._camera = camera_manager
        self._zone_config = zone_config
        self._color_index = len(zone_config.zones) % len(DEFAULT_COLORS)
        self._custom_color: tuple[int, int, int] | None = None
        self._pending_zone: Zone | None = None

        self._setup_ui()
        self._update_table()

        # Capture current frame
        if self._camera.is_connected:
            result = self._camera.get_frame()
            if result is not None:
                frame, timestamp = result
                self._preview.set_frame(frame)
                self._zone_config.config_width = frame.shape[1]
                self._zone_config.config_height = frame.shape[0]

        self._preview.set_zone_configuration(self._zone_config)

    def _setup_ui(self) -> None:
        self.setWindowTitle("Zone Configuration")
        self.setMinimumSize(1000, 700)

        layout = QHBoxLayout(self)

        # Left side - preview
        left_layout = QVBoxLayout()

        # Preview widget
        self._preview = ZonePreviewWidget()
        self._preview.zone_defined.connect(self._on_zone_defined)
        left_layout.addWidget(self._preview, 1)

        # Shape selector
        shape_group = QGroupBox("Zone Shape")
        shape_layout = QHBoxLayout(shape_group)

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
        left_layout.addWidget(shape_group)

        # Drawing controls
        draw_layout = QHBoxLayout()

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
        file_layout = QHBoxLayout()

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
            f"border: 1px solid #666;"
        )

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
                f"border: 1px solid #666;"
            )
            self._table.setCellWidget(i, 2, color_widget)

            # Delete button
            delete_btn = QPushButton("X")
            delete_btn.setFixedWidth(40)
            delete_btn.clicked.connect(lambda checked, idx=i: self._remove_zone(idx))
            self._table.setCellWidget(i, 3, delete_btn)

        self._update_info()

    def _remove_zone(self, index: int) -> None:
        """Remove a zone by index."""
        if 0 <= index < len(self._zone_config.zones):
            zone = self._zone_config.zones[index]
            self._zone_config.remove_zone(zone.id)
            self._update_table()
            self._preview.set_zone_configuration(self._zone_config)
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

    def get_zone_configuration(self) -> ZoneConfiguration:
        """Get the zone configuration."""
        return self._zone_config
