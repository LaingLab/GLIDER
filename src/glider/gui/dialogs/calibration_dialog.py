"""
Calibration Dialog - Draw measurement lines on camera view.

Allows users to draw lines on the camera preview and assign
real-world measurements for pixel-to-distance calibration.
"""

import logging
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from glider.vision.calibration import CameraCalibration, LengthUnit

if TYPE_CHECKING:
    from glider.vision.camera_manager import CameraManager

logger = logging.getLogger(__name__)


class CalibrationPreviewWidget(QLabel):
    """
    Widget for drawing calibration lines on camera preview.

    Handles mouse clicks to define line endpoints.
    """

    line_defined = pyqtSignal(tuple, tuple)  # start, end points

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
        self._calibration: CameraCalibration | None = None
        self._drawing_mode = False
        self._start_point: QPoint | None = None
        self._current_point: QPoint | None = None
        self._image_rect = None  # Actual image area within widget
        self._snap_threshold = 15  # Pixels - snap to endpoint if within this distance

    def set_frame(self, frame: np.ndarray) -> None:
        """Set the current frame to display."""
        self._frame = frame.copy()
        self._update_display()

    def set_calibration(self, calibration: CameraCalibration) -> None:
        """Set calibration data to display existing lines."""
        self._calibration = calibration
        num_lines = len(calibration.lines) if calibration else 0
        logger.info(f"set_calibration called with {num_lines} lines")
        self._update_display()

    def set_drawing_mode(self, enabled: bool) -> None:
        """Enable/disable line drawing mode."""
        self._drawing_mode = enabled
        self._start_point = None
        self._current_point = None
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._update_display()

    def _update_display(self) -> None:
        """Update the display with frame and calibration lines."""
        if self._frame is None:
            self.setText("No camera frame\nStart camera preview first")
            return

        # Draw on a copy of the frame
        display_frame = self._frame.copy()
        h, w = display_frame.shape[:2]

        # Draw existing calibration lines
        num_lines = len(self._calibration.lines) if self._calibration else 0
        logger.info(f"_update_display: frame={w}x{h}, calibration lines={num_lines}")

        if self._calibration and self._calibration.lines:
            for i, line in enumerate(self._calibration.lines):
                x1, y1, x2, y2 = line.get_pixel_coords(w, h)
                logger.info(f"Drawing line {i}: ({x1},{y1}) to ({x2},{y2})")
                # Use bright cyan color for visibility
                color = (255, 255, 0)  # Cyan in BGR
                cv2.line(display_frame, (x1, y1), (x2, y2), color, 3)

                # Draw endpoints - larger and highlighted when in drawing mode
                endpoint_radius = 12 if self._drawing_mode else 8
                endpoint_color = (
                    (0, 255, 255) if self._drawing_mode else color
                )  # Yellow when drawing
                cv2.circle(display_frame, (x1, y1), endpoint_radius, endpoint_color, -1)
                cv2.circle(display_frame, (x2, y2), endpoint_radius, endpoint_color, -1)
                # Add outline for better visibility
                cv2.circle(display_frame, (x1, y1), endpoint_radius, (255, 255, 255), 2)
                cv2.circle(display_frame, (x2, y2), endpoint_radius, (255, 255, 255), 2)

                # Draw label
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                label = f"{line.name}: {line.length:.1f} {line.unit.value}"
                cv2.putText(
                    display_frame,
                    label,
                    (mid_x + 5, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        # Convert to QImage - must copy data to avoid dangling reference
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        bytes_per_line = 3 * w
        q_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()  # Copy to own the data

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

        # Clear any existing content and set new pixmap
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

    def _find_nearest_endpoint(self, x: int, y: int) -> tuple[int, int] | None:
        """
        Find the nearest existing line endpoint within snap threshold.

        Args:
            x, y: Click coordinates in image space

        Returns:
            Endpoint coordinates if within threshold, else None
        """
        if not self._calibration or not self._calibration.lines or self._frame is None:
            return None

        h, w = self._frame.shape[:2]
        min_dist = float("inf")
        nearest = None

        for line in self._calibration.lines:
            x1, y1, x2, y2 = line.get_pixel_coords(w, h)

            # Check start point
            dist1 = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
            if dist1 < min_dist and dist1 <= self._snap_threshold:
                min_dist = dist1
                nearest = (x1, y1)

            # Check end point
            dist2 = math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
            if dist2 < min_dist and dist2 <= self._snap_threshold:
                min_dist = dist2
                nearest = (x2, y2)

        return nearest

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for line drawing."""
        logger.info(f"mousePressEvent: drawing_mode={self._drawing_mode}, button={event.button()}")

        if not self._drawing_mode:
            super().mousePressEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        coords = self._widget_to_image_coords(event.pos())
        logger.info(f"Click coords: widget={event.pos()}, image={coords}")

        if coords is None:
            logger.warning("Click outside image area")
            return

        # Check for snap to existing endpoint
        snapped = self._find_nearest_endpoint(coords[0], coords[1])
        if snapped:
            coords = snapped
            logger.info(f"Snapped to endpoint: {coords}")

        if self._start_point is None:
            # First click - start point
            self._start_point = QPoint(coords[0], coords[1])
            self._current_point = self._start_point
            logger.info(f"First point set: {coords}")
        else:
            # Second click - end point
            end_point = QPoint(coords[0], coords[1])
            logger.info(f"Second point set: {coords}, emitting line_defined signal")
            self.line_defined.emit(
                (self._start_point.x(), self._start_point.y()), (end_point.x(), end_point.y())
            )
            self._start_point = None
            self._current_point = None

        self._update_display()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for line preview."""
        if self._drawing_mode and self._start_point is not None:
            coords = self._widget_to_image_coords(event.pos())
            if coords:
                self._current_point = QPoint(coords[0], coords[1])
                self._draw_preview_line()
        super().mouseMoveEvent(event)

    def _draw_preview_line(self) -> None:
        """Draw preview line while drawing."""
        if self._frame is None or self._start_point is None or self._current_point is None:
            return

        display_frame = self._frame.copy()
        h, w = display_frame.shape[:2]

        # Draw existing calibration lines with highlighted endpoints
        if self._calibration and self._calibration.lines:
            for line in self._calibration.lines:
                x1, y1, x2, y2 = line.get_pixel_coords(w, h)
                color = (255, 255, 0)  # Cyan
                cv2.line(display_frame, (x1, y1), (x2, y2), color, 3)
                # Highlighted snap points
                cv2.circle(display_frame, (x1, y1), 12, (0, 255, 255), -1)
                cv2.circle(display_frame, (x2, y2), 12, (0, 255, 255), -1)
                cv2.circle(display_frame, (x1, y1), 12, (255, 255, 255), 2)
                cv2.circle(display_frame, (x2, y2), 12, (255, 255, 255), 2)

        # Check if current point would snap
        end_x, end_y = self._current_point.x(), self._current_point.y()
        snapped = self._find_nearest_endpoint(end_x, end_y)
        if snapped:
            end_x, end_y = snapped

        # Draw preview line in different color (magenta)
        start = (self._start_point.x(), self._start_point.y())
        end = (end_x, end_y)
        cv2.line(display_frame, start, end, (255, 0, 255), 3)
        cv2.circle(display_frame, start, 8, (255, 0, 255), -1)

        # Show snap indicator if snapping
        if snapped:
            cv2.circle(display_frame, end, 15, (0, 255, 0), 3)  # Green ring = will snap
            cv2.circle(display_frame, end, 8, (0, 255, 0), -1)
        else:
            cv2.circle(display_frame, end, 8, (255, 0, 255), -1)

        # Convert and display - must copy to avoid dangling reference
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


class CalibrationDialog(QDialog):
    """
    Dialog for camera calibration with measurement lines.

    Allows users to:
    - Draw lines on camera preview
    - Assign real-world measurements to lines
    - Save/load calibration files
    """

    def __init__(
        self, camera_manager: "CameraManager", calibration: CameraCalibration, parent=None
    ):
        super().__init__(parent)
        self._camera = camera_manager
        self._calibration = calibration
        self._pending_line: tuple[tuple[int, int], tuple[int, int]] | None = None

        self._setup_ui()
        self._update_table()

        # Capture current frame
        if self._camera.is_connected:
            result = self._camera.get_frame()
            if result is not None:
                frame, timestamp = result
                self._preview.set_frame(frame)
                self._calibration.calibration_width = frame.shape[1]
                self._calibration.calibration_height = frame.shape[0]

        self._preview.set_calibration(self._calibration)

    def _setup_ui(self) -> None:
        self.setWindowTitle("Camera Calibration")
        self.setMinimumSize(900, 600)

        layout = QHBoxLayout(self)

        # Left side - preview
        left_layout = QVBoxLayout()

        # Preview widget
        self._preview = CalibrationPreviewWidget()
        self._preview.line_defined.connect(self._on_line_defined)
        left_layout.addWidget(self._preview, 1)

        # Drawing controls
        draw_layout = QHBoxLayout()

        self._draw_btn = QPushButton("Draw Line")
        self._draw_btn.setCheckable(True)
        self._draw_btn.toggled.connect(self._on_draw_toggled)
        draw_layout.addWidget(self._draw_btn)

        self._capture_btn = QPushButton("Capture Frame")
        self._capture_btn.clicked.connect(self._capture_frame)
        draw_layout.addWidget(self._capture_btn)

        draw_layout.addStretch()

        left_layout.addLayout(draw_layout)
        layout.addLayout(left_layout, 2)

        # Right side - controls and table
        right_layout = QVBoxLayout()

        # New line group (initially hidden)
        self._new_line_group = QGroupBox("New Measurement")
        new_line_layout = QFormLayout(self._new_line_group)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g., 'Apparatus Width'")
        new_line_layout.addRow("Name:", self._name_edit)

        self._length_spin = QDoubleSpinBox()
        self._length_spin.setRange(0.001, 100000)
        self._length_spin.setDecimals(2)
        self._length_spin.setValue(100)
        new_line_layout.addRow("Length:", self._length_spin)

        self._unit_combo = QComboBox()
        for unit in LengthUnit:
            self._unit_combo.addItem(unit.value, unit)
        self._unit_combo.setCurrentIndex(1)  # Default to cm
        new_line_layout.addRow("Unit:", self._unit_combo)

        btn_layout = QHBoxLayout()
        self._add_btn = QPushButton("Add Line")
        self._add_btn.clicked.connect(self._add_pending_line)
        self._add_btn.setEnabled(False)
        btn_layout.addWidget(self._add_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_pending_line)
        btn_layout.addWidget(self._cancel_btn)

        new_line_layout.addRow(btn_layout)

        self._new_line_group.hide()
        right_layout.addWidget(self._new_line_group)

        # Existing lines table
        table_group = QGroupBox("Calibration Lines")
        table_layout = QVBoxLayout(table_group)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Name", "Length", "Unit", ""])
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

        # Calibration info
        info_group = QGroupBox("Calibration Info")
        info_layout = QFormLayout(info_group)

        self._ppm_label = QLabel("--")
        info_layout.addRow("Pixels/mm:", self._ppm_label)

        self._res_label = QLabel("--")
        info_layout.addRow("Resolution:", self._res_label)

        right_layout.addWidget(info_group)

        # File operations
        file_layout = QHBoxLayout()

        self._save_btn = QPushButton("Save...")
        self._save_btn.clicked.connect(self._save_calibration)
        file_layout.addWidget(self._save_btn)

        self._load_btn = QPushButton("Load...")
        self._load_btn.clicked.connect(self._load_calibration)
        file_layout.addWidget(self._load_btn)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_calibration)
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

    def _on_draw_toggled(self, checked: bool) -> None:
        """Handle draw button toggle."""
        self._preview.set_drawing_mode(checked)
        if checked:
            self._draw_btn.setText("Cancel Drawing")
        else:
            self._draw_btn.setText("Draw Line")
            # Only clear pending line if user cancelled (not if line was just defined)
            # _on_line_defined will have already set _pending_line before unchecking
            # So if _pending_line exists, don't clear it
            if self._pending_line is None:
                self._new_line_group.hide()
                self._add_btn.setEnabled(False)

    def _on_line_defined(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Handle line definition from preview widget."""
        logger.info(f"_on_line_defined: start={start}, end={end}")
        self._pending_line = (start, end)
        self._new_line_group.show()
        self._add_btn.setEnabled(True)
        self._draw_btn.setChecked(False)

        # Calculate pixel length for reference
        import math

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        pixel_length = math.sqrt(dx * dx + dy * dy)
        self._new_line_group.setTitle(f"New Measurement ({pixel_length:.0f} pixels)")
        logger.info(f"Pending line set, pixel_length={pixel_length:.0f}")

    def _add_pending_line(self) -> None:
        """Add the pending line to calibration."""
        if self._pending_line is None:
            logger.warning("No pending line to add")
            return

        start, end = self._pending_line
        name = self._name_edit.text().strip() or f"Line {len(self._calibration.lines) + 1}"
        length = self._length_spin.value()
        unit = self._unit_combo.currentData()

        resolution = (self._calibration.calibration_width, self._calibration.calibration_height)
        logger.info(f"Adding line: start={start}, end={end}, resolution={resolution}")

        if resolution[0] == 0 or resolution[1] == 0:
            logger.error("Calibration resolution is 0! Cannot add line.")
            QMessageBox.warning(self, "Error", "No frame captured. Click 'Capture Frame' first.")
            return

        self._calibration.add_line(start, end, length, unit, name, resolution)
        logger.info(f"Calibration now has {len(self._calibration.lines)} lines")

        self._pending_line = None
        self._new_line_group.hide()
        self._name_edit.clear()
        self._update_table()
        self._preview.set_calibration(self._calibration)
        self._preview._update_display()

    def _cancel_pending_line(self) -> None:
        """Cancel pending line addition."""
        self._pending_line = None
        self._new_line_group.hide()
        self._name_edit.clear()
        self._preview._update_display()

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
            self._calibration.calibration_width = frame.shape[1]
            self._calibration.calibration_height = frame.shape[0]
            self._update_info()

    def _update_table(self) -> None:
        """Update the calibration lines table."""
        num_lines = len(self._calibration.lines)
        logger.info(f"_update_table: showing {num_lines} lines")
        self._table.setRowCount(num_lines)

        for i, line in enumerate(self._calibration.lines):
            # Name
            name_item = QTableWidgetItem(line.name)
            self._table.setItem(i, 0, name_item)

            # Length
            length_item = QTableWidgetItem(f"{line.length:.2f}")
            length_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(i, 1, length_item)

            # Unit
            unit_item = QTableWidgetItem(line.unit.value)
            self._table.setItem(i, 2, unit_item)

            # Delete button
            delete_btn = QPushButton("X")
            delete_btn.setFixedWidth(40)
            delete_btn.clicked.connect(lambda checked, idx=i: self._remove_line(idx))
            self._table.setCellWidget(i, 3, delete_btn)

        self._update_info()

    def _remove_line(self, index: int) -> None:
        """Remove a calibration line."""
        if self._calibration.remove_line(index):
            self._update_table()
            self._preview.set_calibration(self._calibration)
            self._preview._update_display()

    def _update_info(self) -> None:
        """Update calibration info display."""
        ppm = self._calibration.pixels_per_mm
        if ppm > 0:
            self._ppm_label.setText(f"{ppm:.2f}")
        else:
            self._ppm_label.setText("--")

        if self._calibration.calibration_width > 0:
            self._res_label.setText(
                f"{self._calibration.calibration_width} x {self._calibration.calibration_height}"
            )
        else:
            self._res_label.setText("--")

    def _save_calibration(self) -> None:
        """Save calibration to file."""
        from pathlib import Path

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._calibration.save(Path(path))
            QMessageBox.information(self, "Saved", f"Calibration saved to {path}")

    def _load_calibration(self) -> None:
        """Load calibration from file."""
        from pathlib import Path

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            if self._calibration.load(Path(path)):
                self._update_table()
                self._preview.set_calibration(self._calibration)
                self._preview._update_display()
                QMessageBox.information(
                    self, "Loaded", f"Loaded {len(self._calibration.lines)} calibration lines"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to load calibration file")

    def _clear_calibration(self) -> None:
        """Clear all calibration lines."""
        if not self._calibration.lines:
            return

        reply = QMessageBox.question(
            self,
            "Clear Calibration",
            "Remove all calibration lines?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._calibration.clear()
            self._update_table()
            self._preview.set_calibration(self._calibration)
            self._preview._update_display()

    def get_calibration(self) -> CameraCalibration:
        """Get the calibration data."""
        return self._calibration
