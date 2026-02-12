"""
Camera Panel - Dock widget for camera preview and controls.

Provides live camera preview with CV overlays, recording status,
and quick access to camera settings.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.vision.calibration import CameraCalibration
    from glider.vision.camera_manager import CameraManager
    from glider.vision.cv_processor import CVProcessor
    from glider.vision.multi_camera_manager import MultiCameraManager
    from glider.vision.multi_video_recorder import MultiVideoRecorder
    from glider.vision.tracking_logger import TrackingDataLogger
    from glider.vision.video_recorder import VideoRecorder
    from glider.vision.zones import ZoneConfiguration

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Thread-safe container for frame data passed via Qt signals."""

    frame: np.ndarray
    timestamp: float
    camera_id: Optional[str] = None  # For multi-camera mode


class CVWorker(QObject):
    """
    Worker for offloading CV processing from the main thread.
    """

    results_ready = pyqtSignal(
        object, list, list, object
    )  # frame_data, detections, tracked, motion

    def __init__(self, cv_processor: "CVProcessor"):
        super().__init__()
        self._cv_processor = cv_processor

    def process_frame(self, frame_data: FrameData):
        """Process a frame and emit results."""
        if not self._cv_processor or not self._cv_processor.is_initialized:
            return

        try:
            detections, tracked, motion = self._cv_processor.process_frame(
                frame_data.frame, frame_data.timestamp
            )
            self.results_ready.emit(frame_data, detections, tracked, motion)
        except Exception as e:
            logger.error(f"Error in CV worker: {e}")


class CameraPreviewWidget(QLabel):
    """
    Widget displaying live camera feed.

    Thread Safety:
    - All methods must be called from the main Qt thread
    - update_frame() creates QPixmap which is not thread-safe
    """

    frame_clicked = pyqtSignal(int, int)  # x, y click position

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #0d0d1a;
                border: 1px solid #2d2d44;
                border-radius: 8px;
            }
        """)
        self._placeholder = True
        self._calibration = None
        self._show_calibration = True
        self._zone_config: Optional[ZoneConfiguration] = None
        self._show_zones = True
        self.setText("No Camera")
        # Prevent the widget from resizing based on pixmap content
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_calibration(self, calibration) -> None:
        """Set calibration to display on preview."""
        self._calibration = calibration

    def set_show_calibration(self, show: bool) -> None:
        """Toggle calibration line display."""
        self._show_calibration = show

    def set_zone_configuration(self, config: "ZoneConfiguration") -> None:
        """Set zone configuration to display on preview."""
        self._zone_config = config

    def set_show_zones(self, show: bool) -> None:
        """Toggle zone display."""
        self._show_zones = show

    def update_frame(self, frame: np.ndarray) -> None:
        """Update display with new frame."""
        self._placeholder = False

        # Draw calibration lines if enabled
        display_frame = frame
        if self._show_calibration and self._calibration and self._calibration.lines:
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            for line in self._calibration.lines:
                x1, y1, x2, y2 = line.get_pixel_coords(w, h)
                cv2.line(display_frame, (x1, y1), (x2, y2), line.color, 2)
                cv2.circle(display_frame, (x1, y1), 4, line.color, -1)
                cv2.circle(display_frame, (x2, y2), 4, line.color, -1)
                # Draw label
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                label = f"{line.length:.1f}{line.unit.value}"
                cv2.putText(
                    display_frame,
                    label,
                    (mid_x + 5, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    line.color,
                    1,
                )

        # Draw zones if enabled
        if self._show_zones and self._zone_config and self._zone_config.zones:
            from glider.vision.zones import draw_zones

            if display_frame is frame:
                display_frame = frame.copy()
            display_frame = draw_zones(
                display_frame, self._zone_config, alpha=0.3, show_labels=True
            )

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def show_placeholder(self, text: str = "No Camera") -> None:
        """Show placeholder text."""
        self._placeholder = True
        self.clear()
        self.setText(text)
        self.setStyleSheet("""
            QLabel {
                background-color: #0d0d1a;
                border: 1px solid #2d2d44;
                border-radius: 8px;
                color: #666;
                font-size: 14px;
            }
        """)

    def mousePressEvent(self, event):
        """Handle mouse clicks on the preview."""
        if not self._placeholder:
            self.frame_clicked.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)


class CameraPanel(QWidget):
    """
    Camera control panel as dock widget content.

    Layout:
    - Camera preview (top, expandable)
    - Status bar (recording indicator, FPS)
    - Control buttons (Settings, Start/Stop Preview)
    - CV toggle checkbox

    Thread Safety:
    - Frame callbacks from CameraManager run in background threads
    - All UI updates are marshaled to main thread via Qt signals
    """

    settings_requested = pyqtSignal()

    # Thread-safe signals for frame updates (background thread -> main thread)
    _frame_received = pyqtSignal(object)  # FrameData for single camera
    _multi_frame_received = pyqtSignal(object)  # FrameData for multi-camera

    def __init__(
        self,
        camera_manager: "CameraManager",
        cv_processor: "CVProcessor",
        multi_camera_manager: Optional["MultiCameraManager"] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._camera = camera_manager
        self._cv_processor = cv_processor
        self._multi_cam = multi_camera_manager
        self._video_recorder: Optional[VideoRecorder] = None
        self._multi_video_recorder: Optional[MultiVideoRecorder] = None
        self._tracking_logger: Optional[TrackingDataLogger] = None
        self._calibration: Optional[CameraCalibration] = None
        self._zone_config: Optional[ZoneConfiguration] = None
        self._preview_active = False
        self._multi_camera_mode = False
        self._last_frame = None
        self._frame_count = 0

        # Initialize CV Worker and Thread
        self._cv_thread = QThread()
        self._cv_worker = CVWorker(self._cv_processor)
        self._cv_worker.moveToThread(self._cv_thread)
        self._cv_thread.start()

        self._setup_ui()
        self._connect_signals()

        # Timer for FPS updates
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps_display)
        self._fps_timer.start(1000)

    def _setup_ui(self) -> None:
        # Main layout with scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area for the entire panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        # Content widget inside scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Camera preview (single camera mode)
        self._preview = CameraPreviewWidget()

        # Multi-camera preview (grid mode)
        from glider.gui.widgets.multi_camera_preview import MultiCameraPreviewWidget

        self._multi_preview = MultiCameraPreviewWidget()
        self._multi_preview.primary_changed.connect(self._on_primary_camera_changed)

        # Stacked widget to switch between single and multi-camera preview
        self._preview_stack = QStackedWidget()
        self._preview_stack.addWidget(self._preview)
        self._preview_stack.addWidget(self._multi_preview)
        layout.addWidget(self._preview_stack, 1)  # Stretch factor 1

        # Status bar
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 4, 8, 4)

        self._recording_indicator = QLabel("REC")
        self._recording_indicator.setStyleSheet("""
            QLabel {
                background-color: #c0392b;
                color: white;
                padding: 2px 8px;
                border-radius: 8px;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        self._recording_indicator.hide()
        status_layout.addWidget(self._recording_indicator)

        self._fps_label = QLabel("-- FPS")
        self._fps_label.setStyleSheet("color: #888; font-size: 11px;")
        status_layout.addWidget(self._fps_label)

        status_layout.addStretch()

        self._resolution_label = QLabel("---")
        self._resolution_label.setStyleSheet("color: #888; font-size: 11px;")
        status_layout.addWidget(self._resolution_label)

        layout.addWidget(status_frame)

        # Camera selector
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera:")
        camera_label.setStyleSheet("font-size: 12px;")
        camera_layout.addWidget(camera_label)

        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(120)
        camera_layout.addWidget(self._camera_combo, 1)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setFixedWidth(80)
        self._refresh_btn.clicked.connect(self._refresh_cameras)
        camera_layout.addWidget(self._refresh_btn)

        layout.addLayout(camera_layout)

        # Control buttons
        control_layout = QHBoxLayout()

        self._preview_btn = QPushButton("Start Preview")
        self._preview_btn.clicked.connect(self._toggle_preview)
        control_layout.addWidget(self._preview_btn)

        self._settings_btn = QPushButton("Settings")
        self._settings_btn.clicked.connect(self.settings_requested.emit)
        control_layout.addWidget(self._settings_btn)

        layout.addLayout(control_layout)

        # CV options
        cv_layout = QHBoxLayout()

        self._cv_enabled_cb = QCheckBox("Computer Vision")
        self._cv_enabled_cb.setChecked(True)
        self._cv_enabled_cb.setFixedWidth(150)
        self._cv_enabled_cb.toggled.connect(self._on_cv_toggle)
        cv_layout.addWidget(self._cv_enabled_cb)

        self._overlay_cb = QCheckBox("Overlays")
        self._overlay_cb.setChecked(True)
        self._overlay_cb.toggled.connect(self._on_overlay_toggle)
        cv_layout.addWidget(self._overlay_cb)

        cv_layout.addStretch()
        layout.addLayout(cv_layout)

        # Multi-camera options
        multi_cam_layout = QHBoxLayout()

        self._multi_cam_cb = QCheckBox("Multi-Camera")
        self._multi_cam_cb.setChecked(False)
        self._multi_cam_cb.toggled.connect(self._on_multi_camera_toggle)
        # Disable if multi-camera manager not provided
        self._multi_cam_cb.setEnabled(self._multi_cam is not None)
        multi_cam_layout.addWidget(self._multi_cam_cb)

        multi_cam_layout.addStretch()
        layout.addLayout(multi_cam_layout)

        # Set up scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        # Initial camera list
        self._refresh_cameras()

    def _connect_signals(self) -> None:
        """Connect camera callbacks and thread-safe signals."""
        # Register callback with camera manager (called from background thread)
        self._camera.on_frame(self._on_frame)

        # Connect thread-safe signals for UI updates (main thread)
        self._frame_received.connect(self._handle_frame_input)
        self._multi_frame_received.connect(self._handle_multi_frame_input)

        # Connect CV worker signals
        self._cv_worker.results_ready.connect(self._process_cv_results_on_main_thread)

    def _handle_frame_input(self, frame_data: FrameData) -> None:
        """Decide whether to process frame with CV or update UI immediately."""
        if self._cv_enabled_cb.isChecked() and self._cv_processor.is_initialized:
            # Offload to CV worker thread
            self._cv_worker.process_frame(frame_data)
        else:
            # Update UI immediately with raw frame
            self._process_frame_on_main_thread(frame_data)

    def _handle_multi_frame_input(self, frame_data: FrameData) -> None:
        """Decide whether to process multi-frame with CV or update UI immediately."""
        if (
            self._cv_enabled_cb.isChecked()
            and self._cv_processor.is_initialized
            and self._multi_cam
            and frame_data.camera_id == self._multi_cam.primary_camera_id
        ):
            # Offload primary camera to CV worker thread
            self._cv_worker.process_frame(frame_data)
        else:
            # Update UI immediately
            self._process_multi_frame_on_main_thread(frame_data)

    def _refresh_cameras(self) -> None:
        """Refresh available camera list."""
        self._camera_combo.clear()
        cameras = self._camera.enumerate_cameras()
        for cam in cameras:
            self._camera_combo.addItem(cam.name, cam.index)
        if not cameras:
            self._camera_combo.addItem("No cameras found", -1)
            self._preview_btn.setEnabled(False)
        else:
            self._preview_btn.setEnabled(True)

    def _toggle_preview(self) -> None:
        """Start/stop camera preview."""
        if self._preview_active:
            if self._multi_camera_mode:
                self._stop_multi_cameras()
            else:
                self._stop_preview()
        else:
            if self._multi_camera_mode:
                self._setup_multi_cameras()
            else:
                self._start_preview()

    def _start_preview(self) -> None:
        """Start camera preview."""
        camera_idx = self._camera_combo.currentData()
        if camera_idx is None or camera_idx < 0:
            return

        # Use the camera manager's existing settings (configured via Settings dialog)
        # but update the camera index to the selected one
        settings = self._camera.settings
        settings.camera_index = camera_idx

        if self._camera.connect(settings):
            self._camera.start_streaming()
            self._preview_btn.setText("Stop Preview")
            self._preview_active = True

            # Update resolution display
            res = settings.resolution
            self._resolution_label.setText(f"{res[0]}x{res[1]}")

            # Initialize CV processor
            if self._cv_enabled_cb.isChecked():
                self._cv_processor.initialize()

            logger.info(f"Started camera preview: {camera_idx}")
        else:
            self._preview.show_placeholder("Failed to connect")

    def _stop_preview(self) -> None:
        """Stop camera preview."""
        self._camera.stop_streaming()
        self._camera.disconnect()
        self._preview_btn.setText("Start Preview")
        self._preview.show_placeholder("No Camera")
        self._preview_active = False
        self._resolution_label.setText("---")
        self._fps_label.setText("-- FPS")
        logger.info("Stopped camera preview")

    def _on_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Handle incoming camera frame (called from background thread).

        This method is called from CameraManager's capture thread.
        It emits a signal to marshal the frame data to the main thread
        for safe UI updates.
        """
        if not self._preview_active:
            return

        # Copy frame data to avoid race conditions
        frame_copy = frame.copy()

        # Emit signal to process on main thread (thread-safe)
        self._frame_received.emit(FrameData(frame=frame_copy, timestamp=timestamp))

    def _process_frame_on_main_thread(
        self,
        frame_data: FrameData,
        detections: Optional[list] = None,
        tracked: Optional[list] = None,
        motion: Optional[Any] = None,
    ) -> None:
        """
        Update UI with frame and CV results (called on main thread).
        """
        if not self._preview_active:
            return

        frame = frame_data.frame
        timestamp = frame_data.timestamp

        self._frame_count += 1
        display_frame = frame
        annotated_frame = None

        # Use provided results or process locally if not provided but enabled
        if (
            detections is None
            and self._cv_enabled_cb.isChecked()
            and self._cv_processor.is_initialized
        ):
            detections, tracked, motion = self._cv_processor.process_frame(frame, timestamp)

        if detections is not None:
            if self._overlay_cb.isChecked():
                display_frame = self._cv_processor.draw_overlays(frame, detections, tracked, motion)
                annotated_frame = display_frame

        self._last_frame = display_frame
        self._preview.update_frame(display_frame)

        # Write annotated frame to video recorder if recording
        if annotated_frame is not None and self._video_recorder is not None:
            # Draw calibration lines on the annotated frame for the video
            if self._calibration and self._calibration.lines:
                annotated_frame = self._draw_calibration_lines(annotated_frame)
            self._video_recorder.write_annotated_frame(annotated_frame)

        # Log tracking data if tracking logger is active
        if (
            self._tracking_logger is not None
            and self._tracking_logger.is_recording
            and tracked is not None
        ):
            # Set frame size and calibration for distance calculations
            h, w = frame.shape[:2]
            self._tracking_logger.set_frame_size(w, h)
            if self._calibration:
                self._tracking_logger.set_calibration(self._calibration)

            motion_detected = motion.motion_detected if motion else False
            motion_area = motion.motion_area if motion else 0.0

            self._tracking_logger.log_frame(timestamp, tracked, motion_detected, motion_area)

    def _process_cv_results_on_main_thread(
        self, frame_data: FrameData, detections: list, tracked: list, motion: Any
    ) -> None:
        """Handle results from CV worker on main thread."""
        if frame_data.camera_id:
            self._process_multi_frame_on_main_thread(frame_data, detections, tracked, motion)
        else:
            self._process_frame_on_main_thread(frame_data, detections, tracked, motion)

    def _draw_calibration_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw calibration lines on a frame for video recording."""
        if not self._calibration or not self._calibration.lines:
            return frame

        output = frame.copy()
        h, w = output.shape[:2]

        for line in self._calibration.lines:
            x1, y1, x2, y2 = line.get_pixel_coords(w, h)
            # Draw the line
            cv2.line(output, (x1, y1), (x2, y2), line.color, 2)
            # Draw endpoint circles
            cv2.circle(output, (x1, y1), 4, line.color, -1)
            cv2.circle(output, (x2, y2), 4, line.color, -1)
            # Draw measurement label at midpoint
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            label = f"{line.length:.1f}{line.unit.value}"
            cv2.putText(
                output, label, (mid_x + 5, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, line.color, 1
            )

        return output

    def _update_fps_display(self) -> None:
        """Update FPS display."""
        if self._preview_active:
            if self._multi_camera_mode and self._multi_cam:
                # Show primary camera FPS in status bar
                primary_id = self._multi_cam.primary_camera_id
                if primary_id:
                    fps = self._multi_cam.get_camera_fps(primary_id)
                    self._fps_label.setText(f"{fps:.1f} FPS")
            else:
                fps = self._camera.current_fps
                self._fps_label.setText(f"{fps:.1f} FPS")

    def _on_cv_toggle(self, enabled: bool) -> None:
        """Handle CV processing toggle."""
        self._cv_processor.settings.enabled = enabled
        self._overlay_cb.setEnabled(enabled)
        if enabled and self._preview_active:
            self._cv_processor.initialize()

    def _on_overlay_toggle(self, enabled: bool) -> None:
        """Handle overlay display toggle."""
        self._cv_processor.settings.draw_overlays = enabled

    def _on_multi_camera_toggle(self, enabled: bool) -> None:
        """Handle multi-camera mode toggle."""
        if self._multi_cam is None:
            return

        self._multi_camera_mode = enabled
        self._multi_cam.enabled = enabled

        if enabled:
            # Switch to multi-camera preview
            self._preview_stack.setCurrentWidget(self._multi_preview)

            # Stop single-camera preview if active
            if self._preview_active:
                self._stop_preview()

            # Add all available cameras to multi-camera manager
            self._setup_multi_cameras()
        else:
            # Switch back to single-camera preview
            self._preview_stack.setCurrentWidget(self._preview)

            # Stop multi-camera streaming
            self._stop_multi_cameras()

        logger.info(f"Multi-camera mode {'enabled' if enabled else 'disabled'}")

    def _setup_multi_cameras(self) -> None:
        """Set up all available cameras in multi-camera mode."""
        if self._multi_cam is None:
            return

        from dataclasses import replace

        # Get base settings from camera manager (configured via Settings dialog)
        base_settings = self._camera.settings

        # Get all available cameras
        cameras = self._camera.enumerate_cameras()

        for i, cam_info in enumerate(cameras):
            camera_id = self._multi_cam.camera_id_from_index(cam_info.index)
            # Copy base settings but change camera index
            settings = replace(base_settings, camera_index=cam_info.index)

            # Add camera to manager
            if self._multi_cam.add_camera(camera_id, settings):
                # Add preview tile
                is_primary = i == 0
                self._multi_preview.add_camera(camera_id, is_primary)

                # Register frame callback
                self._multi_cam.on_frame(camera_id, self._on_multi_camera_frame)

        # Start streaming on all cameras
        self._multi_cam.start_all_streaming()

        # Initialize CV processor for primary camera
        if self._cv_enabled_cb.isChecked():
            self._cv_processor.initialize()

        self._preview_active = True
        self._preview_btn.setText("Stop Preview")

        # Update resolution display for primary camera
        primary_id = self._multi_cam.primary_camera_id
        if primary_id:
            res = self._multi_cam.get_camera_resolution(primary_id)
            if res:
                self._resolution_label.setText(f"{res[0]}x{res[1]}")

        logger.info(f"Started multi-camera preview with {self._multi_cam.camera_count} cameras")

    def _stop_multi_cameras(self) -> None:
        """Stop all cameras in multi-camera mode."""
        if self._multi_cam is None:
            return

        self._multi_cam.stop_all_streaming()
        self._multi_cam.remove_all_cameras()
        self._multi_preview.remove_all_cameras()

        self._preview_active = False
        self._preview_btn.setText("Start Preview")
        self._resolution_label.setText("---")
        self._fps_label.setText("-- FPS")

        logger.info("Stopped multi-camera preview")

    def _on_multi_camera_frame(self, camera_id: str, frame: np.ndarray, timestamp: float) -> None:
        """
        Handle incoming frame from multi-camera manager (called from background thread).

        This method is called from the camera's capture thread.
        It emits a signal to marshal the frame data to the main thread
        for safe UI updates.
        """
        if not self._preview_active or not self._multi_camera_mode:
            return

        # Copy frame data to avoid race conditions
        frame_copy = frame.copy()

        # Emit signal to process on main thread (thread-safe)
        self._multi_frame_received.emit(
            FrameData(frame=frame_copy, timestamp=timestamp, camera_id=camera_id)
        )

    def _process_multi_frame_on_main_thread(
        self,
        frame_data: FrameData,
        detections: Optional[list] = None,
        tracked: Optional[list] = None,
        motion: Optional[Any] = None,
    ) -> None:
        """
        Process multi-camera frame and update UI (called on main thread via signal).
        """
        if not self._preview_active or not self._multi_camera_mode:
            return

        camera_id = frame_data.camera_id
        frame = frame_data.frame
        timestamp = frame_data.timestamp

        # Update the preview tile
        self._multi_preview.update_frame(camera_id, frame)

        # Get FPS for this camera
        if self._multi_cam:
            fps = self._multi_cam.get_camera_fps(camera_id)
            self._multi_preview.update_fps(camera_id, fps)

        # Only process CV on primary camera
        if self._multi_cam and camera_id == self._multi_cam.primary_camera_id:
            self._frame_count += 1
            display_frame = frame
            annotated_frame = None

            # Use provided results or process locally if enabled
            if (
                detections is None
                and self._cv_enabled_cb.isChecked()
                and self._cv_processor.is_initialized
            ):
                detections, tracked, motion = self._cv_processor.process_frame(frame, timestamp)

            if detections is not None:
                if self._overlay_cb.isChecked():
                    display_frame = self._cv_processor.draw_overlays(
                        frame, detections, tracked, motion
                    )
                    annotated_frame = display_frame

                    # Update the preview tile with CV overlays
                    self._multi_preview.update_frame(camera_id, display_frame)

            self._last_frame = display_frame

            # Write annotated frame to video recorder if recording
            if annotated_frame is not None:
                if self._multi_video_recorder is not None:
                    if self._calibration and self._calibration.lines:
                        annotated_frame = self._draw_calibration_lines(annotated_frame)
                    self._multi_video_recorder.write_annotated_frame(annotated_frame)
                elif self._video_recorder is not None:
                    if self._calibration and self._calibration.lines:
                        annotated_frame = self._draw_calibration_lines(annotated_frame)
                    self._video_recorder.write_annotated_frame(annotated_frame)

            # Log tracking data if tracking logger is active
            if (
                self._tracking_logger is not None
                and self._tracking_logger.is_recording
                and tracked is not None
            ):
                h, w = frame.shape[:2]
                self._tracking_logger.set_frame_size(w, h)
                if self._calibration:
                    self._tracking_logger.set_calibration(self._calibration)

                motion_detected = motion.motion_detected if motion else False
                motion_area = motion.motion_area if motion else 0.0

                self._tracking_logger.log_frame(timestamp, tracked, motion_detected, motion_area)

    def _on_primary_camera_changed(self, camera_id: str) -> None:
        """Handle primary camera change from UI."""
        if self._multi_cam is None:
            return

        self._multi_cam.set_primary_camera(camera_id)

        # Update resolution display
        res = self._multi_cam.get_camera_resolution(camera_id)
        if res:
            self._resolution_label.setText(f"{res[0]}x{res[1]}")

        logger.info(f"Primary camera changed to {camera_id}")

    def set_video_recorder(self, recorder: "VideoRecorder") -> None:
        """Set the video recorder for annotated frame writing."""
        self._video_recorder = recorder

    def set_multi_video_recorder(self, recorder: "MultiVideoRecorder") -> None:
        """Set the multi-video recorder for annotated frame writing."""
        self._multi_video_recorder = recorder

    def set_tracking_logger(self, logger: "TrackingDataLogger") -> None:
        """Set the tracking logger for logging CV results."""
        self._tracking_logger = logger

    def set_calibration(self, calibration: "CameraCalibration") -> None:
        """Set the camera calibration for real-world measurements."""
        self._calibration = calibration

    def set_zone_configuration(self, zone_config: "ZoneConfiguration") -> None:
        """Set the zone configuration for zone display and tracking."""
        self._zone_config = zone_config
        self._preview.set_zone_configuration(zone_config)

    def set_recording(self, recording: bool) -> None:
        """Update recording indicator."""
        if recording:
            self._recording_indicator.show()
        else:
            self._recording_indicator.hide()

        # Update multi-camera preview recording indicators
        if self._multi_camera_mode:
            self._multi_preview.set_recording(recording)

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame (for snapshots)."""
        return self._last_frame.copy() if self._last_frame is not None else None

    def closeEvent(self, event):
        """Clean up on close."""
        if self._preview_active:
            if self._multi_camera_mode:
                self._stop_multi_cameras()
            else:
                self._stop_preview()

        # Stop CV thread
        if self._cv_thread.isRunning():
            self._cv_thread.quit()
            self._cv_thread.wait(2000)

        self._fps_timer.stop()
        super().closeEvent(event)
