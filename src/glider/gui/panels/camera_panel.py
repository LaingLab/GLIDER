"""
Camera Panel - Dock widget for camera preview and controls.

Provides live camera preview with CV overlays, recording status,
and quick access to camera settings.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from glider.gui.panels.fps_meter import FpsMeter

if TYPE_CHECKING:
    from glider.vision.calibration import CameraCalibration
    from glider.vision.camera_manager import CameraManager
    from glider.vision.cv_processor import CVProcessor
    from glider.vision.multi_camera_manager import MultiCameraManager
    from glider.vision.multi_video_recorder import MultiVideoRecorder
    from glider.vision.tracking_logger import TrackingDataLogger
    from glider.vision.video_recorder import VideoRecorder
    from glider.vision.video_tracking_runner import VideoTrackingConfig
    from glider.vision.zones import ZoneConfiguration

logger = logging.getLogger(__name__)


#: Shown under the keypoint-names field when the operator must type the names.
#: YOLO checkpoints record class names and a kpt_shape, never body-part names.
_MANUAL_NAMES_HINT = "Names must be in the model's training order."

#: How each detected pose-model format is described in the panel.
_KIND_LABELS = {"yolo": "YOLO", "dlc": "DeepLabCut", "sleap": "SLEAP"}


def _picker_row(label: QLabel, button: QPushButton) -> QHBoxLayout:
    """A file-picker row: stretched label + a tight Browse button.

    Mirrors the ``_row`` helper the behavior-analysis window uses for its
    model/YOLO pickers, keeping the Live Behavior group visually consistent
    with the Apply tab.
    """
    row = QHBoxLayout()
    row.addWidget(label, 1)
    row.addWidget(button, 0)
    return row


@dataclass
class FrameData:
    """Thread-safe container for frame data passed via Qt signals."""

    frame: np.ndarray
    timestamp: float
    camera_id: str | None = None  # For multi-camera mode


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
        self._placeholder = True
        self._calibration = None
        self._show_calibration = True
        self._zone_config: ZoneConfiguration | None = None
        self._show_zones = True
        # Live pose-skeleton + behavior-label overlays (set by the panel from
        # the main thread; drawn onto the BGR frame before RGB conversion).
        self._pose_kps: np.ndarray | None = None
        self._behavior_label = ""
        self._vocab: list[str] | None = None
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

    def set_pose_overlay(self, keypoints: "np.ndarray | None") -> None:
        """Set the pose skeleton to draw on the preview.

        ``keypoints`` is a ``(K, 2)`` ndarray of xy pixel coords, or ``None``
        to clear the skeleton. Main-thread only (like the other setters).
        """
        self._pose_kps = keypoints

    def set_behavior_label(self, label: str) -> None:
        """Set the behavior-label badge text (empty string clears it).

        Main-thread only.
        """
        self._behavior_label = label or ""

    def set_behavior_vocab(self, vocab: list[str]) -> None:
        """Set the ordered class vocabulary for stable label colors.

        The panel calls this once at Start. Main-thread only.
        """
        self._vocab = vocab

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw the pose skeleton + behavior badge onto ``frame`` in place.

        Returns the same array for chaining. No-op when no pose/label is set.
        The overlay module is imported lazily to avoid pulling the heavy
        behavior-analysis package at GUI import time (mirrors the lazy vision
        imports elsewhere in this module).
        """
        if self._pose_kps is None and not self._behavior_label:
            return frame

        from glider.analysis.behavior.classify import overlay

        if self._pose_kps is not None:
            overlay.draw_skeleton(frame, self._pose_kps, edges=None)
        if self._behavior_label:
            overlay.draw_label_badge(
                frame,
                self._behavior_label,
                overlay.color_for_behavior(self._behavior_label, self._vocab),
            )
        return frame

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

        # Draw pose skeleton + behavior badge if set. Copy first so we never
        # mutate the caller's frame (zones/calibration may already have copied).
        if self._pose_kps is not None or self._behavior_label:
            if display_frame is frame:
                display_frame = frame.copy()
            display_frame = self._draw_overlays(display_frame)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

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
        self.setProperty("textRole", "muted")

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
    analysis_requested = pyqtSignal(str)  # output_dir → open the Analysis panel
    draw_zones_requested = pyqtSignal(object)  # scrubbed frame (np.ndarray) for the zone editor

    # Thread-safe signals for frame updates (background thread -> main thread)
    _frame_received = pyqtSignal(object)  # FrameData for single camera
    _multi_frame_received = pyqtSignal(object)  # FrameData for multi-camera

    # Signal to dispatch frame processing to CVWorker on its own thread.
    # Because _cv_worker lives on a different thread (QueuedConnection is used
    # automatically), this ensures process_frame() runs on the worker thread.
    _process_frame_requested = pyqtSignal(object)  # FrameData -> CVWorker.process_frame

    # Live-behavior worker dispatch. Like _process_frame_requested, these cross
    # into the BehaviorInferenceWorker's thread via automatic QueuedConnections.
    _behavior_init_requested = pyqtSignal(str, str, list)  # pkl, pt, keypoint names
    _behavior_frame_requested = pyqtSignal(object)  # FrameData -> worker.process_frame
    # PumpStats from the rehearsal pump's thread -> the GUI thread.
    _rehearsal_finished = pyqtSignal(object)

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
        self._video_recorder: VideoRecorder | None = None
        self._multi_video_recorder: MultiVideoRecorder | None = None
        self._tracking_logger: TrackingDataLogger | None = None
        # data_recorder and event_logger are the two newer per-frame
        # consumers wired in by main_window when the core is constructed.
        # Kept Optional so unit tests / standalone panels still work without
        # them.
        self._data_recorder: Any | None = None
        self._event_logger: Any | None = None
        # Carries live classifications to nodes in the flow. Optional for the
        # same reason as the two above: standalone panels and unit tests build
        # a CameraPanel without a core.
        self._live_signals: Any | None = None
        self._calibration: CameraCalibration | None = None
        self._zone_config: ZoneConfiguration | None = None
        self._preview_active = False
        self._multi_camera_mode = False
        self._last_frame = None
        self._frame_count = 0

        # --- Live behavior inference state ---
        # Chosen model paths (set via the pickers); both required to Start.
        self._behavior_pkl: Path | None = None
        self._pose_model_path: Path | None = None
        self._behavior_thread: QThread | None = None
        self._behavior_worker: Any | None = None
        self._behavior_running = False
        self._rehearsal_pump: Any | None = None

        # Video-file source state (offline tracking)
        from glider.vision.video_source import VideoFileSource

        self._video_source = VideoFileSource()
        self._video_mode = False
        self._video_current_frame = 0
        self._video_frame = None  # most recent scrubbed frame (np.ndarray)

        # Initialize CV Worker and Thread
        self._cv_thread = QThread()
        self._cv_worker = CVWorker(self._cv_processor)
        self._cv_worker.moveToThread(self._cv_thread)
        self._cv_thread.start()

        # Models and videos can be dragged straight onto the panel; the drop is
        # routed by type (see gui/panels/model_drop.py).
        self.setAcceptDrops(True)
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
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 4, 8, 4)

        self._recording_indicator = QLabel("REC")
        self._recording_indicator.setProperty("recording", True)
        self._recording_indicator.hide()
        status_layout.addWidget(self._recording_indicator)

        self._fps_label = QLabel("-- FPS")
        self._fps_label.setProperty("textRole", "muted")
        status_layout.addWidget(self._fps_label)

        status_layout.addStretch()

        self._resolution_label = QLabel("---")
        self._resolution_label.setProperty("textRole", "muted")
        status_layout.addWidget(self._resolution_label)

        layout.addWidget(status_frame)

        # Camera selector
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera:")
        camera_layout.addWidget(camera_label)

        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(120)
        camera_layout.addWidget(self._camera_combo, 1)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setFixedWidth(80)
        self._refresh_btn.clicked.connect(self._refresh_cameras)
        camera_layout.addWidget(self._refresh_btn)

        layout.addLayout(camera_layout)

        # --- Source toggle: Live camera vs Video file ---
        source_layout = QHBoxLayout()
        self._live_radio = QRadioButton("Live")
        self._video_radio = QRadioButton("Video file")
        self._live_radio.setChecked(True)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setEnabled(False)
        source_layout.addWidget(QLabel("Source:"))
        source_layout.addWidget(self._live_radio)
        source_layout.addWidget(self._video_radio)
        source_layout.addWidget(self._browse_btn, 1)
        layout.addLayout(source_layout)

        # --- Video transport (hidden until a file loads) ---
        self._video_controls = QWidget()
        # Two stacked rows so the strip stays within a narrow (Pi) screen width
        # instead of pushing the preview off-screen: seek on top, actions below.
        vctl = QVBoxLayout(self._video_controls)
        vctl.setContentsMargins(0, 0, 0, 0)
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setEnabled(False)
        self._frame_label = QLabel("0 / 0")
        self._draw_zones_btn = QPushButton("Draw Zones…")
        self._draw_zones_btn.setEnabled(False)
        self._run_btn = QPushButton("Run tracking")
        self._run_btn.setEnabled(False)
        # Writing the annotated MP4 is software-encoded (no HW H.264 on Pi 5)
        # and costs ~40% of the per-frame budget. Let operators skip it when
        # they only want the tracking CSV / live preview (e.g. gauging FPS).
        self._save_annotated_cb = QCheckBox("Save annotated video")
        self._save_annotated_cb.setChecked(True)
        self._save_annotated_cb.setToolTip(
            "Encode an annotated .mp4 during the run. Uncheck for faster "
            "processing when you only need the tracking data / live preview."
        )

        seek_row = QHBoxLayout()
        seek_row.addWidget(self._seek_slider, 1)
        seek_row.addWidget(self._frame_label)
        vctl.addLayout(seek_row)

        # Checkbox on its own row; the two action buttons share a row below.
        # Keeps every row narrow enough for a ~460px Pi screen.
        vctl.addWidget(self._save_annotated_cb)

        action_row = QHBoxLayout()
        action_row.addWidget(self._draw_zones_btn, 1)
        action_row.addWidget(self._run_btn, 1)
        vctl.addLayout(action_row)
        self._video_controls.setVisible(False)
        layout.addWidget(self._video_controls)

        # --- Run progress + cancel (hidden until a run starts) ---
        progress_row = QHBoxLayout()
        self._run_progress = QProgressBar()
        self._cancel_btn = QPushButton("Cancel")
        progress_row.addWidget(self._run_progress, 1)
        progress_row.addWidget(self._cancel_btn)
        self._progress_container = QWidget()
        self._progress_container.setLayout(progress_row)
        self._progress_container.setVisible(False)
        layout.addWidget(self._progress_container)

        # Control buttons — stacked vertically so they stay full-width and don't
        # widen the panel past a narrow (Pi touch) screen.
        control_layout = QVBoxLayout()

        self._preview_btn = QPushButton("Start Preview")
        self._preview_btn.clicked.connect(self._toggle_preview)
        control_layout.addWidget(self._preview_btn)

        self._settings_btn = QPushButton("Settings")
        self._settings_btn.setProperty("buttonRole", "secondary")
        self._settings_btn.clicked.connect(self.settings_requested.emit)
        control_layout.addWidget(self._settings_btn)

        layout.addLayout(control_layout)

        # Section divider
        cv_divider = QFrame()
        cv_divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(cv_divider)

        cv_section_label = QLabel("Vision")
        cv_section_label.setProperty("textRole", "section")
        layout.addWidget(cv_section_label)

        cv_layout = QVBoxLayout()
        cv_layout.setSpacing(10)

        self._cv_enabled_cb = QCheckBox("Computer Vision")
        self._cv_enabled_cb.toggled.connect(self._on_cv_toggle)
        cv_layout.addWidget(self._cv_enabled_cb)

        self._overlay_cb = QCheckBox("Overlays")
        self._overlay_cb.toggled.connect(self._on_overlay_toggle)
        cv_layout.addWidget(self._overlay_cb)

        self._vision_cone_cb = QCheckBox("Vision Cone")
        self._vision_cone_cb.toggled.connect(self._on_vision_cone_toggle)
        cv_layout.addWidget(self._vision_cone_cb)

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

        # --- Live Behavior inference ---
        behavior_group = QGroupBox("Live Behavior")
        behavior_layout = QVBoxLayout(behavior_group)

        self._behavior_model_label = QLabel("Behavior model: (none)")
        behavior_model_btn = QPushButton("Browse…")
        behavior_model_btn.clicked.connect(self._on_choose_behavior_model)
        behavior_layout.addLayout(_picker_row(self._behavior_model_label, behavior_model_btn))

        self._pose_model_label = QLabel("Pose model: (none)")
        pose_model_btn = QPushButton("Browse…")
        pose_model_btn.clicked.connect(self._on_choose_pose_model)
        pose_folder_btn = QPushButton("Folder…")
        pose_folder_btn.setToolTip(
            "Choose a DeepLabCut or SLEAP exported-model folder "
            "(or just drag one onto this panel)."
        )
        pose_folder_btn.clicked.connect(self._on_choose_pose_model_folder)
        pose_row = _picker_row(self._pose_model_label, pose_model_btn)
        pose_row.addWidget(pose_folder_btn)
        behavior_layout.addLayout(pose_row)

        self._kp_names_edit = QLineEdit()
        self._kp_names_edit.setPlaceholderText("nose, left_ear, right_ear, ... (comma-separated)")
        self._kp_names_edit.textChanged.connect(self._update_live_controls_enabled)
        kp_row = QHBoxLayout()
        kp_row.addWidget(QLabel("Keypoint names:"))
        kp_row.addWidget(self._kp_names_edit, 1)
        behavior_layout.addLayout(kp_row)

        self._kp_hint = QLabel(_MANUAL_NAMES_HINT)
        self._kp_hint.setProperty("textRole", "muted")
        behavior_layout.addWidget(self._kp_hint)

        self._live_behavior_btn = QPushButton("Start")
        self._live_behavior_btn.setEnabled(False)
        self._live_behavior_btn.clicked.connect(self._toggle_live_behavior)
        behavior_layout.addWidget(self._live_behavior_btn)

        # --- rehearsal: drive the live path from a recording ------------------
        rehearsal_line = QFrame()
        rehearsal_line.setFrameShape(QFrame.Shape.HLine)
        rehearsal_line.setProperty("textRole", "muted")
        behavior_layout.addWidget(rehearsal_line)

        rehearsal_row = QHBoxLayout()
        self._rehearse_btn = QPushButton("Rehearse from video\u2026")
        self._rehearse_btn.setEnabled(False)
        self._rehearse_btn.setToolTip(
            "Play a recording through the live path instead of the camera.\n"
            "Classification, node triggers and hardware all run for real."
        )
        self._rehearse_btn.clicked.connect(self._toggle_rehearsal)
        rehearsal_row.addWidget(self._rehearse_btn, 1)

        self._rehearse_speed = QComboBox()
        # Feature values do not depend on playback rate -- compute_features uses
        # unit frame spacing -- so both modes classify identically. They answer
        # different questions, which is why both exist.
        self._rehearse_speed.addItem("Real time", 1.0)
        self._rehearse_speed.addItem("As fast as possible", 0.0)
        self._rehearse_speed.setToolTip(
            "Real time answers 'does inference keep up, and what is the "
            "latency?'.\nAs fast as possible answers 'is this wired up?' "
            "sooner.\nBoth classify the recording identically."
        )
        rehearsal_row.addWidget(self._rehearse_speed)
        behavior_layout.addLayout(rehearsal_row)

        self._rehearse_status = QLabel(
            "Start live behavior first \u2014 a rehearsal with no classifier "
            "running does nothing."
        )
        self._rehearse_status.setProperty("textRole", "muted")
        self._rehearse_status.setWordWrap(True)
        behavior_layout.addWidget(self._rehearse_status)

        layout.addWidget(behavior_group)

        # Set up scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        # Initial camera list
        self._refresh_cameras()

    def _connect_signals(self) -> None:
        """Connect camera callbacks and thread-safe signals."""
        # Register callback with camera manager (called from background thread)
        self._frame_callback_ref = self._on_frame
        self._camera.on_frame(self._frame_callback_ref)

        # Connect thread-safe signals for UI updates (main thread)
        self._frame_received.connect(self._handle_frame_input)
        self._rehearsal_finished.connect(self._on_rehearsal_finished_main)
        self._multi_frame_received.connect(self._handle_multi_frame_input)

        # Connect CV worker signals
        self._cv_worker.results_ready.connect(self._process_cv_results_on_main_thread)

        # Wire _process_frame_requested to CVWorker.process_frame.  Because
        # _cv_worker lives on a different QThread, Qt automatically uses a
        # QueuedConnection, so process_frame() runs on the worker thread — not
        # the main thread.
        self._process_frame_requested.connect(self._cv_worker.process_frame)

        # Ensure CV thread cleanup on widget destruction
        self.destroyed.connect(self._cleanup_cv_thread)

        # Video-source controls
        self._live_radio.toggled.connect(self._on_source_toggled)
        self._browse_btn.clicked.connect(self._on_browse_video)
        self._seek_slider.valueChanged.connect(self._on_seek)
        self._draw_zones_btn.clicked.connect(self._on_draw_zones)
        self._run_btn.clicked.connect(self._on_run_tracking)
        self._cancel_btn.clicked.connect(self._on_cancel_run)

        # Initialise run-thread attributes so _on_cancel_run / _teardown_run_thread
        # can guard safely before any run has started.
        self._run_thread = None
        self._run_worker = None
        # Live processing-rate readout for batch tracking runs (shown in the
        # FPS field, which is otherwise idle when there is no live camera).
        self._run_fps = FpsMeter()
        self._run_frames_done = 0

    # --- rehearsal from a recording ---------------------------------------

    def _toggle_rehearsal(self) -> None:
        if self._rehearsal_pump is not None and self._rehearsal_pump.is_running:
            self._stop_rehearsal()
        else:
            self._start_rehearsal()

    def _rehearsal_video(self) -> str | None:
        """The video a rehearsal would play: the loaded one, or one chosen now.

        Prefers whatever the panel already has open. Prompting unconditionally
        let a rehearsal run against a *different* video than the one on screen
        -- the one the zones were drawn over and the one being scrubbed -- which
        is wrong in a way nothing on screen would show.
        """
        if self._video_source.is_loaded and self._video_source.path:
            return str(self._video_source.path)

        from PyQt6.QtWidgets import QFileDialog

        from glider.vision.pose.batch import VIDEO_FILTER

        path, _ = QFileDialog.getOpenFileName(self, "Rehearse from video", "", VIDEO_FILTER)
        if not path:
            return None
        # Load it properly rather than only handing it to the pump, so the scrub
        # bar, zone drawing and the preview all refer to what is being played.
        if not self._apply_video(Path(path)):
            return None
        return path

    def _update_rehearse_label(self) -> None:
        """Say which video the button would play, so it is not a surprise."""
        if getattr(self, "_rehearse_btn", None) is None:
            return
        if self._rehearsal_pump is not None and self._rehearsal_pump.is_running:
            return
        if self._video_source.is_loaded and self._video_source.path:
            name = Path(self._video_source.path).name
            self._rehearse_btn.setText(f"Rehearse {name}")
        else:
            self._rehearse_btn.setText("Rehearse from video\u2026")

    def _start_rehearsal(self) -> None:
        """Play a recording through the live path, hardware and all."""
        from glider.vision.video_pump import VideoPump

        path = self._rehearsal_video()
        if not path:
            return

        speed = self._rehearse_speed.currentData()
        self._rehearsal_pump = VideoPump(
            path,
            self._on_rehearsal_frame,
            speed=speed,
            on_finished=self._on_rehearsal_finished,
        )
        if not self._rehearsal_pump.start():
            self._rehearsal_pump = None
            self._rehearse_status.setText(f"Could not open {Path(path).name}.")
            return

        self._rehearse_btn.setText("Stop rehearsal")
        self._rehearse_speed.setEnabled(False)
        self._rehearse_status.setText(
            f"Rehearsing {Path(path).name} \u2014 hardware is being driven for real."
        )
        logger.info("Rehearsal started from %s", path)

    def _stop_rehearsal(self) -> None:
        pump = self._rehearsal_pump
        if pump is not None:
            pump.stop()
        self._rehearsal_pump = None
        self._rehearse_speed.setEnabled(True)
        self._update_rehearse_label()
        logger.info("Rehearsal stopped")

    def _on_rehearsal_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Hand a recorded frame to the live path.

        Called from the pump's thread, exactly as ``_on_frame`` is called from
        the camera's capture thread, and doing the same thing: copy, then emit
        so the work happens on the GUI thread. Emitting ``_frame_received``
        rather than calling ``_on_frame`` deliberately skips its
        ``_preview_active`` guard -- a rehearsal does not need a camera open,
        and requiring one would mean it could not run on a machine that has
        none.
        """
        self._frame_received.emit(FrameData(frame=frame.copy(), timestamp=timestamp))

    def _on_rehearsal_finished(self, stats: Any) -> None:
        """Report how it went. Called from the pump's thread."""
        self._rehearsal_finished.emit(stats)

    def _on_rehearsal_finished_main(self, stats: Any) -> None:
        self._rehearse_speed.setEnabled(True)
        self._rehearsal_pump = None
        self._update_rehearse_label()

        lag_ms = stats.max_lag_s * 1000
        summary = f"Rehearsal finished: {stats.frames_delivered} frames"
        if stats.max_lag_s > 0:
            # The number that says whether the rig will hold up live: a run
            # that ended a second behind will miss stimulus timing on an animal
            # by the same margin.
            summary += f", worst lag {lag_ms:.0f} ms \u2014 inference did not keep up"
        elif self._rehearse_speed.currentData():
            summary += ", kept up with real time"
        self._rehearse_status.setText(summary)
        logger.info("%s", summary)

    def _convert_pose_model(self, folder: Path, converter) -> bool:
        """Run a plugin's converter over *folder*, showing progress. True if usable.

        Blocking on purpose. The researcher just chose this model and cannot do
        anything useful until it is ready; a conversion running quietly in the
        background while the panel claimed to be armed would be worse than a
        wait they can see.

        The converter belongs to a plugin, so every failure here is third-party
        code failing. It is reported on screen and the selection is refused --
        never raised into the panel.
        """
        from PyQt6.QtWidgets import QApplication, QMessageBox

        from glider.vision.pose.converters import converter_label, converter_preflight

        label = converter_label(converter)
        question = (
            f"{folder.name} is a {label} model and needs converting before "
            "GLIDER can run it.\n\nThis happens once \u2014 the result is kept "
            "beside the model and reused until you retrain."
        )
        # A converter that has something costly to disclose says so here, and
        # it goes above the question rather than after it: the point is to be
        # read before the person clicks Yes.
        note = converter_preflight(converter, folder)
        if note:
            question = f"{note}\n\n{question}"

        answer = QMessageBox.question(
            self,
            f"Convert {label} model?",
            f"{question}\n\nConvert now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False

        self._pose_model_label.setText(f"Converting {folder.name}\u2026")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            converter.convert(folder)
        except Exception as exc:  # noqa: BLE001 - a converter is third-party code
            logger.exception("%s conversion failed for %s", label, folder)
            QMessageBox.warning(self, f"Convert {label} model", str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()

        logger.info("Converted %s model %s", label, folder)
        return True

    def _handle_frame_input(self, frame_data: FrameData) -> None:
        """Decide whether to process frame with CV or update UI immediately."""
        # Fan out EVERY frame to the live-behavior worker (independent of the CV
        # pipeline). The worker lives on its own thread, so the emit is a
        # QueuedConnection and never blocks this handler.
        #
        # Do NOT decimate here: the worker's StreamingFeatureExtractor computes
        # centered-gradient velocity/acceleration over *consecutive* frames and
        # its SlidingFeatureBuffer rolls over consecutive frames. Skipping frames
        # would ~Nx-inflate live kinematics and stretch the window's wall-clock
        # span, pushing features out of the trained distribution. This mirrors
        # the offline PoseTracker/FeatureEngine, which push every frame into the
        # extractor and gate only the *emission* cadence. Tradeoff: if pose
        # inference can't keep up with camera fps the queued frames back up and
        # latency grows -- but we must not silently drop frames, because that
        # corrupts the stateful features. (A future bounded/gap-aware path is
        # out of scope.)
        if self._behavior_running:
            self._behavior_frame_requested.emit(frame_data)

        if self._cv_enabled_cb.isChecked() and self._cv_processor.is_initialized:
            # Offload to CV worker thread via signal (QueuedConnection ensures
            # process_frame runs on the worker thread, not the main thread)
            self._process_frame_requested.emit(frame_data)
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
            # Offload primary camera to CV worker thread via signal
            self._process_frame_requested.emit(frame_data)
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
        detections: list | None = None,
        tracked: list | None = None,
        motion: Any | None = None,
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

        # Use provided results (from CVWorker) or skip if not provided
        if (
            detections is None
            and self._cv_enabled_cb.isChecked()
            and self._cv_processor.is_initialized
        ):
            # Skip CV processing on main thread — should be handled by CVWorker
            logger.debug("Skipping main-thread CV processing (should use CVWorker)")

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

        # Log tracking data if the tracking logger is active. We always call
        # log_frame (even when `tracked` is None/empty) so the per-session
        # frame counter advances once per camera frame — this is the
        # canonical frame index that joins the device-state CSV, the event
        # log, and the tracking CSV against the MP4. The existing heartbeat
        # path inside log_frame handles the "no detections, no motion" case.
        if self._tracking_logger is not None and self._tracking_logger.is_recording:
            h, w = frame.shape[:2]
            self._tracking_logger.set_frame_size(w, h)
            if self._calibration:
                self._tracking_logger.set_calibration(self._calibration)

            motion_detected = motion.motion_detected if motion else False
            motion_area = motion.motion_area if motion else 0.0

            self._tracking_logger.log_frame(timestamp, tracked or [], motion_detected, motion_area)

            # Per-frame tick for the device-state recorder + event logger.
            # Fires unconditionally on every processed frame, independent
            # of whether CV produced any tracked objects — otherwise an
            # empty CV result would silently drop the device CSV row for
            # that frame. Failures must not crash the CV pipeline.
            self._dispatch_frame_tick(self._tracking_logger.frame_count, timestamp)

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
        # During a batch-tracking run show the processing rate (frames/sec) in
        # the same field the live camera uses — it's otherwise idle here.
        if getattr(self, "_run_thread", None) is not None:
            fps = self._run_fps.update(self._run_frames_done, time.perf_counter())
            if fps is not None:
                self._fps_label.setText(f"{fps:.1f} FPS")
            return

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

    def _on_vision_cone_toggle(self, enabled: bool) -> None:
        """Handle vision cone overlay toggle."""
        self._cv_processor.settings.vision_cone_enabled = enabled

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
        detections: list | None = None,
        tracked: list | None = None,
        motion: Any | None = None,
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

            # Use provided results (from CVWorker) or skip if not provided
            if (
                detections is None
                and self._cv_enabled_cb.isChecked()
                and self._cv_processor.is_initialized
            ):
                # Skip CV processing on main thread — should be handled by CVWorker
                logger.debug(
                    "Skipping main-thread CV processing for multi-cam (should use CVWorker)"
                )

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

            # Log tracking data + dispatch the per-frame tick. See the
            # single-cam path's comment above for the rationale: we call
            # log_frame even when CV returned no tracked objects so the
            # frame counter advances once per processed frame, and we
            # dispatch unconditionally so the device CSV gets one row per
            # frame regardless of CV output.
            if self._tracking_logger is not None and self._tracking_logger.is_recording:
                h, w = frame.shape[:2]
                self._tracking_logger.set_frame_size(w, h)
                if self._calibration:
                    self._tracking_logger.set_calibration(self._calibration)

                motion_detected = motion.motion_detected if motion else False
                motion_area = motion.motion_area if motion else 0.0

                self._tracking_logger.log_frame(
                    timestamp, tracked or [], motion_detected, motion_area
                )
                self._dispatch_frame_tick(self._tracking_logger.frame_count, timestamp)

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

    def set_data_recorder(self, recorder: Any) -> None:
        """
        Set the device-state data recorder. When set, each processed frame
        triggers a per-frame row via ``record_at_frame``, keeping the
        device-state CSV frame-aligned with the tracking CSV and the MP4.
        """
        self._data_recorder = recorder

    def set_event_logger(self, event_logger: Any) -> None:
        """
        Set the device event logger. When set, each processed frame
        pushes the current frame index into the logger so subsequent
        event rows are stamped with that frame.
        """
        self._event_logger = event_logger

    def set_live_signals(self, bus: Any) -> None:
        """
        Set the live signal bus. When set, every classified frame from live
        behavior inference is published to it, so Behavior Input nodes in the
        running flow can trigger on the animal's behavior.
        """
        self._live_signals = bus

    def _dispatch_frame_tick(self, frame_no: int, frame_ts: float) -> None:
        """
        Fire the per-frame side effects on the data_recorder and event_logger.

        Called from the CV processing path immediately after
        ``tracking_logger.log_frame``. Both consumers are best-effort: an
        exception in either must not break the CV pipeline (which would
        also break the live preview and the video recording).

        ``record_at_frame`` is async; we schedule it as a task on the
        running event loop (qasync provides one on the Qt main thread).
        ``set_current_frame`` is synchronous and is called directly.
        """
        recorder = self._data_recorder
        if recorder is not None and getattr(recorder, "is_recording", False):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(recorder.record_at_frame(frame_no, frame_ts))
            except RuntimeError:
                # No running loop (unit-test / headless scenario); skip
                # silently rather than blow up the camera thread.
                logger.debug("frame tick skipped: no running event loop", exc_info=True)
            except Exception:
                logger.exception("record_at_frame failed")

        events = self._event_logger
        if events is not None and getattr(events, "is_recording", False):
            try:
                events.set_current_frame(frame_no, frame_ts)
            except Exception:
                logger.exception("event_logger.set_current_frame failed")

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

    def get_current_frame(self) -> np.ndarray | None:
        """Get the current frame (for snapshots)."""
        return self._last_frame.copy() if self._last_frame is not None else None

    # ------------------------------------------------------------------
    # Video-file source handlers (Part A)
    # ------------------------------------------------------------------

    def _on_source_toggled(self, live_checked: bool) -> None:
        """Switch between live camera and video-file source."""
        self._video_mode = not live_checked
        self._browse_btn.setEnabled(self._video_mode)
        self._video_controls.setVisible(self._video_mode)
        # Disable live-only controls in video mode.
        self._camera_combo.setEnabled(not self._video_mode)
        self._refresh_btn.setEnabled(not self._video_mode)
        self._preview_btn.setEnabled(not self._video_mode)
        if self._video_mode and self._preview_active:
            self._stop_preview()
        if self._video_mode:
            # Video is a single stream; make sure the single-camera preview is
            # the visible page so scrubbed frames are shown (not the hidden
            # multi-camera preview).
            self._preview_stack.setCurrentWidget(self._preview)

    def _on_browse_video(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        from glider.vision.pose.batch import VIDEO_FILTER

        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if path:
            self._apply_video(Path(path))

    def _apply_video(self, path: Path) -> bool:
        """Load *path* as the video source and arm the scrub/run controls."""
        from PyQt6.QtWidgets import QMessageBox

        path = str(path)
        if not self._video_source.load(path):
            QMessageBox.warning(self, "Video", f"Could not open:\n{path}")
            return False
        n = self._video_source.frame_count
        self._seek_slider.setEnabled(True)
        self._seek_slider.setRange(0, n - 1)
        self._seek_slider.setValue(0)
        self._draw_zones_btn.setEnabled(True)
        self._run_btn.setEnabled(True)
        self._on_seek(0)
        self._update_rehearse_label()
        return True

    def _on_seek(self, n: int) -> None:
        if not self._video_source.is_loaded:
            return
        frame = self._video_source.read_frame(n)
        if frame is None:
            return
        self._video_current_frame = n
        self._video_frame = frame
        self._frame_label.setText(f"{n} / {self._video_source.frame_count - 1}")
        self._preview.set_zone_configuration(self._zone_config)
        self._preview.update_frame(frame)

    # ------------------------------------------------------------------
    # Batch-tracking run handlers (Part B)
    # ------------------------------------------------------------------

    def _on_run_tracking(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        from glider.gui.panels.video_tracking_worker import VideoTrackingWorker

        if not self._video_source.is_loaded:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out_dir:
            return

        cfg = self._build_tracking_config(out_dir)
        # cv_processor=None → the runner builds a fresh CVProcessor (clean IDs).
        self._run_thread = QThread()
        self._run_worker = VideoTrackingWorker(cfg)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.progress.connect(self._on_run_progress)
        self._run_worker.preview.connect(self._on_run_preview)
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.failed.connect(self._on_run_failed)

        self._progress_container.setVisible(True)
        self._run_progress.setRange(0, self._video_source.frame_count)
        self._run_progress.setValue(0)
        self._cancel_btn.setEnabled(True)
        self._run_btn.setEnabled(False)
        # Start the live processing-rate readout for this run.
        self._run_frames_done = 0
        self._run_fps.reset(time.perf_counter())
        self._run_thread.start()

    def _build_tracking_config(self, out_dir: str) -> "VideoTrackingConfig":
        """Assemble the batch-tracking config from the current UI state.

        Extracted from _on_run_tracking so the annotated-video toggle (and the
        rest of the config wiring) is unit-testable without spinning up the
        worker thread or a file dialog.
        """
        from pathlib import Path

        from glider.vision.video_tracking_runner import VideoTrackingConfig

        return VideoTrackingConfig(
            source_path=Path(self._video_source.path),
            output_dir=Path(out_dir),
            zone_config=self._zone_config,
            cv_settings=replace(self._cv_processor.settings),
            write_annotated=self._save_annotated_cb.isChecked(),
        )

    def _on_cancel_run(self) -> None:
        """Cancel the in-flight tracking run (stable slot, connected once)."""
        worker = getattr(self, "_run_worker", None)
        if worker is not None:
            worker.cancel()

    def _on_run_progress(self, done: int, total: int) -> None:
        self._run_progress.setValue(done)
        # Latest cumulative frame count; the FPS timer samples this to show the
        # live processing rate (see _update_fps_display).
        self._run_frames_done = done

    def _on_run_preview(self, frame: np.ndarray, frame_index: int) -> None:
        """Show a batch-tracking frame (with overlays) live as it's processed.

        Emitted from the worker thread (throttled to ~10 fps by the runner), so
        this is a QueuedConnection onto the main thread — safe to touch the
        preview widget here.
        """
        self._preview.update_frame(frame)

    def _on_run_finished(self, output_dir: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        self._teardown_run_thread()
        self._progress_container.setVisible(False)
        self._run_btn.setEnabled(True)
        box = QMessageBox(self)
        box.setWindowTitle("Tracking complete")
        box.setText(f"Wrote results to:\n{output_dir}")
        open_btn = box.addButton("Open in Analysis panel", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is open_btn:
            self.analysis_requested.emit(output_dir)

    def _on_run_failed(self, message: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        self._teardown_run_thread()
        self._progress_container.setVisible(False)
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Tracking failed", message)

    def _on_draw_zones(self) -> None:
        """Ask the host window to open the zone editor on the current video frame."""
        if self._video_frame is not None:
            self.draw_zones_requested.emit(self._video_frame)

    def refresh_scrub_frame(self) -> None:
        """Re-render the current scrubbed frame (e.g. after zones changed)."""
        if self._video_mode and self._video_source.is_loaded:
            self._on_seek(self._video_current_frame)

    def _teardown_run_thread(self) -> None:
        if getattr(self, "_run_thread", None) is not None:
            if self._run_worker is not None:
                self._run_worker.cancel()  # stop the loop between frames
            self._run_thread.quit()
            self._run_thread.wait(5000)
            self._run_thread = None
            self._run_worker = None
            # Clear the processing-rate readout now the run is over.
            self._fps_label.setText("-- FPS")

    def _cleanup_cv_thread(self) -> None:
        """Ensure CV thread is stopped on destruction."""
        if self._cv_thread.isRunning():
            self._cv_thread.quit()
            self._cv_thread.wait(2000)

    # ------------------------------------------------------------------
    # Live-behavior inference (Task 5)
    # ------------------------------------------------------------------

    def _on_choose_behavior_model(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Choose behavior model", "", "Model files (*.pkl);;All files (*)"
        )
        if path:
            self._apply_behavior_model(Path(path))

    def _on_choose_pose_model(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose pose model",
            "",
            "Pose models (*.pt *.onnx);;All files (*)",
        )
        if not path:
            return
        self._apply_pose_model(Path(path))

    def _on_choose_pose_model_folder(self) -> None:
        """Pick a DeepLabCut / SLEAP exported-model *folder*."""
        from PyQt6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Choose exported model folder")
        if path:
            self._apply_pose_model(Path(path))

    # ------------------------------------------------------------------
    # Applying a chosen/dropped model (shared by the pickers and drag-drop)
    # ------------------------------------------------------------------

    def _apply_pose_model(self, path: Path) -> bool:
        """Adopt *path* as the pose model, filling in names when it knows them.

        DeepLabCut and SLEAP folders carry their body-part names in training
        order, so those are filled in and locked — that ordering is the one
        thing an operator cannot reliably retype, and getting it wrong yields
        all-NaN angle features and uniformly blank predictions rather than an
        error. YOLO checkpoints carry no names, so the field stays editable.
        """
        from PyQt6.QtWidgets import QMessageBox

        from glider.vision.cv_processor import DetectionBackend
        from glider.vision.pose.converters import needs_conversion
        from glider.vision.pose.spec import PoseModelError, identify_pose_model

        # A folder a vendor wrote holds their checkpoint, not the ONNX GLIDER
        # runs. Rather than refusing it with instructions to go and convert it
        # by hand -- which is what used to happen, and which nobody did --
        # hand it to whichever plugin claims it. Path-only check: no vendor
        # framework is imported to answer it.
        converter = needs_conversion(path)
        if converter is not None and not self._convert_pose_model(Path(path), converter):
            return False

        try:
            # Pure path/JSON/YAML work — safe on the UI thread, unlike reading
            # a .pt, which would import torch and stall for seconds.
            spec = identify_pose_model(path)
        except PoseModelError as exc:
            QMessageBox.warning(self, "Pose model", str(exc))
            return False

        self._pose_model_path = Path(path)
        kind = _KIND_LABELS.get(spec.kind, spec.kind)

        if spec.keypoint_names:
            self._kp_names_edit.setText(", ".join(spec.keypoint_names))
            self._kp_names_edit.setReadOnly(True)
            self._kp_hint.setText(
                f"Names read from the {kind} model — {spec.n_keypoints} keypoints."
            )
            self._pose_model_label.setText(
                f"Pose model: {Path(path).name} ({kind}, {spec.n_keypoints} kp)"
            )
        else:
            self._kp_names_edit.setReadOnly(False)
            self._kp_hint.setText(_MANUAL_NAMES_HINT)
            self._pose_model_label.setText(f"Pose model: {Path(path).name} ({kind})")

        # Only SLEAP/DeepLabCut point tracking at this model: load_pose_backend
        # needs caller-supplied keypoint names to run a YOLO checkpoint, which
        # this panel does not prompt for, so a .pt selection leaves the CV
        # backend (background subtraction, YOLO, whatever Settings has) alone.
        if spec.kind in ("sleap", "dlc"):
            settings = self._cv_processor.settings.copy()
            settings.backend = DetectionBackend.POSE_MODEL
            settings.model_path = str(spec.model_path)
            settings.keypoint_names = list(spec.keypoint_names)
            self._cv_processor.update_settings(settings)

        self._update_live_controls_enabled()
        return True

    def _apply_behavior_model(self, path: Path) -> bool:
        self._behavior_pkl = Path(path)
        self._behavior_model_label.setText(f"Behavior model: {path}")
        self._update_live_controls_enabled()
        return True

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------

    # Qt camelCase overrides (N802 is suppressed project-wide).
    def dragEnterEvent(self, event):
        from glider.gui.panels.model_drop import DropKind, classify_drop, paths_from_mime

        paths = paths_from_mime(event.mimeData())
        if any(classify_drop(p) is not DropKind.UNKNOWN for p in paths):
            event.acceptProposedAction()
            self._set_drop_active(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):  # noqa: ARG002
        self._set_drop_active(False)

    def dropEvent(self, event):
        from glider.gui.panels.model_drop import DropKind, paths_from_mime, route_drops

        self._set_drop_active(False)
        routed = route_drops(paths_from_mime(event.mimeData()))
        if not routed:
            event.ignore()
            return
        event.acceptProposedAction()

        if DropKind.POSE_MODEL in routed or DropKind.BEHAVIOR_MODEL in routed:
            if self._behavior_running:
                # Swapping models mid-run would leave the worker holding one set
                # and the panel showing another.
                self._show_status("Stop live behavior before changing models.")
                return
        if DropKind.POSE_MODEL in routed:
            self._apply_pose_model(routed[DropKind.POSE_MODEL])
        if DropKind.BEHAVIOR_MODEL in routed:
            self._apply_behavior_model(routed[DropKind.BEHAVIOR_MODEL])
        if DropKind.VIDEO in routed:
            self._apply_video(routed[DropKind.VIDEO])

    def _set_drop_active(self, active: bool) -> None:
        """Toggle the accent border via a dynamic property + repolish.

        Matches the panel's existing property-driven styling (see the "textRole"
        usages) rather than layering an overlay widget over the preview.
        """
        self.setProperty("dropActive", "true" if active else "false")
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)

    def _show_status(self, message: str) -> None:
        """Surface a short message; falls back to the log when there's no bar."""
        window = self.window()
        bar = getattr(window, "statusBar", None)
        if callable(bar):
            try:
                bar().showMessage(message, 5000)
                return
            except Exception:
                pass
        logger.info("%s", message)

    def _parse_keypoint_names(self) -> list[str]:
        """Split the keypoint-names field into a trimmed, non-empty name list."""
        return [name.strip() for name in self._kp_names_edit.text().split(",") if name.strip()]

    def _update_live_controls_enabled(self) -> None:
        """Enable the Start toggle only once both paths + names are provided.

        While a run is active the button stays enabled (it becomes "Stop"), so
        we only gate it in the stopped state.
        """
        if self._behavior_running:
            return
        ready = (
            self._behavior_pkl is not None
            and self._pose_model_path is not None
            and bool(self._parse_keypoint_names())
        )
        self._live_behavior_btn.setEnabled(ready)

    def _toggle_live_behavior(self) -> None:
        """Start the live-behavior worker, or stop it if already running."""
        if self._behavior_running:
            self.stop_live_behavior()
        else:
            self._start_live_behavior()

    def _start_live_behavior(self) -> None:
        """Spin up the BehaviorInferenceWorker on its own thread and load models."""
        names = self._parse_keypoint_names()
        if self._behavior_pkl is None or self._pose_model_path is None or not names:
            return

        from glider.gui.panels.live_behavior import BehaviorInferenceWorker

        self._behavior_thread = QThread()
        self._behavior_worker = BehaviorInferenceWorker()
        self._behavior_worker.moveToThread(self._behavior_thread)

        # Result/status signals come back to the main thread (QueuedConnection).
        self._behavior_worker.ready.connect(self._on_behavior_ready)
        self._behavior_worker.load_failed.connect(self._on_behavior_load_failed)
        self._behavior_worker.result_ready.connect(self._on_behavior_result)
        # Cross into the worker thread for model loading + per-frame inference.
        self._behavior_init_requested.connect(self._behavior_worker.initialize)
        self._behavior_frame_requested.connect(self._behavior_worker.process_frame)

        # Disabled while models load; re-enabled as "Stop" once ready, or reset
        # to an enabled "Start" if loading fails.
        self._live_behavior_btn.setEnabled(False)
        self._behavior_thread.start()
        self._behavior_init_requested.emit(
            str(self._behavior_pkl), str(self._pose_model_path), names
        )

    def _on_behavior_ready(self) -> None:
        """Models loaded successfully — go live."""
        worker = self._behavior_worker
        if worker is None:
            return
        self._preview.set_behavior_vocab(worker.classes)
        self._rehearse_btn.setEnabled(True)
        self._update_rehearse_label()
        self._rehearse_status.setText("Ready to rehearse from a recording.")
        # Also hand the vocabulary to the flow side, so a Behavior Input node's
        # properties can offer the behaviors this model actually emits instead
        # of a free-text box where a typo means "never fires".
        if self._live_signals is not None:
            self._live_signals.set_behaviors(worker.classes)
        self._behavior_running = True
        self._live_behavior_btn.setText("Stop")
        self._live_behavior_btn.setEnabled(True)

    def _on_behavior_load_failed(self, message: str) -> None:
        """Model loading failed — surface it and return to the stopped state."""
        from PyQt6.QtWidgets import QMessageBox

        self._teardown_behavior_thread()
        self._behavior_running = False
        self._live_behavior_btn.setText("Start")
        self._update_live_controls_enabled()
        QMessageBox.warning(self, "Live Behavior", message)

    def _on_behavior_result(self, label: str, keypoints: Any) -> None:
        """Push a classified frame's label + pose overlay onto the preview.

        Also publishes the classification so nodes in the running flow can act
        on it. Until this existed the label was drawn and then discarded, which
        is why behavior could be displayed but never trigger anything.
        """
        self._preview.set_behavior_label(label)
        self._preview.set_pose_overlay(keypoints)
        self._publish_behavior(label, keypoints)

    def _publish_behavior(self, label: str, keypoints: Any) -> None:
        """Hand one classification to the live signal bus, if one is attached."""
        bus = self._live_signals
        if bus is None:
            return
        from glider.core.live_signals import BehaviorEvent

        try:
            bus.publish_behavior(
                BehaviorEvent(
                    behavior=label,
                    frame_index=self._frame_count,
                    timestamp=time.time(),
                    keypoints=keypoints,
                )
            )
        except Exception:
            # A failing subscriber must not take the preview down with it; the
            # bus already logs per-subscriber failures.
            logger.exception("Failed to publish behavior event")

    def stop_live_behavior(self) -> None:
        """Stop live inference: join the worker thread and clear overlays.

        Safe to call when nothing is running (idempotent). Used by both the
        Stop toggle and panel teardown so no worker thread outlives the panel.
        """
        self._behavior_running = False
        self._teardown_behavior_thread()
        self._preview.set_pose_overlay(None)
        self._preview.set_behavior_label("")
        self._live_behavior_btn.setText("Start")
        self._update_live_controls_enabled()

    def _teardown_behavior_thread(self) -> None:
        """Disconnect + quit()/wait() the behavior worker thread (if any)."""
        worker = self._behavior_worker
        if worker is not None:
            for signal, slot in (
                (worker.ready, self._on_behavior_ready),
                (worker.load_failed, self._on_behavior_load_failed),
                (worker.result_ready, self._on_behavior_result),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            for signal, slot in (
                (self._behavior_init_requested, worker.initialize),
                (self._behavior_frame_requested, worker.process_frame),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass

        thread = self._behavior_thread
        if thread is not None:
            # Schedule the worker's deletion via the thread's finished signal
            # *before* quit(): calling worker.deleteLater() after wait() posts
            # the event to an already-dead event loop, so the worker may never
            # be reclaimed. Connecting finished -> deleteLater lets Qt honour the
            # queued deletion as the worker's loop unwinds.
            if worker is not None:
                thread.finished.connect(worker.deleteLater)
            thread.quit()
            thread.wait(5000)
            thread.deleteLater()
            self._behavior_thread = None
        elif worker is not None:
            # No thread ever started for this worker — delete it directly.
            worker.deleteLater()
        self._behavior_worker = None

    def closeEvent(self, event):
        """Clean up on close."""
        # Tear down any in-flight tracking run and release the scrub video source.
        self._teardown_run_thread()
        self.stop_live_behavior()
        if self._rehearsal_pump is not None:
            self._rehearsal_pump.stop()
            self._rehearsal_pump = None
        self._video_source.release()

        if self._preview_active:
            if self._multi_camera_mode:
                self._stop_multi_cameras()
            else:
                self._stop_preview()

        # Deregister frame callback from camera manager to prevent dangling references
        if hasattr(self, "_frame_callback_ref") and self._frame_callback_ref is not None:
            self._camera.remove_frame_callback(self._frame_callback_ref)
            self._frame_callback_ref = None

        # Stop CV thread
        if self._cv_thread.isRunning():
            self._cv_thread.quit()
            self._cv_thread.wait(2000)

        self._fps_timer.stop()
        super().closeEvent(event)
