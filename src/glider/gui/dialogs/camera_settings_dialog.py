"""
Camera Settings Dialog - Configure camera and CV parameters.

Provides tabbed interface for camera settings, computer vision
configuration, and tracking parameters.

Automatically adapts layout for touchscreen (Pi) or desktop environments.
"""

import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScroller,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.gui.view_manager import ViewManager
    from glider.vision.camera_manager import CameraManager

from glider.vision.camera_manager import CameraSettings
from glider.vision.cv_processor import CVSettings, DetectionBackend, parse_keypoint_names

logger = logging.getLogger(__name__)


class CameraSettingsDialog(QDialog):
    """Dialog for configuring camera and CV settings."""

    # Signals for opening external dialogs
    calibration_requested = pyqtSignal()
    zones_requested = pyqtSignal()

    def __init__(
        self,
        camera_settings: CameraSettings | None = None,
        cv_settings: CVSettings | None = None,
        parent=None,
        view_manager: Optional["ViewManager"] = None,
        camera_manager: Optional["CameraManager"] = None,
    ):
        super().__init__(parent)
        self._camera_settings = camera_settings or CameraSettings()
        self._cv_settings = cv_settings or CVSettings()
        self._view_manager = view_manager
        self._camera_manager = camera_manager  # For live LED/EWL control
        self._is_touch_mode = view_manager.is_runner_mode if view_manager else False
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        self.setWindowTitle("Camera Settings")

        # Adaptive sizing based on mode
        if self._is_touch_mode:
            # Pi touchscreen: fill most of the screen
            self.setMinimumSize(460, 700)
            self.setMaximumSize(480, 780)
        else:
            self.setMinimumSize(500, 450)

        layout = QVBoxLayout(self)

        # Adjust layout spacing for touch mode
        if self._is_touch_mode:
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)

        # Tab widget with larger tabs for touch
        self._tabs = QTabWidget()
        if self._is_touch_mode:
            self._tabs.setProperty("touchMode", True)
        layout.addWidget(self._tabs)

        # Camera tab (wrapped in scroll area)
        self._camera_tab = self._create_scrollable_tab(self._create_camera_tab_content())
        self._tabs.addTab(self._camera_tab, "Camera")

        # Computer Vision tab (wrapped in scroll area)
        self._cv_tab = self._create_scrollable_tab(self._create_cv_tab_content())
        self._tabs.addTab(self._cv_tab, "CV")

        # Tools tab (wrapped in scroll area)
        self._tools_tab = self._create_scrollable_tab(self._create_tools_tab_content())
        self._tabs.addTab(self._tools_tab, "Tools")

        # Audio tab (wrapped in scroll area)
        self._audio_tab = self._create_scrollable_tab(self._create_audio_tab_content())
        self._tabs.addTab(self._audio_tab, "Audio")

        # Dialog buttons - larger for touch
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )

        if self._is_touch_mode:
            for button in button_box.buttons():
                button.setMinimumHeight(44)
                button.setProperty("touchMode", True)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        apply_btn = button_box.button(QDialogButtonBox.StandardButton.Apply)
        apply_btn.clicked.connect(self._apply_settings)
        layout.addWidget(button_box)

    def _create_scrollable_tab(self, content_widget: QWidget) -> QScrollArea:
        """Wrap a widget in a scroll area for touch-friendly scrolling."""
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        if self._is_touch_mode:
            # Enable kinetic scrolling for touch
            scroll.setProperty("touchMode", True)
            # Enable kinetic scrolling
            QScroller.grabGesture(
                scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
            )

        return scroll

    def _apply_touch_group_property(self, group: QGroupBox) -> None:
        """Apply touch mode property to a group box."""
        if self._is_touch_mode:
            group.setProperty("touchMode", True)

    def _create_camera_tab_content(self) -> QWidget:
        """Create the camera settings tab content."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Adjust spacing for touch mode
        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(16)

        # Touch mode applied via QSS property

        # Resolution group
        res_group = QGroupBox("Resolution")
        self._apply_touch_group_property(res_group)
        res_layout = QFormLayout(res_group)
        if self._is_touch_mode:
            res_layout.setSpacing(12)
            res_layout.setContentsMargins(12, 20, 12, 12)

        self._resolution_combo = QComboBox()
        self._resolution_combo.addItem("320x240", (320, 240))
        self._resolution_combo.addItem("608x608 (Miniscope)", (608, 608))
        self._resolution_combo.addItem("640x480", (640, 480))
        self._resolution_combo.addItem("720x540", (720, 540))
        self._resolution_combo.addItem("800x600", (800, 600))
        self._resolution_combo.addItem("1280x720 (HD)", (1280, 720))
        self._resolution_combo.addItem("1920x1080 (Full HD)", (1920, 1080))
        res_layout.addRow("Resolution:", self._resolution_combo)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 120)
        self._fps_spin.setValue(30)
        self._fps_spin.setSuffix(" fps")
        res_layout.addRow("Frame Rate:", self._fps_spin)

        layout.addWidget(res_group)

        # Image settings group
        image_group = QGroupBox("Image Settings")
        self._apply_touch_group_property(image_group)
        image_layout = QFormLayout(image_group)
        if self._is_touch_mode:
            image_layout.setSpacing(12)
            image_layout.setContentsMargins(12, 20, 12, 12)

        # Exposure
        exposure_layout = QHBoxLayout()
        exposure_layout.setSpacing(8 if not self._is_touch_mode else 12)
        self._auto_exposure_cb = QCheckBox("Auto")
        self._auto_exposure_cb.toggled.connect(self._on_auto_exposure_toggle)
        exposure_layout.addWidget(self._auto_exposure_cb)

        self._exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self._exposure_slider.setRange(-10, 0)
        self._exposure_slider.setValue(-5)
        if self._is_touch_mode:
            self._exposure_slider.setMinimumHeight(40)
        exposure_layout.addWidget(self._exposure_slider)

        self._exposure_label = QLabel("-5")
        self._exposure_label.setMinimumWidth(40 if self._is_touch_mode else 30)
        self._exposure_slider.valueChanged.connect(lambda v: self._exposure_label.setText(str(v)))
        exposure_layout.addWidget(self._exposure_label)

        image_layout.addRow("Exposure:", exposure_layout)

        # Brightness
        brightness_layout = QHBoxLayout()
        brightness_layout.setSpacing(8 if not self._is_touch_mode else 12)
        self._brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self._brightness_slider.setRange(0, 255)
        self._brightness_slider.setValue(128)
        if self._is_touch_mode:
            self._brightness_slider.setMinimumHeight(40)
        brightness_layout.addWidget(self._brightness_slider)

        self._brightness_label = QLabel("128")
        self._brightness_label.setMinimumWidth(40 if self._is_touch_mode else 30)
        self._brightness_slider.valueChanged.connect(
            lambda v: self._brightness_label.setText(str(v))
        )
        brightness_layout.addWidget(self._brightness_label)

        image_layout.addRow("Brightness:", brightness_layout)

        # Contrast
        contrast_layout = QHBoxLayout()
        contrast_layout.setSpacing(8 if not self._is_touch_mode else 12)
        self._contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self._contrast_slider.setRange(0, 255)
        self._contrast_slider.setValue(128)
        if self._is_touch_mode:
            self._contrast_slider.setMinimumHeight(40)
        contrast_layout.addWidget(self._contrast_slider)

        self._contrast_label = QLabel("128")
        self._contrast_label.setMinimumWidth(40 if self._is_touch_mode else 30)
        self._contrast_slider.valueChanged.connect(lambda v: self._contrast_label.setText(str(v)))
        contrast_layout.addWidget(self._contrast_label)

        image_layout.addRow("Contrast:", contrast_layout)

        layout.addWidget(image_group)

        # Connection settings group (for USB cameras like miniscopes)
        conn_group = QGroupBox("Connection")
        self._apply_touch_group_property(conn_group)
        conn_layout = QFormLayout(conn_group)
        if self._is_touch_mode:
            conn_layout.setSpacing(12)
            conn_layout.setContentsMargins(12, 20, 12, 12)

        self._backend_camera_combo = QComboBox()
        self._backend_camera_combo.addItem("Auto-detect", None)
        self._backend_camera_combo.addItem("DirectShow (Windows)", "dshow")
        self._backend_camera_combo.addItem("MediaFoundation (Windows)", "msmf")
        self._backend_camera_combo.addItem("V4L2 (Linux)", "v4l2")
        self._backend_camera_combo.addItem("Picamera2 (Pi)", "picamera2")
        self._backend_camera_combo.setToolTip(
            "Force a specific camera backend.\n"
            "Windows: DirectShow or MediaFoundation.\n"
            "Linux: V4L2 for USB cameras, Picamera2 for Pi camera module.\n"
            "For cameras that don't work, use OBS Virtual Camera."
        )
        conn_layout.addRow("Backend:", self._backend_camera_combo)

        self._timeout_spin = QDoubleSpinBox()
        self._timeout_spin.setRange(1.0, 30.0)
        self._timeout_spin.setValue(5.0)
        self._timeout_spin.setSingleStep(1.0)
        self._timeout_spin.setSuffix(" s")
        self._timeout_spin.setToolTip(
            "Connection timeout for camera initialization.\n"
            "Increase this for slow USB cameras like miniscopes (try 10-15s)."
        )
        conn_layout.addRow("Timeout:", self._timeout_spin)

        self._pixel_format_combo = QComboBox()
        self._pixel_format_combo.addItem("Auto-detect", None)
        self._pixel_format_combo.addItem("MJPG (Most cameras)", "MJPG")
        self._pixel_format_combo.addItem("YUY2", "YUY2")
        self._pixel_format_combo.addItem("YUYV (Miniscope)", "YUYV")
        self._pixel_format_combo.addItem("NV12", "NV12")
        self._pixel_format_combo.addItem("I420", "I420")
        self._pixel_format_combo.addItem("GREY (Grayscale)", "GREY")
        self._pixel_format_combo.addItem("Y800 (Grayscale)", "Y800")
        self._pixel_format_combo.setToolTip(
            "Pixel format for video capture.\n"
            "Auto-detect: Tries multiple formats automatically.\n"
            "MJPG: Motion JPEG, works with most webcams.\n"
            "YUY2/YUYV: Raw YUV, good for miniscopes and USB cameras.\n"
            "GREY/Y800: Grayscale, for scientific/industrial cameras.\n"
            "Try different formats if your camera doesn't work."
        )
        conn_layout.addRow("Format:", self._pixel_format_combo)

        self._miniscope_mode_cb = QCheckBox("Miniscope Mode")
        self._miniscope_mode_cb.setToolTip(
            "Enable special initialization for UCLA Miniscope cameras.\n"
            "This runs v4l2-ctl commands to wake up the LED and\n"
            "includes a watchdog to re-trigger if image goes dark."
        )
        self._miniscope_mode_cb.toggled.connect(self._on_miniscope_mode_toggle)
        conn_layout.addRow(self._miniscope_mode_cb)

        layout.addWidget(conn_group)

        # Miniscope-specific controls group (visible when miniscope mode enabled)
        self._miniscope_group = QGroupBox("Miniscope Controls")
        self._apply_touch_group_property(self._miniscope_group)
        miniscope_layout = QFormLayout(self._miniscope_group)
        if self._is_touch_mode:
            miniscope_layout.setSpacing(12)
            miniscope_layout.setContentsMargins(12, 20, 12, 12)

        # LED Power (most important control - at top with slider)
        led_layout = QHBoxLayout()
        led_layout.setSpacing(8 if not self._is_touch_mode else 12)
        self._led_power_slider = QSlider(Qt.Orientation.Horizontal)
        self._led_power_slider.setRange(0, 100)
        self._led_power_slider.setValue(0)
        if self._is_touch_mode:
            self._led_power_slider.setMinimumHeight(40)
        led_layout.addWidget(self._led_power_slider)
        self._led_power_label = QLabel("0%")
        self._led_power_label.setMinimumWidth(45 if self._is_touch_mode else 35)
        self._led_power_slider.valueChanged.connect(self._on_led_power_changed)
        led_layout.addWidget(self._led_power_label)
        miniscope_layout.addRow("LED Power:", led_layout)

        # EWL Focus (second most important - with slider)
        ewl_layout = QHBoxLayout()
        ewl_layout.setSpacing(8 if not self._is_touch_mode else 12)
        self._ewl_focus_slider = QSlider(Qt.Orientation.Horizontal)
        self._ewl_focus_slider.setRange(0, 255)
        self._ewl_focus_slider.setValue(128)
        if self._is_touch_mode:
            self._ewl_focus_slider.setMinimumHeight(40)
        ewl_layout.addWidget(self._ewl_focus_slider)
        self._ewl_focus_label = QLabel("128")
        self._ewl_focus_label.setMinimumWidth(45 if self._is_touch_mode else 35)
        self._ewl_focus_slider.valueChanged.connect(self._on_ewl_focus_changed)
        ewl_layout.addWidget(self._ewl_focus_label)
        miniscope_layout.addRow("EWL Focus:", ewl_layout)

        # Exposure time
        self._exposure_time_spin = QSpinBox()
        self._exposure_time_spin.setRange(0, 65535)
        self._exposure_time_spin.setValue(100)
        self._exposure_time_spin.setToolTip("Exposure time (0-65535)")
        miniscope_layout.addRow("Exposure:", self._exposure_time_spin)

        # Gain
        self._gain_spin = QSpinBox()
        self._gain_spin.setRange(0, 65535)
        self._gain_spin.setValue(0)
        self._gain_spin.setToolTip("Sensor gain (0-65535)")
        miniscope_layout.addRow("Gain:", self._gain_spin)

        # Gamma
        self._gamma_spin = QSpinBox()
        self._gamma_spin.setRange(0, 65535)
        self._gamma_spin.setValue(0)
        self._gamma_spin.setToolTip("Gamma correction (0-65535)")
        miniscope_layout.addRow("Gamma:", self._gamma_spin)

        # Hue
        self._hue_spin = QSpinBox()
        self._hue_spin.setRange(-32768, 32767)
        self._hue_spin.setValue(0)
        self._hue_spin.setToolTip("Hue adjustment (-32768 to 32767)")
        miniscope_layout.addRow("Hue:", self._hue_spin)

        # Sharpness
        self._sharpness_spin = QSpinBox()
        self._sharpness_spin.setRange(0, 65535)
        self._sharpness_spin.setValue(0)
        self._sharpness_spin.setToolTip("Sharpness (0-65535)")
        miniscope_layout.addRow("Sharpness:", self._sharpness_spin)

        # Focus (standard V4L2 focus, different from EWL)
        self._focus_spin = QSpinBox()
        self._focus_spin.setRange(0, 65535)
        self._focus_spin.setValue(0)
        self._focus_spin.setToolTip("V4L2 focus position (0-65535)")
        miniscope_layout.addRow("V4L2 Focus:", self._focus_spin)

        # Zoom
        self._zoom_spin = QSpinBox()
        self._zoom_spin.setRange(0, 65535)
        self._zoom_spin.setValue(0)
        self._zoom_spin.setToolTip("Zoom level (0-65535)")
        miniscope_layout.addRow("Zoom:", self._zoom_spin)

        # Iris
        self._iris_spin = QSpinBox()
        self._iris_spin.setRange(0, 65535)
        self._iris_spin.setValue(0)
        self._iris_spin.setToolTip("Iris aperture (0-65535)")
        miniscope_layout.addRow("Iris:", self._iris_spin)

        # Hide by default until miniscope mode is enabled
        self._miniscope_group.setVisible(False)

        layout.addWidget(self._miniscope_group)
        layout.addStretch()

        return widget

    def _create_cv_tab_content(self) -> QWidget:
        """Create the computer vision settings tab content."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Adjust spacing for touch mode
        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(16)

        # Touch mode applied via QSS property

        # Enable CV - prominent checkbox at top
        self._cv_enabled_cb = QCheckBox("Enable CV Processing")
        if self._is_touch_mode:
            self._cv_enabled_cb.setProperty("touchMode", True)
        self._cv_enabled_cb.toggled.connect(self._on_cv_enabled_toggle)
        layout.addWidget(self._cv_enabled_cb)

        # Detection group
        detection_group = QGroupBox("Detection")
        self._apply_touch_group_property(detection_group)
        detection_layout = QFormLayout(detection_group)
        if self._is_touch_mode:
            detection_layout.setSpacing(12)
            detection_layout.setContentsMargins(12, 20, 12, 12)

        self._backend_combo = QComboBox()
        self._backend_combo.addItem("Background Sub", DetectionBackend.BACKGROUND_SUBTRACTION)
        self._backend_combo.addItem("Motion Only", DetectionBackend.MOTION_ONLY)
        self._backend_combo.addItem("YOLO v8", DetectionBackend.YOLO_V8)
        self._backend_combo.addItem("YOLO+ByteTrack", DetectionBackend.YOLO_BYTETRACK)
        self._backend_combo.addItem("Pose Model", DetectionBackend.POSE_MODEL)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        detection_layout.addRow("Backend:", self._backend_combo)

        # Model path (visible for any model-backed backend: YOLO or pose)
        model_layout = QHBoxLayout()
        model_layout.setSpacing(8)
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText("YOLO model (.pt or NCNN .param)")
        model_layout.addWidget(self._model_path_edit)

        self._browse_model_btn = QPushButton("...")
        if self._is_touch_mode:
            self._browse_model_btn.setMinimumSize(50, 40)
        self._browse_model_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(self._browse_model_btn)

        self._model_path_label = QLabel("Model:")
        detection_layout.addRow(self._model_path_label, model_layout)

        # Bodypart names for pose models. Pose weights don't carry keypoint
        # names (Ultralytics `names` is the class map), so they're supplied
        # here; they label the rows of the keypoints CSV. Left blank, the
        # logger falls back to positional indices — data is still recorded.
        self._keypoint_names_edit = QLineEdit()
        self._keypoint_names_edit.setPlaceholderText(
            "nose, left_ear, right_ear, ... (comma-separated, pose models only)"
        )
        self._keypoint_names_edit.setToolTip(
            "Bodypart names in the order your pose model outputs them.\n"
            "Used to label the keypoints CSV. Leave blank to use indices (0, 1, 2, ...).\n"
            "Ignored by detection-only models."
        )
        self._keypoint_names_label = QLabel("Keypoints:")
        detection_layout.addRow(self._keypoint_names_label, self._keypoint_names_edit)

        # Confidence threshold
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.1, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setValue(0.5)
        detection_layout.addRow("Confidence:", self._confidence_spin)

        # Min contour area
        self._min_area_spin = QSpinBox()
        self._min_area_spin.setRange(100, 50000)
        self._min_area_spin.setValue(500)
        self._min_area_spin.setSuffix(" px")
        detection_layout.addRow("Min Area:", self._min_area_spin)

        # Frame skip for performance (process every N frames)
        self._frame_skip_spin = QSpinBox()
        self._frame_skip_spin.setRange(1, 10)
        self._frame_skip_spin.setValue(1)
        self._frame_skip_spin.setToolTip(
            "Process CV every N frames. Higher values improve FPS but reduce tracking accuracy.\n"
            "1 = process every frame, 3 = process every 3rd frame (3x faster)"
        )
        detection_layout.addRow("Skip Frames:", self._frame_skip_spin)

        layout.addWidget(detection_group)

        # Overlay group
        overlay_group = QGroupBox("Display")
        self._apply_touch_group_property(overlay_group)
        overlay_layout = QFormLayout(overlay_group)
        if self._is_touch_mode:
            overlay_layout.setSpacing(12)
            overlay_layout.setContentsMargins(12, 20, 12, 12)

        self._draw_overlays_cb = QCheckBox("Bounding Boxes")
        self._draw_overlays_cb.setChecked(True)
        overlay_layout.addRow(self._draw_overlays_cb)

        self._show_keypoints_cb = QCheckBox("Keypoints")
        self._show_keypoints_cb.setChecked(True)
        self._show_keypoints_cb.setToolTip("Draw pose keypoints as dots (requires a pose model)")
        overlay_layout.addRow(self._show_keypoints_cb)

        self._draw_tracks_cb = QCheckBox("Motion Tracks")
        self._draw_tracks_cb.setChecked(True)
        overlay_layout.addRow(self._draw_tracks_cb)

        self._draw_contours_cb = QCheckBox("Contours")
        self._draw_contours_cb.setChecked(False)
        overlay_layout.addRow(self._draw_contours_cb)

        layout.addWidget(overlay_group)
        layout.addStretch()

        return widget

    def _create_tools_tab_content(self) -> QWidget:
        """Create the tools tab content with calibration and zones buttons."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Adjust spacing for touch mode
        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(16)

        # Touch mode applied via QSS property

        # Calibration group
        calibration_group = QGroupBox("Camera Calibration")
        self._apply_touch_group_property(calibration_group)
        calibration_layout = QVBoxLayout(calibration_group)
        if self._is_touch_mode:
            calibration_layout.setSpacing(12)
            calibration_layout.setContentsMargins(12, 20, 12, 12)

        calibration_desc = QLabel(
            "Draw measurement lines on the camera view to calibrate\n"
            "pixel-to-distance conversion for tracking data."
        )
        calibration_desc.setWordWrap(True)
        calibration_desc.setProperty("textRole", "muted")
        calibration_layout.addWidget(calibration_desc)

        self._calibrate_btn = QPushButton("Open Calibration...")
        if self._is_touch_mode:
            self._calibrate_btn.setMinimumHeight(50)
            self._calibrate_btn.setProperty("touchMode", True)
        self._calibrate_btn.clicked.connect(self._on_calibration_clicked)
        calibration_layout.addWidget(self._calibrate_btn)

        layout.addWidget(calibration_group)

        # Zones group
        zones_group = QGroupBox("Zone Configuration")
        self._apply_touch_group_property(zones_group)
        zones_layout = QVBoxLayout(zones_group)
        if self._is_touch_mode:
            zones_layout.setSpacing(12)
            zones_layout.setContentsMargins(12, 20, 12, 12)

        zones_desc = QLabel(
            "Define regions of interest (zones) on the camera view\n"
            "for tracking object entries, exits, and time spent."
        )
        zones_desc.setWordWrap(True)
        zones_desc.setProperty("textRole", "muted")
        zones_layout.addWidget(zones_desc)

        self._zones_btn = QPushButton("Open Zone Editor...")
        if self._is_touch_mode:
            self._zones_btn.setMinimumHeight(50)
            self._zones_btn.setProperty("touchMode", True)
        self._zones_btn.clicked.connect(self._on_zones_clicked)
        zones_layout.addWidget(self._zones_btn)

        layout.addWidget(zones_group)
        layout.addStretch()

        return widget

    def _on_calibration_clicked(self) -> None:
        """Handle calibration button click."""
        self.calibration_requested.emit()

    def _on_zones_clicked(self) -> None:
        """Handle zones button click."""
        self.zones_requested.emit()

    def _create_audio_tab_content(self) -> QWidget:
        """Create audio recording settings tab content."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Audio device selection
        device_group = QGroupBox("Microphone")
        device_layout = QVBoxLayout(device_group)

        # Device dropdown
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("Device:"))
        self._audio_device_combo = QComboBox()
        mic_layout.addWidget(self._audio_device_combo, 1)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_audio_devices)
        mic_layout.addWidget(refresh_btn)
        device_layout.addLayout(mic_layout)

        # Test button
        test_layout = QHBoxLayout()
        self._audio_test_btn = QPushButton("Test Microphone")
        self._audio_test_btn.clicked.connect(self._test_microphone)
        test_layout.addWidget(self._audio_test_btn)
        self._audio_test_status = QLabel("")
        test_layout.addWidget(self._audio_test_status, 1)
        device_layout.addLayout(test_layout)

        layout.addWidget(device_group)
        layout.addStretch()

        # Populate devices
        self._refresh_audio_devices()

        return widget

    def _refresh_audio_devices(self) -> None:
        """Refresh the audio device dropdown."""
        self._audio_device_combo.clear()
        self._audio_device_combo.addItem("None (no audio recording)", None)

        try:
            from glider.vision.audio_recorder import AudioRecorder

            devices = AudioRecorder.enumerate_devices()
            for idx, name in devices:
                self._audio_device_combo.addItem(name, idx)
        except Exception as e:
            logger.warning(f"Could not enumerate audio devices: {e}")

    def _test_microphone(self) -> None:
        """Record 1 second of audio and play it back."""
        device_index = self._audio_device_combo.currentData()
        if device_index is None:
            self._audio_test_status.setText("No device selected")
            return

        self._audio_test_status.setText("Recording...")
        self._audio_test_btn.setEnabled(False)

        try:
            import numpy as np
            import sounddevice as _sd

            duration = 1.0
            recording = _sd.rec(
                int(duration * 44100),
                samplerate=44100,
                channels=1,
                dtype="int16",
                device=device_index,
                blocking=True,
            )

            self._audio_test_status.setText("Playing back...")
            try:
                _sd.play(recording, samplerate=44100, blocking=True)
                self._audio_test_status.setText("Test complete")
            except Exception:
                # Playback may fail on headless systems
                if np.any(recording):
                    self._audio_test_status.setText("Test recording captured successfully")
                else:
                    self._audio_test_status.setText("Warning: recording was silent")
        except ImportError:
            self._audio_test_status.setText("sounddevice not installed")
        except Exception as e:
            self._audio_test_status.setText(f"Error: {e}")
        finally:
            self._audio_test_btn.setEnabled(True)

    def _load_settings(self):
        """Load current settings into the UI."""
        # Camera settings
        res = self._camera_settings.resolution
        for i in range(self._resolution_combo.count()):
            if self._resolution_combo.itemData(i) == res:
                self._resolution_combo.setCurrentIndex(i)
                break

        self._fps_spin.setValue(self._camera_settings.fps)

        if self._camera_settings.exposure == -1:
            self._auto_exposure_cb.setChecked(True)
        else:
            self._auto_exposure_cb.setChecked(False)
            self._exposure_slider.setValue(self._camera_settings.exposure)

        self._brightness_slider.setValue(self._camera_settings.brightness)
        self._contrast_slider.setValue(self._camera_settings.contrast)

        # Connection settings
        self._timeout_spin.setValue(self._camera_settings.connection_timeout)
        for i in range(self._backend_camera_combo.count()):
            if self._backend_camera_combo.itemData(i) == self._camera_settings.force_backend:
                self._backend_camera_combo.setCurrentIndex(i)
                break
        for i in range(self._pixel_format_combo.count()):
            if self._pixel_format_combo.itemData(i) == self._camera_settings.pixel_format:
                self._pixel_format_combo.setCurrentIndex(i)
                break
        self._miniscope_mode_cb.setChecked(self._camera_settings.miniscope_mode)

        # Miniscope controls
        self._exposure_time_spin.setValue(self._camera_settings.exposure_time)
        self._gain_spin.setValue(self._camera_settings.gain)
        self._gamma_spin.setValue(self._camera_settings.gamma)
        self._hue_spin.setValue(self._camera_settings.hue)
        self._sharpness_spin.setValue(self._camera_settings.sharpness)
        self._focus_spin.setValue(self._camera_settings.focus)
        self._zoom_spin.setValue(self._camera_settings.zoom)
        self._iris_spin.setValue(self._camera_settings.iris)
        # LED and EWL controls
        self._led_power_slider.setValue(self._camera_settings.led_power)
        self._led_power_label.setText(f"{self._camera_settings.led_power}%")
        self._ewl_focus_slider.setValue(self._camera_settings.ewl_focus)
        self._ewl_focus_label.setText(str(self._camera_settings.ewl_focus))
        # Show/hide miniscope group based on mode
        self._miniscope_group.setVisible(self._camera_settings.miniscope_mode)

        # Audio settings
        audio_name = self._camera_settings.audio_device_name
        if audio_name:
            for i in range(self._audio_device_combo.count()):
                if self._audio_device_combo.itemText(i) == audio_name:
                    self._audio_device_combo.setCurrentIndex(i)
                    break

        # CV settings
        self._cv_enabled_cb.setChecked(self._cv_settings.enabled)

        for i in range(self._backend_combo.count()):
            if self._backend_combo.itemData(i) == self._cv_settings.backend:
                self._backend_combo.setCurrentIndex(i)
                break

        if self._cv_settings.model_path:
            self._model_path_edit.setText(self._cv_settings.model_path)

        if self._cv_settings.keypoint_names:
            self._keypoint_names_edit.setText(", ".join(self._cv_settings.keypoint_names))

        self._confidence_spin.setValue(self._cv_settings.confidence_threshold)
        self._min_area_spin.setValue(self._cv_settings.min_detection_area)
        self._frame_skip_spin.setValue(self._cv_settings.process_every_n_frames)
        self._draw_overlays_cb.setChecked(self._cv_settings.draw_overlays)
        self._show_keypoints_cb.setChecked(self._cv_settings.show_keypoints)

        # Update UI state
        self._on_cv_enabled_toggle(self._cv_settings.enabled)
        self._on_backend_changed(self._backend_combo.currentIndex())

    def _on_auto_exposure_toggle(self, checked: bool):
        """Handle auto exposure toggle."""
        self._exposure_slider.setEnabled(not checked)
        self._exposure_label.setEnabled(not checked)

    def _on_cv_enabled_toggle(self, enabled: bool):
        """Handle CV enabled toggle."""
        self._backend_combo.setEnabled(enabled)
        self._confidence_spin.setEnabled(enabled)
        self._min_area_spin.setEnabled(enabled)
        self._frame_skip_spin.setEnabled(enabled)
        self._draw_overlays_cb.setEnabled(enabled)
        self._show_keypoints_cb.setEnabled(enabled)
        self._draw_tracks_cb.setEnabled(enabled)
        self._draw_contours_cb.setEnabled(enabled)
        if enabled:
            self._on_backend_changed(self._backend_combo.currentIndex())

    def _on_backend_changed(self, index: int):
        """Handle backend selection change."""
        backend = self._backend_combo.itemData(index)
        is_yolo = backend in (DetectionBackend.YOLO_V8, DetectionBackend.YOLO_BYTETRACK)
        is_pose = backend == DetectionBackend.POSE_MODEL

        # If the user just switched to a YOLO backend, make sure the
        # ultralytics library is actually available (or offer to install it
        # — see yolo_install.py for the AGPL rationale). If we can't get
        # it, revert the combo to Background Sub so the rest of the UI
        # doesn't end up half-configured. Guarded by an attribute so unit
        # tests and migrations can construct the dialog without triggering
        # a real install prompt. Not a concern for POSE_MODEL: it never
        # touches ultralytics.
        if is_yolo and not getattr(self, "_suppress_yolo_prompt", False):
            from glider.vision.yolo_install import ensure_ultralytics_installed

            if not ensure_ultralytics_installed(self):
                # Block signals while we rewind; otherwise this handler
                # fires re-entrantly on setCurrentIndex.
                self._backend_combo.blockSignals(True)
                try:
                    for i in range(self._backend_combo.count()):
                        if (
                            self._backend_combo.itemData(i)
                            == DetectionBackend.BACKGROUND_SUBTRACTION
                        ):
                            self._backend_combo.setCurrentIndex(i)
                            break
                finally:
                    self._backend_combo.blockSignals(False)
                is_yolo = False

        # The model field is shared by every model-backed backend. Its
        # placeholder used to claim to be YOLO-only, which stopped being true
        # once a pose model could drive tracking too.
        is_model_backed = is_yolo or is_pose
        self._model_path_edit.setPlaceholderText(
            "Pose model (SLEAP/DLC folder or .onnx, or a YOLO-pose .pt)"
            if is_pose
            else "YOLO model (.pt or NCNN .param)"
        )
        self._model_path_edit.setVisible(is_model_backed)
        self._model_path_label.setVisible(is_model_backed)
        self._browse_model_btn.setVisible(is_model_backed)
        # Keypoint names mean something for any model-backed backend now, not
        # only YOLO: a pose model's bodypart names label the keypoints CSV too.
        self._keypoint_names_edit.setVisible(is_model_backed)
        self._keypoint_names_label.setVisible(is_model_backed)

    def _on_miniscope_mode_toggle(self, enabled: bool):
        """Handle miniscope mode toggle - auto-set recommended values."""
        # Show/hide miniscope controls group
        self._miniscope_group.setVisible(enabled)

        if enabled:
            # Auto-set recommended miniscope settings
            # Set resolution to 608x608
            for i in range(self._resolution_combo.count()):
                if self._resolution_combo.itemData(i) == (608, 608):
                    self._resolution_combo.setCurrentIndex(i)
                    break
            # Set pixel format to YUY2/YUYV
            import sys

            pixel_format = "YUY2" if sys.platform == "win32" else "YUYV"
            for i in range(self._pixel_format_combo.count()):
                if self._pixel_format_combo.itemData(i) == pixel_format:
                    self._pixel_format_combo.setCurrentIndex(i)
                    break
            # Set backend based on platform
            backend = "dshow" if sys.platform == "win32" else "v4l2"
            for i in range(self._backend_camera_combo.count()):
                if self._backend_camera_combo.itemData(i) == backend:
                    self._backend_camera_combo.setCurrentIndex(i)
                    break
            # Increase timeout
            self._timeout_spin.setValue(10.0)

    def _on_led_power_changed(self, value: int):
        """Handle LED power slider change.

        Surfaces I2C / out-of-range failures so the slider's displayed
        value cannot silently drift away from the actual hardware state.
        ``set_led_power`` raises ``ValueError`` on out-of-range; the GUI
        slider's bounds *should* prevent this, but a programmatic caller
        or a touchscreen edge case could violate them.
        """
        self._led_power_label.setText(f"{value}%")
        if self._camera_manager is None or not self._camera_manager.is_connected:
            return
        try:
            ok = self._camera_manager.set_led_power(value)
        except ValueError as e:
            self._led_power_label.setText(f"{value}% (rejected)")
            QMessageBox.warning(self, "LED Power Out of Range", str(e))
            return
        if not ok:
            self._led_power_label.setText(f"{value}% (failed)")
            QMessageBox.warning(
                self,
                "LED Command Failed",
                f"Could not set LED power to {value}%. "
                "Check camera connection and that miniscope mode is enabled.",
            )

    def _on_ewl_focus_changed(self, value: int):
        """Handle EWL focus slider change.

        Same fail-loud pattern as LED power — focus drift between the
        slider and the lens silently corrupts experiment focus settings.
        """
        self._ewl_focus_label.setText(str(value))
        if self._camera_manager is None or not self._camera_manager.is_connected:
            return
        try:
            ok = self._camera_manager.set_ewl_focus(value)
        except ValueError as e:
            self._ewl_focus_label.setText(f"{value} (rejected)")
            QMessageBox.warning(self, "EWL Focus Out of Range", str(e))
            return
        if not ok:
            self._ewl_focus_label.setText(f"{value} (failed)")
            QMessageBox.warning(
                self,
                "EWL Focus Command Failed",
                f"Could not set EWL focus to {value}. "
                "Check camera connection and that miniscope mode is enabled.",
            )

    def _browse_model(self):
        """Browse for a YOLO model.

        Accepts a PyTorch ``.pt`` file or an NCNN export. NCNN models live in a
        ``*_ncnn_model/`` folder; since ``getOpenFileName`` can't select a
        folder, the user navigates into it and picks ``model.ncnn.param`` —
        CVProcessor normalizes that to the containing folder when loading.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "YOLO models (*.pt *.param);;PyTorch (*.pt);;" "NCNN param (*.param);;All Files (*)",
        )
        if path:
            self._model_path_edit.setText(path)

    def _apply_settings(self):
        """Apply settings without closing dialog."""
        self._save_settings()
        logger.info("Camera settings applied")

    def _save_settings(self):
        """Save UI values to settings objects."""
        # Camera settings
        self._camera_settings.resolution = self._resolution_combo.currentData()
        self._camera_settings.fps = self._fps_spin.value()

        if self._auto_exposure_cb.isChecked():
            self._camera_settings.exposure = -1
        else:
            self._camera_settings.exposure = self._exposure_slider.value()

        self._camera_settings.brightness = self._brightness_slider.value()
        self._camera_settings.contrast = self._contrast_slider.value()
        self._camera_settings.connection_timeout = self._timeout_spin.value()
        self._camera_settings.force_backend = self._backend_camera_combo.currentData()
        self._camera_settings.pixel_format = self._pixel_format_combo.currentData()
        self._camera_settings.miniscope_mode = self._miniscope_mode_cb.isChecked()

        # Miniscope controls
        self._camera_settings.exposure_time = self._exposure_time_spin.value()
        self._camera_settings.gain = self._gain_spin.value()
        self._camera_settings.gamma = self._gamma_spin.value()
        self._camera_settings.hue = self._hue_spin.value()
        self._camera_settings.sharpness = self._sharpness_spin.value()
        self._camera_settings.focus = self._focus_spin.value()
        self._camera_settings.zoom = self._zoom_spin.value()
        self._camera_settings.iris = self._iris_spin.value()
        # LED and EWL controls
        self._camera_settings.led_power = self._led_power_slider.value()
        self._camera_settings.ewl_focus = self._ewl_focus_slider.value()

        # Audio settings
        selected_idx = self._audio_device_combo.currentData()
        if selected_idx is not None:
            self._camera_settings.audio_device_name = self._audio_device_combo.currentText()
            self._camera_settings.audio_device_index = selected_idx
        else:
            self._camera_settings.audio_device_name = None
            self._camera_settings.audio_device_index = None

        # CV settings
        self._cv_settings.enabled = self._cv_enabled_cb.isChecked()
        self._cv_settings.backend = self._backend_combo.currentData()
        self._cv_settings.model_path = self._model_path_edit.text() or None
        self._cv_settings.keypoint_names = parse_keypoint_names(self._keypoint_names_edit.text())
        self._cv_settings.confidence_threshold = self._confidence_spin.value()
        self._cv_settings.min_detection_area = self._min_area_spin.value()
        self._cv_settings.process_every_n_frames = self._frame_skip_spin.value()
        self._cv_settings.draw_overlays = self._draw_overlays_cb.isChecked()
        self._cv_settings.show_keypoints = self._show_keypoints_cb.isChecked()

    def accept(self):
        """Handle dialog acceptance."""
        self._save_settings()
        super().accept()

    def get_camera_settings(self) -> CameraSettings:
        """Get the camera settings."""
        return self._camera_settings

    def get_cv_settings(self) -> CVSettings:
        """Get the CV settings."""
        return self._cv_settings
