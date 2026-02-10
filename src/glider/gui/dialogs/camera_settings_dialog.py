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

from glider.audio.audio_recorder import AudioRecorder, AudioSettings
from glider.vision.camera_manager import CameraSettings
from glider.vision.cv_processor import CVSettings, DetectionBackend

logger = logging.getLogger(__name__)


class CameraSettingsDialog(QDialog):
    """Dialog for configuring camera and CV settings."""

    # Signals for opening external dialogs
    calibration_requested = pyqtSignal()
    zones_requested = pyqtSignal()

    def __init__(
        self,
        camera_settings: Optional[CameraSettings] = None,
        cv_settings: Optional[CVSettings] = None,
        audio_settings: Optional[AudioSettings] = None,
        parent=None,
        view_manager: Optional["ViewManager"] = None,
        camera_manager: Optional["CameraManager"] = None,
    ):
        super().__init__(parent)
        self._camera_settings = camera_settings or CameraSettings()
        self._cv_settings = cv_settings or CVSettings()
        self._audio_settings = audio_settings or AudioSettings()
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
            self._tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #555;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    padding: 12px 20px;
                    font-size: 16px;
                    font-weight: bold;
                    min-width: 100px;
                }
                QTabBar::tab:selected {
                    background-color: #3a7bd5;
                    color: white;
                }
            """)
        layout.addWidget(self._tabs)

        # Camera tab (wrapped in scroll area)
        self._camera_tab = self._create_scrollable_tab(self._create_camera_tab_content())
        self._tabs.addTab(self._camera_tab, "Camera")

        # Computer Vision tab (wrapped in scroll area)
        self._cv_tab = self._create_scrollable_tab(self._create_cv_tab_content())
        self._tabs.addTab(self._cv_tab, "CV")

        # Tracking tab (wrapped in scroll area)
        self._tracking_tab = self._create_scrollable_tab(self._create_tracking_tab_content())
        self._tabs.addTab(self._tracking_tab, "Tracking")

        # Behavior tab (wrapped in scroll area)
        self._behavior_tab = self._create_scrollable_tab(self._create_behavior_tab_content())
        self._tabs.addTab(self._behavior_tab, "Behavior")

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
            button_box.setStyleSheet("""
                QPushButton {
                    padding: 14px 24px;
                    font-size: 16px;
                    font-weight: bold;
                    min-width: 90px;
                    min-height: 44px;
                }
            """)

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
            # Wide scrollbar and kinetic scrolling for touch
            scroll.setStyleSheet("""
                QScrollArea {
                    background: transparent;
                }
                QScrollBar:vertical {
                    width: 30px;
                    background: #2a2a2a;
                    border-radius: 15px;
                    margin: 4px;
                }
                QScrollBar::handle:vertical {
                    background: #5a5a5a;
                    border-radius: 13px;
                    min-height: 60px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #6a6a6a;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)
            # Enable kinetic scrolling
            QScroller.grabGesture(
                scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
            )

        return scroll

    def _get_touch_group_style(self) -> str:
        """Get stylesheet for group boxes in touch mode."""
        if not self._is_touch_mode:
            return ""
        return """
            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                padding: 16px 8px 8px 8px;
                margin-top: 12px;
                border: 1px solid #444;
                border-radius: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                color: #3a7bd5;
            }
            QLabel {
                font-size: 14px;
            }
            QComboBox {
                padding: 10px;
                font-size: 14px;
                min-height: 36px;
            }
            QComboBox::drop-down {
                width: 30px;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 8px;
                font-size: 14px;
                min-height: 36px;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 30px;
            }
            QCheckBox {
                font-size: 14px;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 28px;
                height: 28px;
            }
            QSlider::groove:horizontal {
                height: 12px;
                background: #3a3a3a;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                width: 28px;
                height: 28px;
                margin: -8px 0;
                background: #3a7bd5;
                border-radius: 14px;
            }
        """

    def _create_camera_tab_content(self) -> QWidget:
        """Create the camera settings tab content."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Adjust spacing for touch mode
        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(16)

        touch_style = self._get_touch_group_style()

        # Resolution group
        res_group = QGroupBox("Resolution")
        res_group.setStyleSheet(touch_style)
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
        image_group.setStyleSheet(touch_style)
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
        conn_group.setStyleSheet(touch_style)
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
        self._miniscope_group.setStyleSheet(touch_style)
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

        touch_style = self._get_touch_group_style()

        # Enable CV - prominent checkbox at top
        self._cv_enabled_cb = QCheckBox("Enable CV Processing")
        if self._is_touch_mode:
            self._cv_enabled_cb.setStyleSheet("""
                QCheckBox {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 8px;
                    spacing: 12px;
                }
                QCheckBox::indicator {
                    width: 32px;
                    height: 32px;
                }
            """)
        self._cv_enabled_cb.toggled.connect(self._on_cv_enabled_toggle)
        layout.addWidget(self._cv_enabled_cb)

        # Detection group
        detection_group = QGroupBox("Detection")
        detection_group.setStyleSheet(touch_style)
        detection_layout = QFormLayout(detection_group)
        if self._is_touch_mode:
            detection_layout.setSpacing(12)
            detection_layout.setContentsMargins(12, 20, 12, 12)

        self._backend_combo = QComboBox()
        self._backend_combo.addItem("Background Sub", DetectionBackend.BACKGROUND_SUBTRACTION)
        self._backend_combo.addItem("Motion Only", DetectionBackend.MOTION_ONLY)
        self._backend_combo.addItem("YOLO v8", DetectionBackend.YOLO_V8)
        self._backend_combo.addItem("YOLO+ByteTrack", DetectionBackend.YOLO_BYTETRACK)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        detection_layout.addRow("Backend:", self._backend_combo)

        # YOLO model path (only visible for YOLO backend)
        model_layout = QHBoxLayout()
        model_layout.setSpacing(8)
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText("YOLO model (.pt)")
        model_layout.addWidget(self._model_path_edit)

        self._browse_model_btn = QPushButton("...")
        if self._is_touch_mode:
            self._browse_model_btn.setMinimumSize(50, 40)
        self._browse_model_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(self._browse_model_btn)

        self._model_path_label = QLabel("Model:")
        detection_layout.addRow(self._model_path_label, model_layout)

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
        overlay_group.setStyleSheet(touch_style)
        overlay_layout = QFormLayout(overlay_group)
        if self._is_touch_mode:
            overlay_layout.setSpacing(12)
            overlay_layout.setContentsMargins(12, 20, 12, 12)

        self._draw_overlays_cb = QCheckBox("Bounding Boxes")
        self._draw_overlays_cb.setChecked(True)
        overlay_layout.addRow(self._draw_overlays_cb)

        self._draw_tracks_cb = QCheckBox("Motion Tracks")
        self._draw_tracks_cb.setChecked(True)
        overlay_layout.addRow(self._draw_tracks_cb)

        self._draw_contours_cb = QCheckBox("Contours")
        self._draw_contours_cb.setChecked(False)
        overlay_layout.addRow(self._draw_contours_cb)

        layout.addWidget(overlay_group)
        layout.addStretch()

        return widget

    def _create_tracking_tab_content(self) -> QWidget:
        """Create the tracking settings tab content."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Adjust spacing for touch mode
        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(16)

        touch_style = self._get_touch_group_style()

        # Enable tracking - prominent checkbox at top
        self._tracking_enabled_cb = QCheckBox("Enable Tracking")
        if self._is_touch_mode:
            self._tracking_enabled_cb.setStyleSheet("""
                QCheckBox {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 8px;
                    spacing: 12px;
                }
                QCheckBox::indicator {
                    width: 32px;
                    height: 32px;
                }
            """)
        self._tracking_enabled_cb.toggled.connect(self._on_tracking_enabled_toggle)
        layout.addWidget(self._tracking_enabled_cb)

        # Tracking parameters
        tracking_group = QGroupBox("Tracking")
        tracking_group.setStyleSheet(touch_style)
        tracking_layout = QFormLayout(tracking_group)
        if self._is_touch_mode:
            tracking_layout.setSpacing(12)
            tracking_layout.setContentsMargins(12, 20, 12, 12)

        self._max_disappeared_spin = QSpinBox()
        self._max_disappeared_spin.setRange(1, 200)
        self._max_disappeared_spin.setValue(50)
        self._max_disappeared_spin.setSuffix(" frames")
        tracking_layout.addRow("Max Lost:", self._max_disappeared_spin)

        # Only show help text in desktop mode (too verbose for touch)
        if not self._is_touch_mode:
            help_label = QLabel(
                "Number of consecutive frames an object can be missing\n"
                "before it is deregistered from tracking."
            )
            help_label.setStyleSheet("color: #888; font-size: 11px;")
            tracking_layout.addRow(help_label)

        self._max_distance_spin = QSpinBox()
        self._max_distance_spin.setRange(10, 500)
        self._max_distance_spin.setValue(100)
        self._max_distance_spin.setSuffix(" px")
        tracking_layout.addRow("Max Distance:", self._max_distance_spin)

        # Only show help text in desktop mode
        if not self._is_touch_mode:
            distance_help = QLabel(
                "Maximum distance (in pixels) an object can move\n"
                "between frames and still be considered the same object."
            )
            distance_help.setStyleSheet("color: #888; font-size: 11px;")
            tracking_layout.addRow(distance_help)

        layout.addWidget(tracking_group)

        # Motion detection
        motion_group = QGroupBox("Motion")
        motion_group.setStyleSheet(touch_style)
        motion_layout = QFormLayout(motion_group)
        if self._is_touch_mode:
            motion_layout.setSpacing(12)
            motion_layout.setContentsMargins(12, 20, 12, 12)

        self._motion_threshold_spin = QSpinBox()
        self._motion_threshold_spin.setRange(1, 100)
        self._motion_threshold_spin.setValue(25)
        motion_layout.addRow("Threshold:", self._motion_threshold_spin)

        self._motion_area_spin = QDoubleSpinBox()
        self._motion_area_spin.setRange(0.1, 50.0)
        self._motion_area_spin.setValue(1.0)
        self._motion_area_spin.setSuffix(" %")
        motion_layout.addRow("Min Area:", self._motion_area_spin)

        layout.addWidget(motion_group)
        layout.addStretch()

        return widget

    def _create_behavior_tab_content(self) -> QWidget:
        """Create the behavior analysis settings tab content."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Adjust spacing for touch mode
        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(16)

        touch_style = self._get_touch_group_style()

        # Enable behavior analysis - prominent checkbox at top
        self._behavior_enabled_cb = QCheckBox("Enable Behavior Analysis")
        if self._is_touch_mode:
            self._behavior_enabled_cb.setStyleSheet("""
                QCheckBox {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 8px;
                    spacing: 12px;
                }
                QCheckBox::indicator {
                    width: 32px;
                    height: 32px;
                }
            """)
        self._behavior_enabled_cb.toggled.connect(self._on_behavior_enabled_toggle)
        layout.addWidget(self._behavior_enabled_cb)

        # Thresholds group
        thresholds_group = QGroupBox("Thresholds (px/frame)")
        thresholds_group.setStyleSheet(touch_style)
        thresholds_layout = QFormLayout(thresholds_group)
        if self._is_touch_mode:
            thresholds_layout.setSpacing(12)
            thresholds_layout.setContentsMargins(12, 20, 12, 12)

        # Freeze threshold
        self._freeze_threshold_spin = QDoubleSpinBox()
        self._freeze_threshold_spin.setRange(0.1, 10.0)
        self._freeze_threshold_spin.setSingleStep(0.5)
        self._freeze_threshold_spin.setValue(1.0)
        self._freeze_threshold_spin.setToolTip(
            "Maximum movement (px/frame) to be considered frozen.\n"
            "Lower = stricter freeze detection."
        )
        thresholds_layout.addRow("Freeze:", self._freeze_threshold_spin)

        # Immobile threshold
        self._immobile_threshold_spin = QDoubleSpinBox()
        self._immobile_threshold_spin.setRange(1.0, 20.0)
        self._immobile_threshold_spin.setSingleStep(1.0)
        self._immobile_threshold_spin.setValue(5.0)
        self._immobile_threshold_spin.setToolTip(
            "Maximum movement (px/frame) to be considered immobile.\n"
            "Covers minor movements like grooming."
        )
        thresholds_layout.addRow("Immobile:", self._immobile_threshold_spin)

        # Dart threshold
        self._dart_threshold_spin = QDoubleSpinBox()
        self._dart_threshold_spin.setRange(20.0, 200.0)
        self._dart_threshold_spin.setSingleStep(5.0)
        self._dart_threshold_spin.setValue(50.0)
        self._dart_threshold_spin.setToolTip(
            "Minimum movement (px/frame) to be considered darting.\n"
            "Detects rapid escape/flight responses."
        )
        thresholds_layout.addRow("Darting:", self._dart_threshold_spin)

        layout.addWidget(thresholds_group)

        # Timing group
        timing_group = QGroupBox("Timing")
        timing_group.setStyleSheet(touch_style)
        timing_layout = QFormLayout(timing_group)
        if self._is_touch_mode:
            timing_layout.setSpacing(12)
            timing_layout.setContentsMargins(12, 20, 12, 12)

        # Freeze duration
        self._freeze_duration_spin = QSpinBox()
        self._freeze_duration_spin.setRange(5, 60)
        self._freeze_duration_spin.setValue(15)
        self._freeze_duration_spin.setSuffix(" frames")
        self._freeze_duration_spin.setToolTip(
            "Number of consecutive low-movement frames required\n" "to confirm a freeze state."
        )
        timing_layout.addRow("Freeze Duration:", self._freeze_duration_spin)

        # Smoothing window
        self._smoothing_window_spin = QSpinBox()
        self._smoothing_window_spin.setRange(3, 15)
        self._smoothing_window_spin.setValue(5)
        self._smoothing_window_spin.setSuffix(" frames")
        self._smoothing_window_spin.setToolTip(
            "Number of frames to average for velocity smoothing.\n"
            "Higher = smoother but less responsive."
        )
        timing_layout.addRow("Smoothing:", self._smoothing_window_spin)

        layout.addWidget(timing_group)

        # State colors reference (read-only info)
        colors_group = QGroupBox("State Colors")
        colors_group.setStyleSheet(touch_style)
        colors_layout = QVBoxLayout(colors_group)
        if self._is_touch_mode:
            colors_layout.setSpacing(8)
            colors_layout.setContentsMargins(12, 20, 12, 12)

        color_labels = [
            ("FREEZE", "#0000FF", "Complete stillness"),
            ("IMMOBILE", "#FFFF00", "Minor movements"),
            ("MOVING", "#00FF00", "Normal locomotion"),
            ("DARTING", "#FF0000", "Rapid movement"),
        ]
        for name, color, desc in color_labels:
            color_row = QHBoxLayout()
            color_box = QLabel()
            color_box.setFixedSize(20, 20)
            color_box.setStyleSheet(f"background-color: {color}; border: 1px solid #666;")
            color_row.addWidget(color_box)
            text_label = QLabel(f"{name}: {desc}")
            if self._is_touch_mode:
                text_label.setStyleSheet("font-size: 13px;")
            color_row.addWidget(text_label)
            color_row.addStretch()
            colors_layout.addLayout(color_row)

        layout.addWidget(colors_group)
        layout.addStretch()

        return widget

    def _on_behavior_enabled_toggle(self, enabled: bool):
        """Handle behavior analysis enabled toggle."""
        self._freeze_threshold_spin.setEnabled(enabled)
        self._immobile_threshold_spin.setEnabled(enabled)
        self._dart_threshold_spin.setEnabled(enabled)
        self._freeze_duration_spin.setEnabled(enabled)
        self._smoothing_window_spin.setEnabled(enabled)

    def _create_tools_tab_content(self) -> QWidget:
        """Create the tools tab content with calibration and zones buttons."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Adjust spacing for touch mode
        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(16)

        touch_style = self._get_touch_group_style()

        # Calibration group
        calibration_group = QGroupBox("Camera Calibration")
        calibration_group.setStyleSheet(touch_style)
        calibration_layout = QVBoxLayout(calibration_group)
        if self._is_touch_mode:
            calibration_layout.setSpacing(12)
            calibration_layout.setContentsMargins(12, 20, 12, 12)

        calibration_desc = QLabel(
            "Draw measurement lines on the camera view to calibrate\n"
            "pixel-to-distance conversion for tracking data."
        )
        calibration_desc.setWordWrap(True)
        if self._is_touch_mode:
            calibration_desc.setStyleSheet("font-size: 13px; color: #aaa;")
        else:
            calibration_desc.setStyleSheet("color: #888; font-size: 11px;")
        calibration_layout.addWidget(calibration_desc)

        self._calibrate_btn = QPushButton("Open Calibration...")
        if self._is_touch_mode:
            self._calibrate_btn.setMinimumHeight(50)
            self._calibrate_btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 12px;
                }
            """)
        self._calibrate_btn.clicked.connect(self._on_calibration_clicked)
        calibration_layout.addWidget(self._calibrate_btn)

        layout.addWidget(calibration_group)

        # Zones group
        zones_group = QGroupBox("Zone Configuration")
        zones_group.setStyleSheet(touch_style)
        zones_layout = QVBoxLayout(zones_group)
        if self._is_touch_mode:
            zones_layout.setSpacing(12)
            zones_layout.setContentsMargins(12, 20, 12, 12)

        zones_desc = QLabel(
            "Define regions of interest (zones) on the camera view\n"
            "for tracking object entries, exits, and time spent."
        )
        zones_desc.setWordWrap(True)
        if self._is_touch_mode:
            zones_desc.setStyleSheet("font-size: 13px; color: #aaa;")
        else:
            zones_desc.setStyleSheet("color: #888; font-size: 11px;")
        zones_layout.addWidget(zones_desc)

        self._zones_btn = QPushButton("Open Zone Editor...")
        if self._is_touch_mode:
            self._zones_btn.setMinimumHeight(50)
            self._zones_btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 12px;
                }
            """)
        self._zones_btn.clicked.connect(self._on_zones_clicked)
        zones_layout.addWidget(self._zones_btn)

        layout.addWidget(zones_group)
        layout.addStretch()

        return widget

    def _create_audio_tab_content(self) -> QWidget:
        """Create the audio recording settings tab content."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(16)

        touch_style = self._get_touch_group_style()

        # Enable audio recording - prominent checkbox at top
        self._audio_enabled_cb = QCheckBox("Enable Audio Recording")
        if self._is_touch_mode:
            self._audio_enabled_cb.setStyleSheet("""
                QCheckBox {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 8px;
                    spacing: 12px;
                }
                QCheckBox::indicator {
                    width: 32px;
                    height: 32px;
                }
            """)
        self._audio_enabled_cb.toggled.connect(self._on_audio_enabled_toggle)
        layout.addWidget(self._audio_enabled_cb)

        # Audio Device group
        device_group = QGroupBox("Audio Device")
        device_group.setStyleSheet(touch_style)
        device_layout = QFormLayout(device_group)
        if self._is_touch_mode:
            device_layout.setSpacing(12)
            device_layout.setContentsMargins(12, 20, 12, 12)

        device_row = QHBoxLayout()
        self._audio_device_combo = QComboBox()
        device_row.addWidget(self._audio_device_combo)

        self._audio_refresh_btn = QPushButton("Refresh")
        if self._is_touch_mode:
            self._audio_refresh_btn.setMinimumSize(80, 40)
        self._audio_refresh_btn.clicked.connect(self._populate_audio_devices)
        device_row.addWidget(self._audio_refresh_btn)

        device_layout.addRow("Device:", device_row)
        layout.addWidget(device_group)

        # Recording Format group
        format_group = QGroupBox("Recording Format")
        format_group.setStyleSheet(touch_style)
        format_layout = QFormLayout(format_group)
        if self._is_touch_mode:
            format_layout.setSpacing(12)
            format_layout.setContentsMargins(12, 20, 12, 12)

        self._audio_sample_rate_combo = QComboBox()
        self._audio_sample_rate_combo.addItem("22050 Hz", 22050)
        self._audio_sample_rate_combo.addItem("44100 Hz", 44100)
        self._audio_sample_rate_combo.addItem("48000 Hz", 48000)
        self._audio_sample_rate_combo.addItem("96000 Hz", 96000)
        self._audio_sample_rate_combo.setCurrentIndex(1)  # Default 44100
        format_layout.addRow("Sample Rate:", self._audio_sample_rate_combo)

        self._audio_channels_combo = QComboBox()
        self._audio_channels_combo.addItem("Mono", 1)
        self._audio_channels_combo.addItem("Stereo", 2)
        format_layout.addRow("Channels:", self._audio_channels_combo)

        layout.addWidget(format_group)

        # Input Gain group
        gain_group = QGroupBox("Input Gain")
        gain_group.setStyleSheet(touch_style)
        gain_layout = QFormLayout(gain_group)
        if self._is_touch_mode:
            gain_layout.setSpacing(12)
            gain_layout.setContentsMargins(12, 20, 12, 12)

        gain_row = QHBoxLayout()
        gain_row.setSpacing(8 if not self._is_touch_mode else 12)
        self._audio_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._audio_gain_slider.setRange(0, 500)
        self._audio_gain_slider.setValue(100)  # 1.0x
        if self._is_touch_mode:
            self._audio_gain_slider.setMinimumHeight(40)
        gain_row.addWidget(self._audio_gain_slider)

        self._audio_gain_label = QLabel("1.0x")
        self._audio_gain_label.setMinimumWidth(45 if self._is_touch_mode else 35)
        self._audio_gain_slider.valueChanged.connect(self._on_audio_gain_changed)
        gain_row.addWidget(self._audio_gain_label)

        gain_layout.addRow("Gain:", gain_row)
        layout.addWidget(gain_group)

        # Status label
        self._audio_status_label = QLabel()
        self._audio_status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._audio_status_label)

        layout.addStretch()

        # Populate devices
        self._populate_audio_devices()

        return widget

    def _populate_audio_devices(self) -> None:
        """Enumerate and fill the audio device combo box."""
        self._audio_device_combo.clear()
        self._audio_device_combo.addItem("System Default", None)

        devices = AudioRecorder.get_available_devices()
        for dev in devices:
            label = f"{dev['name']} ({dev['channels']}ch)"
            self._audio_device_combo.addItem(label, dev["index"])

        self._update_audio_status_label()

    def _on_audio_enabled_toggle(self, enabled: bool) -> None:
        """Handle audio enabled toggle - enable/disable child controls."""
        self._audio_device_combo.setEnabled(enabled)
        self._audio_refresh_btn.setEnabled(enabled)
        self._audio_sample_rate_combo.setEnabled(enabled)
        self._audio_channels_combo.setEnabled(enabled)
        self._audio_gain_slider.setEnabled(enabled)

    def _on_audio_gain_changed(self, value: int) -> None:
        """Handle gain slider change - update label text."""
        gain = value / 100.0
        self._audio_gain_label.setText(f"{gain:.1f}x")

    def _update_audio_status_label(self) -> None:
        """Show device count or error message."""
        recorder = AudioRecorder()
        if not recorder.is_available:
            self._audio_status_label.setText(
                "sounddevice library not available - audio recording disabled"
            )
        else:
            count = self._audio_device_combo.count() - 1  # Subtract "System Default"
            self._audio_status_label.setText(f"{count} audio input device(s) found")

    def _on_calibration_clicked(self) -> None:
        """Handle calibration button click."""
        self.calibration_requested.emit()

    def _on_zones_clicked(self) -> None:
        """Handle zones button click."""
        self.zones_requested.emit()

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

        # CV settings
        self._cv_enabled_cb.setChecked(self._cv_settings.enabled)

        for i in range(self._backend_combo.count()):
            if self._backend_combo.itemData(i) == self._cv_settings.backend:
                self._backend_combo.setCurrentIndex(i)
                break

        if self._cv_settings.model_path:
            self._model_path_edit.setText(self._cv_settings.model_path)

        self._confidence_spin.setValue(self._cv_settings.confidence_threshold)
        self._min_area_spin.setValue(self._cv_settings.min_detection_area)
        self._frame_skip_spin.setValue(self._cv_settings.process_every_n_frames)
        self._draw_overlays_cb.setChecked(self._cv_settings.draw_overlays)

        # Tracking settings
        self._tracking_enabled_cb.setChecked(self._cv_settings.tracking_enabled)
        self._max_disappeared_spin.setValue(self._cv_settings.max_disappeared)

        # Behavior settings
        self._behavior_enabled_cb.setChecked(self._cv_settings.behavior_enabled)
        self._freeze_threshold_spin.setValue(self._cv_settings.freeze_threshold)
        self._immobile_threshold_spin.setValue(self._cv_settings.immobile_threshold)
        self._dart_threshold_spin.setValue(self._cv_settings.dart_threshold)
        self._freeze_duration_spin.setValue(self._cv_settings.freeze_duration)
        self._smoothing_window_spin.setValue(self._cv_settings.smoothing_window)

        # Audio settings
        self._audio_enabled_cb.setChecked(self._audio_settings.enabled)

        # Set device selection
        for i in range(self._audio_device_combo.count()):
            if self._audio_device_combo.itemData(i) == self._audio_settings.device_index:
                self._audio_device_combo.setCurrentIndex(i)
                break

        # Set sample rate
        for i in range(self._audio_sample_rate_combo.count()):
            if self._audio_sample_rate_combo.itemData(i) == self._audio_settings.sample_rate:
                self._audio_sample_rate_combo.setCurrentIndex(i)
                break

        # Set channels
        for i in range(self._audio_channels_combo.count()):
            if self._audio_channels_combo.itemData(i) == self._audio_settings.channels:
                self._audio_channels_combo.setCurrentIndex(i)
                break

        # Set gain slider
        self._audio_gain_slider.setValue(int(self._audio_settings.gain * 100))
        self._audio_gain_label.setText(f"{self._audio_settings.gain:.1f}x")

        # Update UI state
        self._on_cv_enabled_toggle(self._cv_settings.enabled)
        self._on_backend_changed(self._backend_combo.currentIndex())
        self._on_tracking_enabled_toggle(self._cv_settings.tracking_enabled)
        self._on_behavior_enabled_toggle(self._cv_settings.behavior_enabled)
        self._on_audio_enabled_toggle(self._audio_settings.enabled)

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
        self._draw_tracks_cb.setEnabled(enabled)
        self._draw_contours_cb.setEnabled(enabled)
        if enabled:
            self._on_backend_changed(self._backend_combo.currentIndex())

    def _on_backend_changed(self, index: int):
        """Handle backend selection change."""
        backend = self._backend_combo.itemData(index)
        is_yolo = backend in (DetectionBackend.YOLO_V8, DetectionBackend.YOLO_BYTETRACK)
        self._model_path_edit.setVisible(is_yolo)
        self._model_path_label.setVisible(is_yolo)
        self._browse_model_btn.setVisible(is_yolo)

    def _on_tracking_enabled_toggle(self, enabled: bool):
        """Handle tracking enabled toggle."""
        self._max_disappeared_spin.setEnabled(enabled)
        self._max_distance_spin.setEnabled(enabled)

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
        """Handle LED power slider change - update label and apply live if streaming."""
        self._led_power_label.setText(f"{value}%")
        # Apply live if camera manager is available and connected
        if self._camera_manager is not None and self._camera_manager.is_connected:
            self._camera_manager.set_led_power(value)

    def _on_ewl_focus_changed(self, value: int):
        """Handle EWL focus slider change - update label and apply live if streaming."""
        self._ewl_focus_label.setText(str(value))
        # Apply live if camera manager is available and connected
        if self._camera_manager is not None and self._camera_manager.is_connected:
            self._camera_manager.set_ewl_focus(value)

    def _browse_model(self):
        """Browse for YOLO model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "PyTorch Models (*.pt);;All Files (*)"
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

        # CV settings
        self._cv_settings.enabled = self._cv_enabled_cb.isChecked()
        self._cv_settings.backend = self._backend_combo.currentData()
        self._cv_settings.model_path = self._model_path_edit.text() or None
        self._cv_settings.confidence_threshold = self._confidence_spin.value()
        self._cv_settings.min_detection_area = self._min_area_spin.value()
        self._cv_settings.process_every_n_frames = self._frame_skip_spin.value()
        self._cv_settings.draw_overlays = self._draw_overlays_cb.isChecked()
        self._cv_settings.tracking_enabled = self._tracking_enabled_cb.isChecked()
        self._cv_settings.max_disappeared = self._max_disappeared_spin.value()

        # Behavior settings
        self._cv_settings.behavior_enabled = self._behavior_enabled_cb.isChecked()
        self._cv_settings.freeze_threshold = self._freeze_threshold_spin.value()
        self._cv_settings.immobile_threshold = self._immobile_threshold_spin.value()
        self._cv_settings.dart_threshold = self._dart_threshold_spin.value()
        self._cv_settings.freeze_duration = self._freeze_duration_spin.value()
        self._cv_settings.smoothing_window = self._smoothing_window_spin.value()

        # Audio settings
        self._audio_settings.enabled = self._audio_enabled_cb.isChecked()
        self._audio_settings.device_index = self._audio_device_combo.currentData()
        self._audio_settings.device_name = self._audio_device_combo.currentText()
        self._audio_settings.sample_rate = self._audio_sample_rate_combo.currentData()
        self._audio_settings.channels = self._audio_channels_combo.currentData()
        self._audio_settings.gain = self._audio_gain_slider.value() / 100.0

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

    def get_audio_settings(self) -> AudioSettings:
        """Get the audio settings."""
        return self._audio_settings
