"""
Main Window - The primary PyQt6 window for GLIDER.

Manages the high-level layout and view switching between
Desktop (Builder) and Runner modes.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QMimeData, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QDrag, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from glider.core.config import get_config
from glider.gui.commands import (
    Command,
    CreateConnectionCommand,
    CreateNodeCommand,
    DeleteConnectionCommand,
    DeleteNodeCommand,
    MoveNodeCommand,
    PropertyChangeCommand,
    UndoStack,
)
from glider.gui.dialogs.calibration_dialog import CalibrationDialog
from glider.gui.dialogs.camera_settings_dialog import CameraSettingsDialog
from glider.gui.dialogs.experiment_dialog import ExperimentDialog
from glider.gui.dialogs.subject_dialog import SubjectDialog
from glider.gui.dialogs.zone_dialog import ZoneDialog
from glider.gui.node_graph.graph_view import NodeGraphView
from glider.gui.panels.camera_panel import CameraPanel
from glider.gui.view_manager import ViewManager, ViewMode
from glider.hal.base_board import BoardConnectionState
from glider.vision.zones import ZoneConfiguration

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)


class DraggableNodeButton(QPushButton):
    """A button that can be dragged to create nodes in the graph."""

    def __init__(self, node_type: str, display_name: str, category: str, parent=None):
        super().__init__(display_name, parent)
        self._node_type = node_type
        self._category = category

        # Use Qt properties for styling (defined in desktop.qss)
        self.setProperty("nodeCategory", category)
        self.setProperty("nodeButton", True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event):
        """Handle mouse press for drag initiation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for drag operation."""
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        if not hasattr(self, "_drag_start_pos"):
            return

        # Check if we've moved enough to start a drag
        if (event.pos() - self._drag_start_pos).manhattanLength() < 10:
            return

        # Create drag
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(f"node:{self._node_type}")
        drag.setMimeData(mime_data)

        # Execute drag
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        drag.exec(Qt.DropAction.CopyAction)
        self.setCursor(Qt.CursorShape.OpenHandCursor)


class EditableDraggableButton(DraggableNodeButton):
    """A draggable button with context menu for edit/delete."""

    def __init__(
        self,
        node_type: str,
        display_name: str,
        category: str,
        on_edit=None,
        on_delete=None,
        parent=None,
    ):
        super().__init__(node_type, display_name, category, parent)
        self._on_edit = on_edit
        self._on_delete = on_delete
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos):
        """Show context menu with edit/delete options."""
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self)

        edit_action = menu.addAction("Edit")
        edit_action.triggered.connect(self._handle_edit)

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(self._handle_delete)

        menu.exec(self.mapToGlobal(pos))

    def _handle_edit(self):
        """Handle edit action."""
        if self._on_edit:
            self._on_edit()

    def _handle_delete(self):
        """Handle delete action."""
        if self._on_delete:
            self._on_delete()

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to edit."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._on_edit:
                self._on_edit()
        else:
            super().mouseDoubleClickEvent(event)


class MainWindow(QMainWindow):
    """
    Main application window for GLIDER.

    Uses a QStackedWidget to switch between:
    - Index 0: Builder view (Desktop mode with node graph)
    - Index 1: Runner view (Touch-optimized dashboard)
    """

    # Signals
    session_changed = pyqtSignal()
    state_changed = pyqtSignal(str)  # Session state name
    error_occurred = pyqtSignal(str, str)  # source, message
    analog_value_received = pyqtSignal(int, int)  # pin, value - for real-time analog updates

    def __init__(
        self,
        core: "GliderCore",
        view_manager: Optional[ViewManager] = None,
        view_mode: ViewMode = ViewMode.AUTO,
    ):
        """
        Initialize the main window.

        Args:
            core: GliderCore instance
            view_manager: ViewManager instance (creates one if not provided)
            view_mode: Initial view mode (only used if view_manager not provided)
        """
        super().__init__()

        self._core = core
        if view_manager is not None:
            self._view_manager = view_manager
        else:
            self._view_manager = ViewManager(None)
            self._view_manager.mode = view_mode

        # Undo/Redo stack
        self._undo_stack = UndoStack()

        # Async task tracking to prevent garbage collection
        self._pending_tasks: set = set()

        # UI components
        self._stack: Optional[QStackedWidget] = None
        self._builder_view: Optional[QWidget] = None
        self._runner_view: Optional[QWidget] = None
        self._node_library_dock: Optional[QDockWidget] = None
        self._properties_dock: Optional[QDockWidget] = None

        # Zone configuration
        self._zone_config = ZoneConfiguration()

        # Setup UI
        self._setup_window()
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._connect_signals()

        # Apply stylesheet
        self._view_manager.apply_stylesheet(self)

        logger.info(f"MainWindow initialized in {self._view_manager.mode.name} mode")

    @property
    def core(self) -> "GliderCore":
        """Get the GliderCore instance."""
        return self._core

    @property
    def view_manager(self) -> ViewManager:
        """Get the view manager."""
        return self._view_manager

    @property
    def is_runner_mode(self) -> bool:
        """Whether in runner mode."""
        return self._view_manager.is_runner_mode

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle("GLIDER - General Laboratory Interface")
        config = get_config()

        if self._view_manager.is_runner_mode:
            # Runner mode: fullscreen, no frame
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            # Explicitly set geometry to fill the screen (more reliable on Pi)
            screen = QApplication.primaryScreen()
            if screen:
                self.setGeometry(screen.geometry())
            self.show()
        else:
            # Desktop mode: standard window
            self.setMinimumSize(config.ui.min_window_width, config.ui.min_window_height)
            self.resize(config.ui.default_window_width, config.ui.default_window_height)

    def _setup_ui(self) -> None:
        """Set up the main UI components."""
        # Central stacked widget
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Create views
        self._create_builder_view()
        self._create_runner_view()

        # Add to stack
        self._stack.addWidget(self._builder_view)  # Index 0
        self._stack.addWidget(self._runner_view)  # Index 1

        # Set initial view based on mode
        if self._view_manager.is_runner_mode:
            self._stack.setCurrentIndex(1)
        else:
            self._stack.setCurrentIndex(0)
            self._setup_dock_widgets()

    def _create_builder_view(self) -> None:
        """Create the builder (desktop) view."""
        self._builder_view = QWidget()
        layout = QVBoxLayout(self._builder_view)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter for graph and properties
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Node graph view (actual graph editor)
        self._graph_view = NodeGraphView()
        self._graph_view.setMinimumSize(400, 300)

        # Connect graph view signals
        self._graph_view.node_created.connect(self._on_node_created)
        self._graph_view.node_deleted.connect(self._on_node_deleted)
        self._graph_view.node_selected.connect(self._on_node_selected)
        self._graph_view.node_moved.connect(self._on_node_moved)
        self._graph_view.connection_created.connect(self._on_connection_created)
        self._graph_view.connection_deleted.connect(self._on_connection_deleted)

        splitter.addWidget(self._graph_view)
        layout.addWidget(splitter)

    def _create_runner_view(self) -> None:
        """Create the runner (dashboard) view optimized for 480x800 portrait."""
        self._runner_view = QWidget()
        self._runner_view.setObjectName("runnerView")
        layout = QVBoxLayout(self._runner_view)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # === Header Bar ===
        header = QWidget()
        header.setFixedHeight(50)
        header.setProperty("runnerHeader", True)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 4, 12, 4)

        # Experiment name (editable)
        self._runner_exp_name = QLineEdit("Untitled Experiment")
        self._runner_exp_name.setProperty("title", True)
        self._runner_exp_name.setPlaceholderText("Enter experiment name...")
        self._runner_exp_name.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 16px;
                font-weight: bold;
                color: white;
                min-width: 200px;
            }
            QLineEdit:hover {
                border: 1px solid #3d3d5c;
                background-color: rgba(45, 45, 68, 0.5);
            }
            QLineEdit:focus {
                border: 1px solid #4CAF50;
                background-color: #2d2d44;
            }
        """)
        self._runner_exp_name.textChanged.connect(self._on_experiment_name_changed)
        header_layout.addWidget(self._runner_exp_name)

        header_layout.addStretch()

        # Elapsed time timer
        self._runner_timer = QLabel("00:00")
        self._runner_timer.setProperty("timer", True)
        self._runner_timer.setStyleSheet("""
            QLabel[timer] {
                color: #4CAF50;
                font-size: 18px;
                font-weight: bold;
                font-family: "SF Mono", "Menlo", "Consolas", "Monaco", "Courier New";
                padding: 4px 8px;
                background-color: rgba(76, 175, 80, 0.1);
                border-radius: 4px;
            }
        """)
        header_layout.addWidget(self._runner_timer)

        # Status indicator - uses properties for styling
        self._status_label = QLabel("IDLE")
        self._status_label.setProperty("runnerStatus", True)
        self._status_label.setProperty("statusState", "IDLE")
        header_layout.addWidget(self._status_label)

        # Menu button for settings/exit
        self._runner_menu_btn = QPushButton("⚙️")
        # self._runner_menu_btn.setFixedSize(40, 40)
        self._runner_menu_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d44;
                border: none;
                border-radius: 8px;
                font-size: 20px;
                color: white;
            }
            QPushButton:pressed {
                background-color: #3d3d5c;
            }
        """)
        self._runner_menu_btn.clicked.connect(self._show_runner_menu)
        header_layout.addWidget(self._runner_menu_btn)

        layout.addWidget(header)

        # === Recording Indicator ===
        self._runner_recording = QLabel("● REC")
        self._runner_recording.setProperty("recording", True)
        self._runner_recording.setFixedHeight(28)
        self._runner_recording.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._runner_recording.hide()  # Hidden until recording starts
        layout.addWidget(self._runner_recording)

        # === Device Status Area (Scrollable) ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background-color: transparent;")

        # Enable kinetic scrolling
        from PyQt6.QtWidgets import QScroller

        QScroller.grabGesture(
            scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )

        # Content widget for device cards
        self._runner_devices_widget = QWidget()
        self._runner_devices_layout = QVBoxLayout(self._runner_devices_widget)
        self._runner_devices_layout.setContentsMargins(0, 0, 0, 0)
        self._runner_devices_layout.setSpacing(8)

        # Placeholder for devices
        self._runner_no_devices = QLabel("Connect hardware to see devices")
        self._runner_no_devices.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._runner_no_devices.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14px;
                padding: 40px;
            }
        """)
        self._runner_devices_layout.addWidget(self._runner_no_devices)
        self._runner_devices_layout.addStretch()

        scroll.setWidget(self._runner_devices_widget)
        layout.addWidget(scroll, 1)

        # === Control Buttons ===
        controls = QWidget()
        controls.setFixedHeight(160)
        controls.setProperty("runnerControls", True)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(8)

        # Top row: START and STOP
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self._start_btn = QPushButton("▶  START")
        self._start_btn.setFixedHeight(60)
        self._start_btn.setProperty("runnerAction", "start")
        self._start_btn.clicked.connect(self._on_start_clicked)
        top_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  STOP")
        self._stop_btn.setFixedHeight(60)
        self._stop_btn.setProperty("runnerAction", "stop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        top_row.addWidget(self._stop_btn)

        controls_layout.addLayout(top_row)

        # Bottom row: E-STOP (full width)
        self._emergency_btn = QPushButton("EMERGENCY STOP")
        self._emergency_btn.setFixedHeight(60)
        self._emergency_btn.setProperty("runnerAction", "emergency")
        self._emergency_btn.clicked.connect(self._on_emergency_stop)
        controls_layout.addWidget(self._emergency_btn)

        layout.addWidget(controls)

        # Store device widgets for updates
        self._runner_device_cards: dict[str, QWidget] = {}

        # Timer for periodic device state updates in runner view
        config = get_config()
        self._device_refresh_timer = QTimer()
        self._device_refresh_timer.setInterval(config.timing.device_refresh_interval_ms)
        self._device_refresh_timer.timeout.connect(self._update_runner_device_states)

        # Timer for elapsed time display
        self._experiment_start_time: Optional[float] = None
        self._elapsed_timer = QTimer()
        self._elapsed_timer.setInterval(config.timing.elapsed_timer_interval_ms)
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)

    def _setup_dock_widgets(self) -> None:
        """Set up dock widgets for desktop mode."""
        # Node Library dock
        self._node_library_dock = QDockWidget("Node Library", self)
        self._node_library_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        library_widget = self._create_node_library()
        self._node_library_dock.setWidget(library_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._node_library_dock)

        # Properties dock
        self._properties_dock = QDockWidget("Properties", self)
        self._properties_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        properties_widget = QWidget()
        properties_layout = QVBoxLayout(properties_widget)
        properties_layout.addWidget(QLabel("Select a node to view properties"))
        properties_layout.addStretch()
        self._properties_dock.setWidget(properties_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._properties_dock)

        # Hardware Panel dock
        self._hardware_dock = QDockWidget("Hardware", self)
        self._hardware_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        hardware_widget = QWidget()
        hardware_layout = QVBoxLayout(hardware_widget)
        hardware_layout.setContentsMargins(4, 4, 4, 4)

        # Hardware tree
        self._hardware_tree = QTreeWidget()
        self._hardware_tree.setHeaderLabels(["Name", "Type", "Status"])
        self._hardware_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._hardware_tree.customContextMenuRequested.connect(self._on_hardware_context_menu)
        hardware_layout.addWidget(self._hardware_tree)

        # Hardware buttons
        hw_btn_layout = QHBoxLayout()
        add_board_btn = QPushButton("+ Board")
        add_board_btn.clicked.connect(self._on_add_board)
        hw_btn_layout.addWidget(add_board_btn)

        add_device_btn = QPushButton("+ Device")
        add_device_btn.clicked.connect(self._on_add_device)
        hw_btn_layout.addWidget(add_device_btn)

        hardware_layout.addLayout(hw_btn_layout)

        self._hardware_dock.setWidget(hardware_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._hardware_dock)

        # Device Control Panel dock
        self._control_dock = QDockWidget("Device Control", self)
        self._control_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # Wrap in scroll area for touch screens
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        control_widget = QWidget()
        control_widget.setMinimumWidth(200)
        self._control_layout = QVBoxLayout(control_widget)
        self._control_layout.setContentsMargins(6, 6, 6, 6)
        self._control_layout.setSpacing(8)

        # Device selector row
        device_layout = QHBoxLayout()
        device_layout.setSpacing(6)
        device_label = QLabel("Device:")
        device_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._device_combo = QComboBox()
        self._device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._device_combo.currentTextChanged.connect(self._on_device_selected)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self._device_combo, 1)
        self._control_layout.addLayout(device_layout)

        # Output Controls group
        self._control_group = QGroupBox("Output Controls")
        self._control_group_layout = QVBoxLayout(self._control_group)
        self._control_group_layout.setContentsMargins(8, 12, 8, 8)
        self._control_group_layout.setSpacing(8)

        # Digital output controls (container widget for show/hide)
        self._digital_widget = QWidget()
        digital_layout = QHBoxLayout(self._digital_widget)
        digital_layout.setContentsMargins(0, 0, 0, 0)
        digital_layout.setSpacing(4)
        self._on_btn = QPushButton("ON")
        self._on_btn.setMinimumHeight(32)
        self._on_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._on_btn.clicked.connect(lambda: self._set_digital_output(True))
        self._off_btn = QPushButton("OFF")
        self._off_btn.setMinimumHeight(32)
        self._off_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._off_btn.clicked.connect(lambda: self._set_digital_output(False))
        self._toggle_btn = QPushButton("Toggle")
        self._toggle_btn.setMinimumHeight(32)
        self._toggle_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle_btn.clicked.connect(self._toggle_digital_output)
        digital_layout.addWidget(self._on_btn)
        digital_layout.addWidget(self._off_btn)
        digital_layout.addWidget(self._toggle_btn)
        self._control_group_layout.addWidget(self._digital_widget)

        # PWM control row (container widget for show/hide)
        self._pwm_widget = QWidget()
        pwm_layout = QHBoxLayout(self._pwm_widget)
        pwm_layout.setContentsMargins(0, 0, 0, 0)
        pwm_layout.setSpacing(6)
        pwm_label = QLabel("PWM:")
        pwm_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._pwm_spinbox = QSpinBox()
        self._pwm_spinbox.setRange(0, 255)
        self._pwm_spinbox.setMinimumHeight(35)
        self._pwm_spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._pwm_spinbox.valueChanged.connect(self._on_pwm_changed)
        # Hidden slider for compatibility
        self._pwm_slider = QSlider(Qt.Orientation.Horizontal)
        self._pwm_slider.setRange(0, 255)
        self._pwm_slider.hide()
        self._pwm_spinbox.valueChanged.connect(self._pwm_slider.setValue)
        self._pwm_slider.valueChanged.connect(self._pwm_spinbox.setValue)
        pwm_layout.addWidget(pwm_label)
        pwm_layout.addWidget(self._pwm_spinbox, 1)
        self._control_group_layout.addWidget(self._pwm_widget)
        self._pwm_widget.hide()

        self._control_layout.addWidget(self._control_group)

        # Input Reading group (separate from Output Controls)
        input_group = QGroupBox("Input Reading")
        input_group_layout = QVBoxLayout(input_group)
        input_group_layout.setContentsMargins(8, 12, 8, 8)
        input_group_layout.setSpacing(8)

        # Value display
        self._input_value_label = QLabel("--")
        self._input_value_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding: 6px; "
            "background-color: #2d2d2d; border-radius: 4px; color: #00ff00;"
        )
        self._input_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._input_value_label.setMinimumHeight(48)
        self._input_value_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        input_group_layout.addWidget(self._input_value_label)

        # Read controls row
        input_row = QHBoxLayout()
        input_row.setSpacing(4)
        self._read_btn = QPushButton("Read")
        self._read_btn.setMinimumHeight(32)
        self._read_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._read_btn.clicked.connect(self._read_input_once)
        input_row.addWidget(self._read_btn)

        self._continuous_checkbox = QCheckBox("Auto")
        self._continuous_checkbox.setMinimumWidth(35)
        self._continuous_checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._continuous_checkbox.stateChanged.connect(self._on_continuous_changed)
        input_row.addWidget(self._continuous_checkbox)

        self._poll_spinbox = QSpinBox()
        self._poll_spinbox.setRange(50, 5000)
        self._poll_spinbox.setValue(100)
        self._poll_spinbox.setSuffix("ms")
        self._poll_spinbox.setMinimumWidth(75)
        self._poll_spinbox.setMinimumHeight(28)
        self._poll_spinbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._poll_spinbox.valueChanged.connect(self._on_poll_interval_changed)
        input_row.addWidget(self._poll_spinbox)

        input_group_layout.addLayout(input_row)
        self._input_group = input_group
        self._control_layout.addWidget(input_group)

        # Timer for continuous reading (fallback for digital inputs)
        self._input_poll_timer = QTimer(self)
        self._input_poll_timer.timeout.connect(self._poll_input)

        # Real-time callback tracking for analog inputs
        self._analog_callback_board = None  # Board with registered callback
        self._analog_callback_pin = None  # Pin with registered callback
        self._analog_callback_func = None  # The callback function reference
        self.analog_value_received.connect(self._on_analog_value_received)

        # Status display
        self._device_status_label = QLabel("No device selected")
        self._device_status_label.setStyleSheet("font-size: 11px; color: #888; padding: 2px;")
        self._device_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._control_layout.addWidget(self._device_status_label)

        self._control_layout.addStretch()

        # Set scroll area content and dock widget
        control_scroll.setWidget(control_widget)
        self._control_dock.setWidget(control_scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._control_dock)

        # Stack the control dock below the hardware dock
        self.tabifyDockWidget(self._hardware_dock, self._control_dock)
        self._hardware_dock.raise_()

        # Camera Panel dock
        self._camera_dock = QDockWidget("Camera", self)
        self._camera_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._camera_panel = CameraPanel(
            self._core.camera_manager,
            self._core.cv_processor,
            multi_camera_manager=self._core.multi_camera_manager,
        )
        self._camera_panel.settings_requested.connect(self._on_camera_settings)
        self._camera_panel.set_video_recorder(self._core.video_recorder)
        self._camera_panel.set_multi_video_recorder(self._core.multi_video_recorder)
        self._camera_panel.set_tracking_logger(self._core.tracking_logger)
        self._camera_panel.set_calibration(self._core.calibration)
        self._camera_panel._preview.set_calibration(self._core.calibration)
        # Set zone configuration if available
        self._camera_panel.set_zone_configuration(self._zone_config)
        self._camera_dock.setWidget(self._camera_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._camera_dock)

        # Files dock (for Pi touchscreen access to file operations)
        self._files_dock = QDockWidget("Files", self)
        self._files_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # Wrap in scroll area for touch screens
        files_scroll = QScrollArea()
        files_scroll.setWidgetResizable(True)
        files_scroll.setFrameShape(QFrame.Shape.NoFrame)
        files_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        files_widget = QWidget()
        files_layout = QVBoxLayout(files_widget)
        files_layout.setContentsMargins(4, 4, 4, 4)
        files_layout.setSpacing(4)

        # File operation buttons - compact for Pi
        new_btn = QPushButton("New")
        new_btn.setFixedHeight(36)
        new_btn.clicked.connect(self._on_new)
        files_layout.addWidget(new_btn)

        open_btn = QPushButton("Open")
        open_btn.setFixedHeight(36)
        open_btn.clicked.connect(self._on_open)
        files_layout.addWidget(open_btn)

        save_btn = QPushButton("Save")
        save_btn.setFixedHeight(36)
        save_btn.clicked.connect(self._on_save)
        files_layout.addWidget(save_btn)

        save_as_btn = QPushButton("Save As...")
        save_as_btn.setFixedHeight(36)
        save_as_btn.clicked.connect(self._on_save_as)
        files_layout.addWidget(save_as_btn)

        files_layout.addStretch()

        files_scroll.setWidget(files_widget)
        self._files_dock.setWidget(files_scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._files_dock)

        # Hide Files dock by default on desktop - user can enable via View menu
        # (Files dock is mainly for Pi touchscreen where menu isn't easily accessible)
        if not self._view_manager.is_runner_mode:
            self._files_dock.setVisible(False)

        # Experiment dialog (opened from menu)
        self._experiment_dialog: ExperimentDialog | None = None

        # Refresh hardware tree (which also refreshes the device combo)
        self._refresh_hardware_tree()

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        if self._view_manager.is_runner_mode:
            return  # No menu in runner mode

        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new)
        file_menu.addAction(new_action)

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        self._undo_action = QAction("&Undo", self)
        self._undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_action.triggered.connect(self._on_undo)
        self._undo_action.setEnabled(False)
        edit_menu.addAction(self._undo_action)

        self._redo_action = QAction("&Redo", self)
        self._redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self._redo_action.triggered.connect(self._on_redo)
        self._redo_action.setEnabled(False)
        edit_menu.addAction(self._redo_action)

        # Experiment menu
        experiment_menu = menubar.addMenu("E&xperiment")

        experiment_settings_action = QAction("Experiment &Settings...", self)
        experiment_settings_action.triggered.connect(self._on_open_experiment_dialog)
        experiment_menu.addAction(experiment_settings_action)

        experiment_menu.addSeparator()

        add_subject_action = QAction("&Add Subject...", self)
        add_subject_action.triggered.connect(lambda: self._on_edit_subject(""))
        experiment_menu.addAction(add_subject_action)

        experiment_menu.addSeparator()

        analyze_data_action = QAction("&Analyze Data...", self)
        analyze_data_action.triggered.connect(self._on_open_analysis_dialog)
        experiment_menu.addAction(analyze_data_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        switch_view_action = QAction("Switch to &Runner View", self)
        switch_view_action.setShortcut(QKeySequence("F11"))
        switch_view_action.triggered.connect(self._toggle_view)
        view_menu.addAction(switch_view_action)

        view_menu.addSeparator()

        # Panel visibility toggles (only in desktop mode)
        if hasattr(self, "_node_library_dock"):
            node_library_action = self._node_library_dock.toggleViewAction()
            node_library_action.setText("&Node Library")
            view_menu.addAction(node_library_action)

        if hasattr(self, "_properties_dock"):
            properties_action = self._properties_dock.toggleViewAction()
            properties_action.setText("&Properties Panel")
            view_menu.addAction(properties_action)

        if hasattr(self, "_hardware_dock"):
            hardware_action = self._hardware_dock.toggleViewAction()
            hardware_action.setText("&Hardware Panel")
            view_menu.addAction(hardware_action)

        if hasattr(self, "_control_dock"):
            control_action = self._control_dock.toggleViewAction()
            control_action.setText("&Device Control")
            view_menu.addAction(control_action)

        if hasattr(self, "_camera_dock"):
            camera_action = self._camera_dock.toggleViewAction()
            camera_action.setText("&Camera Panel")
            view_menu.addAction(camera_action)

        if hasattr(self, "_files_dock"):
            files_action = self._files_dock.toggleViewAction()
            files_action.setText("&Files Panel")
            view_menu.addAction(files_action)

        view_menu.addSeparator()

        # Layout presets
        pi_view_action = QAction("&Pi Touchscreen (Tabbed)", self)
        pi_view_action.triggered.connect(self._set_pi_touchscreen_layout)
        view_menu.addAction(pi_view_action)

        compact_view_action = QAction("&Compact (1024x768)", self)
        compact_view_action.triggered.connect(lambda: self._set_window_size(1024, 768))
        view_menu.addAction(compact_view_action)

        default_view_action = QAction("&Default Layout", self)
        default_view_action.triggered.connect(self._set_default_layout)
        view_menu.addAction(default_view_action)

        # Hardware menu
        hardware_menu = menubar.addMenu("&Hardware")

        add_board_action = QAction("Add &Board...", self)
        add_board_action.triggered.connect(self._on_add_board)
        hardware_menu.addAction(add_board_action)

        add_device_action = QAction("Add &Device...", self)
        add_device_action.triggered.connect(self._on_add_device)
        hardware_menu.addAction(add_device_action)

        hardware_menu.addSeparator()

        connect_action = QAction("&Connect All", self)
        connect_action.triggered.connect(self._on_connect_hardware)
        hardware_menu.addAction(connect_action)

        disconnect_action = QAction("&Disconnect All", self)
        disconnect_action.triggered.connect(self._on_disconnect_hardware)
        hardware_menu.addAction(disconnect_action)

        # Run menu
        run_menu = menubar.addMenu("&Run")

        start_action = QAction("&Start", self)
        start_action.setShortcut(QKeySequence("F5"))
        start_action.triggered.connect(self._on_start_clicked)
        run_menu.addAction(start_action)

        stop_action = QAction("S&top", self)
        stop_action.setShortcut(QKeySequence("Shift+F5"))
        stop_action.triggered.connect(self._on_stop_clicked)
        run_menu.addAction(stop_action)

        run_menu.addSeparator()

        emergency_action = QAction("&Emergency Stop", self)
        emergency_action.setShortcut(QKeySequence("Ctrl+Shift+Escape"))
        emergency_action.triggered.connect(self._on_emergency_stop)
        run_menu.addAction(emergency_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About GLIDER", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self) -> None:
        """Set up the toolbar."""
        if self._view_manager.is_runner_mode:
            return

        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Add toolbar actions with proper connections
        new_action = toolbar.addAction("New")
        new_action.triggered.connect(self._on_new)

        open_action = toolbar.addAction("Open")
        open_action.triggered.connect(self._on_open)

        save_action = toolbar.addAction("Save")
        save_action.triggered.connect(self._on_save)

        toolbar.addSeparator()

        connect_action = toolbar.addAction("Connect")
        connect_action.triggered.connect(self._on_connect_hardware)

        toolbar.addSeparator()

        start_action = toolbar.addAction("Start")
        start_action.triggered.connect(self._on_start_clicked)

        stop_action = toolbar.addAction("Stop")
        stop_action.triggered.connect(self._on_stop_clicked)

        # Add spacer to push status to the right
        spacer = QWidget()
        spacer.setSizePolicy(
            spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy()
        )
        from PyQt6.QtWidgets import QSizePolicy

        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # Persistent status indicator - uses properties for styling
        self._toolbar_status = QLabel("IDLE")
        self._toolbar_status.setProperty("statusIndicator", True)
        self._toolbar_status.setProperty("statusState", "IDLE")
        toolbar.addWidget(self._toolbar_status)

    def _setup_status_bar(self) -> None:
        """Set up the status bar."""
        if self._view_manager.is_runner_mode:
            return

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("Ready")

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Connect core callbacks
        self._core.on_state_change(self._on_core_state_change)
        self._core.on_error(self._on_core_error)

        # Connect hardware connection change callback for disconnect handling
        self._core.hardware_manager.on_connection_change(self._on_hardware_connection_change)

        # Connect session changed to update runner view
        self.session_changed.connect(self._update_runner_experiment_name)

    def _on_core_state_change(self, state) -> None:
        """Handle core state changes."""
        state_name = state.name
        self.state_changed.emit(state_name)

        # Update runner view status label - uses property for color (defined in QSS)
        if hasattr(self, "_status_label"):
            self._status_label.setText(state_name)
            self._status_label.setProperty("statusState", state_name)
            self._status_label.style().unpolish(self._status_label)
            self._status_label.style().polish(self._status_label)

        # Update toolbar status indicator - uses property for color (defined in QSS)
        if hasattr(self, "_toolbar_status"):
            self._toolbar_status.setText(state_name)
            self._toolbar_status.setProperty("statusState", state_name)
            self._toolbar_status.style().unpolish(self._toolbar_status)
            self._toolbar_status.style().polish(self._toolbar_status)

        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage(f"State: {state_name}")

        # Update runner view recording indicator
        if hasattr(self, "_runner_recording"):
            if state_name == "RUNNING" and self._core.data_recorder.is_recording:
                self._runner_recording.show()
            else:
                self._runner_recording.hide()

        # Start/stop device refresh timer based on state
        if hasattr(self, "_device_refresh_timer"):
            if state_name == "RUNNING":
                self._device_refresh_timer.start()
            else:
                self._device_refresh_timer.stop()
                # Do one final refresh when stopping
                self._update_runner_device_states()

        # Start/stop elapsed timer based on state
        if hasattr(self, "_elapsed_timer"):
            if state_name == "RUNNING":
                import time

                self._experiment_start_time = time.time()
                self._elapsed_timer.start()
                self._update_elapsed_time()  # Immediate update
            else:
                self._elapsed_timer.stop()
                if hasattr(self, "_runner_timer"):
                    # Keep the last time displayed when stopped
                    pass

    def _refresh_runner_devices(self) -> None:
        """Refresh the device cards in runner view."""
        if not hasattr(self, "_runner_devices_layout"):
            return

        # Clear existing device cards
        for card in self._runner_device_cards.values():
            card.setParent(None)
            card.deleteLater()
        self._runner_device_cards.clear()

        # Get devices from hardware manager
        devices = self._core.hardware_manager.devices

        if not devices:
            self._runner_no_devices.show()
            return

        self._runner_no_devices.hide()

        # Create a card for each device
        for device_id, device in devices.items():
            card = self._create_device_card(device_id, device)
            # Insert before the stretch
            self._runner_devices_layout.insertWidget(self._runner_devices_layout.count() - 1, card)
            self._runner_device_cards[device_id] = card

    def _create_device_card(self, device_id: str, device) -> QWidget:
        """Create a device status card for the runner view."""
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                background-color: #1a1a2e;
                border: 2px solid #2d2d44;
                border-radius: 12px;
            }
        """)
        card.setFixedHeight(80)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Device info (left side)
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        name_label = QLabel(device_id)
        name_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #fff; background: transparent; border: none;"
        )
        info_layout.addWidget(name_label)

        device_type = getattr(device, "device_type", "Unknown")
        type_label = QLabel(device_type)
        type_label.setStyleSheet(
            "font-size: 12px; color: #888; background: transparent; border: none;"
        )
        info_layout.addWidget(type_label)

        layout.addLayout(info_layout)
        layout.addStretch()

        # Status indicator (right side)
        initialized = getattr(device, "_initialized", False)
        device_type = getattr(device, "device_type", "Unknown")

        # Check if this is an analog input device
        is_analog_input = device_type == "AnalogInput"

        status_widget = QWidget()
        status_widget.setFixedSize(80 if is_analog_input else 60, 50)
        status_layout = QVBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(2)

        # State value
        if is_analog_input:
            # For analog inputs, show the raw value and voltage
            last_value = getattr(device, "_last_value", None)
            if last_value is not None:
                # Calculate voltage (assuming 10-bit ADC, 5V reference)
                voltage = (last_value / 1023.0) * 5.0
                state_text = f"{last_value}\n{voltage:.2f}V"
                state_color = "#3498db"
            else:
                state_text = "---"
                state_color = "#444"
        else:
            # For other devices, use the state
            state = getattr(device, "_state", None)
            if state is not None:
                if isinstance(state, bool):
                    state_text = "HIGH" if state else "LOW"
                    state_color = "#27ae60" if state else "#7f8c8d"
                else:
                    state_text = str(state)[:6]
                    state_color = "#3498db"
            else:
                state_text = "---"
                state_color = "#444"

        state_label = QLabel(state_text)
        state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_label.setStyleSheet(f"""
            QLabel {{
                background-color: {state_color};
                color: white;
                font-size: {'11px' if is_analog_input else '14px'};
                font-weight: bold;
                border-radius: 8px;
                padding: 4px 8px;
                border: none;
                line-height: 1.2;
            }}
        """)
        status_layout.addWidget(state_label)

        # Ready indicator
        ready_label = QLabel("Ready" if initialized else "---")
        ready_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ready_label.setStyleSheet(
            f"font-size: 10px; color: {'#27ae60' if initialized else '#666'}; background: transparent; border: none;"
        )
        status_layout.addWidget(ready_label)

        layout.addWidget(status_widget)

        # Store references for updates
        card._state_label = state_label
        card._ready_label = ready_label

        return card

    def _update_runner_experiment_name(self) -> None:
        """Update the experiment name in runner view from session."""
        if not hasattr(self, "_runner_exp_name"):
            return

        # Block signals to prevent feedback loop
        self._runner_exp_name.blockSignals(True)
        if self._core.session and self._core.session.metadata.name:
            self._runner_exp_name.setText(self._core.session.metadata.name)
        else:
            self._runner_exp_name.setText("Untitled Experiment")
        self._runner_exp_name.blockSignals(False)

    def _on_experiment_name_changed(self, name: str) -> None:
        """Handle experiment name change from user input."""
        if self._core.session:
            self._core.session.metadata.name = name
            self._core.session._dirty = True

    def _update_elapsed_time(self) -> None:
        """Update the elapsed time display in runner view."""
        if not hasattr(self, "_runner_timer") or self._experiment_start_time is None:
            return

        import time

        elapsed = time.time() - self._experiment_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        if hours > 0:
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            time_str = f"{minutes:02d}:{seconds:02d}"

        self._runner_timer.setText(time_str)

    def _update_runner_device_states(self) -> None:
        """Update the device state displays in runner view."""
        if not hasattr(self, "_runner_device_cards"):
            return

        for device_id, card in self._runner_device_cards.items():
            device = self._core.hardware_manager.get_device(device_id)
            if device is None:
                continue

            # Get device info
            initialized = getattr(device, "_initialized", False)
            device_type = getattr(device, "device_type", "Unknown")
            is_analog_input = device_type == "AnalogInput"

            # Update state label
            if hasattr(card, "_state_label"):
                if is_analog_input:
                    # For analog inputs, show raw value and voltage
                    last_value = getattr(device, "_last_value", None)
                    if last_value is not None:
                        # Calculate voltage (assuming 10-bit ADC, 5V reference)
                        voltage = (last_value / 1023.0) * 5.0
                        state_text = f"{last_value}\n{voltage:.2f}V"
                        state_color = "#3498db"
                        font_size = "11px"
                    else:
                        state_text = "---"
                        state_color = "#444"
                        font_size = "11px"
                else:
                    # For other devices, use the state
                    state = getattr(device, "_state", None)
                    if state is not None:
                        if isinstance(state, bool):
                            state_text = "HIGH" if state else "LOW"
                            state_color = "#27ae60" if state else "#7f8c8d"
                        else:
                            state_text = str(state)[:6]
                            state_color = "#3498db"
                    else:
                        state_text = "---"
                        state_color = "#444"
                    font_size = "14px"

                card._state_label.setText(state_text)
                card._state_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {state_color};
                        color: white;
                        font-size: {font_size};
                        font-weight: bold;
                        border-radius: 8px;
                        padding: 4px 8px;
                        border: none;
                        line-height: 1.2;
                    }}
                """)

            # Update ready label
            if hasattr(card, "_ready_label"):
                card._ready_label.setText("Ready" if initialized else "---")
                card._ready_label.setStyleSheet(
                    f"font-size: 10px; color: {'#27ae60' if initialized else '#666'}; background: transparent; border: none;"
                )

    def _on_core_error(self, source: str, error: Exception) -> None:
        """Handle core errors."""
        self.error_occurred.emit(source, str(error))
        logger.error(f"Error from {source}: {error}")

    def _on_hardware_connection_change(self, board_id: str, state: BoardConnectionState) -> None:
        """
        Handle hardware connection state changes.

        If a board disconnects while running, pause and prompt the user.
        """
        # Only handle disconnection during active experiment
        if state not in (BoardConnectionState.DISCONNECTED, BoardConnectionState.ERROR):
            return

        # Check if we're running an experiment
        if not hasattr(self._core, "state"):
            return

        from glider.core.glider_core import SessionState

        if self._core.state != SessionState.RUNNING:
            # Not running, just log it
            logger.warning(f"Board {board_id} disconnected (state: {state.name})")
            return

        # We're running and a board disconnected - pause and prompt
        logger.warning(f"Board {board_id} disconnected during experiment! Pausing...")

        # Pause the experiment
        self._run_async(self._core.pause())

        # Show the disconnection dialog
        self._show_hardware_disconnection_dialog(board_id, state)

    def _show_hardware_disconnection_dialog(
        self, board_id: str, state: BoardConnectionState
    ) -> None:
        """Show a dialog when hardware disconnects during an experiment."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Hardware Disconnected")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(16)

        # Warning icon and message
        header_layout = QHBoxLayout()
        warning_label = QLabel("Warning")
        warning_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #f39c12;")
        header_layout.addWidget(warning_label)

        message = QLabel(
            f"<b>Board '{board_id}' has disconnected.</b><br><br>"
            f"The experiment has been paused. What would you like to do?"
        )
        message.setWordWrap(True)
        header_layout.addWidget(message, 1)
        layout.addLayout(header_layout)

        # Status info
        status_label = QLabel(f"Connection state: {state.name}")
        status_label.setStyleSheet("color: #888;")
        layout.addWidget(status_label)

        # Buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        retry_btn = QPushButton("Retry Connection")
        retry_btn.setMinimumHeight(40)
        retry_btn.clicked.connect(lambda: self._handle_disconnection_retry(dialog, board_id))
        button_layout.addWidget(retry_btn)

        continue_btn = QPushButton("Continue Without Hardware")
        continue_btn.setMinimumHeight(40)
        continue_btn.setToolTip("Resume the experiment without this board (may cause errors)")
        continue_btn.clicked.connect(lambda: self._handle_disconnection_continue(dialog))
        button_layout.addWidget(continue_btn)

        stop_btn = QPushButton("Stop Experiment")
        stop_btn.setMinimumHeight(40)
        stop_btn.clicked.connect(lambda: self._handle_disconnection_stop(dialog))
        button_layout.addWidget(stop_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def _handle_disconnection_retry(self, dialog: QDialog, board_id: str) -> None:
        """Attempt to reconnect to the disconnected board."""
        dialog.accept()

        # Show progress
        self.statusBar().showMessage(f"Reconnecting to {board_id}...")

        async def retry_connection():
            try:
                success = await self._core.hardware_manager.connect_board(board_id)
                if success:
                    logger.info(f"Reconnected to board {board_id}")
                    self.statusBar().showMessage(f"Reconnected to {board_id}. Resuming...", 3000)
                    # Resume the experiment
                    await self._core.resume()
                else:
                    logger.warning(f"Failed to reconnect to board {board_id}")
                    self.statusBar().showMessage(f"Failed to reconnect to {board_id}", 5000)
                    # Show dialog again
                    self._show_hardware_disconnection_dialog(
                        board_id, BoardConnectionState.DISCONNECTED
                    )
            except Exception as e:
                logger.error(f"Error reconnecting to {board_id}: {e}")
                self.statusBar().showMessage(f"Error: {e}", 5000)
                self._show_hardware_disconnection_dialog(board_id, BoardConnectionState.ERROR)

        self._run_async(retry_connection())

    def _handle_disconnection_continue(self, dialog: QDialog) -> None:
        """Continue the experiment without the disconnected hardware."""
        dialog.accept()

        reply = QMessageBox.warning(
            self,
            "Continue Without Hardware",
            "Continuing without the disconnected hardware may cause errors "
            "or unexpected behavior. Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.warning("User chose to continue experiment without disconnected hardware")
            self._run_async(self._core.resume())

    def _handle_disconnection_stop(self, dialog: QDialog) -> None:
        """Stop the experiment due to hardware disconnection."""
        dialog.accept()
        logger.info("User stopped experiment due to hardware disconnection")
        self._run_async(self._core.stop())
        self.statusBar().showMessage("Experiment stopped", 3000)

    def _toggle_view(self) -> None:
        """Toggle between builder and runner views."""
        current = self._stack.currentIndex()
        self._stack.setCurrentIndex(1 if current == 0 else 0)

    def switch_to_builder(self) -> None:
        """Switch to builder view."""
        self._stack.setCurrentIndex(0)

    def switch_to_runner(self) -> None:
        """Switch to runner view."""
        self._stack.setCurrentIndex(1)

    def _set_window_size(self, width: int, height: int) -> None:
        """Set the window to a specific size."""
        # Adjust minimum size to allow smaller dimensions (e.g., Pi display)
        self.setMinimumSize(min(width, 480), min(height, 480))
        self.resize(width, height)
        self.statusBar().showMessage(f"Window resized to {width}x{height}", 2000)

    def _set_pi_touchscreen_layout(self) -> None:
        """Set up Pi Touchscreen layout with tabbed panels.

        Pi desktop mode focuses on hardware management and experiment running,
        not flow creation. Shows runner view with dock panels for configuration.
        """
        # Resize window for Pi display
        self.setMinimumSize(480, 480)
        self.resize(480, 800)

        # Show runner view as main content (Pi is for running, not creating flows)
        if hasattr(self, "_stack") and self._stack is not None:
            self._stack.setCurrentIndex(1)  # Runner view

        # Collect dock widgets relevant for Pi: Files, Hardware, Control, Camera
        # Excludes: Node Library, Properties (flow creation), Agent
        docks = []
        if getattr(self, "_files_dock", None) is not None:
            docks.append(self._files_dock)
        if getattr(self, "_hardware_dock", None) is not None:
            docks.append(self._hardware_dock)
        if getattr(self, "_control_dock", None) is not None:
            docks.append(self._control_dock)
        if getattr(self, "_camera_dock", None) is not None:
            docks.append(self._camera_dock)

        # Hide flow-creation docks on Pi
        if getattr(self, "_node_library_dock", None) is not None:
            self._node_library_dock.setVisible(False)
        if getattr(self, "_properties_dock", None) is not None:
            self._properties_dock.setVisible(False)

        if len(docks) < 2:
            return

        # Move all docks to bottom area and lock them (no dragging/floating on Pi)
        for dock in docks:
            dock.setVisible(True)
            # Disable dragging and floating - only allow closing via tab
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
            dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        # Tabify all docks together (stack them as tabs)
        first_dock = docks[0]
        for dock in docks[1:]:
            self.tabifyDockWidget(first_dock, dock)

        # Make tabs span the full width of the screen
        for tab_bar in self.findChildren(QTabBar):
            tab_bar.setExpanding(True)

        # Raise the first dock to make it visible
        first_dock.raise_()

        self.statusBar().showMessage("Pi Touchscreen layout applied", 2000)
        logger.info("Applied Pi Touchscreen tabbed layout")

    def _set_default_layout(self) -> None:
        """Restore default desktop layout."""
        self.resize(1400, 900)

        # Default dock features for desktop (movable, floatable, closable)
        default_features = (
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        default_areas = Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea

        # Restore dock positions and features
        if getattr(self, "_node_library_dock", None) is not None:
            self._node_library_dock.setFeatures(default_features)
            self._node_library_dock.setAllowedAreas(default_areas)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._node_library_dock)
            self._node_library_dock.setVisible(True)

        if getattr(self, "_properties_dock", None) is not None:
            self._properties_dock.setFeatures(default_features)
            self._properties_dock.setAllowedAreas(default_areas)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._properties_dock)
            self._properties_dock.setVisible(True)

        if getattr(self, "_hardware_dock", None) is not None:
            self._hardware_dock.setFeatures(default_features)
            self._hardware_dock.setAllowedAreas(default_areas)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._hardware_dock)
            self._hardware_dock.setVisible(True)

        if getattr(self, "_control_dock", None) is not None:
            self._control_dock.setFeatures(default_features)
            self._control_dock.setAllowedAreas(
                default_areas | Qt.DockWidgetArea.BottomDockWidgetArea
            )
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._control_dock)
            if getattr(self, "_hardware_dock", None) is not None:
                self.tabifyDockWidget(self._hardware_dock, self._control_dock)
                self._hardware_dock.raise_()

        if getattr(self, "_camera_dock", None) is not None:
            self._camera_dock.setFeatures(default_features)
            self._camera_dock.setAllowedAreas(default_areas)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._camera_dock)
            self._camera_dock.setVisible(True)

        # Hide files dock on large screens (menu/toolbar available)
        if getattr(self, "_files_dock", None) is not None:
            self._files_dock.setVisible(False)

        self.statusBar().showMessage("Default layout restored", 2000)
        logger.info("Restored default desktop layout")

    # File operations
    def _on_new(self) -> None:
        """Create new experiment."""
        if self._check_save():
            self._core.hardware_manager.clear()
            self._core.new_session()
            self._graph_view.clear_graph()
            # Clear zone configuration
            self._zone_config = ZoneConfiguration()
            self._camera_panel.set_zone_configuration(self._zone_config)
            self.session_changed.emit()
            self._refresh_hardware_tree()
            self._refresh_custom_devices()
            self._refresh_flow_functions()
            self._refresh_zones()
            if self._experiment_dialog:
                self._experiment_dialog.set_session(self._core.session)

    def _on_open(self) -> None:
        """Open experiment file."""
        if not self._check_save():
            return

        # Remember if we were in runner mode (fullscreen/frameless)
        was_runner_mode = self._view_manager.is_runner_mode

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Experiment",
            "",
            "GLIDER Experiments (*.glider);;JSON Files (*.json);;All Files (*)",
        )

        # Restore fullscreen/frameless state on Pi after dialog closes
        if was_runner_mode:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.showFullScreen()

        if file_path:
            try:
                self._core.load_session(file_path)
                # Populate hardware manager from session
                self._populate_hardware_from_session()
                # Populate graph view from session
                self._populate_graph_from_session()
                # Load zone configuration from session
                self._load_zones_from_session()
                self.session_changed.emit()
                self._refresh_hardware_tree()
                self._refresh_custom_devices()
                self._refresh_flow_functions()
                self._refresh_zones()
                if self._experiment_dialog:
                    self._experiment_dialog.set_session(self._core.session)
                # Apply recording directory from loaded session
                rec_dir = self._core.session.metadata.recording_directory
                if rec_dir:
                    self._core.set_recording_directory(Path(rec_dir))
                self.statusBar().showMessage(f"Opened: {file_path}")
            except Exception as e:
                logger.exception(f"Failed to open file: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open file: {e}")

    def _populate_hardware_from_session(self) -> None:
        """Populate hardware manager from session configuration."""
        if not self._core.session:
            return

        # Clear existing hardware
        self._core.hardware_manager.clear()

        # Add boards from session
        for board_config in self._core.session.hardware.boards:
            try:
                # Map driver type to the type expected by add_board
                board_type = "telemetrix" if board_config.driver_type == "arduino" else "pigpio"
                self._core.hardware_manager.add_board(
                    board_config.id,
                    board_type,
                    port=board_config.port,
                )
            except Exception as e:
                logger.warning(f"Failed to add board {board_config.id}: {e}")

        # Add devices from session
        for device_config in self._core.session.hardware.devices:
            try:
                # Get the pin name and value from the pins dict
                # Pins dict is like {"output": 13} or {"input": 2}
                if device_config.pins:
                    pin_name = list(device_config.pins.keys())[0]
                    pin = device_config.pins[pin_name]
                else:
                    pin_name = "pin"
                    pin = 0

                self._core.hardware_manager.add_device(
                    device_config.id,
                    device_config.device_type,
                    device_config.board_id,
                    pin,
                    name=device_config.name,
                    pin_name=pin_name,
                )
            except Exception as e:
                logger.warning(f"Failed to add device {device_config.id}: {e}")

    def _populate_graph_from_session(self) -> None:
        """Populate graph view from session flow configuration."""
        if not self._core.session:
            return

        # Clear existing graph
        self._graph_view.clear_graph()

        # Create visual nodes from session
        for node_config in self._core.session.flow.nodes:
            try:
                x, y = node_config.position
                node_type = node_config.node_type
                display_name = node_type
                definition_id = None

                # Handle CustomDevice/CustomDeviceAction nodes - get display name from definition
                if node_type in ("CustomDevice", "CustomDeviceAction"):
                    definition_id = (
                        node_config.state.get("definition_id") if node_config.state else None
                    )
                    if definition_id:
                        def_dict = self._core.session.get_custom_device_definition(definition_id)
                        if def_dict:
                            display_name = def_dict.get("name", "Custom Device")

                # Handle FlowFunctionCall nodes - get display name from definition
                elif node_type == "FlowFunctionCall":
                    definition_id = (
                        node_config.state.get("definition_id") if node_config.state else None
                    )
                    if definition_id:
                        def_dict = self._core.session.get_flow_function_definition(definition_id)
                        if def_dict:
                            display_name = def_dict.get("name", "Flow Function")

                # Handle ZoneInput nodes - get display name from zone_name in state
                elif node_type == "ZoneInput":
                    zone_name = (
                        node_config.state.get("zone_name", "Zone") if node_config.state else "Zone"
                    )
                    display_name = f"Zone: {zone_name}"

                # Determine category from node type
                category = "default"
                flow_nodes = ["StartExperiment", "EndExperiment", "Delay"]
                control_nodes = ["Loop", "WaitForInput"]
                io_nodes = [
                    "Output",
                    "Input",
                    "MotorGovernor",
                    "CustomDevice",
                    "CustomDeviceAction",
                ]
                function_nodes = [
                    "FlowFunctionCall",
                    "FunctionCall",
                    "StartFunction",
                    "EndFunction",
                ]
                interface_nodes = ["ZoneInput"]

                node_type_normalized = node_type.replace(" ", "")
                if node_type_normalized in flow_nodes:
                    category = "logic"
                elif node_type_normalized in control_nodes:
                    category = "interface"  # Orange color for control nodes
                elif node_type_normalized in io_nodes:
                    category = "hardware"
                elif node_type_normalized in function_nodes:
                    category = "logic"
                elif node_type_normalized in interface_nodes:
                    category = "interface"  # Zone inputs are interface nodes

                # Add visual node to graph (use display_name for visual)
                node_item = self._graph_view.add_node(node_config.id, display_name, x, y)
                node_item._category = category
                node_item._header_color = node_item.CATEGORY_COLORS.get(
                    category, node_item.CATEGORY_COLORS["default"]
                )

                # Store actual node type and definition ID for custom nodes
                node_item._actual_node_type = node_type
                node_item._definition_id = definition_id

                # Add ports based on node type
                self._setup_node_ports(node_item, node_type)

                # Connect port signals
                self._graph_view._connect_port_signals(node_item)

                logger.debug(f"Loaded node: {node_config.id} at ({x}, {y})")

            except Exception as e:
                logger.error(f"Failed to load node {node_config.id}: {e}")

        # Create visual connections from session
        for conn_config in self._core.session.flow.connections:
            try:
                self._graph_view.add_connection(
                    conn_config.id,
                    conn_config.from_node,
                    conn_config.from_output,
                    conn_config.to_node,
                    conn_config.to_input,
                )
                logger.debug(f"Loaded connection: {conn_config.id}")

            except Exception as e:
                logger.error(f"Failed to load connection {conn_config.id}: {e}")

        logger.info(
            f"Loaded {len(self._core.session.flow.nodes)} nodes and {len(self._core.session.flow.connections)} connections from session"
        )

    def _on_save(self) -> None:
        """Save experiment."""
        if self._core.session and self._core.session.file_path:
            try:
                self._core.save_session()
                self.statusBar().showMessage("Saved")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        """Save experiment as new file."""
        # Remember if we were in runner mode (fullscreen/frameless)
        was_runner_mode = self._view_manager.is_runner_mode

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Experiment",
            "",
            "GLIDER Experiments (*.glider);;JSON Files (*.json)",
        )

        # Restore fullscreen/frameless state on Pi after dialog closes
        if was_runner_mode:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.showFullScreen()

        if file_path:
            try:
                self._core.save_session(file_path)
                self.statusBar().showMessage(f"Saved: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _check_save(self) -> bool:
        """Check if current session should be saved. Returns True to proceed."""
        if self._core.session and self._core.session.is_dirty:
            result = QMessageBox.question(
                self,
                "Save Changes?",
                "The current experiment has unsaved changes. Save before continuing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if result == QMessageBox.StandardButton.Save:
                self._on_save()
                return True
            elif result == QMessageBox.StandardButton.Cancel:
                return False

        return True

    # Hardware operations
    def _on_add_board(self) -> None:
        """Show dialog to add a new board."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Board")
        dialog.setMinimumWidth(350)

        layout = QFormLayout(dialog)

        # Board type selection
        type_combo = QComboBox()
        type_combo.addItems(["telemetrix", "pigpio"])
        layout.addRow("Board Type:", type_combo)

        # Board ID
        id_edit = QLineEdit()
        id_edit.setPlaceholderText("e.g., arduino_1")
        layout.addRow("Board ID:", id_edit)

        # Port selection with refresh button
        port_layout = QHBoxLayout()
        port_combo = QComboBox()
        port_combo.setMinimumWidth(200)

        def refresh_ports():
            port_combo.clear()
            port_combo.addItem("Auto-detect", None)
            try:
                import serial.tools.list_ports

                ports = serial.tools.list_ports.comports()
                for port in ports:
                    # Show port with description
                    label = f"{port.device}"
                    if port.description and port.description != "n/a":
                        label += f" - {port.description}"
                    port_combo.addItem(label, port.device)
            except ImportError:
                pass

        refresh_ports()
        port_layout.addWidget(port_combo)

        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(refresh_ports)
        port_layout.addWidget(refresh_btn)

        layout.addRow("Serial Port:", port_layout)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            from glider.core.experiment_session import BoardConfig

            board_id = id_edit.text().strip() or f"board_{len(self._core.hardware_manager.boards)}"
            board_type = type_combo.currentText()
            port = port_combo.currentData()  # Get the actual port device path

            # Map UI type to driver type
            driver_type = "arduino" if board_type == "telemetrix" else "raspberry_pi"

            try:
                # Add to hardware manager for runtime use
                self._core.hardware_manager.add_board(board_id, board_type, port=port)

                # Add to session for persistence
                if self._core.session:
                    board_config = BoardConfig(
                        id=board_id,
                        driver_type=driver_type,
                        port=port,
                        board_type="uno",  # Default board type
                    )
                    self._core.session.add_board(board_config)

                self._refresh_hardware_tree()
                QMessageBox.information(self, "Success", f"Added board: {board_id}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add board: {e}")

    def _on_add_device(self) -> None:
        """Show dialog to add a new device."""
        if not self._core.hardware_manager.boards:
            QMessageBox.warning(self, "No Boards", "Please add a board first.")
            return

        # Map UI names to registry types and pin configurations
        # Format: (registry_type, [list of pin names])
        device_type_map = {
            "Digital Output (LED, Relay)": ("DigitalOutput", ["output"]),
            "Digital Input (Button, Sensor)": ("DigitalInput", ["input"]),
            "Analog Input (Potentiometer)": ("AnalogInput", ["input"]),
            "PWM Output (Dimmable LED, Motor)": ("PWMOutput", ["output"]),
            "Servo Motor": ("Servo", ["signal"]),
            "Motor Governor": ("MotorGovernor", ["up", "down", "signal"]),
            "ADS1115 (I2C ADC)": ("ADS1115", []),  # I2C device, no GPIO pins
        }

        dialog = QDialog(self)
        dialog.setWindowTitle("Add Device")
        dialog.setMinimumWidth(400)

        layout = QFormLayout(dialog)

        # Device type
        type_combo = QComboBox()
        type_combo.setMaxVisibleItems(10)  # Ensure dropdown is scrollable
        type_combo.setMinimumWidth(280)  # Ensure full text is visible
        type_combo.addItems(list(device_type_map.keys()))
        layout.addRow("Device Type:", type_combo)

        # Device ID
        id_edit = QLineEdit()
        id_edit.setPlaceholderText("e.g., led_1")
        layout.addRow("Device ID:", id_edit)

        # Board selection
        board_combo = QComboBox()
        board_combo.addItems(list(self._core.hardware_manager.boards.keys()))
        layout.addRow("Board:", board_combo)

        # Container for pin inputs (dynamic based on device type)
        pin_container = QWidget()
        pin_layout = QFormLayout(pin_container)
        pin_layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow(pin_container)

        # Dictionary to hold pin spinboxes
        pin_spinboxes: dict[str, QSpinBox] = {}
        # Dictionary to hold ADS1115 settings spinboxes
        ads1115_settings: dict[str, QSpinBox] = {}

        def update_pin_inputs():
            """Update pin inputs based on selected device type."""
            # Clear existing pin inputs
            while pin_layout.rowCount() > 0:
                pin_layout.removeRow(0)
            pin_spinboxes.clear()
            ads1115_settings.clear()

            # Get pin names for selected device type
            ui_type = type_combo.currentText()
            device_type, pin_names = device_type_map[ui_type]

            # Check if this is an analog device
            is_analog = device_type == "AnalogInput"
            is_ads1115 = device_type == "ADS1115"

            if is_ads1115:
                # ADS1115 I2C settings instead of pin inputs
                # I2C Address
                addr_spin = QSpinBox()
                addr_spin.setRange(72, 75)  # 0x48-0x4B
                addr_spin.setValue(72)  # Default 0x48
                addr_spin.setToolTip("I2C address: 72=0x48, 73=0x49, 74=0x4A, 75=0x4B")
                ads1115_settings["i2c_address"] = addr_spin
                pin_layout.addRow("I2C Address:", addr_spin)

                # Channel
                chan_spin = QSpinBox()
                chan_spin.setRange(0, 3)
                chan_spin.setValue(0)
                chan_spin.setToolTip("ADC channel to read (0-3)")
                ads1115_settings["channel"] = chan_spin
                pin_layout.addRow("Channel:", chan_spin)

                # Gain
                gain_combo = QComboBox()
                gain_combo.addItems(
                    ["1 (±4.096V)", "2 (±2.048V)", "4 (±1.024V)", "8 (±0.512V)", "16 (±0.256V)"]
                )
                gain_combo.setCurrentIndex(0)
                ads1115_settings["gain_combo"] = gain_combo
                pin_layout.addRow("Gain:", gain_combo)

                # Note about I2C
                note = QLabel("Note: Uses I2C on GPIO2 (SDA) and GPIO3 (SCL)")
                note.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
                note.setWordWrap(True)
                pin_layout.addRow(note)
            else:
                # Create spinbox for each pin
                for pin_name in pin_names:
                    spin = QSpinBox()
                    spin.setRange(0, 53)

                    # Set default value based on device type
                    if is_analog:
                        spin.setValue(14)  # Default to A0 (pin 14)
                        spin.setSpecialValueText("Invalid")
                    else:
                        spin.setValue(0)

                    pin_spinboxes[pin_name] = spin
                    label = f"{pin_name.capitalize()} Pin:"
                    pin_layout.addRow(label, spin)

                # Add helpful note for analog devices
                if is_analog:
                    note = QLabel("Note: A0=14, A1=15, A2=16, A3=17, A4=18, A5=19")
                    note.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
                    note.setWordWrap(True)
                    pin_layout.addRow(note)

        # Connect device type change to update pin inputs
        type_combo.currentTextChanged.connect(lambda: update_pin_inputs())
        update_pin_inputs()  # Initial setup

        # Name
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g., Status LED")
        layout.addRow("Name:", name_edit)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            from glider.core.experiment_session import DeviceConfig

            device_id = (
                id_edit.text().strip() or f"device_{len(self._core.hardware_manager.devices)}"
            )
            ui_device_type = type_combo.currentText()
            board_id = board_combo.currentText()
            name = name_edit.text().strip() or device_id

            # Get the actual device type and pin configuration
            device_type, pin_names = device_type_map[ui_device_type]

            # Build pins dictionary from spinboxes (empty for ADS1115)
            pins = {pin_name: pin_spinboxes[pin_name].value() for pin_name in pin_names}

            # Build settings dictionary for ADS1115
            settings = {}
            if device_type == "ADS1115" and ads1115_settings:
                settings["i2c_address"] = ads1115_settings["i2c_address"].value()
                settings["channel"] = ads1115_settings["channel"].value()
                # Parse gain from combo text (e.g., "1 (±4.096V)" -> 1)
                gain_text = ads1115_settings["gain_combo"].currentText()
                settings["gain"] = int(gain_text.split()[0])

            try:
                # Add to hardware manager for runtime use
                self._core.hardware_manager.add_device_multi_pin(
                    device_id, device_type, board_id, pins, name=name, **settings
                )

                # Auto-initialize if board is connected
                board = self._core.hardware_manager.get_board(board_id)
                if board and board.is_connected:

                    async def init_device():
                        try:
                            await self._core.hardware_manager.initialize_device(device_id)
                            logger.info(f"Auto-initialized device: {device_id}")
                        except Exception as e:
                            logger.error(f"Failed to auto-initialize device {device_id}: {e}")

                    self._run_async(init_device())

                # Add to session for persistence
                if self._core.session:
                    device_config = DeviceConfig(
                        id=device_id,
                        device_type=device_type,
                        name=name,
                        board_id=board_id,
                        pins=pins,
                        settings=settings,
                    )
                    self._core.session.add_device(device_config)

                self._refresh_hardware_tree()
                QMessageBox.information(self, "Success", f"Added device: {device_id}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add device: {e}")

    def _on_edit_board(self, board_id: str) -> None:
        """Show dialog to edit an existing board."""
        board = self._core.hardware_manager.get_board(board_id)
        if board is None:
            QMessageBox.warning(self, "Error", f"Board '{board_id}' not found.")
            return

        # Get session config for current values
        board_config = self._core.session.get_board(board_id) if self._core.session else None
        current_port = board_config.port if board_config else getattr(board, "port", None)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Board: {board_id}")
        dialog.setMinimumWidth(350)

        layout = QFormLayout(dialog)

        # Board ID (read-only)
        id_label = QLabel(board_id)
        id_label.setStyleSheet("color: #888;")
        layout.addRow("Board ID:", id_label)

        # Port selection with refresh button
        port_layout = QHBoxLayout()
        port_combo = QComboBox()
        port_combo.setMinimumWidth(200)

        def refresh_ports():
            port_combo.clear()
            port_combo.addItem("Auto-detect", None)
            try:
                import serial.tools.list_ports

                ports = serial.tools.list_ports.comports()
                for port in ports:
                    label = f"{port.device}"
                    if port.description and port.description != "n/a":
                        label += f" - {port.description}"
                    port_combo.addItem(label, port.device)
            except ImportError:
                pass

            # Select current port if set
            if current_port:
                for i in range(port_combo.count()):
                    if port_combo.itemData(i) == current_port:
                        port_combo.setCurrentIndex(i)
                        break

        refresh_ports()
        port_layout.addWidget(port_combo)

        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(refresh_ports)
        port_layout.addWidget(refresh_btn)

        layout.addRow("Serial Port:", port_layout)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_port = port_combo.currentData()

            try:
                # Update board in hardware manager via internal attribute
                if hasattr(board, "_port"):
                    board._port = new_port

                # Update session config
                if self._core.session:
                    self._core.session.update_board(board_id, port=new_port)

                self._refresh_hardware_tree()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Updated board: {board_id}\n\n"
                    "Note: Port changes take effect after reconnecting.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update board: {e}")

    def _on_edit_device(self, device_id: str) -> None:
        """Show dialog to edit an existing device."""
        device = self._core.hardware_manager.get_device(device_id)
        if device is None:
            QMessageBox.warning(self, "Error", f"Device '{device_id}' not found.")
            return

        # Get session config for current values
        device_config = self._core.session.get_device(device_id) if self._core.session else None

        # Get current values
        current_name = device_config.name if device_config else getattr(device, "name", device_id)
        current_pins = device_config.pins if device_config else {}
        device_type = device_config.device_type if device_config else type(device).__name__

        # If no pins in config, try to get from device
        if not current_pins and hasattr(device, "pin"):
            current_pins = {"output": device.pin} if hasattr(device, "pin") else {}
        if not current_pins and hasattr(device, "pins"):
            current_pins = device.pins if isinstance(device.pins, dict) else {}

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Device: {device_id}")
        dialog.setMinimumWidth(380)

        layout = QFormLayout(dialog)

        # Device ID (read-only)
        id_label = QLabel(device_id)
        id_label.setStyleSheet("color: #888;")
        layout.addRow("Device ID:", id_label)

        # Device type (read-only)
        type_label = QLabel(device_type)
        type_label.setStyleSheet("color: #888;")
        layout.addRow("Device Type:", type_label)

        # Name (editable)
        name_edit = QLineEdit(current_name)
        layout.addRow("Name:", name_edit)

        # Pin inputs
        pin_spinboxes: dict[str, QSpinBox] = {}

        # Determine if this is an analog device
        is_analog = "Analog" in device_type

        for pin_name, pin_value in current_pins.items():
            spin = QSpinBox()
            spin.setRange(0, 53)
            spin.setValue(pin_value)
            pin_spinboxes[pin_name] = spin
            label = f"{pin_name.capitalize()} Pin:"
            layout.addRow(label, spin)

        # Add helpful note for analog devices
        if is_analog:
            note = QLabel("Note: A0=14, A1=15, A2=16, A3=17, A4=18, A5=19")
            note.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
            note.setWordWrap(True)
            layout.addRow(note)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_name = name_edit.text().strip() or device_id
            new_pins = {pin_name: spin.value() for pin_name, spin in pin_spinboxes.items()}

            try:
                # Update device in hardware manager
                # Update name via _name attribute (name property may not have setter)
                if hasattr(device, "_name"):
                    device._name = new_name

                # Update pins via the config object
                if hasattr(device, "config") and hasattr(device.config, "pins"):
                    device.config.pins.update(new_pins)
                elif hasattr(device, "_config") and hasattr(device._config, "pins"):
                    device._config.pins.update(new_pins)

                # Update session config
                if self._core.session:
                    self._core.session.update_device(device_id, name=new_name, pins=new_pins)

                self._refresh_hardware_tree()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Updated device: {device_id}\n\n"
                    "Note: Pin changes take effect after reconnecting the board.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update device: {e}")

    def _refresh_hardware_tree(self) -> None:
        """Refresh the hardware tree widget."""
        if not hasattr(self, "_hardware_tree"):
            return

        self._hardware_tree.clear()

        # Add boards
        for board_id, board in self._core.hardware_manager.boards.items():
            board_item = QTreeWidgetItem(
                [
                    board_id,
                    getattr(board, "name", type(board).__name__),
                    board.state.name if hasattr(board, "state") else "Unknown",
                ]
            )
            board_item.setData(0, Qt.ItemDataRole.UserRole, ("board", board_id))

            # Add devices under this board
            for device_id, device in self._core.hardware_manager.devices.items():
                if hasattr(device, "board") and device.board is board:
                    pins = getattr(device, "_pins", [])
                    pin_str = f"Pin {pins[0]}" if pins else ""
                    device_item = QTreeWidgetItem(
                        [
                            getattr(device, "name", device_id),
                            f"{getattr(device, 'device_type', 'unknown')} ({pin_str})",
                            (
                                "Ready"
                                if getattr(device, "_initialized", False)
                                else "Not initialized"
                            ),
                        ]
                    )
                    device_item.setData(0, Qt.ItemDataRole.UserRole, ("device", device_id))
                    board_item.addChild(device_item)

            self._hardware_tree.addTopLevelItem(board_item)
            board_item.setExpanded(True)

        self._hardware_tree.resizeColumnToContents(0)
        self._hardware_tree.resizeColumnToContents(1)

        # Also refresh the device combo for the control panel
        self._refresh_device_combo()

        # Also refresh runner view devices
        self._refresh_runner_devices()

    def _on_hardware_context_menu(self, position) -> None:
        """Show context menu for hardware tree."""
        item = self._hardware_tree.itemAt(position)
        if item is None:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return

        item_type, item_id = data

        menu = QMenu(self)

        if item_type == "board":
            connect_action = menu.addAction("Connect")
            connect_action.triggered.connect(lambda: self._connect_board(item_id))

            disconnect_action = menu.addAction("Disconnect")
            disconnect_action.triggered.connect(lambda: self._disconnect_board(item_id))

            menu.addSeparator()

            edit_action = menu.addAction("Edit Board")
            edit_action.triggered.connect(lambda: self._on_edit_board(item_id))

            remove_action = menu.addAction("Remove Board")
            remove_action.triggered.connect(lambda: self._remove_board(item_id))

        elif item_type == "device":
            edit_action = menu.addAction("Edit Device")
            edit_action.triggered.connect(lambda: self._on_edit_device(item_id))

            remove_action = menu.addAction("Remove Device")
            remove_action.triggered.connect(lambda: self._remove_device(item_id))

        menu.exec(self._hardware_tree.viewport().mapToGlobal(position))

    def _connect_board(self, board_id: str) -> None:
        """Connect to a specific board and initialize its devices."""

        async def connect():
            try:
                success = await self._core.hardware_manager.connect_board(board_id)
                if success:
                    # Initialize devices for this board
                    for device_id, device in self._core.hardware_manager.devices.items():
                        if hasattr(device, "board") and device.board is not None:
                            if device.board.id == board_id:
                                try:
                                    await self._core.hardware_manager.initialize_device(device_id)
                                except Exception as e:
                                    logger.warning(f"Failed to initialize device {device_id}: {e}")
                    self.statusBar().showMessage(f"Connected to {board_id}", 3000)
                else:
                    QMessageBox.warning(
                        self, "Connection Failed", f"Could not connect to {board_id}"
                    )
                self._refresh_hardware_tree()
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", str(e))

        self._run_async(connect())

    def _disconnect_board(self, board_id: str) -> None:
        """Disconnect from a specific board."""

        async def disconnect():
            try:
                await self._core.hardware_manager.disconnect_board(board_id)
                self._refresh_hardware_tree()
            except Exception as e:
                QMessageBox.critical(self, "Disconnect Error", str(e))

        self._run_async(disconnect())

    def _remove_board(self, board_id: str) -> None:
        """Remove a board."""
        reply = QMessageBox.question(
            self,
            "Remove Board",
            f"Remove board '{board_id}' and all its devices?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:

            async def remove():
                await self._core.hardware_manager.remove_board(board_id)
                # Also remove from session for persistence
                if self._core.session:
                    self._core.session.remove_board(board_id)
                self._refresh_hardware_tree()

            self._run_async(remove())

    def _remove_device(self, device_id: str) -> None:
        """Remove a device."""
        reply = QMessageBox.question(
            self,
            "Remove Device",
            f"Remove device '{device_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:

            async def remove():
                await self._core.hardware_manager.remove_device(device_id)
                # Also remove from session for persistence
                if self._core.session:
                    self._core.session.remove_device(device_id)
                self._refresh_hardware_tree()

            self._run_async(remove())

    def _on_connect_hardware(self) -> None:
        """Connect to all hardware."""
        self._run_async(self._connect_hardware_async())

    async def _connect_hardware_async(self) -> None:
        """Async hardware connection."""
        try:
            await self._core.setup_hardware()
            results = await self._core.connect_hardware()
            self._refresh_hardware_tree()
            failed = [k for k, v in results.items() if not v]
            if failed:
                QMessageBox.warning(
                    self, "Connection Warning", f"Failed to connect: {', '.join(failed)}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))

    def _on_disconnect_hardware(self) -> None:
        """Disconnect all hardware."""
        self._run_async(self._core.hardware_manager.disconnect_all())

    # Camera operations
    def _on_camera_settings(self) -> None:
        """Show camera settings dialog."""
        dialog = CameraSettingsDialog(
            camera_settings=self._core.camera_manager.settings,
            cv_settings=self._core.cv_processor.settings,
            parent=self,
            view_manager=self._view_manager,
            camera_manager=self._core.camera_manager,
        )

        # Connect signals for opening calibration and zones dialogs from Tools tab
        dialog.calibration_requested.connect(self._on_camera_calibration)
        dialog.zones_requested.connect(self._on_zones_requested)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Apply camera settings
            camera_settings = dialog.get_camera_settings()
            self._core.camera_manager.apply_settings(camera_settings)

            # Apply CV settings
            cv_settings = dialog.get_cv_settings()
            self._core.cv_processor.update_settings(cv_settings)

            logger.info("Camera settings updated")

    def _on_camera_calibration(self) -> None:
        """Show camera calibration dialog."""
        dialog = CalibrationDialog(
            camera_manager=self._core.camera_manager,
            calibration=self._core.calibration,
            parent=self,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Calibration is modified in place, just log
            calibration = dialog.get_calibration()
            if calibration.is_calibrated:
                logger.info(
                    f"Camera calibrated: {calibration.pixels_per_mm:.2f} pixels/mm "
                    f"({len(calibration.lines)} lines)"
                )
            else:
                logger.info("Camera calibration cleared")

    def _on_zones_requested(self) -> None:
        """Show zone configuration dialog."""
        dialog = ZoneDialog(
            camera_manager=self._core.camera_manager, zone_config=self._zone_config, parent=self
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._zone_config = dialog.get_zone_configuration()
            # Update camera panel display
            self._camera_panel.set_zone_configuration(self._zone_config)
            # Update CV processor for zone tracking
            self._core.cv_processor.set_zone_configuration(self._zone_config)
            # Update tracking logger for CSV zone columns
            self._core.tracking_logger.set_zone_configuration(self._zone_config)
            # Update data recorder for zone columns
            if hasattr(self._core, "data_recorder"):
                self._core.data_recorder.set_zone_configuration(self._zone_config)
                self._core.data_recorder.set_cv_processor(self._core.cv_processor)

            # Save zones to session
            self._save_zones_to_session()

            # Refresh zone nodes in node library
            self._refresh_zones()

            logger.info(f"Zone configuration updated: {len(self._zone_config.zones)} zones")

    def _on_experiment_metadata_changed(self) -> None:
        """Handle experiment metadata changes from the experiment dialog."""
        # Mark session as dirty so changes are saved
        self._core.session._dirty = True
        logger.debug("Experiment metadata changed")

    def _on_recording_directory_changed(self, directory: str) -> None:
        """Handle recording directory changes from the experiment dialog."""
        if directory:
            self._core.set_recording_directory(Path(directory))
            logger.info(f"Recording directory changed to: {directory}")

    def _on_open_experiment_dialog(self) -> None:
        """Open the experiment settings dialog."""
        if self._experiment_dialog is None:
            self._experiment_dialog = ExperimentDialog(
                session=self._core.session,
                parent=self,
                is_touch_mode=self._view_manager.is_runner_mode,
            )
            self._experiment_dialog.metadata_changed.connect(self._on_experiment_metadata_changed)
            self._experiment_dialog.edit_subject_requested.connect(self._on_edit_subject)
            self._experiment_dialog.recording_directory_changed.connect(
                self._on_recording_directory_changed
            )
        else:
            # Refresh dialog with current session
            self._experiment_dialog.set_session(self._core.session)

        self._experiment_dialog.show()
        self._experiment_dialog.raise_()
        self._experiment_dialog.activateWindow()

    def _on_open_analysis_dialog(self) -> None:
        """Open the CSV analysis dialog."""
        from glider.gui.dialogs.analysis_dialog import AnalysisDialog

        dialog = AnalysisDialog(parent=self)
        dialog.exec()

    def _on_edit_subject(self, subject_id: str) -> None:
        """Handle subject edit request from experiment dialog."""

        subject = None
        if subject_id:
            subject = self._core.session.metadata.get_subject(subject_id)

        dialog = SubjectDialog(
            subject=subject,
            parent=self._experiment_dialog if self._experiment_dialog else self,
            is_touch_mode=self._view_manager.is_runner_mode,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_subject = dialog.get_subject()
            if subject_id and subject:
                # Update existing subject
                metadata = self._core.session.metadata
                for i, s in enumerate(metadata.subjects):
                    if s.id == new_subject.id:
                        metadata.subjects[i] = new_subject
                        self._core.session._dirty = True
                        break
                logger.info(f"Updated subject: {new_subject.subject_id}")
            else:
                # Add new subject
                self._core.session.metadata.add_subject(new_subject)
                self._core.session._dirty = True
                logger.info(f"Added subject: {new_subject.subject_id}")

            # Refresh the experiment dialog if it's open
            if self._experiment_dialog:
                self._experiment_dialog.refresh()

    def _load_zones_from_session(self) -> None:
        """Load zone configuration from the current session."""
        if not self._core.session:
            return

        session_zones = self._core.session.zones
        if session_zones.zones:
            # Convert session ZoneConfig to vision ZoneConfiguration
            from glider.vision.zones import Zone

            self._zone_config.zones.clear()
            for zone_dict in session_zones.zones:
                zone = Zone.from_dict(zone_dict)
                self._zone_config.zones.append(zone)
            self._zone_config.config_width = session_zones.config_width
            self._zone_config.config_height = session_zones.config_height

            # Update components with loaded zones
            self._camera_panel.set_zone_configuration(self._zone_config)
            self._core.cv_processor.set_zone_configuration(self._zone_config)
            self._core.tracking_logger.set_zone_configuration(self._zone_config)

            # Refresh zone nodes in node library
            self._refresh_zones()

            logger.info(f"Loaded {len(self._zone_config.zones)} zones from session")
        else:
            # Clear zones if session has none
            self._zone_config = ZoneConfiguration()
            self._camera_panel.set_zone_configuration(self._zone_config)
            self._refresh_zones()

    def _save_zones_to_session(self) -> None:
        """Save zone configuration to the current session."""
        if not self._core.session:
            return

        # Convert ZoneConfiguration to session ZoneConfig
        self._core.session.zones.zones = [zone.to_dict() for zone in self._zone_config.zones]
        self._core.session.zones.config_width = self._zone_config.config_width
        self._core.session.zones.config_height = self._zone_config.config_height
        self._core.session._mark_dirty()

    # Run operations
    @pyqtSlot()
    def _on_start_clicked(self) -> None:
        """Start experiment."""
        self._run_async(self._start_async())

    async def _start_async(self) -> None:
        """Async start."""
        try:
            # Apply recording directory from session metadata before starting
            rec_dir = self._core.session.metadata.recording_directory
            if rec_dir:
                self._core.set_recording_directory(Path(rec_dir))
            await self._core.start_experiment()
        except Exception as e:
            QMessageBox.critical(self, "Start Error", str(e))

    @pyqtSlot()
    def _on_stop_clicked(self) -> None:
        """Stop experiment."""
        self._run_async(self._stop_async())

    async def _stop_async(self) -> None:
        """Async stop."""
        try:
            await self._core.stop_experiment()
        except Exception as e:
            QMessageBox.critical(self, "Stop Error", str(e))

    @pyqtSlot()
    def _on_emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._run_async(self._core.emergency_stop())

    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About GLIDER",
            "GLIDER - General Laboratory Interface for Design, "
            "Experimentation, and Recording\n\n"
            "Version 1.0.0\n\n"
            "A modular experimental orchestration platform.",
        )

    def _show_runner_menu(self) -> None:
        """Show the runner mode menu with options like Open, Exit, etc."""
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a2e;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 12px 24px;
                font-size: 16px;
                color: white;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #3498db;
            }
        """)

        # Open file action
        open_action = menu.addAction("Open Experiment")
        open_action.triggered.connect(self._on_open)

        # Reload action
        reload_action = menu.addAction("Reload")
        reload_action.triggered.connect(self._refresh_runner_devices)

        # Board settings action (quick port configuration)
        board_action = menu.addAction("Ports")
        board_action.triggered.connect(self._show_board_settings_dialog)

        # Switch to desktop/config mode - shows dock panels for full hardware configuration
        desktop_action = menu.addAction("Hardware Config")
        desktop_action.triggered.connect(self._switch_to_desktop_mode)

        menu.addSeparator()

        # Exit action
        exit_action = menu.addAction("✕  Exit")
        exit_action.triggered.connect(self.close)

        # Show menu at button position
        menu.exec(self._runner_menu_btn.mapToGlobal(self._runner_menu_btn.rect().bottomLeft()))

    def _switch_to_desktop_mode(self) -> None:
        """Switch from runner to desktop mode."""
        self.setWindowFlags(Qt.WindowType.Window)
        self.showNormal()

        # Switch to builder view
        self._stack.setCurrentIndex(0)

        # Create dock widgets if they don't exist yet
        if getattr(self, "_node_library_dock", None) is None:
            self._setup_dock_widgets()

        # Use appropriate size for the screen
        screen_size = self._view_manager.screen_size
        if screen_size.width() <= 800:
            # Small screen - maximize and use tabbed layout
            self.showMaximized()
            # Apply tabbed layout for small screens
            self._set_pi_touchscreen_layout()
        else:
            # Large screen - standard desktop size
            self.resize(1400, 900)

    def _show_board_settings_dialog(self) -> None:
        """Show a dialog to configure board settings (ports, etc.)."""
        import glob
        import sys

        from PyQt6.QtWidgets import (
            QComboBox,
            QDialog,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("Board Settings")
        dialog.setMinimumWidth(350)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QComboBox {
                background-color: #2d2d44;
                color: white;
                border: 2px solid #3498db;
                border-radius: 6px;
                padding: 8px;
                min-height: 36px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d44;
                color: white;
                selection-background-color: #3498db;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                min-height: 40px;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
            QPushButton[secondary="true"] {
                background-color: #34495e;
            }
            QGroupBox {
                color: white;
                font-weight: bold;
                border: 2px solid #2d2d44;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Detect available serial ports
        def get_available_ports():
            ports = []
            if sys.platform.startswith("linux"):
                # Common Linux serial port patterns
                patterns = ["/dev/ttyACM*", "/dev/ttyUSB*", "/dev/ttyAMA*"]
                for pattern in patterns:
                    ports.extend(glob.glob(pattern))
            elif sys.platform == "darwin":
                ports.extend(glob.glob("/dev/tty.usbmodem*"))
                ports.extend(glob.glob("/dev/tty.usbserial*"))
            else:
                # Windows
                for i in range(10):
                    ports.append(f"COM{i}")
            return sorted(ports)

        available_ports = get_available_ports()

        # Get boards from session
        boards = self._core.hardware_manager.boards if self._core else {}
        port_combos = {}

        if not boards:
            no_boards_label = QLabel("No boards configured.\nLoad an experiment first.")
            no_boards_label.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(no_boards_label)
        else:
            for board_id, board in boards.items():
                group = QGroupBox(f"Board: {board_id}")
                group_layout = QVBoxLayout(group)

                # Board type
                board_type = getattr(board, "board_type", "unknown")
                type_label = QLabel(f"Type: {board_type}")
                group_layout.addWidget(type_label)

                # Port selection
                port_layout = QHBoxLayout()
                port_label = QLabel("Port:")
                port_combo = QComboBox()
                port_combo.setEditable(True)  # Allow custom port entry

                # Add available ports
                current_port = getattr(board, "_port", "") or ""
                if available_ports:
                    port_combo.addItems(available_ports)
                # Add current port if not in list
                if current_port and current_port not in available_ports:
                    port_combo.addItem(current_port)
                # Set current port
                port_combo.setCurrentText(current_port)

                port_combos[board_id] = port_combo

                port_layout.addWidget(port_label)
                port_layout.addWidget(port_combo, 1)
                group_layout.addLayout(port_layout)

                # Connection status
                connected = getattr(board, "is_connected", False)
                status_text = "Connected" if connected else "Disconnected"
                status_color = "#2ecc71" if connected else "#e74c3c"
                status_label = QLabel(f"Status: {status_text}")
                status_label.setStyleSheet(f"color: {status_color};")
                group_layout.addWidget(status_label)

                layout.addWidget(group)

        # Detected ports info
        if available_ports:
            ports_label = QLabel(f"Detected ports: {', '.join(available_ports)}")
            ports_label.setStyleSheet("color: #888; font-size: 12px;")
            layout.addWidget(ports_label)
        else:
            ports_label = QLabel("No serial ports detected")
            ports_label.setStyleSheet("color: #e74c3c; font-size: 12px;")
            layout.addWidget(ports_label)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("secondary", True)
        cancel_btn.clicked.connect(dialog.reject)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(dialog.accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted and port_combos:
            # Update board ports
            for board_id, combo in port_combos.items():
                new_port = combo.currentText()
                board = boards.get(board_id)
                if board and new_port:
                    old_port = getattr(board, "_port", "")
                    if new_port != old_port:
                        board._port = new_port
                        logger.info(f"Updated board '{board_id}' port: {old_port} -> {new_port}")

                        # Also update the session config
                        if self._core and self._core.session:
                            for board_config in self._core.session.boards:
                                if board_config.id == board_id:
                                    board_config.config["port"] = new_port
                                    self._core.session._dirty = True
                                    break

    # Node Library methods
    def _create_node_library(self) -> QWidget:
        """Create the node library widget with draggable node items."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Define available nodes by category
        node_categories = {
            "Flow": [
                ("StartExperiment", "Start Experiment", "Entry point - begins the experiment flow"),
                ("EndExperiment", "End Experiment", "Exit point - ends the experiment"),
                ("Delay", "Delay", "Wait for a specified duration"),
            ],
            "Functions": [
                (
                    "StartFunction",
                    "Start Function",
                    "Define a reusable function - set name in properties",
                ),
                ("EndFunction", "End Function", "End of function definition"),
            ],
            "Control": [
                ("Loop", "Loop", "Repeat actions N times (0 = infinite)"),
                ("WaitForInput", "Wait For Input", "Wait for input trigger before continuing"),
            ],
            "I/O": [
                ("Output", "Output", "Write to a device (digital or PWM)"),
                ("Input", "Input", "Read from a device (digital or analog)"),
            ],
        }

        # Category colors for headers
        category_colors = {
            "Flow": "#2d5a7a",  # Blue
            "Functions": "#2d7a7a",  # Teal
            "Control": "#7a5a2d",  # Orange/Brown
            "I/O": "#2d7a2d",  # Green
        }

        for category, nodes in node_categories.items():
            color = category_colors.get(category, "#444")

            # Collapsible category section
            category_widget = QWidget()
            category_layout = QVBoxLayout(category_widget)
            category_layout.setContentsMargins(0, 0, 0, 0)
            category_layout.setSpacing(2)

            # Category header button (clickable to expand/collapse)
            header_btn = QPushButton(f"▼  {category}")
            header_btn.setCheckable(True)
            header_btn.setChecked(True)
            header_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            header_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 12px;
                    font-size: 13px;
                    font-weight: bold;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: {color}cc;
                }}
                QPushButton:checked {{
                    border-bottom-left-radius: 0px;
                    border-bottom-right-radius: 0px;
                }}
            """)
            category_layout.addWidget(header_btn)

            # Container for node items
            nodes_container = QWidget()
            nodes_container.setStyleSheet(f"""
                QWidget {{
                    background-color: {color}40;
                    border-bottom-left-radius: 4px;
                    border-bottom-right-radius: 4px;
                }}
            """)
            nodes_layout = QVBoxLayout(nodes_container)
            nodes_layout.setContentsMargins(4, 4, 4, 4)
            nodes_layout.setSpacing(2)

            # Node items
            for node_type, node_name, tooltip in nodes:
                node_btn = DraggableNodeButton(node_type, node_name, category)
                node_btn.setToolTip(tooltip)
                node_btn.clicked.connect(lambda checked, nt=node_type: self._add_node_to_center(nt))
                nodes_layout.addWidget(node_btn)

            category_layout.addWidget(nodes_container)

            # Connect header button to toggle visibility
            def make_toggle(btn, container):
                def toggle(checked):
                    container.setVisible(checked)
                    btn.setText(f"▼  {btn.text()[3:]}" if checked else f"▶  {btn.text()[3:]}")

                return toggle

            header_btn.toggled.connect(make_toggle(header_btn, nodes_container))

            layout.addWidget(category_widget)

        # Custom Devices section (dynamic)
        self._custom_devices_container = QWidget()
        self._custom_devices_layout = QVBoxLayout(self._custom_devices_container)
        self._custom_devices_layout.setContentsMargins(0, 0, 0, 0)
        self._custom_devices_layout.setSpacing(2)
        self._setup_custom_category(
            self._custom_devices_container,
            self._custom_devices_layout,
            "Custom Devices",
            "#6a4a8a",  # Purple
            layout,
            add_new_callback=self._on_new_custom_device,
        )

        # Flow Functions section (dynamic)
        self._flow_functions_container = QWidget()
        self._flow_functions_layout = QVBoxLayout(self._flow_functions_container)
        self._flow_functions_layout.setContentsMargins(0, 0, 0, 0)
        self._flow_functions_layout.setSpacing(2)
        self._setup_custom_category(
            self._flow_functions_container,
            self._flow_functions_layout,
            "Flow Functions",
            "#4a6a8a",  # Steel blue
            layout,
            add_new_callback=self._on_new_flow_function,
        )

        # Zones section (dynamic - populated from zone configuration)
        self._zones_container = QWidget()
        self._zones_layout = QVBoxLayout(self._zones_container)
        self._zones_layout.setContentsMargins(0, 0, 0, 0)
        self._zones_layout.setSpacing(2)
        self._setup_custom_category(
            self._zones_container,
            self._zones_layout,
            "Zones",
            "#5a4a2d",  # Orange/brown - matches zone color
            layout,
            add_new_callback=None,  # Zones are created via Zone dialog
        )

        layout.addStretch()
        scroll_area.setWidget(container)

        # Store container reference for updates
        self._node_library_container = container
        self._node_library_layout = layout

        return scroll_area

    def _setup_custom_category(
        self,
        nodes_container: QWidget,
        nodes_layout: QVBoxLayout,
        category_name: str,
        color: str,
        parent_layout: QVBoxLayout,
        add_new_callback=None,
    ) -> None:
        """Setup a custom category with a header and add button."""
        category_widget = QWidget()
        category_layout = QVBoxLayout(category_widget)
        category_layout.setContentsMargins(0, 0, 0, 0)
        category_layout.setSpacing(2)

        # Category header with add button
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        header_btn = QPushButton(f"▼  {category_name}")
        header_btn.setCheckable(True)
        header_btn.setChecked(True)
        header_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        header_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: bold;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {color}cc;
            }}
            QPushButton:checked {{
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }}
        """)
        header_layout.addWidget(header_btn, 1)

        if add_new_callback:
            add_btn = QPushButton("+")
            add_btn.setFixedWidth(30)
            add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            add_btn.setToolTip(f"Add new {category_name.lower()[:-1]}")
            add_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-top-right-radius: 4px;
                    border-bottom-right-radius: 4px;
                    padding: 8px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color}cc;
                }}
            """)
            add_btn.clicked.connect(add_new_callback)
            header_layout.addWidget(add_btn)

        category_layout.addWidget(header_widget)

        # Style the nodes container
        nodes_container.setStyleSheet(f"""
            QWidget {{
                background-color: {color}40;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
        """)
        nodes_layout.setContentsMargins(4, 4, 4, 4)

        # Add placeholder label
        placeholder = QLabel("No items defined")
        placeholder.setStyleSheet("color: #888; padding: 8px;")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nodes_layout.addWidget(placeholder)

        category_layout.addWidget(nodes_container)

        # Connect toggle
        def make_toggle(btn, container):
            def toggle(checked):
                container.setVisible(checked)
                btn.setText(f"▼  {btn.text()[3:]}" if checked else f"▶  {btn.text()[3:]}")

            return toggle

        header_btn.toggled.connect(make_toggle(header_btn, nodes_container))

        parent_layout.addWidget(category_widget)

    def _on_new_custom_device(self) -> None:
        """Open dialog to create a new custom device."""
        try:
            from glider.gui.dialogs.custom_device_dialog import CustomDeviceDialog

            dialog = CustomDeviceDialog(parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                definition = dialog.get_definition()
                # Add to session
                if self._core.session:
                    self._core.session.add_custom_device_definition(definition.to_dict())
                    self._refresh_custom_devices()
                    logger.info(f"Created custom device: {definition.name}")
        except ImportError as e:
            logger.warning(f"Could not import CustomDeviceDialog: {e}")
            QMessageBox.warning(self, "Not Available", "Custom device editor not available.")

    def _on_new_flow_function(self) -> None:
        """Open dialog to create a new flow function."""
        try:
            from glider.core.flow_engine import FlowEngine
            from glider.gui.dialogs.flow_function_dialog import FlowFunctionDialog

            # Get available node types
            available_types = FlowEngine.get_available_nodes()
            available_types.extend(["FlowFunctionEntry", "FlowFunctionExit", "Parameter"])

            dialog = FlowFunctionDialog(available_node_types=available_types, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                definition = dialog.get_definition()
                # Add to session
                if self._core.session:
                    self._core.session.add_flow_function_definition(definition.to_dict())
                    self._refresh_flow_functions()
                    logger.info(f"Created flow function: {definition.name}")
        except ImportError as e:
            logger.warning(f"Could not import FlowFunctionDialog: {e}")
            QMessageBox.warning(self, "Not Available", "Flow function editor not available.")

    def _refresh_custom_devices(self) -> None:
        """Refresh the custom devices in the node library."""
        if not hasattr(self, "_custom_devices_layout"):
            return

        # Clear existing items
        while self._custom_devices_layout.count():
            item = self._custom_devices_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add devices from session
        if self._core.session:
            definitions = self._core.session.custom_device_definitions
            if definitions:
                for def_id, def_dict in definitions.items():
                    name = def_dict.get("name", "Unknown")
                    btn = EditableDraggableButton(
                        f"CustomDevice:{def_id}",
                        name,
                        "Custom Devices",
                        on_edit=lambda did=def_id: self._edit_custom_device(did),
                        on_delete=lambda did=def_id: self._delete_custom_device(did),
                    )
                    btn.setToolTip(
                        f"{def_dict.get('description', '')}\n(Right-click to edit/delete)"
                    )
                    btn.clicked.connect(
                        lambda checked, did=def_id: self._add_custom_device_node(did)
                    )
                    self._custom_devices_layout.addWidget(btn)
            else:
                placeholder = QLabel("No devices defined")
                placeholder.setStyleSheet("color: #888; padding: 8px;")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._custom_devices_layout.addWidget(placeholder)
        else:
            placeholder = QLabel("No devices defined")
            placeholder.setStyleSheet("color: #888; padding: 8px;")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._custom_devices_layout.addWidget(placeholder)

    def _refresh_flow_functions(self) -> None:
        """Refresh the flow functions in the node library."""
        if not hasattr(self, "_flow_functions_layout"):
            return

        # Clear existing items
        while self._flow_functions_layout.count():
            item = self._flow_functions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        has_functions = False

        # Detect graph-defined functions (StartFunction -> EndFunction chains)
        detected_functions = self._detect_graph_functions()
        if detected_functions:
            for func_info in detected_functions:
                has_functions = True
                func_name = func_info["name"]
                start_node_id = func_info["start_node_id"]

                btn = DraggableNodeButton(f"FunctionCall:{start_node_id}", func_name, "Functions")
                btn.setToolTip(f"Call function '{func_name}'")
                btn.clicked.connect(
                    lambda checked, nid=start_node_id, name=func_name: self._add_function_call_node(
                        nid, name
                    )
                )
                self._flow_functions_layout.addWidget(btn)

        if not has_functions:
            placeholder = QLabel("Define functions with StartFunction → EndFunction")
            placeholder.setStyleSheet("color: #888; padding: 8px; font-size: 10px;")
            placeholder.setWordWrap(True)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._flow_functions_layout.addWidget(placeholder)

    def _refresh_zones(self) -> None:
        """Refresh the zones in the node library."""
        if not hasattr(self, "_zones_layout"):
            return

        # Clear existing items
        while self._zones_layout.count():
            item = self._zones_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add zones from zone configuration
        if self._zone_config.zones:
            for zone in self._zone_config.zones:
                btn = DraggableNodeButton(f"ZoneInput:{zone.id}", f"Zone: {zone.name}", "Zones")
                btn.setToolTip(
                    f"Monitor zone '{zone.name}' for object occupancy\n"
                    f"Outputs: Occupied (bool), Object Count (int), On Enter, On Exit"
                )
                btn.clicked.connect(lambda checked, zid=zone.id: self._add_zone_node(zid))
                self._zones_layout.addWidget(btn)
        else:
            placeholder = QLabel("Create zones in Camera → Zones...")
            placeholder.setStyleSheet("color: #888; padding: 8px; font-size: 10px;")
            placeholder.setWordWrap(True)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._zones_layout.addWidget(placeholder)

    def _add_zone_node(self, zone_id: str) -> None:
        """Add a zone input node to the graph."""
        if hasattr(self, "_graph_view"):
            center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
            # Create node with zone ID encoded in the type
            self._graph_view.node_created.emit(f"ZoneInput:{zone_id}", center.x(), center.y())

    def _detect_graph_functions(self) -> list:
        """
        Detect complete function definitions in the graph.

        A function is defined when a StartFunction node is connected
        (directly or through other nodes) to an EndFunction node.

        Returns:
            List of function info dicts with 'name', 'start_node_id', 'end_node_ids'
        """
        if not self._core.session:
            return []

        functions = []
        flow = self._core.session.flow

        # Find all StartFunction nodes
        start_nodes = [n for n in flow.nodes if n.node_type == "StartFunction"]

        for start_node in start_nodes:
            # Get function name from state
            func_name = "MyFunction"
            if start_node.state:
                func_name = start_node.state.get("function_name", "MyFunction")

            # Check if this StartFunction leads to an EndFunction
            if self._trace_to_end_function(start_node.id, flow):
                functions.append(
                    {
                        "name": func_name,
                        "start_node_id": start_node.id,
                    }
                )

        return functions

    def _trace_to_end_function(self, start_id: str, flow) -> bool:
        """
        Trace from a node to see if it eventually reaches an EndFunction.

        Args:
            start_id: Starting node ID
            flow: The flow configuration

        Returns:
            True if an EndFunction is reachable
        """
        visited = set()
        to_visit = [start_id]

        while to_visit:
            current_id = to_visit.pop()
            if current_id in visited:
                continue
            visited.add(current_id)

            # Check if this node is an EndFunction
            for node in flow.nodes:
                if node.id == current_id and node.node_type == "EndFunction":
                    return True

            # Find outgoing connections
            for conn in flow.connections:
                if conn.from_node == current_id:
                    to_visit.append(conn.to_node)

        return False

    def _add_function_call_node(self, start_node_id: str, func_name: str) -> None:
        """Add a FunctionCall node to the graph."""
        if hasattr(self, "_graph_view"):
            center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
            # Create a FunctionCall node with the function reference
            self._graph_view.node_created.emit(
                f"FunctionCall:{start_node_id}", center.x(), center.y()
            )

    def _add_custom_device_node(self, definition_id: str) -> None:
        """Add a custom device action node to the graph."""
        if hasattr(self, "_graph_view"):
            center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
            # Create node with custom device definition ID encoded in the type
            self._graph_view.node_created.emit(
                f"CustomDevice:{definition_id}", center.x(), center.y()
            )

    def _add_flow_function_node(self, definition_id: str) -> None:
        """Add a flow function node to the graph."""
        if hasattr(self, "_graph_view"):
            center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
            self._graph_view.node_created.emit(
                f"FlowFunction:{definition_id}", center.x(), center.y()
            )

    def _edit_custom_device(self, definition_id: str) -> None:
        """Edit an existing custom device definition."""
        if not self._core.session:
            return

        def_dict = self._core.session.get_custom_device_definition(definition_id)
        if not def_dict:
            QMessageBox.warning(self, "Error", "Custom device not found.")
            return

        try:
            from glider.core.custom_device import CustomDeviceDefinition
            from glider.gui.dialogs.custom_device_dialog import CustomDeviceDialog

            definition = CustomDeviceDefinition.from_dict(def_dict)
            dialog = CustomDeviceDialog(definition=definition, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                updated_def = dialog.get_definition()
                # Update in session (remove old, add new)
                self._core.session.remove_custom_device_definition(definition_id)
                self._core.session.add_custom_device_definition(updated_def.to_dict())
                self._refresh_custom_devices()
                logger.info(f"Updated custom device: {updated_def.name}")
        except ImportError as e:
            logger.warning(f"Could not import CustomDeviceDialog: {e}")
            QMessageBox.warning(self, "Not Available", "Custom device editor not available.")

    def _delete_custom_device(self, definition_id: str) -> None:
        """Delete a custom device definition."""
        if not self._core.session:
            return

        def_dict = self._core.session.get_custom_device_definition(definition_id)
        name = def_dict.get("name", "Unknown") if def_dict else "Unknown"

        result = QMessageBox.question(
            self,
            "Delete Custom Device",
            f"Are you sure you want to delete '{name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if result == QMessageBox.StandardButton.Yes:
            self._core.session.remove_custom_device_definition(definition_id)
            self._refresh_custom_devices()
            logger.info(f"Deleted custom device: {name}")

    def _edit_flow_function(self, definition_id: str) -> None:
        """Edit an existing flow function definition."""
        if not self._core.session:
            return

        def_dict = self._core.session.get_flow_function_definition(definition_id)
        if not def_dict:
            QMessageBox.warning(self, "Error", "Flow function not found.")
            return

        try:
            from glider.core.flow_engine import FlowEngine
            from glider.core.flow_function import FlowFunctionDefinition
            from glider.gui.dialogs.flow_function_dialog import FlowFunctionDialog

            definition = FlowFunctionDefinition.from_dict(def_dict)
            available_types = FlowEngine.get_available_nodes()
            available_types.extend(["FlowFunctionEntry", "FlowFunctionExit", "Parameter"])

            dialog = FlowFunctionDialog(
                definition=definition, available_node_types=available_types, parent=self
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                updated_def = dialog.get_definition()
                # Update in session (remove old, add new)
                self._core.session.remove_flow_function_definition(definition_id)
                self._core.session.add_flow_function_definition(updated_def.to_dict())
                self._refresh_flow_functions()
                logger.info(f"Updated flow function: {updated_def.name}")
        except ImportError as e:
            logger.warning(f"Could not import FlowFunctionDialog: {e}")
            QMessageBox.warning(self, "Not Available", "Flow function editor not available.")

    def _delete_flow_function(self, definition_id: str) -> None:
        """Delete a flow function definition."""
        if not self._core.session:
            return

        def_dict = self._core.session.get_flow_function_definition(definition_id)
        name = def_dict.get("name", "Unknown") if def_dict else "Unknown"

        result = QMessageBox.question(
            self,
            "Delete Flow Function",
            f"Are you sure you want to delete '{name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if result == QMessageBox.StandardButton.Yes:
            self._core.session.remove_flow_function_definition(definition_id)
            self._refresh_flow_functions()
            logger.info(f"Deleted flow function: {name}")

    def _add_node_to_center(self, node_type: str) -> None:
        """Add a node to the center of the graph view."""
        if hasattr(self, "_graph_view"):
            # Get center of current view
            center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
            self._graph_view.node_created.emit(node_type, center.x(), center.y())

    # Node graph event handlers
    def _on_node_created(self, node_type: str, x: float, y: float) -> None:
        """Handle node creation from graph view."""
        import uuid

        from glider.core.experiment_session import NodeConfig

        # Handle CustomDevice: and FlowFunction: prefixed types
        display_name = node_type
        actual_node_type = node_type
        definition_id = None
        initial_state = {}

        if node_type.startswith("CustomDevice:"):
            definition_id = node_type.split(":", 1)[1]
            actual_node_type = "CustomDevice"
            # Get device name for display
            if self._core.session:
                def_dict = self._core.session.get_custom_device_definition(definition_id)
                if def_dict:
                    display_name = def_dict.get("name", "Custom Device")
                    initial_state["definition_id"] = definition_id
        elif node_type.startswith("FunctionCall:"):
            # FunctionCall: is followed by the StartFunction node ID
            start_node_id = node_type.split(":", 1)[1]
            actual_node_type = "FunctionCall"
            # Get function name from the StartFunction node
            if self._core.session:
                start_node = self._core.session.get_node(start_node_id)
                if start_node and start_node.state:
                    display_name = start_node.state.get("function_name", "Function")
                else:
                    display_name = "Function"
                initial_state["function_start_id"] = start_node_id
                initial_state["function_name"] = display_name
        elif node_type.startswith("FlowFunction:"):
            definition_id = node_type.split(":", 1)[1]
            actual_node_type = "FlowFunctionCall"
            # Get function name for display
            if self._core.session:
                def_dict = self._core.session.get_flow_function_definition(definition_id)
                if def_dict:
                    display_name = def_dict.get("name", "Flow Function")
                    initial_state["definition_id"] = definition_id
        elif node_type.startswith("ZoneInput:"):
            # ZoneInput: is followed by zone_id
            zone_id = node_type.split(":", 1)[1]
            actual_node_type = "ZoneInput"
            # Get zone name for display
            for zone in self._zone_config.zones:
                if zone.id == zone_id:
                    display_name = f"Zone: {zone.name}"
                    initial_state["zone_id"] = zone_id
                    initial_state["zone_name"] = zone.name
                    break
            else:
                display_name = "Zone Input"
                initial_state["zone_id"] = zone_id

        node_type_normalized = actual_node_type.replace(" ", "")

        node_id = f"{actual_node_type.lower()}_{uuid.uuid4().hex[:8]}"

        # Determine category from node type
        category = "default"
        flow_nodes = ["StartExperiment", "EndExperiment", "Delay"]
        control_nodes = ["Loop", "WaitForInput"]
        io_nodes = ["Output", "Input", "MotorGovernor", "CustomDeviceAction"]
        function_nodes = ["FlowFunctionCall", "FunctionCall", "StartFunction", "EndFunction"]
        interface_nodes = ["ZoneInput"]

        if node_type_normalized in flow_nodes:
            category = "logic"  # Use blue color
        elif node_type_normalized in control_nodes:
            category = "interface"  # Use orange color for control nodes
        elif node_type_normalized in io_nodes:
            category = "hardware"  # Use green color
        elif node_type_normalized in function_nodes:
            category = "logic"  # Flow functions are logic nodes
        elif node_type_normalized in interface_nodes:
            category = "interface"  # Zone inputs are interface nodes

        # Add visual node to graph (use display_name for the visual)
        node_item = self._graph_view.add_node(node_id, display_name, x, y)
        node_item._category = category
        node_item._header_color = node_item.CATEGORY_COLORS.get(
            category, node_item.CATEGORY_COLORS["default"]
        )
        # Store the actual node type and definition ID for later use
        node_item._actual_node_type = actual_node_type
        node_item._definition_id = definition_id

        # Add default ports based on node type
        self._setup_node_ports(node_item, actual_node_type)

        # Connect port signals for connection creation (after ports are added)
        self._graph_view._connect_port_signals(node_item)

        # Add to session for persistence
        if self._core.session:
            node_config = NodeConfig(
                id=node_id,
                node_type=actual_node_type,
                position=(x, y),
                state=initial_state,
                device_id=None,
                visible_in_runner=category == "interface",
            )
            self._core.session.add_node(node_config)

        # Add to undo stack
        command = CreateNodeCommand(self, node_id, actual_node_type, x, y)
        self._undo_stack.push(command)
        self._update_undo_redo_actions()

        self.statusBar().showMessage(f"Created node: {display_name}", 2000)

    def _setup_node_ports(self, node_item, node_type: str) -> None:
        """Set up input/output ports for a node based on its type."""
        from glider.gui.node_graph.port_item import PortType

        # Normalize node type (handle spaces)
        nt = node_type.replace(" ", "")

        # Define ports for each node type
        # Format: (input_ports, output_ports)
        # Port names starting with ">" are EXEC type, others are DATA type
        port_configs = {
            # Flow nodes
            "StartExperiment": ([], [">next"]),  # No inputs, one exec output
            "EndExperiment": ([">exec"], []),  # One exec input, no outputs
            "Delay": ([">exec"], [">next"]),  # Exec in, exec out (duration set via properties)
            # Control nodes (properties controlled via panel, not ports)
            "Loop": ([">exec"], [">body", ">done"]),  # Exec in, body and done exec outputs
            "WaitForInput": (
                [">exec"],
                [">triggered"],
            ),  # Exec in, triggered exec output (timeout set via properties)
            # I/O nodes
            "Output": ([">exec"], [">next"]),  # Exec in, exec out (value set via properties)
            "Input": ([">exec"], ["value", ">next"]),  # Exec in, outputs value and exec
            "MotorGovernor": (
                [">exec"],
                [">next"],
            ),  # Exec in, exec out (action set via properties: up/down/stop)
            # Custom device action node
            "CustomDevice": ([">exec"], ["value", ">next"]),  # Exec in, value out, exec out
            "CustomDeviceAction": (
                [">exec"],
                ["value", ">next"],
            ),  # Alias for backward compatibility
            # Flow function definition nodes
            "StartFunction": ([], [">next"]),  # No inputs, exec out
            "EndFunction": ([">exec"], []),  # Exec in, no outputs
            # Flow function call node
            "FunctionCall": ([">exec"], [">next"]),  # Exec in, exec out
            "FlowFunctionCall": ([">exec"], [">next"]),  # Legacy alias
            # Zone input node - outputs zone state
            "ZoneInput": ([], ["Occupied", "Object Count", ">On Enter", ">On Exit"]),
        }

        inputs, outputs = port_configs.get(nt, ([">in"], [">out"]))

        for port_name in inputs:
            if port_name.startswith(">"):
                node_item.add_input_port(port_name[1:], PortType.EXEC)
            else:
                node_item.add_input_port(port_name, PortType.DATA)

        for port_name in outputs:
            if port_name.startswith(">"):
                node_item.add_output_port(port_name[1:], PortType.EXEC)
            else:
                node_item.add_output_port(port_name, PortType.DATA)

    def _on_node_deleted(self, node_id: str) -> None:
        """Handle node deletion from graph view."""
        # Save node data for undo before deletion
        node_data = {}
        node_item = self._graph_view.nodes.get(node_id)
        if node_item:
            node_data = {
                "id": node_id,
                "node_type": node_item.node_type,
                "x": node_item.pos().x(),
                "y": node_item.pos().y(),
            }

        # Get additional data from session
        if self._core.session:
            node_config = self._core.session.get_node(node_id)
            if node_config:
                node_data["state"] = node_config.state
                node_data["device_id"] = node_config.device_id
                node_data["visible_in_runner"] = node_config.visible_in_runner

            # Remove from session
            self._core.session.remove_node(node_id)

        # Add to undo stack
        command = DeleteNodeCommand(self, node_id, node_data)
        self._undo_stack.push(command)
        self._update_undo_redo_actions()

        self.statusBar().showMessage(f"Deleted node: {node_id}", 2000)

    def _on_node_selected(self, node_id: str) -> None:
        """Handle node selection from graph view."""
        self._update_properties_panel(node_id)
        self.statusBar().showMessage(f"Selected: {node_id}", 1000)

    def _on_node_moved(self, node_id: str, x: float, y: float) -> None:
        """Handle node movement from graph view."""
        if self._core.session:
            self._core.session.update_node_position(node_id, x, y)

    def _on_connection_created(
        self, from_node: str, from_port: int, to_node: str, to_port: int, conn_type: str = "data"
    ) -> None:
        """Handle connection creation from graph view."""
        import uuid

        from glider.core.experiment_session import ConnectionConfig

        connection_id = f"conn_{uuid.uuid4().hex[:8]}"

        # Add visual connection
        self._graph_view.add_connection(connection_id, from_node, from_port, to_node, to_port)

        # Add to session for persistence
        if self._core.session:
            conn_config = ConnectionConfig(
                id=connection_id,
                from_node=from_node,
                from_output=from_port,
                to_node=to_node,
                to_input=to_port,
                connection_type=conn_type,
            )
            self._core.session.add_connection(conn_config)
            logger.info(
                f"Saved connection: {from_node}:{from_port} -> {to_node}:{to_port} (type: {conn_type})"
            )

        # Add to undo stack
        command = CreateConnectionCommand(
            self, connection_id, from_node, from_port, to_node, to_port, conn_type
        )
        self._undo_stack.push(command)
        self._update_undo_redo_actions()

        self.statusBar().showMessage(f"Connected: {from_node} -> {to_node}", 2000)

        # Refresh flow functions in case a function definition was completed
        self._refresh_flow_functions()

    def _on_connection_deleted(self, connection_id: str) -> None:
        """Handle connection deletion from graph view."""
        # Save connection data for undo
        conn_data = {"id": connection_id}

        # Get connection data from session before deletion
        if self._core.session:
            conn_config = self._core.session.get_connection(connection_id)
            if conn_config:
                conn_data["from_node"] = conn_config.from_node
                conn_data["from_port"] = conn_config.from_output
                conn_data["to_node"] = conn_config.to_node
                conn_data["to_port"] = conn_config.to_input
                conn_data["conn_type"] = conn_config.connection_type

            self._core.session.remove_connection(connection_id)

        # Add to undo stack
        command = DeleteConnectionCommand(self, connection_id, conn_data)
        self._undo_stack.push(command)
        self._update_undo_redo_actions()

        self.statusBar().showMessage(f"Deleted connection: {connection_id}", 2000)

        # Refresh flow functions in case a function definition was broken
        self._refresh_flow_functions()

    def _update_properties_panel(self, node_id: str) -> None:
        """Update the properties panel for the selected node."""
        if not hasattr(self, "_properties_dock"):
            return

        # Get node from graph view
        node_item = self._graph_view.nodes.get(node_id)
        if node_item is None:
            return

        # Get node config from session to load saved values
        node_config = None
        if self._core.session:
            node_config = self._core.session.get_node(node_id)

        # Create properties widget
        props_widget = QWidget()
        props_layout = QFormLayout(props_widget)
        props_layout.setContentsMargins(8, 8, 8, 8)

        # Node info
        props_layout.addRow("ID:", QLabel(node_id))
        props_layout.addRow("Type:", QLabel(node_item.node_type))

        # Use actual node type if available (for custom devices/flow functions)
        # Otherwise use the display node type
        if hasattr(node_item, "_actual_node_type") and node_item._actual_node_type:
            node_type = node_item._actual_node_type.replace(" ", "")
        else:
            node_type = node_item.node_type.replace(" ", "")

        # Add device selector for I/O nodes, WaitForInput, and MotorGovernor
        if node_type in ["Output", "Input", "WaitForInput", "MotorGovernor"]:
            device_combo = QComboBox()
            device_combo.addItem("-- Select Device --", None)
            current_device_id = node_config.device_id if node_config else None
            current_index = 0

            for i, (dev_id, device) in enumerate(self._core.hardware_manager.devices.items()):
                device_name = getattr(device, "name", dev_id)
                device_type = getattr(device, "device_type", "")
                device_combo.addItem(f"{device_name} ({device_type})", dev_id)
                if dev_id == current_device_id:
                    current_index = i + 1  # +1 because of "-- Select Device --" item

            device_combo.setCurrentIndex(current_index)
            device_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=device_combo: self._on_node_device_changed(
                    nid, combo.currentData()
                )
            )
            props_layout.addRow("Device:", device_combo)

        # Add duration input for Delay node
        elif node_type == "Delay":
            duration_spin = QSpinBox()
            duration_spin.setRange(0, 3600)
            # Load saved duration from session
            saved_duration = 1
            if node_config and node_config.state:
                saved_duration = node_config.state.get("duration", 1)
            duration_spin.setValue(saved_duration)
            duration_spin.setSuffix(" sec")
            duration_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "duration", val)
            )
            props_layout.addRow("Duration:", duration_spin)

        # Add function name editor for StartFunction node
        elif node_type == "StartFunction":
            name_edit = QLineEdit()
            name_edit.setPlaceholderText("Enter function name")
            # Load saved function name from session
            saved_name = "MyFunction"
            if node_config and node_config.state:
                saved_name = node_config.state.get("function_name", "MyFunction")
            name_edit.setText(saved_name)
            name_edit.textChanged.connect(
                lambda text, nid=node_id: self._on_node_property_changed(nid, "function_name", text)
            )
            props_layout.addRow("Function Name:", name_edit)

            # Info label
            info_label = QLabel("Connect to EndFunction to define a reusable function.")
            info_label.setWordWrap(True)
            info_label.setStyleSheet("color: #888; font-size: 10px;")
            props_layout.addRow(info_label)

        # Add value control for Output node (PWM spinbox or HIGH/LOW radios)
        if node_type == "Output":
            # Determine if the bound device is a PWM output
            bound_device_type = None
            if node_config and node_config.device_id:
                bound_device = self._core.hardware_manager.get_device(node_config.device_id)
                if bound_device:
                    bound_device_type = getattr(bound_device, "device_type", None)

            if bound_device_type == "PWMOutput":
                # PWM device: show 0-255 spinbox
                pwm_spin = QSpinBox()
                pwm_spin.setRange(0, 255)
                saved_value = 0
                if node_config and node_config.state:
                    saved_value = node_config.state.get("value", 0)
                pwm_spin.setValue(int(saved_value))
                pwm_spin.valueChanged.connect(
                    lambda val, nid=node_id: self._on_node_property_changed(nid, "value", val)
                )
                props_layout.addRow("PWM Value (0-255):", pwm_spin)
            else:
                # Digital device: show HIGH/LOW radios
                from PyQt6.QtWidgets import QHBoxLayout, QRadioButton

                value_layout = QHBoxLayout()
                high_radio = QRadioButton("HIGH")
                low_radio = QRadioButton("LOW")

                # Load saved value from session (1=HIGH, 0=LOW)
                saved_value = 1  # Default to HIGH
                if node_config and node_config.state:
                    saved_value = node_config.state.get("value", 1)

                if saved_value:
                    high_radio.setChecked(True)
                else:
                    low_radio.setChecked(True)

                value_layout.addWidget(high_radio)
                value_layout.addWidget(low_radio)
                high_radio.toggled.connect(
                    lambda checked, nid=node_id: self._on_node_property_changed(
                        nid, "value", 1 if checked else 0
                    )
                )
                value_widget = QWidget()
                value_widget.setLayout(value_layout)
                props_layout.addRow("Value:", value_widget)

        # Add action selector for MotorGovernor node
        elif node_type == "MotorGovernor":
            action_combo = QComboBox()
            action_combo.addItem("Move Up", "up")
            action_combo.addItem("Move Down", "down")
            action_combo.addItem("Stop", "stop")

            # Load saved action from session
            saved_action = "stop"
            if node_config and node_config.state:
                saved_action = node_config.state.get("action", "stop")

            # Set current index based on saved action
            action_map = {"up": 0, "down": 1, "stop": 2}
            action_combo.setCurrentIndex(action_map.get(saved_action, 2))

            action_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=action_combo: self._on_node_property_changed(
                    nid, "action", combo.currentData()
                )
            )
            props_layout.addRow("Action:", action_combo)

        # Add properties for Loop node
        elif node_type == "Loop":
            # Loop count (0 = infinite)
            count_spin = QSpinBox()
            count_spin.setRange(0, 10000)
            count_spin.setSpecialValueText("Infinite")
            saved_count = 0
            if node_config and node_config.state:
                saved_count = node_config.state.get("count", 0)
            count_spin.setValue(saved_count)
            count_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "count", val)
            )
            props_layout.addRow("Iterations:", count_spin)

            # Loop delay
            delay_spin = QDoubleSpinBox()
            delay_spin.setRange(0.0, 3600.0)
            delay_spin.setDecimals(2)
            delay_spin.setSingleStep(0.1)
            saved_delay = 1.0
            if node_config and node_config.state:
                saved_delay = node_config.state.get("delay", 1.0)
            delay_spin.setValue(saved_delay)
            delay_spin.setSuffix(" sec")
            delay_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "delay", val)
            )
            props_layout.addRow("Delay:", delay_spin)

        # Add properties for WaitForInput node
        elif node_type == "WaitForInput":
            # Threshold Mode: Digital or Analog
            mode_combo = QComboBox()
            mode_combo.addItem("Digital (Rising Edge)", "digital")
            mode_combo.addItem("Analog (Threshold)", "analog")

            saved_mode = "digital"
            if node_config and node_config.state:
                saved_mode = node_config.state.get("threshold_mode", "digital")

            mode_combo.setCurrentIndex(0 if saved_mode == "digital" else 1)
            mode_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=mode_combo: self._on_node_property_changed(
                    nid, "threshold_mode", combo.currentData()
                )
            )
            props_layout.addRow("Mode:", mode_combo)

            # Threshold Value
            threshold_spin = QSpinBox()
            threshold_spin.setRange(0, 1023)
            saved_threshold = 512
            if node_config and node_config.state:
                saved_threshold = node_config.state.get("threshold", 512)
            threshold_spin.setValue(saved_threshold)
            threshold_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "threshold", val)
            )
            props_layout.addRow("Threshold:", threshold_spin)

            # Threshold Direction
            direction_combo = QComboBox()
            direction_combo.addItem("Above Threshold", "above")
            direction_combo.addItem("Below Threshold", "below")

            saved_direction = "above"
            if node_config and node_config.state:
                saved_direction = node_config.state.get("threshold_direction", "above")

            direction_combo.setCurrentIndex(0 if saved_direction == "above" else 1)
            direction_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=direction_combo: self._on_node_property_changed(
                    nid, "threshold_direction", combo.currentData()
                )
            )
            props_layout.addRow("Direction:", direction_combo)

            # Timeout (0 = no timeout)
            timeout_spin = QDoubleSpinBox()
            timeout_spin.setRange(0.0, 3600.0)
            timeout_spin.setDecimals(1)
            timeout_spin.setSpecialValueText("No timeout")
            saved_timeout = 0.0
            if node_config and node_config.state:
                saved_timeout = node_config.state.get("timeout", 0.0)
            timeout_spin.setValue(saved_timeout)
            timeout_spin.setSuffix(" sec")
            timeout_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "timeout", val)
            )
            props_layout.addRow("Timeout:", timeout_spin)

            # Info label
            info_label = QLabel(
                "Digital mode: waits for rising edge (LOW → HIGH)\n"
                "Analog mode: waits for value to cross threshold"
            )
            info_label.setWordWrap(True)
            info_label.setStyleSheet("color: #888; font-size: 10px; margin-top: 8px;")
            props_layout.addRow(info_label)

        # Add properties for AnalogRead node
        elif node_type == "AnalogRead":
            # Continuous reading mode
            continuous_check = QCheckBox("Enable continuous reading")
            saved_continuous = False
            if node_config and node_config.state:
                saved_continuous = node_config.state.get("continuous", False)
            continuous_check.setChecked(saved_continuous)
            continuous_check.toggled.connect(
                lambda checked, nid=node_id: self._on_node_property_changed(
                    nid, "continuous", checked
                )
            )
            props_layout.addRow(continuous_check)

            # Poll interval
            poll_spin = QDoubleSpinBox()
            poll_spin.setRange(0.01, 10.0)
            poll_spin.setDecimals(2)
            poll_spin.setSingleStep(0.05)
            saved_poll = 0.05
            if node_config and node_config.state:
                saved_poll = node_config.state.get("poll_interval", 0.05)
            poll_spin.setValue(saved_poll)
            poll_spin.setSuffix(" sec")
            poll_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "poll_interval", val)
            )
            props_layout.addRow("Poll Interval:", poll_spin)

            # Threshold enabled
            threshold_check = QCheckBox("Enable threshold checking")
            saved_threshold_enabled = False
            if node_config and node_config.state:
                saved_threshold_enabled = node_config.state.get("threshold_enabled", False)
            threshold_check.setChecked(saved_threshold_enabled)
            threshold_check.toggled.connect(
                lambda checked, nid=node_id: self._on_node_property_changed(
                    nid, "threshold_enabled", checked
                )
            )
            props_layout.addRow(threshold_check)

            # Threshold value
            threshold_spin = QSpinBox()
            threshold_spin.setRange(0, 1023)
            saved_threshold = 512
            if node_config and node_config.state:
                saved_threshold = node_config.state.get("threshold", 512)
            threshold_spin.setValue(saved_threshold)
            threshold_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "threshold", val)
            )
            props_layout.addRow("Threshold:", threshold_spin)

            # Show in runner dashboard
            visible_check = QCheckBox("Show live value in dashboard")
            saved_visible = False
            if node_config and node_config.state:
                saved_visible = node_config.state.get("visible_in_runner", False)
            visible_check.setChecked(saved_visible)
            visible_check.toggled.connect(
                lambda checked, nid=node_id: self._on_node_property_changed(
                    nid, "visible_in_runner", checked
                )
            )
            props_layout.addRow(visible_check)

            # Info label
            info_label = QLabel(
                "Continuous mode: automatically polls sensor at poll interval.\n"
                "Threshold: output 'threshold_exceeded' will be True when value > threshold.\n"
                "Dashboard: enable to show live analog value in runner view."
            )
            info_label.setWordWrap(True)
            info_label.setStyleSheet("color: #888; font-size: 10px; margin-top: 8px;")
            props_layout.addRow(info_label)

        # Add properties for CustomDevice/CustomDeviceAction node (shows device info and pins)
        elif node_type in ("CustomDevice", "CustomDeviceAction"):
            # Get the definition ID from node state or node_item
            definition_id = None
            if node_config and node_config.state:
                definition_id = node_config.state.get("definition_id")
            if not definition_id and hasattr(node_item, "_definition_id"):
                definition_id = node_item._definition_id

            if definition_id and self._core.session:
                def_dict = self._core.session.get_custom_device_definition(definition_id)
                if def_dict:
                    # Show device name
                    device_name = def_dict.get("name", "Unknown")
                    props_layout.addRow("Device:", QLabel(device_name))

                    # Show description if any
                    desc = def_dict.get("description", "")
                    if desc:
                        desc_label = QLabel(desc)
                        desc_label.setWordWrap(True)
                        props_layout.addRow("Description:", desc_label)

                    # Pin selector dropdown
                    pins = def_dict.get("pins", [])
                    if pins:
                        pin_combo = QComboBox()
                        pin_combo.addItem("(Select a pin)", "")
                        for pin in pins:
                            pin_name = pin.get("name", "")
                            pin_number = pin.get("pin_number")
                            pin_type = pin.get("pin_type", "")
                            pin_desc = pin.get("description", "")
                            # Show pin number if available
                            if pin_number is not None:
                                display_text = f"{pin_name} [Pin {pin_number}] ({pin_type})"
                            else:
                                display_text = f"{pin_name} ({pin_type})"
                            if pin_desc:
                                display_text += f" - {pin_desc}"
                            pin_combo.addItem(display_text, pin_name)

                        # Load saved pin selection
                        saved_pin = ""
                        if node_config and node_config.state:
                            saved_pin = node_config.state.get("pin", "")

                        # Find and set current index
                        for i in range(pin_combo.count()):
                            if pin_combo.itemData(i) == saved_pin:
                                pin_combo.setCurrentIndex(i)
                                break

                        pin_combo.currentIndexChanged.connect(
                            lambda idx, nid=node_id, combo=pin_combo: self._on_node_property_changed(
                                nid, "pin", combo.currentData()
                            )
                        )
                        props_layout.addRow("Pin:", pin_combo)

                        # Value control for output pins
                        saved_pin_type = None
                        for pin in pins:
                            if pin.get("name") == saved_pin:
                                saved_pin_type = pin.get("pin_type")
                                break

                        if saved_pin_type in ("digital_output",):
                            value_combo = QComboBox()
                            value_combo.addItem("LOW (0)", 0)
                            value_combo.addItem("HIGH (1)", 1)

                            saved_value = 0
                            if node_config and node_config.state:
                                saved_value = node_config.state.get("value", 0)
                            value_combo.setCurrentIndex(1 if saved_value else 0)

                            value_combo.currentIndexChanged.connect(
                                lambda idx, nid=node_id, combo=value_combo: self._on_node_property_changed(
                                    nid, "value", combo.currentData()
                                )
                            )
                            props_layout.addRow("Value:", value_combo)

                        elif saved_pin_type in ("analog_output", "pwm"):
                            value_spin = QSpinBox()
                            value_spin.setRange(0, 255)
                            saved_value = 0
                            if node_config and node_config.state:
                                saved_value = node_config.state.get("value", 0)
                            value_spin.setValue(int(saved_value))

                            value_spin.valueChanged.connect(
                                lambda val, nid=node_id: self._on_node_property_changed(
                                    nid, "value", val
                                )
                            )
                            props_layout.addRow("Value:", value_spin)

                    # Edit device button
                    edit_btn = QPushButton("Edit Device Definition")
                    edit_btn.clicked.connect(
                        lambda checked, did=definition_id: self._edit_custom_device(did)
                    )
                    props_layout.addRow(edit_btn)
            else:
                props_layout.addRow(QLabel("(Custom device not found)"))

        # Add properties for FlowFunctionCall node
        elif node_type == "FlowFunctionCall":
            # Get the definition ID from node state
            definition_id = None
            if node_config and node_config.state:
                definition_id = node_config.state.get("definition_id")

            if definition_id and self._core.session:
                def_dict = self._core.session.get_flow_function_definition(definition_id)
                if def_dict:
                    # Show function name
                    func_name = def_dict.get("name", "Unknown")
                    props_layout.addRow("Function:", QLabel(func_name))

                    # Show description if any
                    desc = def_dict.get("description", "")
                    if desc:
                        desc_label = QLabel(desc)
                        desc_label.setWordWrap(True)
                        props_layout.addRow("Description:", desc_label)

                    # Edit function button
                    edit_btn = QPushButton("Edit Flow Function")
                    edit_btn.clicked.connect(lambda: self._edit_flow_function(definition_id))
                    props_layout.addRow(edit_btn)
            else:
                props_layout.addRow(QLabel("(Flow function not found)"))

        self._properties_dock.setWidget(props_widget)

    def _on_node_device_changed(self, node_id: str, device_id: str) -> None:
        """Handle device selection change for a node."""
        if self._core.session:
            node_config = self._core.session.get_node(node_id)
            if node_config:
                # Update session config for persistence
                node_config.device_id = device_id
                self._core.session._mark_dirty()
                logger.info(f"Node {node_id} device changed to: {device_id}")

                # Also bind device to the runtime node in flow engine
                if device_id and hasattr(self._core, "flow_engine") and self._core.flow_engine:
                    runtime_node = self._core.flow_engine.get_node(node_id)
                    if runtime_node and hasattr(runtime_node, "bind_device"):
                        device = self._core.hardware_manager.get_device(device_id)
                        if device:
                            runtime_node.bind_device(device)
                            logger.info(f"Bound device '{device_id}' to runtime node {node_id}")

                # Refresh properties panel to update value controls
                # (e.g., switching between HIGH/LOW and PWM spinbox)
                self._update_properties_panel(node_id)

    def _on_node_property_changed(self, node_id: str, prop_name: str, value) -> None:
        """Handle property change for a node."""
        if self._core.session:
            self._core.session.update_node_state(node_id, {prop_name: value})
            logger.info(f"Node {node_id} property '{prop_name}' changed to: {value}")

            # Refresh flow functions if function name changed
            if prop_name == "function_name":
                self._refresh_flow_functions()

    def _run_async(self, coro) -> asyncio.Task:
        """
        Run an async coroutine with proper task tracking.

        This prevents garbage collection of the task before completion
        and ensures proper cleanup on application exit.
        """
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    # Undo/Redo methods
    def _on_undo(self) -> None:
        """Handle undo action."""
        command = self._undo_stack.undo()
        if command:
            self.statusBar().showMessage(f"Undo: {command.description()}", 2000)
            self._update_undo_redo_actions()

    def _on_redo(self) -> None:
        """Handle redo action."""
        command = self._undo_stack.redo()
        if command:
            # For redo, we need to re-apply the action
            # Most commands just need to be re-created
            self._redo_command(command)
            self.statusBar().showMessage(f"Redo: {command.description()}", 2000)
            self._update_undo_redo_actions()

    def _redo_command(self, command: Command) -> None:
        """Re-apply a command for redo."""
        if isinstance(command, CreateNodeCommand):
            # Re-create the node
            self._on_node_created(command._node_type, command._x, command._y)
            # Remove the duplicate undo entry that was just added
            if self._undo_stack._undo_stack:
                self._undo_stack._undo_stack.pop()
        elif isinstance(command, DeleteNodeCommand):
            # Re-delete the node
            node_id = command._node_id
            self._graph_view.remove_node(node_id)
            if self._core.session:
                self._core.session.remove_node(node_id)
        elif isinstance(command, MoveNodeCommand):
            # Re-apply the move
            node_item = self._graph_view.nodes.get(command._node_id)
            if node_item:
                node_item.setPos(command._new_x, command._new_y)
            if self._core.session:
                self._core.session.update_node_position(
                    command._node_id, command._new_x, command._new_y
                )
        elif isinstance(command, CreateConnectionCommand):
            # Re-create the connection
            self._graph_view.add_connection(
                command._conn_id,
                command._from_node,
                command._from_port,
                command._to_node,
                command._to_port,
            )
            if self._core.session:
                from glider.core.experiment_session import ConnectionConfig

                conn_config = ConnectionConfig(
                    id=command._conn_id,
                    from_node=command._from_node,
                    from_output=command._from_port,
                    to_node=command._to_node,
                    to_input=command._to_port,
                    connection_type=command._conn_type,
                )
                self._core.session.add_connection(conn_config)
        elif isinstance(command, DeleteConnectionCommand):
            # Re-delete the connection
            self._graph_view.remove_connection(command._conn_id)
            if self._core.session:
                self._core.session.remove_connection(command._conn_id)
        elif isinstance(command, PropertyChangeCommand):
            # Re-apply the property change
            if self._core.session:
                self._core.session.update_node_state(
                    command._node_id, {command._prop_name: command._new_value}
                )

    def _update_undo_redo_actions(self) -> None:
        """Update the enabled state of undo/redo menu actions."""
        if hasattr(self, "_undo_action"):
            can_undo = self._undo_stack.can_undo()
            self._undo_action.setEnabled(can_undo)
            if can_undo:
                self._undo_action.setText(f"&Undo {self._undo_stack.undo_description()}")
            else:
                self._undo_action.setText("&Undo")

        if hasattr(self, "_redo_action"):
            can_redo = self._undo_stack.can_redo()
            self._redo_action.setEnabled(can_redo)
            if can_redo:
                self._redo_action.setText(f"&Redo {self._undo_stack.redo_description()}")
            else:
                self._redo_action.setText("&Redo")

    # Device Control Panel methods
    def _refresh_device_combo(self) -> None:
        """Refresh the device selector combo box."""
        if not hasattr(self, "_device_combo"):
            return

        self._device_combo.clear()
        self._device_combo.addItem("-- Select Device --", None)

        for device_id, device in self._core.hardware_manager.devices.items():
            device_name = getattr(device, "name", device_id)
            device_type = getattr(device, "device_type", "unknown")
            self._device_combo.addItem(f"{device_name} ({device_type})", device_id)

    def _on_device_selected(self, text: str) -> None:
        """Handle device selection change."""
        # Stop any continuous input reading
        if hasattr(self, "_input_poll_timer"):
            self._input_poll_timer.stop()
        if hasattr(self, "_continuous_checkbox"):
            self._continuous_checkbox.setChecked(False)
        if hasattr(self, "_input_value_label"):
            self._input_value_label.setText("--")

        device_id = self._device_combo.currentData()
        if device_id is None:
            self._device_status_label.setText("Status: No device selected")
            if hasattr(self, "_input_group"):
                self._input_group.setEnabled(False)
            return

        device = self._core.hardware_manager.get_device(device_id)
        if device is None:
            self._device_status_label.setText("Status: Device not found")
            if hasattr(self, "_input_group"):
                self._input_group.setEnabled(False)
            return

        device_type = getattr(device, "device_type", "unknown")
        board = getattr(device, "board", None)
        connected = board.is_connected if board else False
        initialized = getattr(device, "_initialized", False)

        status = "Connected" if connected else "Disconnected"
        if connected and initialized:
            status = "Ready"
        elif connected and not initialized:
            status = "Not initialized"

        self._device_status_label.setText(f"Status: {status} | Type: {device_type}")

        # Enable/disable input reading based on device type
        if hasattr(self, "_input_group"):
            is_input_device = device_type in ("DigitalInput", "AnalogInput", "ADS1115")
            self._input_group.setEnabled(is_input_device)

        # Show/hide appropriate output controls based on device type
        is_digital_output = device_type == "DigitalOutput"
        is_pwm_output = device_type == "PWMOutput"
        is_output = is_digital_output or is_pwm_output

        if hasattr(self, "_control_group"):
            self._control_group.setVisible(is_output)
        if hasattr(self, "_digital_widget"):
            self._digital_widget.setVisible(is_digital_output)
        if hasattr(self, "_pwm_widget"):
            self._pwm_widget.setVisible(is_pwm_output)

    def _get_selected_device(self):
        """Get the currently selected device."""
        device_id = self._device_combo.currentData()
        if device_id is None:
            return None
        return self._core.hardware_manager.get_device(device_id)

    def _set_digital_output(self, value: bool) -> None:
        """Set digital output to HIGH or LOW."""
        device = self._get_selected_device()
        if device is None:
            QMessageBox.warning(self, "No Device", "Please select a device first.")
            return

        async def set_output():
            try:
                if hasattr(device, "set_state"):
                    await device.set_state(value)
                elif hasattr(device, "turn_on") and hasattr(device, "turn_off"):
                    if value:
                        await device.turn_on()
                    else:
                        await device.turn_off()
                else:
                    # Direct pin write
                    pin = list(device.pins.values())[0] if device.pins else 0
                    await device.board.write_digital(pin, value)

                state = "ON" if value else "OFF"
                self._device_status_label.setText(f"Status: Output set to {state}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to set output: {e}")

        self._run_async(set_output())

    def _toggle_digital_output(self) -> None:
        """Toggle digital output."""
        device = self._get_selected_device()
        if device is None:
            QMessageBox.warning(self, "No Device", "Please select a device first.")
            return

        async def toggle():
            try:
                if hasattr(device, "toggle"):
                    await device.toggle()
                elif hasattr(device, "state"):
                    new_state = not device.state
                    if hasattr(device, "set_state"):
                        await device.set_state(new_state)
                self._device_status_label.setText("Status: Output toggled")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to toggle: {e}")

        self._run_async(toggle())

    def _on_pwm_changed(self, value: int) -> None:
        """Handle PWM slider change."""
        device = self._get_selected_device()
        if device is None:
            return

        async def set_pwm():
            try:
                if hasattr(device, "set_value"):
                    await device.set_value(value)
                elif hasattr(device, "board"):
                    pin = list(device.pins.values())[0] if device.pins else 0
                    await device.board.write_analog(pin, value)
                self._device_status_label.setText(f"Status: PWM set to {value}")
            except Exception as e:
                logger.error(f"PWM error: {e}")

        self._run_async(set_pwm())

    def _on_servo_changed(self, angle: int) -> None:
        """Handle servo slider change."""
        device = self._get_selected_device()
        if device is None:
            return

        async def set_servo():
            try:
                if hasattr(device, "set_angle"):
                    await device.set_angle(angle)
                elif hasattr(device, "board"):
                    pin = list(device.pins.values())[0] if device.pins else 0
                    await device.board.write_servo(pin, angle)
                self._device_status_label.setText(f"Status: Servo set to {angle}°")
            except Exception as e:
                logger.error(f"Servo error: {e}")

        self._run_async(set_servo())

    def _read_input_once(self) -> None:
        """Read the input value once."""
        device = self._get_selected_device()
        if device is None:
            QMessageBox.warning(self, "No Device", "Please select a device first.")
            return

        device_type = getattr(device, "device_type", "")
        if device_type not in ("DigitalInput", "AnalogInput", "ADS1115"):
            QMessageBox.warning(
                self,
                "Invalid Device",
                "Please select a DigitalInput, AnalogInput, or ADS1115 device.",
            )
            return

        async def read_value():
            try:
                # Auto-initialize if not initialized
                if not getattr(device, "_initialized", False):
                    self._device_status_label.setText("Status: Initializing device...")
                    await device.initialize()
                    logger.info(f"Auto-initialized device for reading: {device.id}")

                if device_type == "DigitalInput":
                    if hasattr(device, "read"):
                        value = await device.read()
                        display = "HIGH (1)" if value else "LOW (0)"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(f"Status: Digital input = {display}")
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        value = await device.board.read_digital(pin)
                        display = "HIGH (1)" if value else "LOW (0)"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(f"Status: Digital input = {display}")
                elif device_type == "AnalogInput":
                    if hasattr(device, "read") and hasattr(device, "read_voltage"):
                        raw_value = await device.read()
                        voltage = await device.read_voltage()
                        display = f"{raw_value}\n{voltage:.2f}V"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(
                            f"Status: Analog = {raw_value} ({voltage:.2f}V)"
                        )
                    elif hasattr(device, "read"):
                        raw_value = await device.read()
                        voltage = (raw_value / 1023.0) * 5.0
                        display = f"{raw_value}\n{voltage:.2f}V"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(
                            f"Status: Analog = {raw_value} ({voltage:.2f}V)"
                        )
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        raw_value = await device.board.read_analog(pin)
                        voltage = (raw_value / 1023.0) * 5.0
                        display = f"{raw_value}\n{voltage:.2f}V"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(
                            f"Status: Analog = {raw_value} ({voltage:.2f}V)"
                        )
                elif device_type == "ADS1115":
                    # ADS1115 16-bit ADC via I2C
                    channel = device._config.settings.get("channel", 0)
                    raw_value = await device.read(channel)
                    voltage = await device.read_voltage(channel)
                    display = f"{raw_value}\n{voltage:.3f}V"
                    self._input_value_label.setText(display)
                    self._device_status_label.setText(
                        f"Status: ADS1115 Ch{channel} = {raw_value} ({voltage:.3f}V)"
                    )
            except Exception as e:
                logger.error(f"Read error: {e}")
                self._input_value_label.setText("ERROR")
                self._device_status_label.setText(f"Status: Read failed - {e}")

        self._run_async(read_value())

    def _on_continuous_changed(self, state: int) -> None:
        """Handle continuous checkbox state change."""
        if state == Qt.CheckState.Checked.value:
            device = self._get_selected_device()
            if device is None:
                self._continuous_checkbox.setChecked(False)
                QMessageBox.warning(self, "No Device", "Please select a device first.")
                return

            device_type = getattr(device, "device_type", "")
            if device_type not in ("DigitalInput", "AnalogInput", "ADS1115"):
                self._continuous_checkbox.setChecked(False)
                QMessageBox.warning(
                    self,
                    "Invalid Device",
                    "Please select a DigitalInput, AnalogInput, or ADS1115 device.",
                )
                return

            if device_type == "AnalogInput":
                # Use real-time callbacks for analog inputs (much faster than polling)
                self._start_analog_callback(device)
            elif device_type == "ADS1115":
                # ADS1115 uses I2C polling (no pin-based callbacks)
                interval = self._poll_spinbox.value()
                self._input_poll_timer.start(interval)
                self._device_status_label.setText(f"Status: ADS1115 polling ({interval}ms)")
            else:
                # Use polling for digital inputs
                interval = self._poll_spinbox.value()
                self._input_poll_timer.start(interval)
                self._device_status_label.setText(f"Status: Continuous reading ({interval}ms)")
        else:
            # Stop both callback and polling
            self._stop_analog_callback()
            self._input_poll_timer.stop()
            self._device_status_label.setText("Status: Continuous reading stopped")

    def _start_analog_callback(self, device) -> None:
        """Start real-time analog monitoring using board callbacks."""
        # First, stop any existing callback
        self._stop_analog_callback()

        pin = device.pins.get("input", list(device.pins.values())[0])
        board = device.board

        # Create a callback that emits the Qt signal (thread-safe)
        def analog_callback(callback_pin: int, value: int) -> None:
            # This runs in the telemetrix thread - emit signal to cross to Qt thread
            # Emit signal to update UI from main thread
            self.analog_value_received.emit(callback_pin, value)

        # Register the callback with the board
        board.register_callback(pin, analog_callback)
        logger.debug(f"Registered analog UI callback for pin {pin}")

        # Track the registration so we can unregister later
        self._analog_callback_board = board
        self._analog_callback_pin = pin
        self._analog_callback_func = analog_callback

        self._device_status_label.setText("Status: Real-time monitoring (callback)")
        logger.info(f"Started real-time analog callback for pin {pin}")

    def _stop_analog_callback(self) -> None:
        """Stop real-time analog monitoring."""
        if self._analog_callback_board is not None and self._analog_callback_func is not None:
            try:
                self._analog_callback_board.unregister_callback(
                    self._analog_callback_pin, self._analog_callback_func
                )
                logger.info(f"Stopped analog callback for pin {self._analog_callback_pin}")
            except Exception as e:
                logger.debug(f"Error unregistering callback: {e}")

        self._analog_callback_board = None
        self._analog_callback_pin = None
        self._analog_callback_func = None

    @pyqtSlot(int, int)
    def _on_analog_value_received(self, pin: int, value: int) -> None:
        """Handle real-time analog value updates (called via Qt signal from callback)."""
        # Get reference voltage from current device if available
        device = self._get_selected_device()
        if device is not None and hasattr(device, "_reference_voltage"):
            ref_voltage = device._reference_voltage
        else:
            ref_voltage = 5.0

        voltage = (value / 1023.0) * ref_voltage
        display = f"{value}\n{voltage:.2f}V"
        self._input_value_label.setText(display)

    def _on_poll_interval_changed(self, value: int) -> None:
        """Handle poll interval change."""
        if self._input_poll_timer.isActive():
            self._input_poll_timer.setInterval(value)
            self._device_status_label.setText(f"Status: Poll interval changed to {value}ms")

    def _poll_input(self) -> None:
        """Poll the input value (called by timer)."""
        device = self._get_selected_device()
        if device is None:
            self._input_poll_timer.stop()
            self._continuous_checkbox.setChecked(False)
            return

        device_type = getattr(device, "device_type", "")
        if device_type not in ("DigitalInput", "AnalogInput", "ADS1115"):
            self._input_poll_timer.stop()
            self._continuous_checkbox.setChecked(False)
            return

        async def read_value():
            try:
                if device_type == "DigitalInput":
                    if hasattr(device, "read"):
                        value = await device.read()
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        value = await device.board.read_digital(pin)
                    display = "HIGH (1)" if value else "LOW (0)"
                    self._input_value_label.setText(display)
                elif device_type == "AnalogInput":
                    if hasattr(device, "read"):
                        raw_value = await device.read()
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        raw_value = await device.board.read_analog(pin)

                    # Get voltage
                    if hasattr(device, "_reference_voltage"):
                        ref_voltage = device._reference_voltage
                    else:
                        ref_voltage = 5.0
                    voltage = (raw_value / 1023.0) * ref_voltage
                    display = f"{raw_value}\n{voltage:.2f}V"
                    self._input_value_label.setText(display)
                elif device_type == "ADS1115":
                    # ADS1115 has read() and read_voltage() methods
                    # Get channel from settings (default 0)
                    channel = device._config.settings.get("channel", 0)
                    raw_value = await device.read(channel)
                    voltage = await device.read_voltage(channel)
                    # ADS1115 is 16-bit, show full value
                    display = f"{raw_value}\n{voltage:.3f}V"
                    self._input_value_label.setText(display)
            except Exception as e:
                logger.error(f"Poll read error: {e}")
                self._input_poll_timer.stop()
                self._continuous_checkbox.setChecked(False)
                self._input_value_label.setText("ERROR")

        self._run_async(read_value())

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Stop device refresh timer
        if hasattr(self, "_device_refresh_timer"):
            self._device_refresh_timer.stop()

        # Stop input poll timer
        if hasattr(self, "_input_poll_timer"):
            self._input_poll_timer.stop()

        # Stop analog callback (if initialized)
        if hasattr(self, "_analog_callback_board"):
            self._stop_analog_callback()

        if self._check_save():
            # Cancel any pending async tasks
            for task in self._pending_tasks:
                if not task.done():
                    task.cancel()
            self._pending_tasks.clear()

            # Shutdown core - must complete before closing to ensure devices are LOW
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    # Loop not running, can use run_until_complete
                    loop.run_until_complete(self._core.shutdown())
                else:
                    # Loop is running (qasync) - schedule and wait with processEvents
                    future = asyncio.ensure_future(self._core.shutdown())
                    import time

                    from PyQt6.QtWidgets import QApplication

                    # Process events until shutdown completes (max 10 seconds)
                    timeout = time.time() + 10
                    while not future.done() and time.time() < timeout:
                        QApplication.processEvents()
                        time.sleep(0.01)  # Small sleep to prevent busy loop
                    if not future.done():
                        logger.warning("Shutdown timed out")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
            event.accept()
        else:
            event.ignore()
