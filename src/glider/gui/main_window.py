"""
Main Window - The primary PyQt6 window for GLIDER.

Thin coordinator that manages the high-level layout, view switching,
and signal wiring between extracted panel components.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from glider.core.config import get_config
from glider.gui.commands import UndoStack
from glider.gui.dialogs.calibration_dialog import CalibrationDialog
from glider.gui.dialogs.camera_settings_dialog import CameraSettingsDialog
from glider.gui.dialogs.experiment_dialog import ExperimentDialog
from glider.gui.dialogs.help_dialog import HelpDialog
from glider.gui.dialogs.subject_dialog import SubjectDialog
from glider.gui.dialogs.zone_dialog import ZoneDialog
from glider.gui.node_graph.graph_view import NodeGraphView
from glider.gui.panels.camera_panel import CameraPanel
from glider.gui.panels.device_control_panel import DeviceControlPanel
from glider.gui.panels.hardware_panel import HardwarePanel
from glider.gui.panels.node_editor_controller import NodeEditorController
from glider.gui.panels.node_library_panel import NodeLibraryPanel
from glider.gui.panels.runner_panel import RunnerPanel
from glider.gui.runner.runner_setup_page import RunnerSetupPage
from glider.gui.styles import colors
from glider.gui.view_manager import ViewManager, ViewMode
from glider.hal.base_board import BoardConnectionState
from glider.vision.zones import ZoneConfiguration

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)


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

    # Internal cross-thread marshalling signals
    _core_state_changed = pyqtSignal(object)
    _core_error_occurred = pyqtSignal(str, object)
    _hardware_connection_changed = pyqtSignal(str, object)

    def __init__(
        self,
        core: "GliderCore",
        view_manager: ViewManager | None = None,
        view_mode: ViewMode = ViewMode.AUTO,
    ):
        super().__init__()

        # Set per-window icon as a fallback for window managers (notably some
        # Wayland compositors and older KDE) that read the window icon rather
        # than the QApplication-level icon. Safe even if the app-level icon
        # was already applied in __main__.
        try:
            from glider.assets import get_app_icon

            self.setWindowIcon(get_app_icon())
        except Exception:  # pragma: no cover — cosmetic fallback
            pass

        self._core = core
        if view_manager is not None:
            self._view_manager = view_manager
        else:
            self._view_manager = ViewManager(None)
            self._view_manager.mode = view_mode

        # Undo/Redo stack
        self._undo_stack = UndoStack()

        # Async task tracking
        self._pending_tasks: set = set()

        # UI components
        self._stack: QStackedWidget | None = None
        self._builder_view: QWidget | None = None
        self._node_library_dock: QDockWidget | None = None
        self._properties_dock: QDockWidget | None = None

        # Toolbar status (initialised here so _on_core_state_change can test)
        self._toolbar_status: QLabel | None = None

        # Status bar widgets
        self._conn_dot: QLabel | None = None
        self._conn_label: QLabel | None = None
        self._state_label: QLabel | None = None
        self._stats_label: QLabel | None = None

        # Zone configuration
        self._zone_config = ZoneConfiguration()

        # Reconnection retry tracking
        self._reconnect_retries: dict[str, int] = {}
        self._max_reconnect_retries = 3

        # Panels (created in _setup_ui)
        self._hardware_panel: HardwarePanel | None = None
        self._device_control_panel: DeviceControlPanel | None = None
        self._node_library_panel: NodeLibraryPanel | None = None
        self._runner_panel: RunnerPanel | None = None
        self._node_editor: NodeEditorController | None = None
        self._camera_panel: CameraPanel | None = None
        # Lazily created when the camera panel hands off a finished video
        # tracking run for review (analysis_requested signal).
        self._analysis_dock: QDockWidget | None = None
        self._analysis_panel = None  # AnalysisPanel, imported + created lazily

        # Experiment dialog
        self._experiment_dialog: ExperimentDialog | None = None

        # Update checker — created eagerly so the Help → Check for Updates
        # action and the silent startup check share the same instance and
        # thereby coalesce if both fire at once. Lazy-imported to keep the
        # module import cheap in test contexts that don't need it.
        from glider.updater import UpdateChecker

        self._update_checker = UpdateChecker(self)

        # Setup UI
        self._setup_window()
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._connect_signals()

        # Apply stylesheet
        self._view_manager.apply_stylesheet(self)

        logger.info(f"MainWindow initialized in {self._view_manager.mode.name} mode")

    # --- Properties ---

    @property
    def core(self) -> "GliderCore":
        return self._core

    @property
    def view_manager(self) -> ViewManager:
        return self._view_manager

    @property
    def is_runner_mode(self) -> bool:
        return self._view_manager.is_runner_mode

    # --- Window setup ---

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle("GLIDER - General Laboratory Interface")
        config = get_config()

        if self._view_manager.is_runner_mode:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            screen = QApplication.primaryScreen()
            if screen:
                self.setGeometry(screen.geometry())
            self.show()
        else:
            self.setMinimumSize(config.ui.min_window_width, config.ui.min_window_height)
            self.resize(config.ui.default_window_width, config.ui.default_window_height)

    def _setup_ui(self) -> None:
        """Set up the main UI components."""
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._create_builder_view()
        self._create_runner_view()

        self._stack.addWidget(self._builder_view)  # Index 0
        self._stack.addWidget(self._runner_shell)  # Index 1

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

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._graph_view = NodeGraphView()
        self._graph_view.setMinimumSize(400, 300)

        # Create node editor controller
        self._node_editor = NodeEditorController(
            graph_view=self._graph_view,
            session_fn=lambda: self._core.session,
            hardware_manager=self._core.hardware_manager,
            undo_stack=self._undo_stack,
            core=self._core,
        )
        self._node_editor.connect_graph_signals()
        self._node_editor.set_zone_configuration(self._zone_config)

        # Connect controller signals
        self._node_editor.status_message.connect(self._show_status_message)
        self._node_editor.undo_redo_changed.connect(self._update_undo_redo_actions)

        splitter.addWidget(self._graph_view)
        layout.addWidget(splitter)

    def _create_runner_view(self) -> None:
        """Create the runner (dashboard) view."""
        self._runner_panel = RunnerPanel(self._core, self._view_manager)

        # Connect runner panel signals
        self._runner_panel.start_requested.connect(self._on_start_clicked)
        self._runner_panel.stop_requested.connect(self._on_stop_clicked)

        # Runner-mode Hardware Panel (also reused by desktop dock setup — see
        # _setup_dock_widgets, which re-hosts this same instance rather than
        # constructing a second one).
        self._hardware_panel = HardwarePanel(
            hardware_manager=self._core.hardware_manager,
            session_fn=lambda: self._core.session,
            run_async_fn=self._run_async,
        )
        self._hardware_panel.status_message.connect(self._show_status_message)

        # Runner-mode Camera Panel (also reused by desktop dock setup).
        self._camera_panel = self._build_camera_panel()

        # Manual control page + tab container
        from glider.gui.runner.manual_control_panel import ManualControlPanel
        from glider.gui.runner.manual_control_runner import ManualControlRunner
        from glider.gui.runner.runner_shell import RunnerShell

        self._manual_control_panel = ManualControlPanel(self._core)
        self._manual_control_runner = ManualControlRunner(self._core)
        self._manual_control_panel.function_run_requested.connect(self._on_manual_run)
        self._manual_control_panel.function_run_requested_param.connect(self._on_manual_run_param)
        self._manual_control_panel.set_digital_requested.connect(
            lambda dev_id, v: self._run_async(self._drive_digital(dev_id, v))
        )
        self._manual_control_panel.toggle_digital_requested.connect(
            lambda dev_id: self._run_async(self._drive_toggle(dev_id))
        )
        self._manual_control_panel.set_pwm_requested.connect(
            lambda dev_id, v: self._run_async(self._drive_pwm(dev_id, v))
        )

        self._runner_setup_page = RunnerSetupPage(self._core, hardware_widget=self._hardware_panel)

        self._runner_shell = RunnerShell(
            self._core,
            self._runner_setup_page,
            self._runner_panel,
            self._manual_control_panel,
            self._camera_panel,
        )
        self._runner_panel.elapsed_updated.connect(self._runner_shell.set_banner_time)
        self._runner_shell.stop_requested.connect(self._on_stop_clicked)

        # Setup page signal wiring
        self._runner_setup_page.new_requested.connect(self._on_new)
        self._runner_setup_page.open_requested.connect(self._on_open)
        self._runner_setup_page.save_requested.connect(self._on_save)
        self._runner_setup_page.save_as_requested.connect(self._on_save_as)
        self._runner_setup_page.help_requested.connect(self._on_help)
        self._runner_setup_page.close_requested.connect(self.close)
        self._runner_setup_page.switch_to_desktop_requested.connect(self._switch_to_desktop_mode)
        self._runner_setup_page.board_settings_requested.connect(
            self._hardware_panel.show_board_settings_dialog
        )

        # Hardware-change fan-out (owned here since the runner-mode Hardware
        # Panel is constructed here; _setup_dock_widgets reuses this instance
        # and must NOT re-wire these, to avoid double-firing).
        self._hardware_panel.hardware_changed.connect(self._runner_panel.refresh_devices)
        self._hardware_panel.hardware_changed.connect(self._runner_setup_page.refresh)
        self._hardware_panel.refresh_tree()

    def _build_camera_panel(self) -> CameraPanel:
        """Construct and fully configure a CameraPanel.

        Extracted so both the runner view and the desktop dock setup can
        share a single CameraPanel instance instead of building duplicates.
        """
        camera_panel = CameraPanel(
            self._core.camera_manager,
            self._core.cv_processor,
            multi_camera_manager=self._core.multi_camera_manager,
        )
        camera_panel.settings_requested.connect(self._on_camera_settings)
        camera_panel.analysis_requested.connect(self._on_open_analysis_panel)
        camera_panel.draw_zones_requested.connect(self._on_video_zones_requested)
        camera_panel.set_video_recorder(self._core.video_recorder)
        camera_panel.set_multi_video_recorder(self._core.multi_video_recorder)
        camera_panel.set_tracking_logger(self._core.tracking_logger)
        # Frame-aligned device-state CSV + per-edge event log. CameraPanel
        # ticks both on every processed frame so the device CSV inherits the
        # tracking CSV's `frame` column for one-key joins to the MP4.
        camera_panel.set_data_recorder(self._core.data_recorder)
        camera_panel.set_event_logger(self._core.event_logger)
        camera_panel.set_calibration(self._core.calibration)
        camera_panel._preview.set_calibration(self._core.calibration)
        camera_panel.set_zone_configuration(self._zone_config)
        return camera_panel

    def _setup_dock_widgets(self) -> None:
        """Set up dock widgets for desktop mode."""

        def session_fn():
            return self._core.session

        # Node Library dock
        self._node_library_panel = NodeLibraryPanel(
            session_fn=session_fn,
            graph_view=self._graph_view,
        )
        self._node_library_panel.status_message.connect(self._show_status_message)
        self._node_library_panel._zone_config = self._zone_config

        self._node_library_dock = QDockWidget("Node Library", self)
        self._node_library_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._node_library_dock.setWidget(self._node_library_panel)
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

        # Wire properties dock to node editor
        self._node_editor.set_properties_dock(self._properties_dock)

        # Hardware Panel dock. The runner view (_create_runner_view) already
        # builds this panel unconditionally, so on the runner→desktop switch
        # it already exists here — only construct (and wire its owner
        # connection) if it doesn't.
        if getattr(self, "_hardware_panel", None) is None:
            self._hardware_panel = HardwarePanel(
                hardware_manager=self._core.hardware_manager,
                session_fn=session_fn,
                run_async_fn=self._run_async,
            )
            self._hardware_panel.status_message.connect(self._show_status_message)

        self._hardware_dock = QDockWidget("Hardware", self)
        self._hardware_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._hardware_dock.setWidget(self._hardware_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._hardware_dock)

        # Device Control Panel dock
        self._device_control_panel = DeviceControlPanel(
            hardware_manager=self._core.hardware_manager,
            run_async_fn=self._run_async,
        )
        self._device_control_panel.status_message.connect(self._show_status_message)

        self._control_dock = QDockWidget("Device Control", self)
        self._control_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._control_dock.setWidget(self._device_control_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._control_dock)

        # Group left docks: Node Library + Hardware + Device Control
        self.tabifyDockWidget(self._node_library_dock, self._hardware_dock)
        self.tabifyDockWidget(self._hardware_dock, self._control_dock)
        self._node_library_dock.raise_()

        # Wire hardware_changed → device control refresh. (hardware_changed →
        # runner_panel.refresh_devices / runner_setup_page.refresh are wired
        # once in _create_runner_view, which owns this HardwarePanel
        # instance; re-adding here would double-fire.)
        self._hardware_panel.hardware_changed.connect(self._device_control_panel.refresh_devices)

        # Wire flow_functions_changed from node editor to node library
        self._node_editor.flow_functions_changed.connect(
            self._node_library_panel.refresh_flow_functions
        )

        # Camera Panel dock
        self._camera_dock = QDockWidget("Camera", self)
        self._camera_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        if getattr(self, "_camera_panel", None) is None:
            self._camera_panel = self._build_camera_panel()
        self._camera_dock.setWidget(self._camera_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._camera_dock)

        # Group right docks: Properties + Camera
        self.tabifyDockWidget(self._properties_dock, self._camera_dock)
        self._properties_dock.raise_()

        # Files dock
        from PyQt6.QtWidgets import QFrame, QScrollArea

        self._files_dock = QDockWidget("Files", self)
        self._files_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )

        from PyQt6.QtWidgets import QPushButton

        files_scroll = QScrollArea()
        files_scroll.setWidgetResizable(True)
        files_scroll.setFrameShape(QFrame.Shape.NoFrame)
        files_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        files_widget = QWidget()
        files_layout = QVBoxLayout(files_widget)
        files_layout.setContentsMargins(4, 4, 4, 4)
        files_layout.setSpacing(4)

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

        if not self._view_manager.is_runner_mode:
            self._files_dock.setVisible(False)

        # Refresh hardware tree (which also triggers device combo + runner refresh)
        self._hardware_panel.refresh_tree()

        # The desktop docks have now adopted the shared Hardware/Camera panels,
        # so the runner view would render stripped. Disable the toggle action
        # (single-mode-per-process — see _runner_view_available).
        if getattr(self, "_switch_view_action", None) is not None:
            self._switch_view_action.setEnabled(False)

    # --- Menu / Toolbar / Status bar ---

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        if self._view_manager.is_runner_mode:
            return

        menubar = self.menuBar()

        branding = QLabel("GLIDER")
        branding.setStyleSheet(
            f"font-weight: 600; color: {colors.ACCENT}; font-size: 13px; "
            f"letter-spacing: 0.5px; padding: 0 12px 0 4px;"
        )
        menubar.setCornerWidget(branding, Qt.Corner.TopLeftCorner)

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
        # Kept so _setup_dock_widgets can disable it once the desktop docks
        # have adopted the shared Hardware/Camera panels — see _toggle_view.
        # On a desktop boot the docks already exist by the time this menu is
        # built, so reflect that state immediately too.
        switch_view_action.setEnabled(self._runner_view_available())
        self._switch_view_action = switch_view_action

        view_menu.addSeparator()

        if hasattr(self, "_node_library_dock") and self._node_library_dock:
            node_library_action = self._node_library_dock.toggleViewAction()
            node_library_action.setText("&Node Library")
            view_menu.addAction(node_library_action)

        if hasattr(self, "_properties_dock") and self._properties_dock:
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
        add_board_action.triggered.connect(
            lambda: self._hardware_panel and self._hardware_panel._on_add_board()
        )
        hardware_menu.addAction(add_board_action)

        add_device_action = QAction("Add &Device...", self)
        add_device_action.triggered.connect(
            lambda: self._hardware_panel and self._hardware_panel._on_add_device()
        )
        hardware_menu.addAction(add_device_action)

        new_custom_device_action = QAction("New Custom Device &Type...", self)
        new_custom_device_action.triggered.connect(self._on_new_custom_device)
        hardware_menu.addAction(new_custom_device_action)

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

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        behavior_action = QAction("&Behavior Analysis…", self)
        behavior_action.triggered.connect(self._open_behavior_analysis)
        # Lazy import: keep GLIDER startup free of the behavior/PyQt-heavy
        # window and the optional [behavior] dependency probe until the menu
        # is actually built.
        from glider.gui.behavior.availability import (
            behavior_available,
            missing_behavior_deps,
        )

        if not behavior_available():
            behavior_action.setEnabled(False)
            behavior_action.setToolTip(
                "Install the behavior extra: pip install glider[behavior] "
                f"(missing: {', '.join(missing_behavior_deps())})"
            )
        tools_menu.addAction(behavior_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        help_action = QAction("&GLIDER Help", self)
        help_action.setShortcut(QKeySequence("F1"))
        help_action.triggered.connect(self._on_help)
        help_menu.addAction(help_action)

        # "Check for Updates…" uses the same checker instance as the silent
        # startup check, so an in-flight check is coalesced rather than run
        # twice. The ellipsis follows platform convention for an action that
        # opens a dialog.
        check_updates_action = QAction("Check for &Updates…", self)
        check_updates_action.triggered.connect(lambda: self.check_for_updates(silent=False))
        help_menu.addAction(check_updates_action)

        help_menu.addSeparator()

        about_action = QAction("&About GLIDER", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _open_behavior_analysis(self) -> None:
        """Open (or re-surface) the Behavior Analysis window.

        Constructed lazily on first use and kept on ``self`` so it isn't
        garbage-collected; the import is deferred so GLIDER startup never
        pulls in the behavior/PyQt-heavy window unless the user asks for it.
        """
        from glider.gui.behavior.window import BehaviorAnalysisWindow

        if getattr(self, "_behavior_window", None) is None:
            self._behavior_window = BehaviorAnalysisWindow(parent=None)
        self._behavior_window.show()
        self._behavior_window.raise_()
        self._behavior_window.activateWindow()

    def _setup_toolbar(self) -> None:
        """Set up the toolbar."""
        if self._view_manager.is_runner_mode:
            return

        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

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

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self._toolbar_status = QLabel("IDLE")
        self._toolbar_status.setProperty("statusIndicator", True)
        self._toolbar_status.setProperty("statusState", "IDLE")
        toolbar.addWidget(self._toolbar_status)

    def _setup_status_bar(self) -> None:
        """Set up the status bar with connection, state, and stats."""
        if self._view_manager.is_runner_mode:
            return

        status_bar = QStatusBar()
        status_bar.setSizeGripEnabled(False)
        self.setStatusBar(status_bar)

        # Connection indicator (left)
        conn_widget = QWidget()
        conn_layout = QHBoxLayout(conn_widget)
        conn_layout.setContentsMargins(4, 0, 4, 0)
        conn_layout.setSpacing(4)

        self._conn_dot = QLabel("\u2022")
        self._conn_dot.setStyleSheet(f"color: {colors.ERROR}; font-size: 16px;")
        conn_layout.addWidget(self._conn_dot)

        self._conn_label = QLabel("No board")
        self._conn_label.setProperty("textRole", "muted")
        conn_layout.addWidget(self._conn_label)

        status_bar.addWidget(conn_widget)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {colors.BORDER};")
        sep.setFixedHeight(14)
        status_bar.addWidget(sep)

        # State label
        self._state_label = QLabel("State: IDLE")
        self._state_label.setProperty("textRole", "muted")
        status_bar.addWidget(self._state_label)

        # Stats (right-aligned)
        self._stats_label = QLabel("")
        self._stats_label.setProperty("textRole", "disabled")
        status_bar.addPermanentWidget(self._stats_label)

    def _show_status_message(self, message: str, timeout: int = 0) -> None:
        """Show a status bar message if not in runner mode."""
        if self._view_manager.is_runner_mode:
            logger.debug(f"Status (runner mode): {message}")
            return
        self.statusBar().showMessage(message, timeout)

    def _notify_user(
        self,
        title: str,
        message: str,
        level: str = "warning",
        timeout_ms: int = 8000,
    ) -> None:
        """
        Surface a user-visible notification without blocking the event loop.

        In desktop mode this schedules a non-modal ``QMessageBox`` on the next
        Qt event-loop tick (so it can't freeze an in-flight async coroutine).
        In runner mode (touch kiosk, usually on the Pi) we avoid modal popups
        entirely — runner mode is responsiveness-sensitive and dialogs don't
        work well with touch input — and instead post to the status bar +
        log at the appropriate level.

        Args:
            title: Short title for the notification.
            message: Body text.
            level: "info", "warning", or "critical" — affects log level and,
                in desktop mode, the QMessageBox icon.
            timeout_ms: How long to show the status message in runner mode.
        """
        log_message = f"{title}: {message}"
        if level == "critical":
            logger.error(log_message)
        elif level == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)

        if self._view_manager.is_runner_mode:
            # No modal dialogs on the touch dashboard — just the status bar.
            try:
                self.statusBar().showMessage(f"{title}: {message}", timeout_ms)
            except Exception:
                # Runner mode may not have a conventional status bar; the log
                # above is still the source of truth.
                pass
            return

        # Desktop mode: schedule the dialog on the next event-loop tick so the
        # calling async coroutine gets to finish returning control to the loop
        # before the modal box grabs focus.
        def _show() -> None:
            icon = QMessageBox.Icon.Warning
            if level == "critical":
                icon = QMessageBox.Icon.Critical
            elif level == "info":
                icon = QMessageBox.Icon.Information
            box = QMessageBox(self)
            box.setIcon(icon)
            box.setWindowTitle(title)
            box.setText(message)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            # Non-modal: don't block the event loop.
            box.setWindowModality(Qt.WindowModality.NonModal)
            box.show()

        QTimer.singleShot(0, _show)

    # --- Signal wiring ---

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._core_state_changed.connect(self._on_core_state_change)
        self._core_error_occurred.connect(self._on_core_error)
        self._hardware_connection_changed.connect(self._on_hardware_connection_change)

        self._core.on_state_change(lambda state: self._core_state_changed.emit(state))
        self._core.on_error(lambda source, error: self._core_error_occurred.emit(source, error))
        self._core.hardware_manager.on_connection_change(
            lambda board_id, state: self._hardware_connection_changed.emit(board_id, state)
        )

        self.session_changed.connect(lambda: self._runner_panel.update_experiment_name())
        self.session_changed.connect(self._manual_control_panel.refresh)
        self.session_changed.connect(self._runner_setup_page.refresh)

    @pyqtSlot(object)
    def _on_core_state_change(self, state) -> None:
        """Handle core state changes.

        On entry into ERROR, surface an unmissable critical notification.
        ERROR after a STOP can mean ``_set_all_devices_low`` failed and at
        least one hardware output may still be active — the operator
        cannot distinguish that from a transient error from the status-bar
        color alone, so we explicitly call ``_notify_user`` at ``critical``
        level (which opens a non-modal critical dialog in desktop mode,
        or a long-timeout status message in runner mode).
        Dedup via ``_last_session_state`` so repeated state callbacks for
        the same state don't spam the user.
        """
        state_name = state.name
        prev_state = getattr(self, "_last_session_state", None)
        self._last_session_state = state_name
        self.state_changed.emit(state_name)

        # Update runner panel
        self._runner_shell.update_state(state_name)

        # Update toolbar status indicator
        if self._toolbar_status is not None:
            self._toolbar_status.setText(state_name)
            self._toolbar_status.setProperty("statusState", state_name)
            self._toolbar_status.style().unpolish(self._toolbar_status)
            self._toolbar_status.style().polish(self._toolbar_status)

        self._show_status_message(f"State: {state_name}")

        if self._state_label is not None:
            self._state_label.setText(f"State: {state_name}")

        # Surface unsafe-state modal on transition INTO ERROR. Skip if we
        # were already in ERROR (e.g., repeat callback fired by a listener).
        if state_name == "ERROR" and prev_state != "ERROR":
            self._notify_user(
                "UNSAFE STATE — Hardware may still be active",
                "The experiment session entered an ERROR state.\n\n"
                "If this happened during or immediately after STOP, one or more "
                "hardware outputs (heater, relay, PWM, valve, motor) may still "
                "be active. Verify the physical state of every output device "
                "before powering down. Check the log for diagnostic details.",
                level="critical",
            )

    @pyqtSlot(str, object)
    def _on_core_error(self, source: str, error: Exception) -> None:
        """Handle core errors.

        ``error_occurred`` is kept for backwards compatibility with any
        external subscribers, but the operator-visible path is now the
        explicit ``_notify_user`` call below — previously the signal had
        zero subscribers and every core error vanished into the log.
        """
        self.error_occurred.emit(source, str(error))
        logger.error(f"Error from {source}: {error}")
        # Warning-level rather than critical: not every core error means
        # hardware is in a bad state. The state-change handler above
        # escalates to critical when the session actually enters ERROR.
        self._notify_user(
            f"GLIDER error: {source}",
            str(error),
            level="warning",
        )

    @pyqtSlot(str, object)
    def _on_hardware_connection_change(self, board_id: str, state: BoardConnectionState) -> None:
        """Handle hardware connection state changes.

        Two jobs:
          1. Refresh the status-bar connection indicator (every transition —
             CONNECTING, CONNECTED, DISCONNECTED, ERROR, RECONNECTING). This
             must run unconditionally; previously it was gated behind the
             disconnect path and the dot never turned green on connect.
          2. If the board dropped *during* a running experiment, pause and
             surface the hardware-disconnection dialog.
        """
        # (1) Always refresh the indicator — this is the only wire-up point.
        self._update_connection_status()

        # (2) Pause-on-drop guard only applies to terminal-failure states
        # reached while an experiment is actually running.
        if state not in (BoardConnectionState.DISCONNECTED, BoardConnectionState.ERROR):
            return

        if not hasattr(self._core, "state"):
            return

        from glider.core.glider_core import SessionState

        if self._core.state != SessionState.RUNNING:
            logger.warning(f"Board {board_id} disconnected (state: {state.name})")
            return

        logger.warning(f"Board {board_id} disconnected during experiment! Pausing...")
        self._run_async(self._core.pause())
        self._show_hardware_disconnection_dialog(board_id, state)

    def _update_connection_status(self) -> None:
        """Update the status bar connection indicator.

        Reads ``board.is_connected`` (and ``board.state`` for the transient
        CONNECTING case). The old check probed ``_connected`` / ``connected``
        attributes that don't exist on any BaseBoard subclass, so even when
        this method *was* reached, it always reported "No board".
        """
        if self._conn_dot is None or self._conn_label is None:
            return

        boards = self._core.hardware_manager.boards
        if not boards:
            self._conn_dot.setStyleSheet(f"color: {colors.ERROR}; font-size: 16px;")
            self._conn_label.setText("No board")
            return

        # Prefer a connected board; fall back to CONNECTING so the user sees
        # feedback during the connect handshake.
        connected_board = None
        connecting_board = None
        for board_id, board in boards.items():
            name = getattr(board, "name", board_id)
            if getattr(board, "is_connected", False):
                connected_board = (name, board)
                break
            if getattr(board, "state", None) in (
                BoardConnectionState.CONNECTING,
                BoardConnectionState.RECONNECTING,
            ):
                connecting_board = (name, board)

        if connected_board is not None:
            name, _ = connected_board
            self._conn_dot.setStyleSheet(f"color: {colors.SUCCESS}; font-size: 16px;")
            self._conn_label.setText(f"{name} \u2014 Connected")
        elif connecting_board is not None:
            name, _ = connecting_board
            self._conn_dot.setStyleSheet(f"color: {colors.WARNING}; font-size: 16px;")
            self._conn_label.setText(f"{name} \u2014 Connecting\u2026")
        else:
            self._conn_dot.setStyleSheet(f"color: {colors.ERROR}; font-size: 16px;")
            self._conn_label.setText("No board")

    def update_status_stats(self, node_count: int, connection_count: int) -> None:
        """Update the node/connection count in the status bar."""
        if self._stats_label is not None:
            self._stats_label.setText(f"{node_count} nodes \u2022 {connection_count} connections")

    def _show_hardware_disconnection_dialog(
        self, board_id: str, state: BoardConnectionState
    ) -> None:
        """Show a dialog when hardware disconnects during an experiment."""
        from PyQt6.QtWidgets import QDialog, QHBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Hardware Disconnected")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        warning_label = QLabel("Warning")
        warning_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {colors.WARNING};")
        header_layout.addWidget(warning_label)

        message = QLabel(
            f"<b>Board '{board_id}' has disconnected.</b><br><br>"
            f"The experiment has been paused. What would you like to do?"
        )
        message.setWordWrap(True)
        header_layout.addWidget(message, 1)
        layout.addLayout(header_layout)

        status_label = QLabel(f"Connection state: {state.name}")
        status_label.setProperty("textRole", "muted")
        layout.addWidget(status_label)

        from PyQt6.QtWidgets import QPushButton

        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        retry_btn = QPushButton("Retry Connection")
        retry_btn.setMinimumHeight(40)
        retry_btn.clicked.connect(lambda: self._handle_disconnection_retry(dialog, board_id))
        button_layout.addWidget(retry_btn)

        continue_btn = QPushButton("Continue Without Hardware")
        continue_btn.setMinimumHeight(40)
        continue_btn.clicked.connect(lambda: self._handle_disconnection_continue(dialog))
        button_layout.addWidget(continue_btn)

        stop_btn = QPushButton("Stop Experiment")
        stop_btn.setMinimumHeight(40)
        stop_btn.clicked.connect(lambda: self._handle_disconnection_stop(dialog))
        button_layout.addWidget(stop_btn)

        layout.addLayout(button_layout)
        dialog.exec()

    def _handle_disconnection_retry(self, dialog: QDialog, board_id: str) -> None:
        dialog.accept()
        retries = self._reconnect_retries.get(board_id, 0) + 1
        self._reconnect_retries[board_id] = retries

        if retries > self._max_reconnect_retries:
            self._show_status_message(
                f"Max retries reached for {board_id}. Stopping experiment.", 5000
            )
            self._reconnect_retries.pop(board_id, None)
            self._run_async(self._core.stop())
            return

        self._show_status_message(
            f"Reconnecting to {board_id} (attempt {retries}/{self._max_reconnect_retries})..."
        )

        async def retry_connection():
            try:
                success = await self._core.hardware_manager.connect_board(board_id)
                if success:
                    self._reconnect_retries.pop(board_id, None)
                    self._show_status_message(f"Reconnected to {board_id}. Resuming...", 3000)
                    await self._core.resume()
                else:
                    self._show_status_message(f"Failed to reconnect to {board_id}", 5000)
                    self._show_hardware_disconnection_dialog(
                        board_id, BoardConnectionState.DISCONNECTED
                    )
            except Exception as e:
                self._show_status_message(f"Error: {e}", 5000)
                self._show_hardware_disconnection_dialog(board_id, BoardConnectionState.ERROR)

        self._run_async(retry_connection())

    def _handle_disconnection_continue(self, dialog: QDialog) -> None:
        dialog.accept()
        reply = QMessageBox.warning(
            self,
            "Continue Without Hardware",
            "Continuing without the disconnected hardware may cause errors. Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._run_async(self._core.resume())

    def _handle_disconnection_stop(self, dialog: QDialog) -> None:
        dialog.accept()
        self._run_async(self._core.stop())
        self._show_status_message("Experiment stopped", 3000)

    # --- View switching ---

    def _runner_view_available(self) -> bool:
        """Whether the runner view can still be shown intact.

        GLIDER is single-mode-per-process in practice. The runner shell and its
        Setup page share the Hardware/Camera panel instances with the desktop
        docks; once _setup_dock_widgets has run (on a runner→desktop switch) it
        reparents those panels into the docks, leaving the runner tabs stripped.
        So once the desktop docks exist, switching back to the runner view is
        disallowed rather than showing an empty Camera tab + hardware section.
        """
        return getattr(self, "_node_library_dock", None) is None

    def _toggle_view(self) -> None:
        current = self._stack.currentIndex()
        if current == 0 and not self._runner_view_available():
            self._show_status_message(
                "Runner view is unavailable after switching to desktop mode.", 3000
            )
            return
        self._stack.setCurrentIndex(1 if current == 0 else 0)

    def switch_to_builder(self) -> None:
        self._stack.setCurrentIndex(0)

    def switch_to_runner(self) -> None:
        if not self._runner_view_available():
            self._show_status_message(
                "Runner view is unavailable after switching to desktop mode.", 3000
            )
            return
        self._stack.setCurrentIndex(1)

    def _set_window_size(self, width: int, height: int) -> None:
        self.setMinimumSize(min(width, 480), min(height, 480))
        self.resize(width, height)
        self._show_status_message(f"Window resized to {width}x{height}", 2000)

    def _set_pi_touchscreen_layout(self) -> None:
        """Set up Pi Touchscreen layout with tabbed panels."""
        self.setMinimumSize(480, 480)
        self.resize(480, 800)

        if self._stack is not None:
            self._stack.setCurrentIndex(1)

        docks = []
        if getattr(self, "_files_dock", None) is not None:
            docks.append(self._files_dock)
        if getattr(self, "_hardware_dock", None) is not None:
            docks.append(self._hardware_dock)
        if getattr(self, "_control_dock", None) is not None:
            docks.append(self._control_dock)
        if getattr(self, "_camera_dock", None) is not None:
            docks.append(self._camera_dock)

        if getattr(self, "_node_library_dock", None) is not None:
            self._node_library_dock.setVisible(False)
        if getattr(self, "_properties_dock", None) is not None:
            self._properties_dock.setVisible(False)

        if len(docks) < 2:
            return

        for dock in docks:
            dock.setVisible(True)
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
            dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        first_dock = docks[0]
        for dock in docks[1:]:
            self.tabifyDockWidget(first_dock, dock)

        for tab_bar in self.findChildren(QTabBar):
            tab_bar.setExpanding(True)

        first_dock.raise_()
        self._show_status_message("Pi Touchscreen layout applied", 2000)

    def _set_default_layout(self) -> None:
        """Restore default desktop layout."""
        self.resize(1400, 900)

        default_features = (
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        default_areas = Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea

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

        if getattr(self, "_files_dock", None) is not None:
            self._files_dock.setVisible(False)

        self._show_status_message("Default layout restored", 2000)

    def _switch_to_desktop_mode(self) -> None:
        """Switch from runner to desktop mode."""
        self.setWindowFlags(Qt.WindowType.Window)
        self.showNormal()

        self._stack.setCurrentIndex(0)

        if getattr(self, "_node_library_dock", None) is None:
            self._setup_dock_widgets()

        screen_size = self._view_manager.screen_size
        if screen_size.width() <= 800:
            self.showMaximized()
            self._set_pi_touchscreen_layout()
        else:
            self.resize(1400, 900)

    # --- File operations ---

    def _on_new(self) -> None:
        """Create new experiment."""
        if self._check_save():
            self._core.hardware_manager.clear()
            self._core.new_session()
            self._graph_view.clear_graph()
            self._zone_config = ZoneConfiguration()
            if self._camera_panel:
                self._camera_panel.set_zone_configuration(self._zone_config)
            self.session_changed.emit()
            if self._hardware_panel:
                self._hardware_panel.refresh_tree()
            if self._node_library_panel:
                self._node_library_panel.refresh_flow_functions()
                self._node_library_panel.refresh_zones(self._zone_config)
            if self._node_editor:
                self._node_editor.set_zone_configuration(self._zone_config)
            if self._experiment_dialog:
                self._experiment_dialog.set_session(self._core.session)

    def _on_open(self) -> None:
        """Open experiment file."""
        if not self._check_save():
            return

        was_runner_mode = self._view_manager.is_runner_mode

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Experiment",
            "",
            "GLIDER Experiments (*.glider);;JSON Files (*.json);;All Files (*)",
        )

        if was_runner_mode:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.showFullScreen()

        if file_path:
            try:
                self._core.load_session(file_path)
                self._populate_hardware_from_session()
                self._populate_graph_from_session()
                self._load_zones_from_session()
                self.session_changed.emit()
                if self._hardware_panel:
                    self._hardware_panel.refresh_tree()
                if self._node_library_panel:
                    self._node_library_panel.refresh_flow_functions()
                    self._node_library_panel.refresh_zones(self._zone_config)
                if self._experiment_dialog:
                    self._experiment_dialog.set_session(self._core.session)
                rec_dir = self._core.session.metadata.recording_directory
                if rec_dir:
                    self._core.set_recording_directory(Path(rec_dir))
                self._show_status_message(f"Opened: {file_path}")
            except Exception as e:
                logger.exception(f"Failed to open file: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open file: {e}")

    def _populate_hardware_from_session(self) -> None:
        """Populate hardware manager from session configuration."""
        if not self._core.session:
            return

        self._core.hardware_manager.clear()

        for board_config in self._core.session.hardware.boards:
            try:
                board_type = "telemetrix" if board_config.driver_type == "arduino" else "pigpio"
                self._core.hardware_manager.add_board(
                    board_config.id,
                    board_type,
                    port=board_config.port,
                )
            except Exception as e:
                logger.warning(f"Failed to add board {board_config.id}: {e}")

        for device_config in self._core.session.hardware.devices:
            try:
                settings = device_config.settings or {}
                if device_config.pins:
                    self._core.hardware_manager.add_device_multi_pin(
                        device_config.id,
                        device_config.device_type,
                        device_config.board_id,
                        device_config.pins,
                        name=device_config.name,
                        **settings,
                    )
                else:
                    self._core.hardware_manager.add_device_multi_pin(
                        device_config.id,
                        device_config.device_type,
                        device_config.board_id,
                        {},
                        name=device_config.name,
                        **settings,
                    )
            except Exception as e:
                logger.warning(f"Failed to add device {device_config.id}: {e}")

    def _populate_graph_from_session(self) -> None:
        """Populate graph view from session flow configuration."""
        if not self._core.session:
            return

        self._graph_view.clear_graph()

        for node_config in self._core.session.flow.nodes:
            try:
                x, y = node_config.position
                node_type = node_config.node_type
                display_name = node_type
                definition_id = None

                if node_type == "ZoneInput":
                    zone_name = (
                        node_config.state.get("zone_name", "Zone") if node_config.state else "Zone"
                    )
                    display_name = f"Zone: {zone_name}"

                category = "default"
                flow_nodes = ["StartExperiment", "EndExperiment", "Delay"]
                control_nodes = ["Loop", "WaitForInput"]
                io_nodes = [
                    "Output",
                    "Input",
                    "MotorGovernor",
                ]
                function_nodes = [
                    "FunctionCall",
                    "StartFunction",
                    "EndFunction",
                ]
                interface_nodes = ["ZoneInput"]

                node_type_normalized = node_type.replace(" ", "")
                if node_type_normalized in flow_nodes:
                    category = "logic"
                elif node_type_normalized in control_nodes:
                    category = "interface"
                elif node_type_normalized in io_nodes:
                    category = "hardware"
                elif node_type_normalized in function_nodes:
                    category = "logic"
                elif node_type_normalized in interface_nodes:
                    category = "interface"

                node_item = self._graph_view.add_node(node_config.id, display_name, x, y)
                node_item._category = category
                node_item._header_color = node_item.CATEGORY_COLORS.get(
                    category, node_item.CATEGORY_COLORS["default"]
                )
                node_item._actual_node_type = node_type
                node_item._definition_id = definition_id

                self._node_editor.setup_node_ports(node_item, node_type)
                self._graph_view._connect_port_signals(node_item)

            except Exception as e:
                logger.error(f"Failed to load node {node_config.id}: {e}")

        for conn_config in self._core.session.flow.connections:
            try:
                self._graph_view.add_connection(
                    conn_config.id,
                    conn_config.from_node,
                    conn_config.from_output,
                    conn_config.to_node,
                    conn_config.to_input,
                )
            except Exception as e:
                logger.error(f"Failed to load connection {conn_config.id}: {e}")

        logger.info(
            f"Loaded {len(self._core.session.flow.nodes)} nodes and "
            f"{len(self._core.session.flow.connections)} connections from session"
        )

    def _on_save(self) -> None:
        if self._core.session and self._core.session.file_path:
            try:
                self._core.save_session()
                self._show_status_message("Saved")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        was_runner_mode = self._view_manager.is_runner_mode

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Experiment",
            "",
            "GLIDER Experiments (*.glider);;JSON Files (*.json)",
        )

        if was_runner_mode:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.showFullScreen()

        if file_path:
            try:
                self._core.save_session(file_path)
                self._show_status_message(f"Saved: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _check_save(self) -> bool:
        if self._core.session and self._core.session.is_dirty:
            was_runner_mode = self._view_manager.is_runner_mode

            result = QMessageBox.question(
                self,
                "Save Changes?",
                "The current experiment has unsaved changes. Save before continuing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if was_runner_mode:
                self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
                self.showFullScreen()

            if result == QMessageBox.StandardButton.Save:
                self._on_save()
                return True
            elif result == QMessageBox.StandardButton.Cancel:
                return False

        return True

    # --- Hardware operations ---

    def _on_connect_hardware(self) -> None:
        self._run_async(self._connect_hardware_async())

    async def _connect_hardware_async(self) -> None:
        try:
            await self._core.setup_hardware()
            results = await self._core.connect_hardware()
            if self._hardware_panel:
                self._hardware_panel.refresh_tree()
            failed = [k for k, v in results.items() if not v]
            if failed:
                # Non-blocking notification: modal dialogs called from an async
                # coroutine freeze the event loop (especially bad in runner
                # mode on a Pi). _notify_user schedules the dialog for the
                # next tick or routes to the status bar on the runner.
                self._notify_user(
                    "Connection Warning",
                    f"Failed to connect: {', '.join(failed)}",
                    level="warning",
                )
        except Exception as e:
            self._notify_user("Connection Error", str(e), level="critical")

    def _on_disconnect_hardware(self) -> None:
        self._run_async(self._core.hardware_manager.disconnect_all())

    # --- Camera operations ---

    def _on_camera_settings(self) -> None:
        dialog = CameraSettingsDialog(
            camera_settings=self._core.camera_manager.settings,
            cv_settings=self._core.cv_processor.settings,
            parent=self,
            view_manager=self._view_manager,
            camera_manager=self._core.camera_manager,
        )

        dialog.calibration_requested.connect(self._on_camera_calibration)
        dialog.zones_requested.connect(self._on_zones_requested)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            camera_settings = dialog.get_camera_settings()
            self._core.camera_manager.apply_settings(camera_settings)
            cv_settings = dialog.get_cv_settings()
            self._core.cv_processor.update_settings(cv_settings)

    def _on_camera_calibration(self) -> None:
        dialog = CalibrationDialog(
            camera_manager=self._core.camera_manager,
            calibration=self._core.calibration,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            calibration = dialog.get_calibration()
            if calibration.is_calibrated:
                logger.info(
                    f"Camera calibrated: {calibration.pixels_per_mm:.2f} pixels/mm "
                    f"({len(calibration.lines)} lines)"
                )

    def _on_zones_requested(self) -> None:
        self._open_zone_dialog(frame=None)

    def _on_video_zones_requested(self, frame) -> None:
        """Open the zone editor on a scrubbed video frame (offline mode)."""
        self._open_zone_dialog(frame=frame)

    def _open_zone_dialog(self, frame=None) -> None:
        dialog = ZoneDialog(
            camera_manager=self._core.camera_manager,
            zone_config=self._zone_config,
            parent=self,
            frame=frame,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._zone_config = dialog.get_zone_configuration()
            if self._camera_panel:
                self._camera_panel.set_zone_configuration(self._zone_config)
            self._core.cv_processor.set_zone_configuration(self._zone_config)
            self._core.tracking_logger.set_zone_configuration(self._zone_config)
            if hasattr(self._core, "data_recorder"):
                self._core.data_recorder.set_zone_configuration(self._zone_config)
                self._core.data_recorder.set_cv_processor(self._core.cv_processor)

            self._save_zones_to_session()

            if self._node_library_panel:
                self._node_library_panel.refresh_zones(self._zone_config)
            if self._node_editor:
                self._node_editor.set_zone_configuration(self._zone_config)

            # In offline video mode, re-render the scrubbed frame so the newly
            # drawn zones appear immediately on the preview.
            if frame is not None and self._camera_panel:
                self._camera_panel.refresh_scrub_frame()

    # --- Experiment operations ---

    def _on_open_experiment_dialog(self) -> None:
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
            self._experiment_dialog.set_session(self._core.session)

        self._experiment_dialog.show()
        self._experiment_dialog.raise_()
        self._experiment_dialog.activateWindow()

    def _on_open_analysis_dialog(self) -> None:
        from glider.gui.dialogs.analysis_dialog import AnalysisDialog

        dialog = AnalysisDialog(parent=self)
        dialog.exec()

    def _on_new_custom_device(self) -> None:
        """Open the no-code custom device builder; refresh on save."""
        from glider.gui.dialogs.custom_device_dialog import CustomDeviceDialog

        dialog = CustomDeviceDialog(parent=self)
        if dialog.exec():
            # The new type is registered in DEVICE_REGISTRY and appears the next
            # time Add Device is opened.
            self._show_status_message(
                f"Custom device '{dialog.device_name}' created — add it via Hardware → Add Device",
                5000,
            )

    def _on_open_analysis_panel(self, directory: str) -> None:
        """Open (or reuse) the Analysis dock and load a finished recording.

        Wired to CameraPanel.analysis_requested — fired when the operator
        clicks "Open in Analysis panel" after a video tracking run.
        """
        from glider.gui.panels.analysis import AnalysisPanel

        if self._analysis_dock is None:
            self._analysis_panel = AnalysisPanel()
            self._analysis_dock = QDockWidget("Analysis", self)
            self._analysis_dock.setWidget(self._analysis_panel)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._analysis_dock)

        if not self._analysis_panel.load_recording(Path(directory)):
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Analysis", f"No GLIDER recording found in:\n{directory}")
            return

        self._analysis_dock.show()
        self._analysis_dock.raise_()

    def _on_experiment_metadata_changed(self) -> None:
        self._core.session._dirty = True

    def _on_recording_directory_changed(self, directory: str) -> None:
        if directory:
            self._core.set_recording_directory(Path(directory))

    def _on_edit_subject(self, subject_id: str) -> None:
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
                metadata = self._core.session.metadata
                for i, s in enumerate(metadata.subjects):
                    if s.id == new_subject.id:
                        metadata.subjects[i] = new_subject
                        self._core.session._dirty = True
                        break
            else:
                self._core.session.metadata.add_subject(new_subject)
                self._core.session._dirty = True

            if self._experiment_dialog:
                self._experiment_dialog.refresh()

    def _load_zones_from_session(self) -> None:
        if not self._core.session:
            return

        session_zones = self._core.session.zones
        if session_zones.zones:
            from glider.vision.zones import Zone

            self._zone_config.zones.clear()
            for zone_dict in session_zones.zones:
                zone = Zone.from_dict(zone_dict)
                self._zone_config.zones.append(zone)
            self._zone_config.config_width = session_zones.config_width
            self._zone_config.config_height = session_zones.config_height

            if self._camera_panel:
                self._camera_panel.set_zone_configuration(self._zone_config)
            self._core.cv_processor.set_zone_configuration(self._zone_config)
            self._core.tracking_logger.set_zone_configuration(self._zone_config)

            if self._node_library_panel:
                self._node_library_panel.refresh_zones(self._zone_config)
            if self._node_editor:
                self._node_editor.set_zone_configuration(self._zone_config)
        else:
            self._zone_config = ZoneConfiguration()
            if self._camera_panel:
                self._camera_panel.set_zone_configuration(self._zone_config)
            if self._node_library_panel:
                self._node_library_panel.refresh_zones(self._zone_config)

    def _save_zones_to_session(self) -> None:
        if not self._core.session:
            return

        self._core.session.zones.zones = [zone.to_dict() for zone in self._zone_config.zones]
        self._core.session.zones.config_width = self._zone_config.config_width
        self._core.session.zones.config_height = self._zone_config.config_height
        self._core.session._mark_dirty()

    # --- Run operations ---

    @pyqtSlot()
    def _on_start_clicked(self) -> None:
        self._run_async(self._start_async())

    async def _start_async(self) -> None:
        try:
            rec_dir = self._core.session.metadata.recording_directory
            if rec_dir:
                self._core.set_recording_directory(Path(rec_dir))
            await self._core.start_experiment()
        except Exception as e:
            # Non-blocking: avoid freezing the qasync loop during start.
            self._notify_user("Start Error", str(e), level="critical")

    @pyqtSlot()
    def _on_stop_clicked(self) -> None:
        self._run_async(self._stop_async())

    async def _stop_async(self) -> None:
        try:
            await self._core.stop_experiment()
        except Exception as e:
            # Non-blocking: avoid freezing the qasync loop during stop.
            self._notify_user("Stop Error", str(e), level="critical")

    @pyqtSlot()
    def _on_emergency_stop(self) -> None:
        self._run_async(self._core.emergency_stop())

    @pyqtSlot(str)
    def _on_manual_run(self, start_node_id: str) -> None:
        self._run_async(self._manual_run_async(start_node_id))

    def _on_manual_run_param(self, start_node_id: str, param: dict) -> None:
        self._run_async(self._manual_run_async(start_node_id, param))

    async def _manual_run_async(self, start_node_id: str, param: dict | None = None) -> None:
        from glider.gui.runner.manual_control_runner import RunOutcome

        self._manual_control_panel.set_running(start_node_id)
        try:
            result = await self._manual_control_runner.run(start_node_id, param=param)
        finally:
            self._manual_control_panel.set_running(None)

        if result.outcome is not RunOutcome.SUCCESS:
            self._show_status_message(f"Manual run: {result.outcome.value}")

    async def _drive_digital(self, dev_id, value):
        from glider.core.device_drive import set_digital

        dev = self._core.hardware_manager.get_device(dev_id)
        try:
            await set_digital(dev, value)
        except Exception as e:  # noqa: BLE001
            self._show_status_message(f"Device control failed: {e}")

    async def _drive_toggle(self, dev_id):
        from glider.core.device_drive import toggle_digital

        dev = self._core.hardware_manager.get_device(dev_id)
        try:
            await toggle_digital(dev)
        except Exception as e:  # noqa: BLE001
            self._show_status_message(f"Device control failed: {e}")

    async def _drive_pwm(self, dev_id, value):
        from glider.core.device_drive import set_pwm

        dev = self._core.hardware_manager.get_device(dev_id)
        try:
            await set_pwm(dev, value)
        except Exception as e:  # noqa: BLE001
            self._show_status_message(f"Device control failed: {e}")

    def _on_help(self) -> None:
        dialog = HelpDialog(self)
        dialog.exec()

    def check_for_updates(self, *, silent: bool = False) -> None:
        """Trigger a release check.

        ``silent=True`` is used by the post-launch timer in ``__main__`` — it
        only prompts if a newer version exists and the user hasn't skipped it.
        ``silent=False`` is bound to Help → Check for Updates… and always
        surfaces some feedback (up-to-date, error, or update-available).
        """
        self._update_checker.check(silent=silent)

    def _on_about(self) -> None:
        # Pull from the single source of truth so the About box never drifts
        # out of sync with pyproject.toml or installer metadata at release time.
        from glider import __version__

        QMessageBox.about(
            self,
            "About GLIDER",
            "GLIDER - General Laboratory Interface for Design, "
            "Experimentation, and Recording\n\n"
            f"Version {__version__}\n\n"
            "A modular experimental orchestration platform.",
        )

    # --- Undo/Redo ---

    def _on_undo(self) -> None:
        command = self._undo_stack.undo()
        if command:
            self._show_status_message(f"Undo: {command.description()}", 2000)
            self._update_undo_redo_actions()

    def _on_redo(self) -> None:
        command = self._undo_stack.redo()
        if command:
            self._show_status_message(f"Redo: {command.description()}", 2000)
            self._update_undo_redo_actions()

    def _update_undo_redo_actions(self) -> None:
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

    # --- Utilities ---

    def _run_async(self, coro) -> asyncio.Task:
        """Run an async coroutine with proper task tracking.

        The done-callback both removes the task from the pending set AND
        inspects ``task.exception()`` so that exceptions raised by the
        coroutine surface in the log instead of being silently swallowed
        with the standard "Task exception was never retrieved" warning at
        GC time. Without this, emergency-stop / hardware-write / network
        failures vanish into asyncio's default handler with no operator
        visibility.
        """
        from glider.core.async_utils import log_task_exception

        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        task.add_done_callback(log_task_exception)
        return task

    def closeEvent(self, event) -> None:
        """Handle window close event.

        Cancellation is advisory in asyncio: ``task.cancel()`` only
        schedules ``CancelledError`` to be raised at the next ``await``
        point. The previous implementation immediately cleared the
        pending-task set and launched ``_core.shutdown()`` — so in-flight
        cancellations raced the shutdown sequence (a write task started
        50ms earlier could still drive a pin after the shutdown loop set
        outputs LOW). We now schedule cancellation, await drain via
        ``asyncio.gather(..., return_exceptions=True)``, and only then
        kick off ``_core.shutdown()``. A 10-second budget bounds the
        whole sequence so an unresponsive task can't block app exit.
        """
        import time

        # Stop device control panel polling
        if self._device_control_panel:
            self._device_control_panel.stop_polling()

        if not self._check_save():
            event.ignore()
            return

        # Deterministically stop the CameraPanel CV thread now that we are
        # committed to closing. Runner mode always builds a CameraPanel (nested
        # inside RunnerShell), and its CV QThread is only stopped from
        # CameraPanel.closeEvent/destroyed — which Qt does not fire for a nested
        # child during MainWindow teardown. Calling close() here synchronously
        # joins the thread (idempotent via its isRunning() guard), so the Pi
        # kiosk-exit path can't abort with "QThread destroyed while still
        # running".
        if getattr(self, "_camera_panel", None) is not None:
            try:
                self._camera_panel.close()
            except Exception:
                pass

        async def _drain_and_shutdown() -> None:
            pending = [t for t in self._pending_tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            self._pending_tasks.clear()
            await self._core.shutdown()

        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(_drain_and_shutdown())
            else:
                future = asyncio.ensure_future(_drain_and_shutdown())
                deadline = time.monotonic() + 10.0
                # Bounded event-pump loop with a small sleep so we don't
                # 100%-CPU-spin (previous implementation did).
                while not future.done() and time.monotonic() < deadline:
                    QApplication.processEvents()
                    time.sleep(0.01)
                if not future.done():
                    logger.warning("Shutdown timed out after 10s; cancelling.")
                    future.cancel()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        event.accept()
