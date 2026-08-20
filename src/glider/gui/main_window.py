"""
Main Window - The primary PyQt6 window for GLIDER.

Thin coordinator that manages the high-level layout, view switching,
and signal wiring between extracted panel components.
"""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QSettings, Qt, QTimer, pyqtSignal, pyqtSlot
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
    QStackedWidget,
    QStatusBar,
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
from glider.gui.panels.node_editor_controller import NodeEditorController, node_category_for_type
from glider.gui.panels.node_library_panel import NodeLibraryPanel
from glider.gui.shell import AppShell
from glider.gui.styles import colors
from glider.gui.view_manager import ViewManager, ViewMode
from glider.hal.base_board import BoardConnectionState
from glider.vision.frame_provider import CameraFrameProvider
from glider.vision.zones import ZoneConfiguration

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)

# QSettings flag, namespaced alongside the existing first_run/* keys and kept
# separate from ``first_run/tour_complete``: one shared flag would mean sitting
# through the walkthrough silences the setup form nobody has seen yet.
LAB_SETUP_COMPLETE_KEY = "first_run/setup_complete"

# How a board's connection state is painted on the status strip.
#
# **Only CONNECTED is ever green.** RECONNECTING is a board that has already
# dropped once and is trying to come back -- painting that the same green as a
# working board is the single behaviour that would make the strip actively
# harmful, because the whole point of it is that a board failing 40 minutes into
# an unattended run is visible to whoever walks past the rig. CONNECTING gets
# the same amber: an in-flight handshake is not yet a working board either.
# DISCONNECTED is red rather than neutral, matching the status bar's existing
# reading of "no board" as a problem to act on. Anything the strip does not
# recognise -- a driver vocabulary that widened without this file changing --
# falls through to the strip's neutral grey with its raw value in the tooltip.
DEVICE_STATE_BY_BOARD_STATE = {
    BoardConnectionState.CONNECTED: "ok",
    BoardConnectionState.CONNECTING: "warn",
    BoardConnectionState.RECONNECTING: "warn",
    BoardConnectionState.DISCONNECTED: "error",
    BoardConnectionState.ERROR: "error",
}

# How a session state is shown on the run-state pill: ``(pill, detail)``.
#
# The pill has four words and the session has seven states, so three of them
# ride on a detail rather than getting a colour of their own. PAUSED, STOPPING
# and INITIALIZING are all "a run is live and mid-something", which is closer to
# Running than to Idle -- the failure this pill exists to prevent is reading
# "Idle" while hardware is driven. RUNNING becomes "recording" whenever the data
# recorder is actually recording; that is resolved at call time, not here.
RUN_PILL_BY_SESSION_STATE = {
    "IDLE": ("idle", ""),
    "READY": ("idle", "Ready"),
    "INITIALIZING": ("running", "Starting"),
    "RUNNING": ("running", ""),
    "PAUSED": ("running", "Paused"),
    "STOPPING": ("running", "Stopping"),
    "ERROR": ("error", ""),
}


class PropertiesHost(QWidget):
    """A swappable container for the node editor's properties form.

    ``NodeEditorController`` builds a fresh form on every selection and hands it
    over with ``setWidget`` -- the API it used when the properties surface was a
    ``QDockWidget``. Keeping that call shape means the controller did not have
    to change when the dock did, and it is the only thing the controller ever
    asked of the dock.

    Unlike ``QDockWidget.setWidget``, the outgoing form is deleted rather than
    left parented and hidden. A dock quietly accumulated one dead form per node
    the user ever clicked; nothing referenced them again.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("propertiesHost")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._widget: QWidget | None = None

    def widget(self) -> QWidget | None:
        """The form currently on show, or ``None`` before the first one."""
        return self._widget

    def setWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt-shaped API
        """Show *widget*, destroying whatever it replaces."""
        previous = self._widget
        if previous is widget:
            return
        if previous is not None:
            self._layout.removeWidget(previous)
            previous.setParent(None)
            previous.deleteLater()
        self._widget = widget
        self._layout.addWidget(widget)


def lab_setup_complete(
    settings: QSettings | None = None,
    key: str = LAB_SETUP_COMPLETE_KEY,
) -> bool:
    """Return True once the lab setup form has been shown -- skipped or filled in.

    Mirrors :func:`glider.gui.onboarding.tour.tour_complete`. "Seen" is the
    question, not "answered": Skip is a first-class exit from that form, so a
    skipped setup must never be offered again.
    """
    s = settings if settings is not None else QSettings()
    return bool(s.value(key, False, type=bool))


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
        settings: QSettings | None = None,
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
        # Injectable so a test never reads or writes the developer's real
        # first_run/* state -- and, more sharply, so the one-time Lab Setup
        # offer below cannot pop a modal dialog in the middle of a test run.
        self._settings = settings if settings is not None else QSettings()
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
        # The Builder frame, and a page of _stack rather than something around
        # it. That is the whole reason there is no "hide the Builder panels"
        # helper: switching the stack takes the entire frame off screen, so
        # nothing can linger over the operator view (issue #39).
        self._builder_view: AppShell | None = None
        # The node editor's properties form lives inside this; see PropertiesHost.
        self._properties_host: PropertiesHost | None = None
        self._properties_widget: QWidget | None = None
        self._files_panel: QWidget | None = None
        # View menu panel toggles, kept as attributes so their checked state can
        # follow a panel collapsed by any other route.
        self._left_panel_action: QAction | None = None
        self._right_panel_action: QAction | None = None

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
        self._dash_hardware_panel: HardwarePanel | None = None
        self._device_control_panel: DeviceControlPanel | None = None
        self._node_library_panel: NodeLibraryPanel | None = None
        self._node_editor: NodeEditorController | None = None
        self._camera_panel: CameraPanel | None = None
        # Dashboard camera-quadrant occupant: a lightweight container that is
        # ALWAYS the dashboard's "camera" panel. The single CameraPanel is
        # reparented into this slot (dashboard shown) or the Builder's Camera
        # tab slot (Builder shown) on view switch — never duplicated (Task 12).
        # Both ends are slots rather than the panel itself: a SidePanel tab
        # whose widget were carried off would leave the panel's stack holding
        # nothing to come back to.
        self._camera_slot: QWidget | None = None
        self._camera_tab_slot: QWidget | None = None
        # Lazily created when the camera panel hands off a finished video
        # tracking run for review (analysis_requested signal).
        self._analysis_dock: QDockWidget | None = None
        self._analysis_panel = None  # AnalysisPanel, imported + created lazily

        # The Plugins window, and the in-flight open that will produce it.
        # Both are needed to answer "is one already coming?": the catalogue
        # fetch takes seconds, and the menu item is clickable throughout.
        self._plugins_dialog = None  # PluginManagerDialog, created lazily
        self._plugins_task = None  # asyncio.Task for the open in progress

        # Operator (non-Builder) view. Exactly ONE of these is built, chosen by
        # startup mode: the 4-tab RunnerShell in runner mode (Pi touchscreen),
        # the 2x2 DashboardView in desktop mode. The other stays None, so every
        # site that touches a mode-specific widget must guard on it.
        self._dashboard_view = None  # DashboardView (desktop mode)
        self._run_control_panel = None  # RunControlPanel (dashboard)
        self._device_states_panel = None  # DeviceStatesPanel (dashboard)
        self._experiment_info_panel = None  # ExperimentInfoPanel (dashboard)
        self._runner_shell = None  # RunnerShell (runner mode)
        self._runner_panel = None  # RunnerPanel — Run tab (runner mode)
        self._runner_setup_page = None  # RunnerSetupPage — Setup tab (runner mode)

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

        # Offer the one-time lab setup on a launch where it has never been seen.
        # Deferred so it lands after this window is shown, and after
        # ``first_run.run_first_run_if_needed`` has had its say -- on a fresh
        # install this fires inside the welcome dialog's nested event loop,
        # where the first-run gate turns it away.
        #
        # Without this second call site the offer would reach new installs only:
        # every existing install already has first_run/tour_complete set, so
        # nothing would ever ask them, and they are the people who reported not
        # being able to find these fields in the first place.
        QTimer.singleShot(0, self.offer_lab_setup_once)

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
        # Index 1 is the operator view: RunnerShell (runner) or DashboardView
        # (desktop). _create_runner_view built exactly one of them.
        self._operator_view = self._runner_shell or self._dashboard_view
        self._stack.addWidget(self._operator_view)  # Index 1

        if self._view_manager.is_runner_mode:
            self._stack.setCurrentIndex(1)
        else:
            self._stack.setCurrentIndex(0)
            self._populate_builder_panels()

    def _create_builder_view(self) -> None:
        """Build the Builder frame: the shell, with the graph as its centre.

        The shell is a page of ``_stack``, not a wrapper around the window.
        Everything the Builder shows lives inside it, so switching the stack to
        the operator view removes the lot in one move — which is why there is no
        helper here that hides Builder panels. Its two side panels start empty;
        ``_populate_builder_panels`` fills them once the panel widgets exist.
        """
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

        self._builder_view = AppShell(centre=self._graph_view)
        self._builder_view.strip.palette_requested.connect(self._on_palette_requested)

    def _on_palette_requested(self) -> None:
        """The strip's ``Ctrl K`` hint was pressed.

        The palette itself is a later task. Until it exists the hint says so
        rather than doing nothing at all, because a control that teaches a
        shortcut and then silently ignores it teaches the wrong thing.
        """
        self._show_status_message("The command palette is not available yet", 3000)

    def _create_runner_view(self) -> None:
        """Build the operator (non-Builder) view for the detected mode.

        Runner mode (Pi touchscreen) gets the 4-tab RunnerShell — one full-width
        page at a time, which fits the 480px screen. Desktop mode gets the 2x2
        DashboardView. Exactly one is built, so the panels they share
        (CameraPanel, RunnerDeviceControls, HardwarePanel) are never double-owned.
        """
        if self._view_manager.is_runner_mode:
            self._create_runner_shell_view()
        else:
            self._create_dashboard_view()

    def _create_dashboard_view(self) -> None:
        """Create the desktop 2x2 quadrant DashboardView.

        Builds the DashboardView from five independent panels plus a persistent
        run banner. The dashboard owns its OWN HardwarePanel
        (``_dash_hardware_panel``) so the desktop docks can build a separate one
        without stealing it. The Camera panel is (for now) the single shared
        instance the desktop camera dock also re-hosts.
        """
        from glider.gui.dashboard.dashboard_view import DashboardView
        from glider.gui.dashboard.panels.device_states_panel import DeviceStatesPanel
        from glider.gui.dashboard.panels.experiment_info_panel import ExperimentInfoPanel
        from glider.gui.dashboard.panels.run_control_panel import RunControlPanel
        from glider.gui.runner.device_controls import RunnerDeviceControls
        from glider.gui.runner.run_banner import RunBanner

        # --- Run Control panel ---
        run_control = RunControlPanel(self._core)
        run_control.start_requested.connect(self._on_start_clicked)
        run_control.stop_requested.connect(self._on_stop_clicked)
        self._run_control_panel = run_control

        # --- Device States panel ---
        device_states = DeviceStatesPanel(self._core)
        self._device_states_panel = device_states

        # --- Manual Controls panel (RunnerDeviceControls) ---
        self._runner_device_controls = RunnerDeviceControls(
            self._core.hardware_manager, session_fn=lambda: self._core.session
        )
        self._runner_device_controls.action_write_requested.connect(
            lambda dev_id, action, value: self._run_async(self._drive_action(dev_id, action, value))
        )
        self._runner_device_controls.action_fire_requested.connect(
            lambda dev_id, action: self._run_async(self._drive_action(dev_id, action))
        )
        self._runner_device_controls.read_requested.connect(
            lambda dev_id, action: self._run_async(self._drive_read(dev_id, action))
        )
        self._runner_device_controls.function_run_requested.connect(
            lambda start_node_id: self._run_async(self._run_function_async(start_node_id))
        )
        # Only one manual function runs at a time: the run briefly forces the
        # engine's exec-propagation gate to RUNNING and restores it after, so
        # two overlapping runs would corrupt each other's saved state.
        self._manual_run_busy = False

        # --- Experiment Info panel (owns the dashboard's own Hardware panel) ---
        self._dash_hardware_panel = HardwarePanel(
            hardware_manager=self._core.hardware_manager,
            session_fn=lambda: self._core.session,
            run_async_fn=self._run_async,
            show_add_buttons=False,
        )
        self._dash_hardware_panel.status_message.connect(self._show_status_message)
        experiment_info = ExperimentInfoPanel(self._core, hardware_widget=self._dash_hardware_panel)
        self._experiment_info_panel = experiment_info

        # --- Camera panel (single reparented singleton). The dashboard's
        # camera quadrant hosts a lightweight container slot; the real
        # CameraPanel lives inside whichever view is currently visible (the
        # slot when the dashboard is shown, the desktop dock when Builder is).
        self._camera_slot = QWidget()
        camera_slot_layout = QVBoxLayout(self._camera_slot)
        camera_slot_layout.setContentsMargins(0, 0, 0, 0)
        self._camera_panel = self._build_camera_panel()

        # --- Persistent run banner ---
        banner = RunBanner()
        banner.stop_requested.connect(self._on_stop_clicked)

        panels = {
            "run_control": run_control,
            "device_states": device_states,
            "camera": self._camera_slot,
            "manual_controls": self._runner_device_controls,
            "experiment_info": experiment_info,
        }

        self._dashboard_view = DashboardView(
            panels,
            save_path=get_config().paths.user_config_dir / "dashboard_layout.json",
            banner=banner,
        )
        run_control.elapsed_updated.connect(self._dashboard_view.set_banner_time)

        # Initial camera placement: park the single CameraPanel in the dashboard
        # slot. Desktop startup later re-hosts it into the camera dock (see
        # _setup_dock_widgets), which is correct because Builder is shown then.
        self._move_camera_to_operator_view()

        # Banner show/hide re-evaluates whenever the layout changes (e.g. an
        # operator benches/unbenches Run Control mid-run). Init the cached
        # state BEFORE connecting so the first emit has something to read.
        self._last_dashboard_state = ("IDLE", False)
        self._dashboard_view.layout_changed.connect(
            lambda: self._dashboard_view.update_banner(*self._last_dashboard_state)
        )

        # Hardware-change fan-out to the dashboard panels (the dashboard owns
        # _dash_hardware_panel; _setup_dock_widgets builds a separate panel and
        # wires its own fan-out, so there is no double-firing).
        self._dash_hardware_panel.hardware_changed.connect(device_states.refresh_devices)
        self._dash_hardware_panel.hardware_changed.connect(experiment_info.refresh)
        self._dash_hardware_panel.hardware_changed.connect(self._runner_device_controls.refresh)
        self._dash_hardware_panel.refresh_tree()

        # Experiment Info file-action wiring (preserves the old Setup-page
        # buttons — without these the New/Open/Save/etc. buttons are dead).
        experiment_info.new_requested.connect(self._on_new)
        experiment_info.open_requested.connect(self._on_open)
        experiment_info.save_requested.connect(self._on_save)
        experiment_info.save_as_requested.connect(self._on_save_as)
        experiment_info.help_requested.connect(self._on_help)
        experiment_info.close_requested.connect(self.close)
        experiment_info.switch_to_desktop_requested.connect(self._switch_to_desktop_mode)
        experiment_info.board_settings_requested.connect(
            self._dash_hardware_panel.show_board_settings_dialog
        )

    def _create_runner_shell_view(self) -> None:
        """Create the Pi 4-tab RunnerShell view (Setup / Run / Manual / Camera).

        One full-width page at a time behind a bottom tab bar, so it fits the
        480px touchscreen. Builds the shared HardwarePanel and CameraPanel that
        the desktop docks re-host on a runner->desktop switch (no duplicates).
        """
        from glider.gui.panels.runner_panel import RunnerPanel
        from glider.gui.runner.device_controls import RunnerDeviceControls
        from glider.gui.runner.runner_setup_page import RunnerSetupPage
        from glider.gui.runner.runner_shell import RunnerShell

        # --- Run page ---
        self._runner_panel = RunnerPanel(self._core, self._view_manager)
        self._runner_panel.start_requested.connect(self._on_start_clicked)
        self._runner_panel.stop_requested.connect(self._on_stop_clicked)

        # --- Shared Hardware panel (desktop docks re-host this same instance) ---
        self._hardware_panel = HardwarePanel(
            hardware_manager=self._core.hardware_manager,
            session_fn=lambda: self._core.session,
            run_async_fn=self._run_async,
            show_add_buttons=False,
        )
        self._hardware_panel.status_message.connect(self._show_status_message)

        # --- Shared Camera panel (the single reparented CameraPanel) ---
        self._camera_panel = self._build_camera_panel()

        # --- Manual page ---
        self._runner_device_controls = RunnerDeviceControls(
            self._core.hardware_manager, session_fn=lambda: self._core.session
        )
        self._runner_device_controls.action_write_requested.connect(
            lambda dev_id, action, value: self._run_async(self._drive_action(dev_id, action, value))
        )
        self._runner_device_controls.action_fire_requested.connect(
            lambda dev_id, action: self._run_async(self._drive_action(dev_id, action))
        )
        self._runner_device_controls.read_requested.connect(
            lambda dev_id, action: self._run_async(self._drive_read(dev_id, action))
        )
        self._runner_device_controls.function_run_requested.connect(
            lambda start_node_id: self._run_async(self._run_function_async(start_node_id))
        )
        # Only one manual function runs at a time: the run briefly forces the
        # engine's exec-propagation gate to RUNNING and restores it after, so
        # two overlapping runs would corrupt each other's saved state.
        self._manual_run_busy = False

        # --- Setup page ---
        self._runner_setup_page = RunnerSetupPage(self._core, hardware_widget=self._hardware_panel)

        # --- Shell (owns the bottom tab bar + run banner) ---
        self._runner_shell = RunnerShell(
            self._core,
            self._runner_setup_page,
            self._runner_panel,
            self._runner_device_controls,
            self._camera_panel,
        )
        self._runner_panel.elapsed_updated.connect(self._runner_shell.set_banner_time)
        self._runner_shell.stop_requested.connect(self._on_stop_clicked)

        # Setup page file-action wiring (without these the buttons are dead).
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
        self._hardware_panel.hardware_changed.connect(self._runner_device_controls.refresh)
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

    def _move_camera_to_operator_view(self) -> None:
        """Host the single CameraPanel in the active operator view.

        Desktop mode has a dashboard camera slot; runner mode has the
        RunnerShell's Camera tab. Only one exists, so this targets whichever is
        built — used on entry to the operator view (including after a
        desktop->operator switch that had moved the camera into the dock).
        """
        if self._camera_panel is None:
            return
        if self._camera_slot is not None:
            self._camera_slot.layout().addWidget(self._camera_panel)  # dashboard slot
        elif self._runner_shell is not None:
            self._runner_shell.rehost_camera()  # RunnerShell Camera tab

    def _move_camera_to_builder(self) -> None:
        """Host the single CameraPanel in the Builder's Camera tab slot.

        No-op in runner-only mode, which has not built the Builder's panels yet
        — the camera then stays in the operator view, which is correct.

        Nothing needs hiding at the far end any more. The empty slot the camera
        leaves behind sits inside the Builder frame, and the Builder frame is
        the stack page that has just been switched away from, so it is not on
        screen to look empty.
        """
        if self._camera_tab_slot is not None and self._camera_panel is not None:
            self._camera_tab_slot.layout().addWidget(self._camera_panel)  # reparents

    def _populate_builder_panels(self) -> None:
        """Build the Builder's panel widgets and host them in the shell.

        Each widget here is the same object the Builder's dock widgets used to
        wrap — this method re-hosts rather than rewrites. Left panel: Nodes,
        Hardware, Control, Files. Right panel: Properties, Camera.
        """

        def session_fn():
            return self._core.session

        # --- Nodes ---
        self._node_library_panel = NodeLibraryPanel(
            session_fn=session_fn,
            graph_view=self._graph_view,
        )
        self._node_library_panel.status_message.connect(self._show_status_message)
        self._node_library_panel._zone_config = self._zone_config

        # --- Properties ---
        # The node editor replaces this form on every selection, so the tab
        # hosts a container and the form lives inside it. See PropertiesHost.
        self._properties_host = PropertiesHost()
        self._properties_widget = QWidget()
        properties_layout = QVBoxLayout(self._properties_widget)
        properties_layout.addWidget(QLabel("Select a node to view properties"))
        properties_layout.addStretch()
        self._properties_host.setWidget(self._properties_widget)
        self._node_editor.set_properties_dock(self._properties_host)

        # --- Hardware. The Builder owns its own HardwarePanel, distinct from
        # the dashboard's _dash_hardware_panel. Build it if not already present.
        if getattr(self, "_hardware_panel", None) is None:
            self._hardware_panel = HardwarePanel(
                hardware_manager=self._core.hardware_manager,
                session_fn=session_fn,
                run_async_fn=self._run_async,
            )
            self._hardware_panel.status_message.connect(self._show_status_message)

        # --- Control ---
        self._device_control_panel = DeviceControlPanel(
            hardware_manager=self._core.hardware_manager,
            run_async_fn=self._run_async,
        )
        self._device_control_panel.status_message.connect(self._show_status_message)

        # The Builder's HardwarePanel is a distinct instance from the
        # dashboard's _dash_hardware_panel; its hardware_changed fans out only
        # to the Builder's own _device_control_panel.
        self._hardware_panel.hardware_changed.connect(self._device_control_panel.refresh_devices)

        # Wire flow_functions_changed from node editor to node library
        self._node_editor.flow_functions_changed.connect(
            self._node_library_panel.refresh_flow_functions
        )

        # --- Camera. A slot rather than the panel itself: the single
        # CameraPanel commutes between here and the operator view, and a tab
        # whose widget were carried off would have nothing to come back to.
        self._camera_tab_slot = QWidget()
        camera_tab_layout = QVBoxLayout(self._camera_tab_slot)
        camera_tab_layout.setContentsMargins(0, 0, 0, 0)
        if getattr(self, "_camera_panel", None) is None:
            self._camera_panel = self._build_camera_panel()
        # Only take the single CameraPanel when Builder is the active view;
        # otherwise it stays in the operator view. This runs at desktop startup
        # (and on runner->desktop switch) with Builder shown, so the camera
        # correctly lands here then.
        if self._stack is not None and self._stack.currentIndex() == 0:
            self._move_camera_to_builder()

        # --- Files ---
        from PyQt6.QtWidgets import QFrame, QPushButton, QScrollArea

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
        self._files_panel = files_scroll

        # --- Into the shell. Files was a dock the desktop kept hidden by
        # default; as a tab it costs nothing to leave in place, and the rail
        # keeps it discoverable rather than hidden behind a menu item.
        left = self._builder_view.left
        left.add_tab("nodes", "Nodes", self._node_library_panel, "N")
        left.add_tab("hardware", "Hardware", self._hardware_panel, "H")
        left.add_tab("control", "Control", self._device_control_panel, "D")
        left.add_tab("files", "Files", self._files_panel, "F")

        right = self._builder_view.right
        right.add_tab("properties", "Properties", self._properties_host, "P")
        right.add_tab("camera", "Camera", self._camera_tab_slot, "C")

        # Refresh hardware tree (which also triggers device combo + runner refresh)
        self._hardware_panel.refresh_tree()

        self._refresh_strip_devices()
        self._refresh_strip_experiment()

        # The layout the user left the Builder in last time. Applied after the
        # tabs exist, because a saved tab key means nothing before then.
        self._builder_view.restore_layout(self._settings)
        self._sync_panel_actions()

    def _heal_stale_paint(self, *_args) -> None:
        """Schedule a deferred full-window repaint after a central relayout.

        On Windows, relayouts that move QMainWindow separators (switching the
        central stack, showing the Analysis dock, maximize/restore) can leave
        the 1px column at an old separator position unrepainted — it shows as a
        stale #0f1419 hairline over whatever now occupies that area. Deferred
        so it runs after the relayout settles; update() coalesces, so repeated
        triggers cost one repaint.
        """
        QTimer.singleShot(0, self.update)

    # --- The status strip ---

    def _strip(self):
        """The Builder's status strip, or ``None`` before the frame exists.

        Every refresh below is reached from a signal that can fire during
        construction or in runner mode, so none of them may assume a frame.
        """
        return getattr(self._shell(), "strip", None)

    def _refresh_strip_experiment(self) -> None:
        """Put the session's name and unsaved state on the strip."""
        strip = self._strip()
        if strip is None:
            return
        session = self._core.session
        if session is None:
            strip.set_experiment("", False)
            return
        strip.set_experiment(session.metadata.name or "", bool(session.is_dirty))

    def _refresh_strip_devices(self) -> None:
        """Put one dot per registered board on the strip, from real state.

        Read from ``board.state`` rather than from whatever last transition was
        reported, so the strip describes the rig as it is now. See
        :data:`DEVICE_STATE_BY_BOARD_STATE` for why only CONNECTED is green.
        """
        strip = self._strip()
        if strip is None:
            return
        devices = []
        for board_id, board in self._core.hardware_manager.boards.items():
            name = getattr(board, "name", None) or board_id
            state = DEVICE_STATE_BY_BOARD_STATE.get(getattr(board, "state", None))
            # An unmapped state keeps its raw text: the strip renders anything
            # it does not recognise neutral, with the real value in the tooltip.
            devices.append((str(name), state or str(getattr(board, "state", "unknown"))))
        strip.set_devices(devices)

    def _refresh_strip_run_state(self, state_name: str) -> None:
        """Move the run pill to match the session state."""
        strip = self._strip()
        if strip is None:
            return
        pill, detail = RUN_PILL_BY_SESSION_STATE.get(state_name, (None, ""))
        if pill is None:
            # Run state comes from our own code, so an unrecognised one is a
            # bug here rather than a wider vocabulary. Show it as an error
            # carrying the raw name: never as a state that reads as healthy.
            logger.warning("Unrecognised session state %r on the status strip", state_name)
            strip.set_run_state("error", state_name)
            return
        if pill == "running" and not detail and self._core.data_recorder.is_recording:
            pill = "recording"
        strip.set_run_state(pill, detail or None)

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

        lab_setup_action = QAction("&Lab Setup...", self)
        lab_setup_action.setToolTip(
            "Define the groups, strains, solutions and routes this lab uses"
        )
        lab_setup_action.triggered.connect(self._on_lab_setup)
        experiment_menu.addAction(lab_setup_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        switch_view_action = QAction("Switch to &Runner View", self)
        switch_view_action.setShortcut(QKeySequence("F11"))
        switch_view_action.triggered.connect(self._toggle_view)
        view_menu.addAction(switch_view_action)

        view_menu.addSeparator()

        # Six dock toggles became two panel toggles. Every panel that used to
        # need its own menu item is now a tab, and a collapsed panel is a rail
        # of those tabs rather than nothing — so the menu no longer has to be
        # the way back to a panel somebody closed.
        self._left_panel_action = QAction("&Left Panel", self)
        self._left_panel_action.setCheckable(True)
        self._left_panel_action.setChecked(True)
        self._left_panel_action.triggered.connect(
            lambda checked: self._set_panel_expanded("left", checked)
        )
        view_menu.addAction(self._left_panel_action)

        self._right_panel_action = QAction("&Right Panel", self)
        self._right_panel_action.setCheckable(True)
        self._right_panel_action.setChecked(True)
        self._right_panel_action.triggered.connect(
            lambda checked: self._set_panel_expanded("right", checked)
        )
        view_menu.addAction(self._right_panel_action)

        # The return path. A panel collapsed from the strip, from its rail or by
        # a restored layout has to move the menu item too, or the tick describes
        # a state that is not on screen. (The strip's own buttons are already
        # kept in step by AppShell; this is the same idea one level out.)
        shell = self._shell()
        if shell is not None:
            shell.left.expanded_changed.connect(self._left_panel_action.setChecked)
            shell.right.expanded_changed.connect(self._right_panel_action.setChecked)
        self._sync_panel_actions()

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

        pose_batch_action = QAction("Batch &Pose Tracking…", self)
        pose_batch_action.setStatusTip(
            "Run a pose model over directories of videos and write DeepLabCut CSVs"
        )
        pose_batch_action.triggered.connect(self._open_pose_batch)
        # Lazy import for the same reason as the behavior probe above.
        from glider.gui.pose_batch.availability import (
            missing_pose_batch_deps,
            pose_batch_available,
        )

        if not pose_batch_available():
            pose_batch_action.setEnabled(False)
            pose_batch_action.setToolTip(
                "Install the vision extra: pip install glider[vision] "
                f"(missing: {', '.join(missing_pose_batch_deps())})"
            )
        tools_menu.addAction(pose_batch_action)

        review_action = QAction("&Session Review…", self)
        review_action.setStatusTip(
            "Scrub an analyzed session, select a window, and read what is in it"
        )
        review_action.triggered.connect(self._open_session_review)
        # Shares the behavior extra's probe: it reads the same outputs.
        if not behavior_available():
            review_action.setEnabled(False)
            review_action.setToolTip(
                "Install the behavior extra: pip install glider[behavior] "
                f"(missing: {', '.join(missing_behavior_deps())})"
            )
        tools_menu.addAction(review_action)

        # GPU / device diagnostics. Always enabled — it's most useful precisely
        # when torch or a GPU is missing, and it reports that cleanly.
        gpu_check_action = QAction("&GPU / Device Check…", self)
        gpu_check_action.triggered.connect(self._on_gpu_check)
        tools_menu.addAction(gpu_check_action)

        plugins_action = QAction("&Plugins…", self)
        plugins_action.setStatusTip("Browse the plugin catalogue and install from it")
        plugins_action.triggered.connect(self._on_open_plugins)
        tools_menu.addAction(plugins_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        help_action = QAction("&GLIDER Help", self)
        help_action.setShortcut(QKeySequence("F1"))
        help_action.triggered.connect(self._on_help)
        help_menu.addAction(help_action)

        replay_tour_action = QAction("&Replay Tutorial", self)
        replay_tour_action.triggered.connect(self._start_tour)
        help_menu.addAction(replay_tour_action)

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

    def _open_pose_batch(self) -> None:
        """Open (or re-surface) the Batch Pose Tracking window.

        Same lazy-import + keep-on-self pattern as the behavior window: the
        import pulls in the pose stack (and transitively torch), so it stays
        out of GLIDER startup.
        """
        from glider.gui.pose_batch.window import PoseBatchWindow

        if getattr(self, "_pose_batch_window", None) is None:
            self._pose_batch_window = PoseBatchWindow(parent=None)
        self._pose_batch_window.show()
        self._pose_batch_window.raise_()
        self._pose_batch_window.activateWindow()

    def _open_session_review(self) -> None:
        """Open (or re-surface) the session review window.

        Same lazy-import + keep-on-self pattern as the other tool windows, so
        the pandas/pose imports stay out of GLIDER startup.
        """
        from glider.gui.behavior.analysis_window import AnalysisWindow

        if getattr(self, "_session_review_window", None) is None:
            self._session_review_window = AnalysisWindow(parent=None)
        self._session_review_window.show()
        self._session_review_window.raise_()
        self._session_review_window.activateWindow()

    def _shell(self) -> AppShell | None:
        """The Builder frame, or ``None`` if there is not one yet.

        ``getattr`` rather than the attribute, because ``_setup_menu`` is
        exercised on its own against an instance whose ``__init__`` was bypassed
        -- the pattern the dock toggles used before it, for the same reason.
        """
        return getattr(self, "_builder_view", None)

    def _set_panel_expanded(self, side: str, expanded: bool) -> None:
        """Expand or collapse one side panel from the View menu."""
        shell = self._shell()
        if shell is None:
            return
        panel = shell.left if side == "left" else shell.right
        panel.set_expanded(expanded)

    def _sync_panel_actions(self) -> None:
        """Make the View menu ticks match the panels as they actually are."""
        shell = self._shell()
        if shell is None:
            return
        for action, panel in (
            (getattr(self, "_left_panel_action", None), shell.left),
            (getattr(self, "_right_panel_action", None), shell.right),
        ):
            if action is not None:
                action.setChecked(panel.expanded)

    def _setup_toolbar(self) -> None:
        """Set up the toolbar."""
        if self._view_manager.is_runner_mode:
            return

        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        # Kept as attributes so the onboarding tour can spotlight the Start
        # button (via toolbar.widgetForAction) — see tour_targets().
        self._toolbar = toolbar

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
        self._start_action = start_action

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

        # _runner_device_controls is built in both operator views (shared).
        self.session_changed.connect(self._runner_device_controls.refresh)
        self.session_changed.connect(self._surface_load_warnings)
        self.session_changed.connect(self._refresh_strip_experiment)
        # Dashboard-only panels (desktop mode).
        if self._run_control_panel is not None:
            self.session_changed.connect(lambda: self._run_control_panel.update_experiment_name())
        if self._experiment_info_panel is not None:
            self.session_changed.connect(self._experiment_info_panel.refresh)
        # Runner-only Setup page (runner mode).
        if self._runner_setup_page is not None:
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

        # Update the operator view (whichever was built for this mode).
        if self._dashboard_view is not None:
            self._dashboard_view.update_state(state_name)
            recording = bool(self._core.data_recorder.is_recording)
            self._last_dashboard_state = (state_name, recording)
            self._dashboard_view.update_banner(state_name, recording)
        if self._runner_shell is not None:
            self._runner_shell.update_state(state_name)

        self._refresh_strip_run_state(state_name)

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
        self._refresh_strip_devices()

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

    def _enter_dashboard(self) -> None:
        """Show the operator view (stack index 1) and refresh its hardware-derived
        panels, so a hardware change made via the desktop dock in Builder mode is
        reflected on entry. Shared by the menu toggle and programmatic switch."""
        self._stack.setCurrentIndex(1)
        self._move_camera_to_operator_view()
        # Nothing to hide: the Builder frame is the stack page that just went
        # away, and everything the Builder shows is inside it (issue #39).
        if self._dash_hardware_panel:
            self._dash_hardware_panel.refresh_tree()
        self._heal_stale_paint()

    def _toggle_view(self) -> None:
        if self._stack.currentIndex() == 0:
            self._enter_dashboard()
        else:
            self.switch_to_builder()

    def switch_to_builder(self) -> None:
        self._stack.setCurrentIndex(0)
        self._move_camera_to_builder()
        self._heal_stale_paint()

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Maximize/restore reflows the dock areas the same way a dock
        # show/hide does — same stale-hairline risk, same heal.
        if event.type() == QEvent.Type.WindowStateChange:
            self._heal_stale_paint()
        super().changeEvent(event)

    def switch_to_runner(self) -> None:
        self._enter_dashboard()

    def _set_window_size(self, width: int, height: int) -> None:
        self.setMinimumSize(min(width, 480), min(height, 480))
        self.resize(width, height)
        self._show_status_message(f"Window resized to {width}x{height}", 2000)

    def _set_pi_touchscreen_layout(self) -> None:
        """Shrink to the Pi touchscreen size and show the operator view.

        There is nothing to re-tabify any more: the Builder's panels are tabs
        inside its own frame, and 480 px is not a Builder-sized window in any
        case. So this now does the one thing that was ever load-bearing — put
        the window on the surface that fits the screen.
        """
        self.setMinimumSize(480, 480)
        self.resize(480, 800)

        if self._stack is not None:
            self._enter_dashboard()

        self._show_status_message("Pi Touchscreen layout applied", 2000)

    def _set_default_layout(self) -> None:
        """Restore the default desktop layout: both panels open, first tabs."""
        self.resize(1400, 900)

        if self._builder_view is not None:
            for panel in (self._builder_view.left, self._builder_view.right):
                panel.set_expanded(True)
                keys = panel.keys()
                if keys:
                    panel.set_current(keys[0])
            self._sync_panel_actions()

        self._show_status_message("Default layout restored", 2000)

    def _switch_to_desktop_mode(self) -> None:
        """Switch from runner to desktop mode."""
        self._view_manager.mode = ViewMode.DESKTOP
        self.setWindowFlags(Qt.WindowType.Window)
        self.showNormal()

        self._stack.setCurrentIndex(0)

        if self._node_library_panel is None:
            self._populate_builder_panels()

        # Ensure the single reparented CameraPanel lands in the Builder's Camera
        # tab. On first invocation _populate_builder_panels already did this
        # (idempotent); on repeat invocations (the panels already exist) the
        # camera is still in the operator view from switch_to_runner, so this
        # explicit move prevents an empty Camera tab. Mirrors switch_to_builder().
        self._move_camera_to_builder()

        screen_size = self._view_manager.screen_size
        if screen_size.width() <= 800:
            self.showMaximized()
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
            if self._dash_hardware_panel:
                self._dash_hardware_panel.refresh_tree()
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
                if self._dash_hardware_panel:
                    self._dash_hardware_panel.refresh_tree()
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

                category = node_category_for_type(node_type)

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
                self._refresh_strip_experiment()
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
                self._refresh_strip_experiment()
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
            if self._dash_hardware_panel:
                self._dash_hardware_panel.refresh_tree()
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
        # Hand the dialog COPIES. It edits its settings objects in place, so
        # passing the live ones let the edit land on the manager's own state
        # before apply/update_settings ever ran — their "did anything change?"
        # checks then compared a value to itself and always said no. Net
        # effect: picking a new CV backend, model, or camera index in this
        # dialog silently never took effect on the running app, while the UI
        # and the saved .glider file both reported the new choice.
        #
        # replace() suffices for CameraSettings (every field is an immutable
        # scalar or tuple); CVSettings needs .copy() because keypoint_names is
        # a list a shallow copy would share.
        dialog = CameraSettingsDialog(
            camera_settings=replace(self._core.camera_manager.settings),
            cv_settings=self._core.cv_processor.settings.copy(),
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
            frame_provider=CameraFrameProvider(self._core.camera_manager),
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

    def _on_open_plugins(self) -> None:
        """Open the plugin browser. Non-modal: an install takes minutes.

        Before ``discover_plugins`` has run there is no manager, and opening a
        window then would list nothing — which reads as a broken index rather
        than a cold start. Say so instead.
        """
        if self._core.plugin_manager is None:
            QMessageBox.information(
                self,
                "Plugins",
                "The plugin system has not started yet. Finish loading the "
                "session and try again.",
            )
            return

        if self._plugins_dialog is not None:
            self._plugins_dialog.raise_()
            self._plugins_dialog.activateWindow()
            return
        if self._plugins_task is not None and not self._plugins_task.done():
            # The catalogue fetch takes seconds and the menu stays clickable
            # throughout, so without this three clicks are three windows and
            # three fetches.
            return

        # Lazy import, as with the other tool windows: keeps startup free of the
        # dialog and its registry/installer imports until the menu is used.
        from glider.gui.dialogs.plugin_manager_dialog import PluginManagerDialog

        # Keep the task and read its result. A bare `ensure_future` is the exact
        # hazard the dialog's own `_spawn` documents: asyncio holds only a weak
        # reference, and an exception with nobody to raise to becomes a GC-time
        # log line the user never sees.
        task = asyncio.ensure_future(
            PluginManagerDialog.open_for(parent=self, plugin_manager=self._core.plugin_manager)
        )
        self._plugins_task = task
        task.add_done_callback(self._on_plugins_window_ready)

    def _on_plugins_window_ready(self, task) -> None:
        """Take delivery of the Plugins window, or say why there isn't one."""
        self._plugins_task = None
        try:
            dialog = task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception("Opening the Plugins window failed")
            QMessageBox.warning(
                self,
                "Plugins",
                f"The Plugins window could not be opened:\n\n{exc}",
            )
            return

        if dialog is None:
            # `open_for` already told the user why.
            return
        self._plugins_dialog = dialog
        dialog.finished.connect(self._on_plugins_window_closed)

    def _on_plugins_window_closed(self, _result: int = 0) -> None:
        self._plugins_dialog = None

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
        self._refresh_strip_experiment()

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
        # A manual function run briefly forces the flow engine RUNNING to gate
        # exec propagation; starting a real experiment on top of that would make
        # FlowEngine.start() early-return and never run the flow. Refuse instead.
        if self._manual_run_busy:
            self._notify_user(
                "Function running",
                "A function is running on the Manual tab. Let it finish before starting.",
                level="warning",
            )
            return
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

    async def _drive_action(self, dev_id, action, *value):
        """Drive any device action from the generated runner controls.

        Routes through ``execute_action`` — the one chokepoint that clamps the
        value to the action's declared range and serializes commands per device.
        ``value`` is empty for a no-value command action.
        """
        dev = self._core.hardware_manager.get_device(dev_id)
        if dev is None:
            self._runner_device_controls.on_action_failed(
                dev_id, action, f"{dev_id} is no longer available"
            )
            return
        try:
            await dev.execute_action(action, *value)
            self._runner_device_controls.on_action_succeeded(dev_id, action, *value)
        except Exception as e:  # noqa: BLE001
            self._runner_device_controls.on_action_failed(dev_id, action, f"{action} failed: {e}")

    def _surface_load_warnings(self) -> None:
        """Show out-of-range-at-load warnings (D11) on the runner status strip."""
        warnings = self._core.flow_engine.consume_load_warnings()
        if not warnings:
            return
        msg = warnings[0]
        if len(warnings) > 1:
            msg = f"{msg}  (+{len(warnings) - 1} more)"
        self._runner_device_controls.show_status(msg, level="warn")

    async def _drive_read(self, dev_id, action):
        dev = self._core.hardware_manager.get_device(dev_id)
        if dev is None:
            self._runner_device_controls.show_status(f"{dev_id} is no longer available")
            return
        try:
            value = await dev.execute_action(action)
            self._runner_device_controls.set_read_value(dev_id, action, str(value))
        except Exception as e:  # noqa: BLE001
            self._runner_device_controls.show_status(f"Read failed: {e}")

    async def _run_function_async(self, start_node_id: str) -> None:
        """Run a graph function from a Runner Functions button.

        Lazily instantiates the graph if the engine is idle, optionally prompts
        for a parameter (e.g. N revolutions), then runs the function through the
        flow engine's shared single-in-flight runner. Exec-flow propagation is
        gated on ``FlowState.RUNNING``, so the engine is briefly forced RUNNING
        for the duration and restored after — this does not start a recorded
        experiment or change session state.
        """
        from glider.core.experiment_session import SessionState
        from glider.core.flow_engine import FlowState
        from glider.core.graph_functions import find_run_param

        if self._manual_run_busy:
            self._runner_device_controls.show_status("A function is already running", level="info")
            return
        # A real experiment and a manual run both drive the engine's exec gate;
        # they must not overlap. Refuse if one is live or mid-start/stop.
        if self._core.state in (SessionState.RUNNING, SessionState.PAUSED) or (
            self._core.is_experiment_busy
        ):
            self._runner_device_controls.show_status(
                "Stop the experiment before running a function manually"
            )
            return
        if not self._core.hardware_manager.is_any_board_connected():
            self._runner_device_controls.show_status("Connect hardware to run a function")
            return

        self._manual_run_busy = True
        engine = self._core.flow_engine
        try:
            # Lazy load: instantiate the graph if idle (engine cleared on load).
            if not engine.nodes:
                self._core.setup_flow()
            if engine.get_node(start_node_id) is None:
                self._runner_device_controls.show_status("That function is no longer in the graph")
                return

            # Optional touchscreen parameter (e.g. revolutions/counts target),
            # injected onto the live node after setup so it drives this run.
            param = find_run_param(start_node_id, self._core.session.flow)
            if param is not None and not self._prompt_run_param(engine, param):
                return  # operator cancelled the prompt

            prev_state = engine.state
            engine.state = FlowState.RUNNING
            self._runner_device_controls.set_function_running(start_node_id, True)
            try:
                runner = engine.get_function_runner(start_node_id)
                completed = await runner.execute(
                    on_timeout=lambda: self._runner_device_controls.show_status(
                        "Function is unresponsive — cancelled", level="warn"
                    )
                )
                if completed:
                    self._runner_device_controls.clear_status()
            finally:
                self._runner_device_controls.set_function_running(start_node_id, False)
                engine.state = prev_state
        except Exception as e:  # noqa: BLE001
            self._runner_device_controls.show_status(f"Function failed: {e}")
        finally:
            self._manual_run_busy = False

    def _prompt_run_param(self, engine, param) -> bool:
        """Prompt for a run parameter and inject it onto the live node.

        Returns True to proceed with the run, False if the operator cancelled.
        """
        from PyQt6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getInt(
            self, param.label, f"{param.label}:", param.value, 1, 1_000_000
        )
        if not ok:
            return False
        target = engine.get_node(param.node_id)
        if target is not None and hasattr(target, "_state"):
            target._state[param.state_key] = value
        return True

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

    # --- Onboarding tour ---

    def tour_targets(self) -> dict[str, QWidget | None]:
        """Registry mapping tour step keys to live widgets to spotlight.

        Values may be None (e.g. in runner mode the Builder's panels are not
        built); the tour renders those steps centered with no spotlight rather
        than failing.
        """
        start_btn: QWidget | None = None
        toolbar = getattr(self, "_toolbar", None)
        start_action = getattr(self, "_start_action", None)
        if toolbar is not None and start_action is not None:
            start_btn = toolbar.widgetForAction(start_action)
        return {
            "node_library": self._node_library_panel,
            "dock_tabs": self._panel_tab_strip("left"),
            "canvas": getattr(self, "_graph_view", None),
            "hardware": self._hardware_panel,
            "properties": self._properties_host,
            "camera": self._camera_panel,
            "run": start_btn,
        }

    def _panel_tab_strip(self, side: str) -> QWidget | None:
        """One side panel's row of tab buttons, for the tour to spotlight.

        Found by the objectName ``SidePanel`` publishes rather than by reaching
        into the panel's internals.
        """
        shell = self._shell()
        if shell is None:
            return None
        panel = shell.left if side == "left" else shell.right
        return panel.findChild(QFrame, "sidePanelTabs")

    def _start_tour(self) -> None:
        """Launch the interactive walkthrough (Help ▸ Replay Tutorial)."""
        from glider.gui.onboarding import start_tour

        tour = start_tour(self)
        # Lab Setup follows the walkthrough rather than racing it. Gating on the
        # flag alone would never fire: nothing else asks after the tour resolves,
        # and asking any earlier puts a modal form over the spotlight.
        tour.finished.connect(self._on_tour_finished)

    def _on_tour_finished(self) -> None:
        """Offer the one-time lab setup now the walkthrough has resolved."""
        # Deferred a tick: the overlay is deleteLater'd, and a replay started
        # from the Tutorial button finishes the previous tour before registering
        # itself, so an immediate offer could land on top of the new one.
        QTimer.singleShot(0, self.offer_lab_setup_once)

    def offer_lab_setup_once(self, settings: QSettings | None = None) -> bool:
        """Show the lab setup form the first time, and never again.

        Returns True if it was shown. Called from three places, because no one
        of them reaches everyone:

        * **At launch.** The only path that reaches an existing install, whose
          walkthrough was finished long ago and will never resolve again.
        * **When the walkthrough resolves.** On a fresh install the launch-time
          offer comes due inside the welcome dialog's nested event loop, where
          the first-run gate turns it away.
        * **When the welcome resolves without starting a tour**
          (:func:`glider.first_run.run_first_run_if_needed`). Otherwise the
          fresh install that declines the tour gets no offer at all that
          session -- and someone who skips the tour is exactly the person who
          later cannot find these fields.

        The flag is recorded *before* the dialog opens, so every way out of it
        -- Done, Skip, Esc, the window close button, even a crash -- counts as
        seen. Recording only on ``QDialog.Accepted`` would re-ask at every launch
        until something was typed in, which is how a skippable form becomes a
        nag and how junk treatment groups get entered to make it go away. It is
        also what keeps the two call sites from double-offering in one session:
        whichever runs first closes the gate.
        """
        from glider.first_run import is_first_run

        s = settings if settings is not None else self._settings
        if lab_setup_complete(s):
            return False
        # Not on the Pi runner: a 480px touch surface with no menu bar, where a
        # modal form of five editable lists is the wrong shape entirely. Left
        # deliberately unreachable there rather than overlooked -- the runner's
        # vocabulary is whatever the desktop side already defined.
        if self._view_manager.is_runner_mode:
            return False
        # Not while the first-launch welcome is still up: this fires from a
        # timer inside that dialog's own nested event loop.
        if is_first_run(s):
            return False
        # Not over the walkthrough -- a modal form covers the very widget the
        # spotlight is pointing at.
        if getattr(self, "_active_tour", None) is not None:
            return False

        s.setValue(LAB_SETUP_COMPLETE_KEY, True)
        self._on_lab_setup()
        return True

    def _on_lab_setup(self) -> None:
        """Open the lab vocabulary form (Experiment ▸ Lab Setup...).

        Unconditional: the person doing first launch is often not the person who
        knows the lab's strains, so this is how that knowledge gets in after the
        one-time offer has already been answered.
        """
        from glider.gui.dialogs.lab_setup_dialog import LabSetupDialog

        dialog = LabSetupDialog(
            parent=self,
            is_touch_mode=self._view_manager.is_runner_mode,
        )
        dialog.exec()

    def _on_gpu_check(self) -> None:
        """Show accelerator diagnostics (Tools ▸ GPU / Device Check).

        Reuses the pose subsystem's device utilities so the report matches what
        inference resolves at runtime (CUDA > MPS > CPU). Works — and is worth
        opening — even without torch, where it reports what's missing.
        """
        try:
            from glider.vision.pose.device import diagnose, format_gpu_info, resolve_device
        except Exception as e:  # pragma: no cover - only on a broken vision install
            QMessageBox.warning(self, "GPU / Device Check", f"Diagnostics unavailable: {e}")
            return

        marks = {"ok": "✓", "warn": "⚠", "fail": "✗", "info": "·"}
        lines = [format_gpu_info(), ""]
        lines += [
            f"{marks.get(status, '?')} {check}: {detail}" for check, status, detail in diagnose()
        ]
        try:
            selected = resolve_device(None)
        except Exception as e:
            selected = f"unavailable ({e.__class__.__name__})"
        lines += ["", f"Inference will use: {selected}"]

        box = QMessageBox(self)
        box.setWindowTitle("GPU / Device Check")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("GLIDER GPU / device check")
        box.setInformativeText("\n".join(lines))
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

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

        # The Builder frame is a page of the stack, not a window, so the window
        # is what remembers its layout. Never raises; see AppShell.save_layout.
        if self._builder_view is not None:
            self._builder_view.save_layout(self._settings)

        # Deterministically stop the CameraPanel CV thread now that we are
        # committed to closing. Runner mode always builds a CameraPanel (nested
        # inside the DashboardView's camera quadrant), and its CV QThread is only stopped from
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
