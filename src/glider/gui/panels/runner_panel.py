"""
Runner Panel - Touch-optimized dashboard view for experiment execution.

Provides device status cards, experiment controls (start/stop/e-stop),
elapsed timer, and runner-mode menu.
"""

import logging
import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from glider.core.config import get_config
from glider.gui.runner.readiness import compute_readiness
from glider.gui.runner.run_timer import format_elapsed
from glider.gui.styles import colors

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore
    from glider.gui.view_manager import ViewManager

logger = logging.getLogger(__name__)


class RunnerPanel(QWidget):
    """Touch-optimized dashboard view for experiment execution."""

    experiment_name_changed = pyqtSignal(str)
    open_requested = pyqtSignal()
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    emergency_stop_requested = pyqtSignal()
    board_settings_requested = pyqtSignal()
    switch_to_desktop_requested = pyqtSignal()
    help_requested = pyqtSignal()
    close_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    elapsed_updated = pyqtSignal(str)

    def __init__(self, core: "GliderCore", view_manager: "ViewManager", parent=None):
        super().__init__(parent)
        self._core = core
        self._view_manager = view_manager

        # Store device widgets for updates
        self._runner_device_cards: dict[str, QWidget] = {}
        self._experiment_start_time: float | None = None
        self._state_name = "IDLE"

        self.setObjectName("runnerView")
        self._setup_ui()

    def _setup_ui(self):
        """Build the runner panel UI optimized for 480x800 portrait."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # === Header Bar ===
        header = QWidget()
        header.setFixedHeight(50)
        header.setProperty("runnerHeader", True)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 4, 12, 4)

        self._runner_exp_name = QLineEdit("Untitled Experiment")
        self._runner_exp_name.setProperty("title", True)
        self._runner_exp_name.setPlaceholderText("Enter experiment name...")
        self._runner_exp_name.textChanged.connect(self._on_experiment_name_changed)
        header_layout.addWidget(self._runner_exp_name)

        header_layout.addStretch()

        self._runner_timer = QLabel("00:00.00")
        self._runner_timer.setProperty("timer", True)
        self._runner_timer.setStyleSheet(
            f"color: {colors.SUCCESS}; font-size: 36px; font-weight: bold; font-family: monospace;"
        )
        header_layout.addWidget(self._runner_timer)

        self._status_label = QLabel("IDLE")
        self._status_label.setProperty("runnerStatus", True)
        self._status_label.setProperty("statusState", "IDLE")
        header_layout.addWidget(self._status_label)

        self._runner_menu_btn = QPushButton("\u2699\ufe0f")
        self._runner_menu_btn.setProperty("buttonRole", "secondary")
        self._runner_menu_btn.clicked.connect(self._show_runner_menu)
        header_layout.addWidget(self._runner_menu_btn)

        layout.addWidget(header)

        # === Readiness Strip ===
        self._readiness_strip = QWidget()
        self._readiness_strip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._readiness_strip.setProperty("readinessStrip", True)
        strip_layout = QVBoxLayout(self._readiness_strip)
        strip_layout.setContentsMargins(0, 0, 0, 0)
        strip_layout.setSpacing(6)

        self._board_row = QPushButton()
        self._board_row.setProperty("readinessRow", "blocked")
        self._board_row.setCursor(Qt.CursorShape.PointingHandCursor)
        self._board_row.clicked.connect(lambda: self.board_settings_requested.emit())
        strip_layout.addWidget(self._board_row)

        exp_row_container = QWidget()
        exp_row_layout = QHBoxLayout(exp_row_container)
        exp_row_layout.setContentsMargins(0, 0, 0, 0)
        exp_row_layout.setSpacing(0)

        self._exp_row = QPushButton()
        self._exp_row.setProperty("readinessRow", "blocked")
        self._exp_row.setCursor(Qt.CursorShape.PointingHandCursor)
        self._exp_row.clicked.connect(lambda: self.open_requested.emit())
        exp_row_layout.addWidget(self._exp_row, 1)

        self._reload_btn = QPushButton("⟳")
        self._reload_btn.setProperty("buttonRole", "secondary")
        self._reload_btn.setFixedSize(44, 44)
        self._reload_btn.clicked.connect(lambda: self.reload_requested.emit())
        self._reload_btn.hide()
        exp_row_layout.addWidget(self._reload_btn)

        strip_layout.addWidget(exp_row_container)

        layout.addWidget(self._readiness_strip)

        # === Recording Indicator ===
        self._runner_recording = QLabel("\u25cf REC")
        self._runner_recording.setProperty("recording", True)
        self._runner_recording.setFixedHeight(28)
        self._runner_recording.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._runner_recording.hide()
        layout.addWidget(self._runner_recording)

        # === Device Status Area (Scrollable) ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background-color: transparent;")

        from PyQt6.QtWidgets import QScroller

        QScroller.grabGesture(
            scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )

        self._runner_devices_widget = QWidget()
        self._runner_devices_layout = QVBoxLayout(self._runner_devices_widget)
        self._runner_devices_layout.setContentsMargins(0, 0, 0, 0)
        self._runner_devices_layout.setSpacing(8)

        self._runner_no_devices = QLabel("Connect hardware to see devices")
        self._runner_no_devices.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._runner_no_devices.setProperty("textRole", "muted")
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

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self._start_btn = QPushButton("\u25b6  START")
        self._start_btn.setFixedHeight(60)
        self._start_btn.setProperty("runnerAction", "start")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self.start_requested.emit)
        top_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("\u25a0  STOP")
        self._stop_btn.setFixedHeight(60)
        self._stop_btn.setProperty("runnerAction", "stop")
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        top_row.addWidget(self._stop_btn)

        controls_layout.addLayout(top_row)

        self._emergency_btn = QPushButton("EMERGENCY STOP")
        self._emergency_btn.setFixedHeight(60)
        self._emergency_btn.setProperty("runnerAction", "emergency")
        self._emergency_btn.clicked.connect(self.emergency_stop_requested.emit)
        controls_layout.addWidget(self._emergency_btn)

        layout.addWidget(controls)

        # Timers
        config = get_config()
        self._device_refresh_timer = QTimer()
        self._device_refresh_timer.setInterval(config.timing.device_refresh_interval_ms)
        self._device_refresh_timer.timeout.connect(self._update_runner_device_states)

        self._elapsed_timer = QTimer()
        self._elapsed_timer.setInterval(config.timing.elapsed_timer_interval_ms)
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)

        self._readiness_timer = QTimer(self)
        self._readiness_timer.setInterval(500)
        self._readiness_timer.timeout.connect(self.refresh_readiness)
        self._readiness_timer.start()

        # --- TEMPORARY: main-thread stall instrument (remove in Task 9) ---
        self._stall_last_tick: float | None = None
        self._stall_timer = QTimer(self)
        self._stall_timer.setInterval(50)
        self._stall_timer.timeout.connect(self._check_main_thread_stall)
        self._stall_timer.start()

        self.refresh_readiness()

    # --- Public API ---

    def refresh_devices(self) -> None:
        """Refresh the device cards in runner view."""
        for card in self._runner_device_cards.values():
            card.setParent(None)
            card.deleteLater()
        self._runner_device_cards.clear()

        devices = self._core.hardware_manager.devices

        if not devices:
            self._runner_no_devices.show()
            return

        self._runner_no_devices.hide()

        for device_id, device in devices.items():
            card = self._create_device_card(device_id, device)
            self._runner_devices_layout.insertWidget(self._runner_devices_layout.count() - 1, card)
            self._runner_device_cards[device_id] = card

        self.refresh_readiness()

    def refresh_readiness(self) -> None:
        """Recompute board/experiment readiness and update the strip + START button."""
        r = compute_readiness(self._core)
        # PAUSED is live too (auto-pause on mid-run board disconnect); the setup
        # rows must not reappear mid-experiment.
        live = self._state_name in ("RUNNING", "PAUSED")
        # Visibility depends on run-state, which is NOT part of `r`. Apply it BEFORE
        # the change-guard, or entering a live state (readiness unchanged) would
        # return early and never hide the strip.
        self._readiness_strip.setVisible(not live)
        if r == getattr(self, "_last_readiness", None):
            return
        self._last_readiness = r
        self._board_row.setText(
            f"✓ {r.board_label}" if r.board_ready else "🔌 Board not connected — tap to connect"
        )
        self._board_row.setProperty("readinessRow", "ok" if r.board_ready else "blocked")
        self._exp_row.setText(
            f"✓ {r.experiment_label or 'Experiment loaded'}"
            if r.experiment_ready
            else "📄 No experiment loaded — tap to open"
        )
        self._exp_row.setProperty("readinessRow", "ok" if r.experiment_ready else "blocked")
        self._reload_btn.setVisible(r.experiment_ready)
        for w in (self._board_row, self._exp_row):
            w.style().unpolish(w)
            w.style().polish(w)
        self._start_btn.setEnabled(r.all_ready)

    def update_state(self, state_name: str) -> None:
        """Update UI based on core state changes."""
        self._state_name = state_name

        # Update status label
        self._status_label.setText(state_name)
        self._status_label.setProperty("statusState", state_name)
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

        self.refresh_readiness()

        # The header timer is hidden while live (RUNNING or PAUSED) because the
        # persistent run banner (shown across both Runner pages) owns the
        # visible timer then; it is reshown once idle/stopped, where the
        # snap-to-duration repaint below already keeps it in sync. PAUSED counts
        # as live: main_window auto-pauses on a mid-run board disconnect, and
        # the banner (not the header) stays visible through that.
        if state_name in ("RUNNING", "PAUSED"):
            self._runner_timer.hide()
        else:
            self._runner_timer.show()

        # Update recording indicator
        if state_name == "RUNNING" and self._core.data_recorder.is_recording:
            self._runner_recording.show()
        else:
            self._runner_recording.hide()

        # Start/stop device refresh timer
        if state_name == "RUNNING":
            self._device_refresh_timer.start()
        else:
            self._device_refresh_timer.stop()
            self._update_runner_device_states()

        # Start/stop elapsed timer
        if state_name == "RUNNING":
            self._experiment_start_time = time.time()
            self._elapsed_timer.start()
            self._update_elapsed_time()
        else:
            self._elapsed_timer.stop()
            # When the flow has completed, snap the displayed elapsed time
            # to the flow's *logical* duration rather than leaving it on the
            # last QTimer tick (which includes teardown latency — closing
            # recorder files, atomic-renaming output, driving devices low,
            # and so on, which adds a variable 100-400ms on a Pi). This is
            # what keeps a ``Delay(10s)`` flow display 10.00s instead of
            # 10.11s / 10.43s run-to-run.
            self._snap_timer_to_flow_duration()

    def update_experiment_name(self, name: str | None = None) -> None:
        """Update the experiment name from session."""
        self._runner_exp_name.blockSignals(True)
        if name:
            self._runner_exp_name.setText(name)
        elif self._core.session and self._core.session.metadata.name:
            self._runner_exp_name.setText(self._core.session.metadata.name)
        else:
            self._runner_exp_name.setText("Untitled Experiment")
        self._runner_exp_name.blockSignals(False)

    # --- Internal methods ---

    def _on_experiment_name_changed(self, name: str) -> None:
        """Handle experiment name change from user input."""
        if self._core.session:
            self._core.session.metadata.name = name
            self._core.session.mark_dirty()
        self.experiment_name_changed.emit(name)

    def _check_main_thread_stall(self) -> None:
        """TEMPORARY: log gaps in the Qt event loop > 200ms."""
        now = time.monotonic()
        if self._stall_last_tick is not None:
            gap = now - self._stall_last_tick
            if gap > 0.200:
                logger.warning(
                    "Main-thread stall: %.3fs (QTimer coalesced %d ticks)",
                    gap,
                    max(0, int(gap / 0.050) - 1),
                )
        self._stall_last_tick = now

    def _update_elapsed_time(self) -> None:
        """Update the elapsed time display.

        Format is ``MM:SS.cc`` (or ``HH:MM:SS.cc`` past one hour), where ``cc``
        is centiseconds — two decimal digits of seconds. We deliberately round
        *toward zero* (truncate) rather than rounding nearest so the display
        never jumps ahead of the wall clock and the centiseconds field never
        reads "60" on a boundary.
        """
        if self._experiment_start_time is None:
            return
        self._set_timer_display(time.time() - self._experiment_start_time)

    def _snap_timer_to_flow_duration(self) -> None:
        """On flow end, freeze the timer on the flow's logical duration.

        This is the operator-visible piece of the timing fix. The QTimer's
        last live-tick was a few hundred ms before the state change
        actually fired (timer ticks at the configured interval), and the
        state change itself fired *after* the entire teardown sequence.
        Without this, the display ends on a stale wall-clock value that
        includes I/O latency. ``core.last_flow_duration_s`` is anchored
        to flow-engine start/end and is the truth-of-record.
        """
        duration = self._core.last_flow_duration_s
        if duration is None:
            # No flow ran (or in progress) — leave the last live tick as
            # the display. Happens on cleanup paths that don't correspond
            # to a flow termination.
            return
        self._set_timer_display(duration)

    def _set_timer_display(self, elapsed: float) -> None:
        """Format ``elapsed`` (seconds) and paint it into the timer label."""
        text = format_elapsed(elapsed)
        self._runner_timer.setText(text)
        self.elapsed_updated.emit(text)

    def _update_runner_device_states(self) -> None:
        """Update the device state displays in runner view."""
        for device_id, card in self._runner_device_cards.items():
            device = self._core.hardware_manager.get_device(device_id)
            if device is None:
                continue

            initialized = getattr(device, "_initialized", False)
            device_type = getattr(device, "device_type", "Unknown")
            is_analog_input = device_type == "AnalogInput"

            if hasattr(card, "_state_label"):
                if is_analog_input:
                    last_value = getattr(device, "_last_value", None)
                    if last_value is not None:
                        voltage = (last_value / 1023.0) * 5.0
                        state_text = f"{last_value}\n{voltage:.2f}V"
                        state_color = colors.ACCENT
                        font_size = "11px"
                    else:
                        state_text = "---"
                        state_color = colors.BORDER
                        font_size = "11px"
                else:
                    state = getattr(device, "_state", None)
                    if state is not None:
                        if isinstance(state, bool):
                            state_text = "HIGH" if state else "LOW"
                            state_color = colors.SUCCESS if state else colors.TEXT_MUTED
                        else:
                            state_text = str(state)[:6]
                            state_color = colors.ACCENT
                    else:
                        state_text = "---"
                        state_color = colors.BORDER
                    font_size = "14px"

                card._state_label.setText(state_text)
                card._state_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {state_color};
                        color: {colors.TEXT_PRIMARY};
                        font-size: {font_size};
                        font-weight: bold;
                        border-radius: 8px;
                        padding: 4px 8px;
                        border: none;
                        line-height: 1.2;
                    }}
                """)

            if hasattr(card, "_ready_label"):
                card._ready_label.setText("Ready" if initialized else "---")
                card._ready_label.setStyleSheet(
                    f"font-size: 10px; color: {colors.SUCCESS if initialized else colors.TEXT_DISABLED}; background: transparent; border: none;"
                )

    def _create_device_card(self, device_id: str, device) -> QWidget:
        """Create a device status card for the runner view."""
        card = QWidget()
        card.setProperty("deviceCard", True)
        card.setFixedHeight(80)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        name_label = QLabel(device_id)
        name_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {colors.TEXT_PRIMARY}; background: transparent; border: none;"
        )
        info_layout.addWidget(name_label)

        device_type = getattr(device, "device_type", "Unknown")
        type_label = QLabel(device_type)
        type_label.setStyleSheet(
            f"font-size: 12px; color: {colors.TEXT_MUTED}; background: transparent; border: none;"
        )
        info_layout.addWidget(type_label)

        layout.addLayout(info_layout)
        layout.addStretch()

        initialized = getattr(device, "_initialized", False)
        device_type = getattr(device, "device_type", "Unknown")

        is_analog_input = device_type == "AnalogInput"

        status_widget = QWidget()
        status_widget.setFixedSize(80 if is_analog_input else 60, 50)
        status_layout = QVBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(2)

        if is_analog_input:
            last_value = getattr(device, "_last_value", None)
            if last_value is not None:
                voltage = (last_value / 1023.0) * 5.0
                state_text = f"{last_value}\n{voltage:.2f}V"
                state_color = colors.ACCENT
            else:
                state_text = "---"
                state_color = colors.BORDER
        else:
            state = getattr(device, "_state", None)
            if state is not None:
                if isinstance(state, bool):
                    state_text = "HIGH" if state else "LOW"
                    state_color = colors.SUCCESS if state else colors.TEXT_MUTED
                else:
                    state_text = str(state)[:6]
                    state_color = colors.ACCENT
            else:
                state_text = "---"
                state_color = colors.BORDER

        state_label = QLabel(state_text)
        state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_label.setStyleSheet(f"""
            QLabel {{
                background-color: {state_color};
                color: {colors.TEXT_PRIMARY};
                font-size: {"11px" if is_analog_input else "14px"};
                font-weight: bold;
                border-radius: 8px;
                padding: 4px 8px;
                border: none;
                line-height: 1.2;
            }}
        """)
        status_layout.addWidget(state_label)

        ready_label = QLabel("Ready" if initialized else "---")
        ready_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ready_label.setStyleSheet(
            f"font-size: 10px; color: {colors.SUCCESS if initialized else colors.TEXT_DISABLED}; background: transparent; border: none;"
        )
        status_layout.addWidget(ready_label)

        layout.addWidget(status_widget)

        card._state_label = state_label
        card._ready_label = ready_label

        return card

    def _show_runner_menu(self) -> None:
        """Show the runner mode menu."""
        menu = QMenu(self)

        desktop_action = menu.addAction("Hardware Config")
        desktop_action.triggered.connect(self.switch_to_desktop_requested.emit)

        help_action = menu.addAction("Help")
        help_action.triggered.connect(self.help_requested.emit)

        menu.addSeparator()

        exit_action = menu.addAction("\u2715  Exit")
        exit_action.triggered.connect(self.close_requested.emit)

        menu.exec(self._runner_menu_btn.mapToGlobal(self._runner_menu_btn.rect().bottomLeft()))

    # --- Cleanup ---

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """
        Stop all QTimers before the widget is destroyed.

        Timers started in __init__ (particularly the 50ms stall-instrument
        timer that runs unconditionally) keep firing against a dead widget
        if we don't explicitly stop them — polluting logs and blocking
        garbage collection of this panel.
        """
        for attr in ("_stall_timer", "_elapsed_timer", "_device_refresh_timer", "_readiness_timer"):
            timer = getattr(self, attr, None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    # Qt objects may already be partially torn down by the time
                    # closeEvent fires; swallow to guarantee the other timers
                    # still get stopped.
                    pass
        super().closeEvent(event)
