"""
Run Control Panel - Dashboard quadrant panel for experiment run control.

Provides the run-control header (experiment name, elapsed timer, status pill),
the recording indicator, and the START/STOP controls. Readiness is computed
here to gate START, but the readiness strip (tap-to-fix rows) and gear/setup
menu live on the Setup page. Emergency stop is deliberately not offered here;
it remains a desktop-only menu action (see MainWindow._on_emergency_stop).

Device status cards are a separate concern (DeviceStatesPanel).
"""

import logging
import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from glider.core.config import get_config
from glider.gui.runner.readiness import compute_readiness
from glider.gui.runner.run_timer import format_elapsed
from glider.gui.styles import colors

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)


class RunControlPanel(QWidget):
    """Dashboard panel providing run-control header and START/STOP."""

    experiment_name_changed = pyqtSignal(str)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    elapsed_updated = pyqtSignal(str)

    def __init__(self, core: "GliderCore", parent=None):
        super().__init__(parent)
        self._core = core

        self._experiment_start_time: float | None = None
        self._state_name = "IDLE"

        self.setObjectName("runControlPanel")
        self._setup_ui()

    def _setup_ui(self):
        """Build the run-control panel UI."""
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
        self._runner_exp_name.setReadOnly(True)
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

        layout.addWidget(header)

        # === Recording Indicator ===
        self._runner_recording = QLabel("● REC")
        self._runner_recording.setProperty("recording", True)
        self._runner_recording.setFixedHeight(28)
        self._runner_recording.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._runner_recording.hide()
        layout.addWidget(self._runner_recording)

        layout.addStretch(1)

        # === Control Buttons ===
        controls = QWidget()
        controls.setMinimumHeight(110)
        controls.setProperty("runnerControls", True)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(8)

        self._not_ready_hint = QLabel("Not ready — check Setup")
        self._not_ready_hint.setProperty("textRole", "muted")
        self._not_ready_hint.hide()
        controls_layout.addWidget(self._not_ready_hint)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self._start_btn = QPushButton("▶  START")
        self._start_btn.setFixedHeight(60)
        self._start_btn.setProperty("runnerAction", "start")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self.start_requested.emit)
        top_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  STOP")
        self._stop_btn.setFixedHeight(60)
        self._stop_btn.setProperty("runnerAction", "stop")
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        top_row.addWidget(self._stop_btn)

        controls_layout.addLayout(top_row)

        layout.addWidget(controls)

        # Timers
        config = get_config()
        self._elapsed_timer = QTimer()
        self._elapsed_timer.setInterval(config.timing.elapsed_timer_interval_ms)
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)

        self._readiness_timer = QTimer(self)
        self._readiness_timer.setInterval(500)
        self._readiness_timer.timeout.connect(self._refresh_run_readiness)
        self._readiness_timer.start()

        self._refresh_run_readiness()

    # --- Public API ---

    def _refresh_run_readiness(self) -> None:
        """Recompute board/experiment readiness and update the START button + hint."""
        r = compute_readiness(self._core)
        if r == getattr(self, "_last_readiness", None):
            return
        self._last_readiness = r
        self._start_btn.setEnabled(r.all_ready)
        self._not_ready_hint.setVisible(not r.all_ready)

    def update_state(self, state_name: str) -> None:
        """Update UI based on core state changes."""
        self._state_name = state_name

        # Update status label
        self._status_label.setText(state_name)
        self._status_label.setProperty("statusState", state_name)
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

        self._refresh_run_readiness()

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

    # --- Cleanup ---

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """
        Stop all QTimers before the widget is destroyed.

        Timers started in __init__ keep firing against a dead widget if we
        don't explicitly stop them — polluting logs and blocking garbage
        collection of this panel.
        """
        for attr in ("_elapsed_timer", "_readiness_timer"):
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
