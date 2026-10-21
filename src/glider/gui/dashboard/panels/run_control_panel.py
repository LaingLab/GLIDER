"""
Run Control Panel - the rail's "run box".

The run clock, a board sub-line, Start / Pause-Resume / Stop, and a readiness
hint. Experiment metadata lives in RunHeader and the file actions live in the
menus.

It owns and shows the run clock: it starts/stops the elapsed timer on state
changes and snaps it to ``core.last_flow_duration_s`` at the end of a flow.
``elapsed_updated`` is still emitted for anything else that wants the text.

Emergency stop is deliberately not offered here; it remains a desktop-only
menu action (see MainWindow._on_emergency_stop).
"""

import logging
import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from glider.core.config import get_config
from glider.gui.runner.readiness import compute_readiness
from glider.gui.runner.run_timer import format_elapsed

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)

_LIVE_STATES = ("RUNNING", "PAUSED")


class RunControlPanel(QWidget):
    """Rail run box: clock, Start/Pause/Resume/Stop, readiness hint."""

    start_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    elapsed_updated = pyqtSignal(str)
    # Board + runnable flow (not the run fields): feeds the header's
    # READY / NOT READY pill so it never contradicts a disabled Start.
    readiness_changed = pyqtSignal(bool)

    def __init__(self, core: "GliderCore", parent=None, *, touch: bool = False):
        """``touch``: the Pi's run box. The page shows the clock itself, so this
        one hides its own label (``elapsed_updated`` still fires), and the
        hints name the ⚙ menu rather than the desktop's Hardware menu."""
        super().__init__(parent)
        self._core = core
        self._touch = touch

        self._experiment_start_time: float | None = None
        self._state_name = "IDLE"
        self._last_readiness = None
        self._missing_field: str | None = None
        self._was_ready: bool | None = None
        self._paused_at = 0.0
        self._paused_total = 0.0  # seconds paused so far in this run

        self.setObjectName("runControlPanel")
        if touch:
            # touch.qss gives the Pi's run bar a card; a QWidget subclass paints
            # a stylesheet background only with this set.
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        """Build the run box."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self._timer = QLabel("00:00.00")
        self._timer.setProperty("timer", True)
        if self._touch:
            self._timer.hide()
        layout.addWidget(self._timer)
        self.elapsed_updated.connect(self._timer.setText)

        self._sub = QLabel()
        self._sub.setProperty("textRole", "muted")
        if self._touch:
            # A long board description wraps rather than widening the 480 px page.
            self._sub.setWordWrap(True)
        layout.addWidget(self._sub)

        self._start_btn = QPushButton("▶  Start run")
        self._start_btn.setFixedHeight(54)
        self._start_btn.setProperty("runnerAction", "start")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self.start_requested.emit)
        layout.addWidget(self._start_btn)

        live_row = QHBoxLayout()
        live_row.setSpacing(8)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedHeight(54)
        self._pause_btn.setProperty("runnerAction", "pause")
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        live_row.addWidget(self._pause_btn)
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedHeight(54)
        self._stop_btn.setProperty("runnerAction", "stop")
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        live_row.addWidget(self._stop_btn)
        layout.addLayout(live_row)

        self._not_ready_hint = QLabel("Not ready — connect a board and load an experiment")
        self._not_ready_hint.setProperty("textRole", "warning")
        self._not_ready_hint.setWordWrap(True)
        self._not_ready_hint.hide()
        layout.addWidget(self._not_ready_hint)

        # Timers
        config = get_config()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(config.timing.elapsed_timer_interval_ms)
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)

        self._readiness_timer = QTimer(self)
        self._readiness_timer.setInterval(500)
        self._readiness_timer.timeout.connect(self._refresh_run_readiness)
        self._readiness_timer.start()

    # --- Public API ---

    @property
    def is_live(self) -> bool:
        return self._state_name in _LIVE_STATES

    def refresh(self) -> None:
        """Re-run readiness (the session may have changed)."""
        self._last_readiness = None
        self._refresh_run_readiness()

    def is_ready(self) -> bool:
        """Board and runnable flow in place (the last value readiness_changed sent)."""
        return bool(self._was_ready)

    def set_missing_field(self, label: str | None) -> None:
        """Block Start (with a hint) until the named required field is filled."""
        self._missing_field = label
        self._last_readiness = None
        self._refresh_run_readiness()

    def update_state(self, state_name: str) -> None:
        """Update UI based on core state changes."""
        previous = self._state_name
        self._state_name = state_name
        self._refresh_run_readiness()

        # Start/stop elapsed timer. Paused stretches are not run time.
        now = time.time()
        if previous == "PAUSED" and state_name != "PAUSED":
            self._paused_total += now - self._paused_at
        if state_name == "RUNNING":
            if previous != "PAUSED" or self._experiment_start_time is None:
                self._experiment_start_time = now
                self._paused_total = 0.0
            self._elapsed_timer.start()
            self._update_elapsed_time()
        elif state_name == "PAUSED":
            # Freeze the display on its current value; no flow-duration snap.
            if previous != "PAUSED":
                self._paused_at = now
            self._elapsed_timer.stop()
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

    # --- Internal methods ---

    def _on_pause_clicked(self) -> None:
        if self._state_name == "PAUSED":
            self.resume_requested.emit()
        else:
            self.pause_requested.emit()

    def _refresh_run_readiness(self) -> None:
        """Recompute readiness; update the sub-line, buttons and hint."""
        r = compute_readiness(self._core)
        live = self.is_live
        missing = self._missing_field or r.missing_field
        key = (r, live, self._state_name, missing)
        if key == self._last_readiness:
            return
        self._last_readiness = key
        if r.all_ready != self._was_ready:
            self._was_ready = r.all_ready
            self.readiness_changed.emit(r.all_ready)
        self._sub.setText(str(r.board_label) if r.board_ready else "No board connected")
        self._start_btn.setVisible(not live)
        self._pause_btn.setVisible(live)
        self._stop_btn.setVisible(live)
        self._pause_btn.setText("Resume" if self._state_name == "PAUSED" else "Pause")
        ready = r.all_ready and not missing
        self._start_btn.setEnabled(ready and not live)
        self._stop_btn.setEnabled(live)
        if missing:
            self._not_ready_hint.setText(f"Fill in {missing} to start.")
        elif not r.board_ready:
            self._not_ready_hint.setText(
                "Connect a board to start (⚙ › Connect boards…)."
                if self._touch
                else "Connect the board to start (Hardware > Connect All)."
            )
        else:
            self._not_ready_hint.setText("Add a Start Experiment node to the flow to start.")
        self._not_ready_hint.setVisible(not ready and not live)

    def _update_elapsed_time(self) -> None:
        """Emit the live elapsed time (see ``format_elapsed`` for the format)."""
        if self._experiment_start_time is None:
            return
        self._set_timer_display(time.time() - self._experiment_start_time - self._paused_total)

    def _snap_timer_to_flow_duration(self) -> None:
        """On flow end, freeze the timer on the flow's logical duration.

        The QTimer's last tick was up to one interval before the state change,
        and the state change fired *after* teardown. ``core.last_flow_duration_s``
        is anchored to flow-engine start/end and is the truth-of-record.
        """
        duration = self._core.last_flow_duration_s
        if duration is None:
            # No flow ran (or in progress) — leave the last live tick as
            # the display. Happens on cleanup paths that don't correspond
            # to a flow termination.
            return
        # The flow's duration is wall time; take this run's pauses back out.
        self._set_timer_display(max(0.0, duration - self._paused_total))

    def _set_timer_display(self, elapsed: float) -> None:
        """Format ``elapsed`` (seconds) and publish it (also shown on the clock label)."""
        self.elapsed_updated.emit(format_elapsed(elapsed))

    # --- Cleanup ---

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Stop the QTimers so they don't fire against a dying widget."""
        for timer in (self._elapsed_timer, self._readiness_timer):
            try:
                timer.stop()
            except Exception:
                # Qt objects may already be partially torn down.
                pass
        super().closeEvent(event)
