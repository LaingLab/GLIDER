"""
Run Control Panel - the dashboard's bottom-left quadrant.

Experiment name and metadata (bound to ``session.metadata``), file actions,
the board/experiment status line, the housekeeping menu, and START/STOP.

It also owns the run clock: it starts/stops the elapsed timer on state changes
and snaps it to ``core.last_flow_duration_s`` at the end of a flow. It does not
display the time itself — it emits ``elapsed_updated`` and the dashboard header
(top-right) shows it, so there is exactly one place the time text comes from.

Emergency stop is deliberately not offered here; it remains a desktop-only
menu action (see MainWindow._on_emergency_stop).
"""

import logging
import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from glider.core.config import get_config
from glider.gui.runner.readiness import compute_readiness
from glider.gui.runner.run_timer import format_elapsed

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)

_LIVE_STATES = ("RUNNING", "PAUSED")
# Single-line metadata fields edited in place: (attribute, label).
_LINE_FIELDS = (("experimenter", "Experimenter"), ("protocol", "Protocol"))


def _caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("textRole", "muted")
    return label


class RunControlPanel(QWidget):
    """Dashboard panel: experiment metadata, file actions, START/STOP."""

    experiment_name_changed = pyqtSignal(str)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    elapsed_updated = pyqtSignal(str)

    # File actions and housekeeping (wired by MainWindow).
    new_requested = pyqtSignal()
    open_requested = pyqtSignal()
    save_requested = pyqtSignal()
    save_as_requested = pyqtSignal()
    board_settings_requested = pyqtSignal()
    help_requested = pyqtSignal()
    switch_to_desktop_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, core: "GliderCore", parent=None):
        super().__init__(parent)
        self._core = core

        self._experiment_start_time: float | None = None
        self._state_name = "IDLE"
        self._last_readiness = None

        self.setObjectName("runControlPanel")
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        """Build the run-control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Everything above START/STOP scrolls, so the run buttons never get
        # squeezed off a short quadrant.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)

        # === Status line + housekeeping ⚙ ===
        status_row = QHBoxLayout()
        self._board_status = QLabel()
        self._exp_status = QLabel()
        status_row.addWidget(self._board_status)
        status_row.addWidget(self._exp_status)
        status_row.addStretch(1)
        self._menu_btn = QPushButton("⚙")
        self._menu_btn.setFixedSize(48, 48)
        self._menu_btn.setStyleSheet(
            "min-width:48px; max-width:48px; min-height:48px; max-height:48px; "
            "padding:0px; border:none; font-size: 20px;"
        )
        self._menu_btn.clicked.connect(self._open_housekeeping_menu)
        status_row.addWidget(self._menu_btn)
        body.addLayout(status_row)

        # === File actions (above the metadata, so they never scroll away) ===
        file_row = QHBoxLayout()
        file_row.setSpacing(6)
        self._file_buttons: dict[str, QPushButton] = {}
        for label, signal in (
            ("New", self.new_requested),
            ("Open", self.open_requested),
            ("Save", self.save_requested),
            ("Save As", self.save_as_requested),
            ("Connect / Ports", self.board_settings_requested),
        ):
            btn = QPushButton(label)
            btn.setMinimumHeight(48)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(signal)
            self._file_buttons[label] = btn
            file_row.addWidget(btn)
        body.addLayout(file_row)

        # === Experiment name + metadata ===
        self._name_edit = QLineEdit()
        self._name_edit.setProperty("title", True)
        self._name_edit.setPlaceholderText("Enter experiment name...")
        self._name_edit.setMinimumHeight(40)
        self._name_edit.textChanged.connect(self._on_name_edited)
        body.addWidget(_caption("Experiment"))
        body.addWidget(self._name_edit)

        meta = QGridLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        meta.setHorizontalSpacing(8)
        self._meta_edits: dict[str, QLineEdit] = {}
        for col, (attr, label) in enumerate(_LINE_FIELDS):
            edit = QLineEdit()
            edit.setPlaceholderText(label)
            edit.setMinimumHeight(40)
            edit.textChanged.connect(lambda text, a=attr: self._set_metadata(a, text))
            self._meta_edits[attr] = edit
            # Captions, not just placeholders: a filled field hides its
            # placeholder, and "G. Bradham" / "OF-v2" do not say what they are.
            meta.addWidget(_caption(label), 0, col)
            meta.addWidget(edit, 1, col)
        self._notes_edit = QPlainTextEdit()
        self._notes_edit.setPlaceholderText("Notes")
        self._notes_edit.setFixedHeight(64)
        self._notes_edit.textChanged.connect(
            lambda: self._set_metadata("notes", self._notes_edit.toPlainText())
        )
        meta.addWidget(_caption("Notes"), 2, 0, 1, 2)
        meta.addWidget(self._notes_edit, 3, 0, 1, 2)
        body.addLayout(meta)

        body.addStretch(1)

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # === START / STOP ===
        controls = QWidget()
        controls.setProperty("runnerControls", True)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(6)

        self._not_ready_hint = QLabel("Not ready — connect a board and load an experiment")
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
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        top_row.addWidget(self._stop_btn)

        controls_layout.addLayout(top_row)
        layout.addWidget(controls)

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
        """Reload the metadata fields from the (possibly new) session."""
        metadata = self._metadata()

        def value(attr: str) -> str:
            return str(getattr(metadata, attr, "") or "")

        edits = (self._name_edit, self._notes_edit, *self._meta_edits.values())
        for edit in edits:
            edit.blockSignals(True)
        self._name_edit.setText(value("name"))
        for attr, edit in self._meta_edits.items():
            edit.setText(value(attr))
        self._notes_edit.setPlainText(value("notes"))
        for edit in edits:
            edit.blockSignals(False)
        self._apply_lock()
        self._last_readiness = None
        self._refresh_run_readiness()

    def update_state(self, state_name: str) -> None:
        """Update UI based on core state changes."""
        self._state_name = state_name
        self._apply_lock()
        self._refresh_run_readiness()

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

    # --- Internal methods ---

    def _metadata(self):
        session = self._core.session
        return getattr(session, "metadata", None) if session is not None else None

    def _apply_lock(self) -> None:
        """Metadata is read-only while a run is live (it is being recorded)."""
        live = self.is_live
        self._name_edit.setReadOnly(live)
        self._notes_edit.setReadOnly(live)
        for edit in self._meta_edits.values():
            edit.setReadOnly(live)

    def _set_metadata(self, attr: str, value: str) -> None:
        metadata = self._metadata()
        if metadata is None or self.is_live:
            return
        setattr(metadata, attr, value)
        self._core.session.mark_dirty()

    def _on_name_edited(self, name: str) -> None:
        self._set_metadata("name", name)
        self.experiment_name_changed.emit(name)

    def _refresh_run_readiness(self) -> None:
        """Recompute readiness; update the status line and START/STOP."""
        r = compute_readiness(self._core)
        live = self.is_live
        key = (r, live)
        if key == self._last_readiness:
            return
        self._last_readiness = key
        if r.board_ready:
            self._board_status.setText(f"Board: ✓ {r.board_label}")
        else:
            self._board_status.setText("Board: ✗ not connected")
        if r.experiment_ready:
            self._exp_status.setText("Experiment: ✓ ready")
        else:
            self._exp_status.setText("Experiment: ✗ no flow")
        # START only when ready and not already running; STOP only mid-run.
        self._start_btn.setEnabled(r.all_ready and not live)
        self._stop_btn.setEnabled(live)
        self._not_ready_hint.setVisible(not r.all_ready and not live)

    def _open_housekeeping_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Help").triggered.connect(self.help_requested)
        menu.addAction("Switch to Desktop").triggered.connect(self.switch_to_desktop_requested)
        menu.addAction("Exit").triggered.connect(self.close_requested)
        menu.exec(self._menu_btn.mapToGlobal(self._menu_btn.rect().center()))

    def _update_elapsed_time(self) -> None:
        """Emit the live elapsed time (see ``format_elapsed`` for the format)."""
        if self._experiment_start_time is None:
            return
        self._set_timer_display(time.time() - self._experiment_start_time)

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
        self._set_timer_display(duration)

    def _set_timer_display(self, elapsed: float) -> None:
        """Format ``elapsed`` (seconds) and publish it to the dashboard header."""
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
