"""
Runner Setup Page - single-tab landing view for the Runner shell.

Shows a readiness status line, experiment file actions, an embedded hardware
panel, and a housekeeping menu. This widget is PURE UI: it emits signals and
hosts a hardware widget passed in by the caller; it does not touch file
dialogs or hardware itself. main_window wires the signals up separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from glider.gui.runner.readiness import compute_readiness

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

# How often (ms) to poll readiness so the status line tracks board connect/disconnect.
_POLL_MS = 500


class RunnerSetupPage(QWidget):
    """Scrollable Setup landing page: status, file actions, hardware, housekeeping."""

    new_requested = pyqtSignal()
    open_requested = pyqtSignal()
    save_requested = pyqtSignal()
    save_as_requested = pyqtSignal()
    help_requested = pyqtSignal()
    close_requested = pyqtSignal()
    switch_to_desktop_requested = pyqtSignal()
    board_settings_requested = pyqtSignal()

    def __init__(self, core: GliderCore, hardware_widget: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self._core = core
        self.setObjectName("runnerSetupPage")

        self._setup_ui(hardware_widget)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_MS)
        self._poll_timer.timeout.connect(self.refresh)
        self._poll_timer.start()

        self.refresh()

    # --- UI scaffolding ---

    def _setup_ui(self, hardware_widget: QWidget) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        QScroller.grabGesture(
            scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(12)

        # 1. Status line.
        status_row = QHBoxLayout()
        self._board_status = QLabel()
        self._exp_status = QLabel()
        status_row.addWidget(self._board_status)
        status_row.addWidget(self._exp_status)
        status_row.addStretch(1)
        content_layout.addLayout(status_row)

        # 2. Experiment section.
        self._exp_name = QLabel()
        content_layout.addWidget(self._exp_name)

        file_row = QHBoxLayout()
        self._new_btn = QPushButton("New")
        self._open_btn = QPushButton("Open")
        self._save_btn = QPushButton("Save")
        self._save_as_btn = QPushButton("Save As")
        self._connect_btn = QPushButton("Connect / Ports")
        for btn in (
            self._new_btn,
            self._open_btn,
            self._save_btn,
            self._save_as_btn,
            self._connect_btn,
        ):
            btn.setMinimumHeight(48)
            file_row.addWidget(btn)
        content_layout.addLayout(file_row)

        self._new_btn.clicked.connect(self.new_requested)
        self._open_btn.clicked.connect(self.open_requested)
        self._save_btn.clicked.connect(self.save_requested)
        self._save_as_btn.clicked.connect(self.save_as_requested)
        self._connect_btn.clicked.connect(self.board_settings_requested)

        # 3. Hardware section.
        content_layout.addWidget(hardware_widget)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # 4. Housekeeping.
        self._menu_btn = QPushButton("⚙")
        self._menu_btn.clicked.connect(self._open_housekeeping_menu)
        outer.addWidget(self._menu_btn, 0)

    # --- Public API ---

    def refresh(self) -> None:
        r = compute_readiness(self._core)
        if r.board_ready:
            self._board_status.setText(f"Board: ✓ {r.board_label}")
        else:
            self._board_status.setText("Board: ✗ not connected")
        if r.experiment_ready:
            self._exp_status.setText(f"Experiment: ✓ {r.experiment_label}")
        else:
            self._exp_status.setText("Experiment: ✗ none loaded")
        self._exp_name.setText(r.experiment_label or "Untitled")

    # --- Housekeeping menu ---

    def _open_housekeeping_menu(self) -> None:
        menu = QMenu(self)
        help_action = menu.addAction("Help")
        help_action.triggered.connect(self.help_requested)
        desktop_action = menu.addAction("Switch to Desktop")
        desktop_action.triggered.connect(self.switch_to_desktop_requested)
        exit_action = menu.addAction("Exit")
        exit_action.triggered.connect(self.close_requested)
        menu.exec(self._menu_btn.mapToGlobal(self._menu_btn.rect().center()))

    # --- Lifecycle ---

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        super().closeEvent(event)
