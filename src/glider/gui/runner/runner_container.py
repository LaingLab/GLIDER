# src/glider/gui/runner/runner_container.py
"""Wraps the Runner-mode pages (Dashboard + Manual) behind a bottom tab bar."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from glider.gui.runner.run_banner import RunBanner


class RunnerContainer(QWidget):
    """Holds the existing RunnerPanel (Dashboard) and the ManualControlPanel,
    switched by a persistent bottom tab bar.

    Also owns the persistent run banner (timer/state/REC/STOP), shown across
    both pages while the core is RUNNING so the operator never loses it when
    switching tabs.
    """

    stop_requested = pyqtSignal()

    def __init__(self, core, dashboard_page: QWidget, manual_page: QWidget, parent=None):
        super().__init__(parent)
        self._core = core
        self._dashboard = dashboard_page
        self._manual = manual_page
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._banner = RunBanner()
        self._banner.hide()
        self._banner.stop_requested.connect(self.stop_requested)
        layout.addWidget(self._banner)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._dashboard)  # index 0
        self._stack.addWidget(self._manual)  # index 1
        layout.addWidget(self._stack, 1)

        tabs = QWidget()
        tabs.setProperty("runnerTabBar", True)
        # Plain QWidget only paints a QSS background/border when styled-background
        # is enabled; without this the bar's top divider would not render.
        tabs.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        tabs.setFixedHeight(56)
        tabs_layout = QHBoxLayout(tabs)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.setSpacing(0)

        self._dashboard_btn = QPushButton("Dashboard")
        self._dashboard_btn.setCheckable(True)
        self._dashboard_btn.setChecked(True)
        self._dashboard_btn.clicked.connect(lambda: self._select(0))
        tabs_layout.addWidget(self._dashboard_btn, 1)  # equal-width tabs

        self._manual_btn = QPushButton("Manual")
        self._manual_btn.setCheckable(True)
        self._manual_btn.clicked.connect(lambda: self._select(1))
        tabs_layout.addWidget(self._manual_btn, 1)

        layout.addWidget(tabs)

    def _select(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        self._dashboard_btn.setChecked(index == 0)
        self._manual_btn.setChecked(index == 1)
        # Re-sync the Manual page's enable state each time it is shown, so buttons
        # reflect the current hardware-connected status (the user may have
        # connected a board on the Dashboard/Ports flow since last viewing it).
        if index == 1 and hasattr(self._manual, "refresh"):
            self._manual.refresh()

    def set_banner_time(self, text: str) -> None:
        """Forward a pre-formatted elapsed-time string to the run banner."""
        self._banner.set_time(text)

    def update_state(self, state_name: str) -> None:
        """Forward state changes to both pages (each ignores what it doesn't use)."""
        for page in (self._dashboard, self._manual):
            if hasattr(page, "update_state"):
                page.update_state(state_name)

        running = state_name == "RUNNING"
        self._banner.setVisible(running)
        if running:
            self._banner.set_state(
                state_name, recording=bool(self._core.data_recorder.is_recording)
            )
