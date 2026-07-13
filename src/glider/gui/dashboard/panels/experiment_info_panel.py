"""Dashboard panel wrapping the existing RunnerSetupPage (experiment name,
Open/New/Save, board status, readiness strip) and re-exposing its signals.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from glider.gui.runner.runner_setup_page import RunnerSetupPage


class ExperimentInfoPanel(QWidget):
    # RunnerSetupPage exposes exactly these eight (verified in
    # runner/runner_setup_page.py:41-48); re-expose all so the main window can
    # wire the dashboard the same way it wired the old Setup page.
    new_requested = pyqtSignal()
    open_requested = pyqtSignal()
    save_requested = pyqtSignal()
    save_as_requested = pyqtSignal()
    help_requested = pyqtSignal()
    close_requested = pyqtSignal()
    switch_to_desktop_requested = pyqtSignal()
    board_settings_requested = pyqtSignal()

    def __init__(self, core, hardware_widget: QWidget, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._page = RunnerSetupPage(core, hardware_widget=hardware_widget)
        layout.addWidget(self._page)
        # re-expose all eight signals 1:1
        for name in (
            "new_requested",
            "open_requested",
            "save_requested",
            "save_as_requested",
            "help_requested",
            "close_requested",
            "switch_to_desktop_requested",
            "board_settings_requested",
        ):
            getattr(self._page, name).connect(getattr(self, name))

    def refresh(self) -> None:
        if hasattr(self._page, "refresh"):
            self._page.refresh()

    def update_state(self, state_name: str) -> None:
        if hasattr(self._page, "update_state"):
            self._page.update_state(state_name)
