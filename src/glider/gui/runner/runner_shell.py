# src/glider/gui/runner/runner_shell.py
"""Four-tab Runner-mode shell: Setup / Run / Manual / Camera behind a bottom tab bar."""

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

_TAB_LABELS = ("Setup", "Run", "Manual", "Camera")


class RunnerShell(QWidget):
    """Holds the four Runner-mode pages, switched by a persistent bottom tab bar.

    Also owns the persistent run banner (timer/state/REC/STOP). The banner is
    shown whenever the experiment is live (RUNNING/PAUSED) AND the operator is
    not on the Run tab, so the timer/STOP follow them onto Setup/Manual/Camera
    mid-run without duplicating the Run tab's own controls.
    """

    stop_requested = pyqtSignal()

    def __init__(
        self,
        core,
        setup_page: QWidget,
        run_page: QWidget,
        manual_page: QWidget,
        camera_page: QWidget,
        parent=None,
    ):
        super().__init__(parent)
        self._core = core
        self._state_name = "IDLE"
        self._setup = setup_page
        self._run = run_page
        self._manual = manual_page
        self._camera = camera_page
        self._setup_ui()
        self.select_tab(0)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._banner = RunBanner()
        self._banner.hide()
        self._banner.stop_requested.connect(self.stop_requested)
        layout.addWidget(self._banner)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._setup)  # index 0
        self._stack.addWidget(self._run)  # index 1
        self._stack.addWidget(self._manual)  # index 2
        self._stack.addWidget(self._camera)  # index 3
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

        self._tab_btns: list[QPushButton] = []
        for i, label in enumerate(_TAB_LABELS):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked=False, idx=i: self.select_tab(idx))
            tabs_layout.addWidget(btn, 1)  # equal-width tabs
            self._tab_btns.append(btn)

        layout.addWidget(tabs)

    def select_tab(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        for j, btn in enumerate(self._tab_btns):
            btn.setChecked(j == index)
        # Re-sync the Manual page's enable state each time it is shown, so buttons
        # reflect the current hardware-connected status (the user may have
        # connected a board on the Setup/Ports flow since last viewing it).
        if index == 2 and hasattr(self._manual, "refresh"):
            self._manual.refresh()
        self._update_banner()

    def update_state(self, state_name: str) -> None:
        """Forward state changes to any page that has an update_state method."""
        self._state_name = state_name
        for page in (self._setup, self._run, self._manual, self._camera):
            if hasattr(page, "update_state"):
                page.update_state(state_name)
        self._update_banner()

    def _update_banner(self) -> None:
        # PAUSED is a live state (main_window auto-pauses on mid-run board
        # disconnect); the banner carries the timer/STOP off the Run tab, so it
        # must stay up through a pause too.
        live = self._state_name in ("RUNNING", "PAUSED")
        on_run = self._stack.currentIndex() == 1
        visible = live and not on_run
        self._banner.setVisible(visible)
        if visible:
            self._banner.set_state(
                self._state_name, recording=bool(self._core.data_recorder.is_recording)
            )

    def set_banner_time(self, text: str) -> None:
        """Forward a pre-formatted elapsed-time string to the run banner."""
        self._banner.set_time(text)
