"""The Pi's Run page: header, big clock, device cards, camera slot, run controls.

Self-contained. MainWindow owns the session, the core and the camera; it feeds
this page ``update_state`` / ``refresh``, drops a camera widget into
``camera_slot`` and listens to the four forwarded run signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from glider.gui.dashboard.panels.run_control_panel import RunControlPanel
from glider.gui.dashboard.panels.run_header import RunHeader
from glider.gui.runner.device_grid import DeviceCardGrid

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore


class PiRunPage(QWidget):
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()

    def __init__(self, core: GliderCore, parent=None):
        super().__init__(parent)
        self.setObjectName("piRunPage")

        self.header = RunHeader(core, touch=True)
        self.clock = QLabel("00:00.00")
        self.clock.setObjectName("piRunClock")
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock.setProperty("live", False)
        self.devices = DeviceCardGrid(core)
        self.camera_slot = QWidget()
        QVBoxLayout(self.camera_slot).setContentsMargins(0, 0, 0, 0)
        # Like the cards, the video gives way on a short screen (it clips).
        self.camera_slot.setMinimumHeight(1)
        self.camera_slot.hide()
        self.control = RunControlPanel(core, touch=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.header)
        layout.addWidget(self.clock)
        layout.addWidget(self.devices, 1)
        layout.addWidget(self.camera_slot)
        layout.addWidget(self.control)

        self.control.elapsed_updated.connect(self.clock.setText)
        self.control.readiness_changed.connect(self.header.set_ready)
        # The control computed its first readiness before this connection.
        self.header.set_ready(self.control.is_ready())
        self.control.start_requested.connect(self.start_requested)
        self.control.stop_requested.connect(self.stop_requested)
        self.control.pause_requested.connect(self.pause_requested)
        self.control.resume_requested.connect(self.resume_requested)

    def update_state(self, state_name: str) -> None:
        self.header.update_state(state_name)
        self.control.update_state(state_name)
        live = state_name == "RUNNING"
        self.devices.set_live(live)
        self.clock.setProperty("live", live)
        self.clock.style().unpolish(self.clock)
        self.clock.style().polish(self.clock)

    def refresh(self) -> None:
        """The session changed: reread the title block, readiness and devices."""
        self.header.refresh()
        self.control.refresh()
        self.devices.refresh_devices()

    def set_camera_visible(self, visible: bool) -> None:
        self.camera_slot.setVisible(visible)
