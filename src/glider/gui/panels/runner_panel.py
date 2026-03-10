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

    def __init__(self, core: "GliderCore", view_manager: "ViewManager", parent=None):
        super().__init__(parent)
        self._core = core
        self._view_manager = view_manager

        # Store device widgets for updates
        self._runner_device_cards: dict[str, QWidget] = {}
        self._experiment_start_time: float | None = None

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
        self._runner_exp_name.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 16px;
                font-weight: bold;
                color: white;
                min-width: 200px;
            }
            QLineEdit:hover {
                border: 1px solid #3d3d5c;
                background-color: rgba(45, 45, 68, 0.5);
            }
            QLineEdit:focus {
                border: 1px solid #4CAF50;
                background-color: #2d2d44;
            }
        """)
        self._runner_exp_name.textChanged.connect(self._on_experiment_name_changed)
        header_layout.addWidget(self._runner_exp_name)

        header_layout.addStretch()

        self._runner_timer = QLabel("00:00")
        self._runner_timer.setProperty("timer", True)
        self._runner_timer.setStyleSheet("""
            QLabel[timer] {
                color: #4CAF50;
                font-size: 18px;
                font-weight: bold;
                font-family: "SF Mono", "Menlo", "Consolas", "Monaco", "Courier New";
                padding: 4px 8px;
                background-color: rgba(76, 175, 80, 0.1);
                border-radius: 4px;
            }
        """)
        header_layout.addWidget(self._runner_timer)

        self._status_label = QLabel("IDLE")
        self._status_label.setProperty("runnerStatus", True)
        self._status_label.setProperty("statusState", "IDLE")
        header_layout.addWidget(self._status_label)

        self._runner_menu_btn = QPushButton("\u2699\ufe0f")
        self._runner_menu_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d44;
                border: none;
                border-radius: 8px;
                font-size: 20px;
                color: white;
            }
            QPushButton:pressed {
                background-color: #3d3d5c;
            }
        """)
        self._runner_menu_btn.clicked.connect(self._show_runner_menu)
        header_layout.addWidget(self._runner_menu_btn)

        layout.addWidget(header)

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
        self._runner_no_devices.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14px;
                padding: 40px;
            }
        """)
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

    def update_state(self, state_name: str) -> None:
        """Update UI based on core state changes."""
        # Update status label
        self._status_label.setText(state_name)
        self._status_label.setProperty("statusState", state_name)
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

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
        """Update the elapsed time display."""
        if self._experiment_start_time is None:
            return

        elapsed = time.time() - self._experiment_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        if hours > 0:
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            time_str = f"{minutes:02d}:{seconds:02d}"

        self._runner_timer.setText(time_str)

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
                        state_color = "#3498db"
                        font_size = "11px"
                    else:
                        state_text = "---"
                        state_color = "#444"
                        font_size = "11px"
                else:
                    state = getattr(device, "_state", None)
                    if state is not None:
                        if isinstance(state, bool):
                            state_text = "HIGH" if state else "LOW"
                            state_color = "#27ae60" if state else "#7f8c8d"
                        else:
                            state_text = str(state)[:6]
                            state_color = "#3498db"
                    else:
                        state_text = "---"
                        state_color = "#444"
                    font_size = "14px"

                card._state_label.setText(state_text)
                card._state_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {state_color};
                        color: white;
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
                    f"font-size: 10px; color: {'#27ae60' if initialized else '#666'}; background: transparent; border: none;"
                )

    def _create_device_card(self, device_id: str, device) -> QWidget:
        """Create a device status card for the runner view."""
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                background-color: #1a1a2e;
                border: 2px solid #2d2d44;
                border-radius: 12px;
            }
        """)
        card.setFixedHeight(80)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        name_label = QLabel(device_id)
        name_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #fff; background: transparent; border: none;"
        )
        info_layout.addWidget(name_label)

        device_type = getattr(device, "device_type", "Unknown")
        type_label = QLabel(device_type)
        type_label.setStyleSheet(
            "font-size: 12px; color: #888; background: transparent; border: none;"
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
                state_color = "#3498db"
            else:
                state_text = "---"
                state_color = "#444"
        else:
            state = getattr(device, "_state", None)
            if state is not None:
                if isinstance(state, bool):
                    state_text = "HIGH" if state else "LOW"
                    state_color = "#27ae60" if state else "#7f8c8d"
                else:
                    state_text = str(state)[:6]
                    state_color = "#3498db"
            else:
                state_text = "---"
                state_color = "#444"

        state_label = QLabel(state_text)
        state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state_label.setStyleSheet(f"""
            QLabel {{
                background-color: {state_color};
                color: white;
                font-size: {'11px' if is_analog_input else '14px'};
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
            f"font-size: 10px; color: {'#27ae60' if initialized else '#666'}; background: transparent; border: none;"
        )
        status_layout.addWidget(ready_label)

        layout.addWidget(status_widget)

        card._state_label = state_label
        card._ready_label = ready_label

        return card

    def _show_runner_menu(self) -> None:
        """Show the runner mode menu."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a2e;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 12px 24px;
                font-size: 16px;
                color: white;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #3498db;
            }
        """)

        open_action = menu.addAction("Open Experiment")
        open_action.triggered.connect(self.open_requested.emit)

        reload_action = menu.addAction("Reload")
        reload_action.triggered.connect(self.reload_requested.emit)

        board_action = menu.addAction("Ports")
        board_action.triggered.connect(self.board_settings_requested.emit)

        desktop_action = menu.addAction("Hardware Config")
        desktop_action.triggered.connect(self.switch_to_desktop_requested.emit)

        help_action = menu.addAction("Help")
        help_action.triggered.connect(self.help_requested.emit)

        menu.addSeparator()

        exit_action = menu.addAction("\u2715  Exit")
        exit_action.triggered.connect(self.close_requested.emit)

        menu.exec(self._runner_menu_btn.mapToGlobal(self._runner_menu_btn.rect().bottomLeft()))
