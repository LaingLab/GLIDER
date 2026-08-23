"""
Device States Panel - Dashboard quadrant panel for live device status cards.

Provides a read-only, scrollable view of connected devices and their current
state. It refreshes on a poll timer while the experiment is RUNNING, and once
on any other state change. Device control and readiness computation are not
part of this panel — see RunControlPanel for run control.
"""

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from glider.core.config import get_config
from glider.gui.device_status import link_status_color, link_status_text
from glider.gui.styles import colors

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)


class DeviceStatesPanel(QWidget):
    """Dashboard panel showing read-only, live device status cards."""

    def __init__(self, core: "GliderCore", parent=None):
        super().__init__(parent)
        self._core = core

        # Store device widgets for updates
        self._runner_device_cards: dict[str, QWidget] = {}
        self._state_name = "IDLE"

        self.setObjectName("deviceStatesPanel")
        self._setup_ui()

    def _setup_ui(self):
        """Build the device states panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # === Device Status Area (Scrollable) ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background-color: transparent;")

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

        # Timers
        config = get_config()
        self._device_refresh_timer = QTimer()
        self._device_refresh_timer.setInterval(config.timing.device_refresh_interval_ms)
        self._device_refresh_timer.timeout.connect(self._update_device_states)

    # --- Public API ---

    def refresh_devices(self) -> None:
        """Refresh the device cards in this panel."""
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
        self._state_name = state_name

        # Start/stop device refresh timer
        if state_name == "RUNNING":
            self._device_refresh_timer.start()
        else:
            self._device_refresh_timer.stop()
            self._update_device_states()

    # --- Internal methods ---

    def _update_device_states(self) -> None:
        """Update the device state displays in this panel."""
        for device_id, card in self._runner_device_cards.items():
            device = self._core.hardware_manager.get_device(device_id)
            if device is None:
                continue

            # link_state, not _initialized -- the same swap the hardware tree
            # made. "Has this been set up" is a question a peripheral that
            # walked out of range never stops answering yes to.
            link = getattr(device, "link_state", None)
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
                card._ready_label.setText(link_status_text(link))
                card._ready_label.setStyleSheet(
                    f"font-size: 10px; color: {link_status_color(link)}; background: transparent; border: none;"
                )

    def _create_device_card(self, device_id: str, device) -> QWidget:
        """Create a device status card for this panel."""
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

        link = getattr(device, "link_state", None)
        device_type = getattr(device, "device_type", "Unknown")

        is_analog_input = device_type == "AnalogInput"

        status_widget = QWidget()
        # 80 for both: the link words ("Disconnected", "Reconnecting…") are
        # longer than the "Ready"/"---" the 60 was sized for, and a status that
        # paints as "Disconnec" is not one anybody can act on.
        status_widget.setFixedSize(80, 50)
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

        ready_label = QLabel(link_status_text(link))
        ready_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ready_label.setStyleSheet(
            f"font-size: 10px; color: {link_status_color(link)}; background: transparent; border: none;"
        )
        status_layout.addWidget(ready_label)

        layout.addWidget(status_widget)

        card._state_label = state_label
        card._ready_label = ready_label

        return card

    # --- Cleanup ---

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """
        Stop the device refresh QTimer before the widget is destroyed.

        The timer started in __init__ keeps firing against a dead widget if
        we don't explicitly stop it — polluting logs and blocking garbage
        collection of this panel.
        """
        timer = getattr(self, "_device_refresh_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                # Qt objects may already be partially torn down by the time
                # closeEvent fires; swallow so cleanup never raises here.
                pass
        super().closeEvent(event)
