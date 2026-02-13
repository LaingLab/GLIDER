"""
Device Card Widget for GLIDER.

Provides a reusable widget for displaying device state in the runner view.
Reduces code duplication between creation and update methods.
"""

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from glider.core.config import get_config
from glider.core.types import DeviceType


@dataclass
class DeviceStateInfo:
    """Formatted device state information for display."""

    state_text: str
    state_color: str
    font_size: str
    is_ready: bool


def get_device_state_info(device) -> DeviceStateInfo:
    """
    Get formatted device state information.

    Args:
        device: The device to get state info for

    Returns:
        DeviceStateInfo with formatted display values
    """
    config = get_config()
    initialized = getattr(device, "_initialized", False)
    device_type_str = getattr(device, "device_type", "Unknown")

    # Check if this is an analog input device
    device_type = DeviceType.from_string_safe(device_type_str)
    is_analog_input = device_type == DeviceType.ANALOG_INPUT

    is_pwm_output = device_type == DeviceType.PWM_OUTPUT

    if is_analog_input:
        last_value = getattr(device, "_last_value", None)
        if last_value is not None:
            voltage = (
                last_value / config.hardware.adc_resolution
            ) * config.hardware.adc_reference_voltage
            state_text = f"{last_value}\n{voltage:.2f}V"
            state_color = "#3498db"
        else:
            state_text = "---"
            state_color = "#444"
        font_size = "11px"
    elif is_pwm_output:
        state = getattr(device, "_state", None)
        if state is not None:
            pwm_val = int(state) if not isinstance(state, bool) else (255 if state else 0)
            state_text = str(pwm_val)
            state_color = "#3498db" if pwm_val > 0 else "#7f8c8d"
        else:
            state_text = "---"
            state_color = "#444"
        font_size = "14px"
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

    return DeviceStateInfo(
        state_text=state_text, state_color=state_color, font_size=font_size, is_ready=initialized
    )


def create_state_label_style(state_color: str, font_size: str) -> str:
    """
    Create stylesheet for state label.

    Args:
        state_color: Background color for the label
        font_size: Font size for the label

    Returns:
        CSS stylesheet string
    """
    return f"""
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
    """


def create_ready_label_style(is_ready: bool) -> str:
    """
    Create stylesheet for ready indicator label.

    Args:
        is_ready: Whether the device is ready

    Returns:
        CSS stylesheet string
    """
    color = "#27ae60" if is_ready else "#666"
    return f"font-size: 10px; color: {color}; background: transparent; border: none;"


class DeviceCard(QWidget):
    """
    Widget for displaying device status in the runner view.

    Displays device name, type, current state, and ready status.
    """

    CARD_HEIGHT = 80

    def __init__(self, device_id: str, device, parent: Optional[QWidget] = None):
        """
        Initialize the device card.

        Args:
            device_id: Unique identifier for the device
            device: Device instance to display
            parent: Parent widget
        """
        super().__init__(parent)
        self._device_id = device_id
        self._device = device

        self._setup_ui()
        self.update_state()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a2e;
                border: 2px solid #2d2d44;
                border-radius: 12px;
            }
        """)
        self.setFixedHeight(self.CARD_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Device info (left side)
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        self._name_label = QLabel(self._device_id)
        self._name_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #fff; "
            "background: transparent; border: none;"
        )
        info_layout.addWidget(self._name_label)

        device_type = getattr(self._device, "device_type", "Unknown")
        self._type_label = QLabel(device_type)
        self._type_label.setStyleSheet(
            "font-size: 12px; color: #888; background: transparent; border: none;"
        )
        info_layout.addWidget(self._type_label)

        layout.addLayout(info_layout)
        layout.addStretch()

        # Status indicator (right side)
        device_type_str = getattr(self._device, "device_type", "Unknown")
        device_type = DeviceType.from_string_safe(device_type_str)
        is_analog_input = device_type == DeviceType.ANALOG_INPUT

        status_widget = QWidget()
        status_widget.setFixedSize(80 if is_analog_input else 60, 50)
        status_layout = QVBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(2)

        self._state_label = QLabel("---")
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._state_label)

        self._ready_label = QLabel("---")
        self._ready_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._ready_label)

        layout.addWidget(status_widget)

    def update_state(self) -> None:
        """Update the displayed state from the device."""
        state_info = get_device_state_info(self._device)

        self._state_label.setText(state_info.state_text)
        self._state_label.setStyleSheet(
            create_state_label_style(state_info.state_color, state_info.font_size)
        )

        self._ready_label.setText("Ready" if state_info.is_ready else "---")
        self._ready_label.setStyleSheet(create_ready_label_style(state_info.is_ready))

    @property
    def device_id(self) -> str:
        """Get the device ID."""
        return self._device_id

    def set_device(self, device) -> None:
        """
        Set a new device for this card.

        Args:
            device: New device instance
        """
        self._device = device
        self.update_state()
