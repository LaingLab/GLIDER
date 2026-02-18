"""
Device Control Controller for GLIDER.

This controller encapsulates the device control panel functionality,
extracted from MainWindow to improve modularity.
"""

import logging
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from glider.core.config import get_config
from glider.core.types import DeviceType

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)


class DeviceControlController(QWidget):
    """
    Controller for the device control panel.

    Provides interactive controls for testing and manually operating
    hardware devices.

    Signals:
        device_selected: Emitted when a device is selected (device_id)
        value_read: Emitted when an input value is read (device_id, value)
    """

    device_selected = pyqtSignal(str)
    value_read = pyqtSignal(str, object)

    def __init__(self, core: "GliderCore", run_async: Callable, parent: Optional[QWidget] = None):
        """
        Initialize the device control controller.

        Args:
            core: GliderCore instance
            run_async: Function to run async coroutines
            parent: Parent widget
        """
        super().__init__(parent)
        self._core = core
        self._run_async = run_async
        self._config = get_config()

        # Input polling state
        self._analog_callback_board = None
        self._analog_callback_pin = None
        self._analog_callback_func = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Device selector
        device_layout = QHBoxLayout()
        device_layout.setSpacing(6)
        device_label = QLabel("Device:")
        device_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._device_combo = QComboBox()
        self._device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._device_combo.currentTextChanged.connect(self._on_device_selected)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self._device_combo, 1)
        layout.addLayout(device_layout)

        # Output Controls group
        self._control_group = QGroupBox("Output Controls")
        control_group_layout = QVBoxLayout(self._control_group)
        control_group_layout.setContentsMargins(8, 12, 8, 8)
        control_group_layout.setSpacing(8)

        # Digital output controls (container widget for show/hide)
        self._digital_widget = QWidget()
        digital_layout = QHBoxLayout(self._digital_widget)
        digital_layout.setContentsMargins(0, 0, 0, 0)
        digital_layout.setSpacing(4)
        self._on_btn = QPushButton("ON")
        self._on_btn.setMinimumHeight(32)
        self._on_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._on_btn.clicked.connect(lambda: self._set_digital_output(True))
        self._off_btn = QPushButton("OFF")
        self._off_btn.setMinimumHeight(32)
        self._off_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._off_btn.clicked.connect(lambda: self._set_digital_output(False))
        self._toggle_btn = QPushButton("Toggle")
        self._toggle_btn.setMinimumHeight(32)
        self._toggle_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle_btn.clicked.connect(self._toggle_digital_output)
        digital_layout.addWidget(self._on_btn)
        digital_layout.addWidget(self._off_btn)
        digital_layout.addWidget(self._toggle_btn)
        control_group_layout.addWidget(self._digital_widget)

        # PWM control (container widget for show/hide)
        self._pwm_widget = QWidget()
        pwm_layout = QHBoxLayout(self._pwm_widget)
        pwm_layout.setContentsMargins(0, 0, 0, 0)
        pwm_layout.setSpacing(6)
        pwm_label = QLabel("PWM:")
        pwm_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._pwm_spinbox = QSpinBox()
        self._pwm_spinbox.setRange(self._config.hardware.pwm_min, self._config.hardware.pwm_max)
        self._pwm_spinbox.setMinimumHeight(28)
        self._pwm_spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._pwm_spinbox.valueChanged.connect(self._on_pwm_changed)
        pwm_layout.addWidget(pwm_label)
        pwm_layout.addWidget(self._pwm_spinbox, 1)
        control_group_layout.addWidget(self._pwm_widget)
        self._pwm_widget.hide()

        # Servo control (container widget for show/hide)
        self._servo_widget = QWidget()
        servo_layout = QHBoxLayout(self._servo_widget)
        servo_layout.setContentsMargins(0, 0, 0, 0)
        servo_layout.setSpacing(6)
        servo_label = QLabel("Servo:")
        servo_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._servo_spinbox = QSpinBox()
        self._servo_spinbox.setRange(
            self._config.hardware.servo_min_angle, self._config.hardware.servo_max_angle
        )
        self._servo_spinbox.setValue(self._config.hardware.servo_default_angle)
        self._servo_spinbox.setSuffix("°")
        self._servo_spinbox.setMinimumHeight(28)
        self._servo_spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._servo_spinbox.valueChanged.connect(self._on_servo_changed)
        servo_layout.addWidget(servo_label)
        servo_layout.addWidget(self._servo_spinbox, 1)
        control_group_layout.addWidget(self._servo_widget)
        self._servo_widget.hide()

        layout.addWidget(self._control_group)

        # Input Reading group
        self._input_group = QGroupBox("Input Reading")
        input_group_layout = QVBoxLayout(self._input_group)
        input_group_layout.setContentsMargins(8, 12, 8, 8)
        input_group_layout.setSpacing(8)

        # Value display
        self._input_value_label = QLabel("--")
        self._input_value_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding: 6px; "
            "background-color: #2d2d2d; border-radius: 4px; color: #00ff00;"
        )
        self._input_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._input_value_label.setMinimumHeight(36)
        input_group_layout.addWidget(self._input_value_label)

        # Read controls
        input_row = QHBoxLayout()
        input_row.setSpacing(6)
        self._read_btn = QPushButton("Read")
        self._read_btn.setMinimumHeight(32)
        self._read_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._read_btn.clicked.connect(self._read_input_once)
        input_row.addWidget(self._read_btn)

        self._continuous_checkbox = QCheckBox("Auto")
        self._continuous_checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._continuous_checkbox.stateChanged.connect(self._on_continuous_changed)
        input_row.addWidget(self._continuous_checkbox)

        self._poll_spinbox = QSpinBox()
        self._poll_spinbox.setRange(
            self._config.hardware.min_poll_interval_ms, self._config.hardware.max_poll_interval_ms
        )
        self._poll_spinbox.setValue(self._config.hardware.input_poll_interval_ms)
        self._poll_spinbox.setSuffix("ms")
        self._poll_spinbox.setMinimumWidth(75)
        self._poll_spinbox.setMinimumHeight(28)
        self._poll_spinbox.valueChanged.connect(self._on_poll_interval_changed)
        input_row.addWidget(self._poll_spinbox)

        input_group_layout.addLayout(input_row)
        layout.addWidget(self._input_group)

        # Input poll timer
        self._input_poll_timer = QTimer(self)
        self._input_poll_timer.timeout.connect(self._poll_input)

        # Status display
        self._status_label = QLabel("No device selected")
        self._status_label.setStyleSheet("font-size: 11px; color: #888; padding: 2px;")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        layout.addStretch()

    def refresh_devices(self) -> None:
        """Refresh the device selector combo box."""
        self._device_combo.clear()
        self._device_combo.addItem("-- Select Device --", None)

        for device_id, device in self._core.hardware_manager.devices.items():
            device_name = getattr(device, "name", device_id)
            device_type = getattr(device, "device_type", "unknown")
            self._device_combo.addItem(f"{device_name} ({device_type})", device_id)

    def _on_device_selected(self, text: str) -> None:
        """Handle device selection change."""
        # Stop any continuous input reading
        self._input_poll_timer.stop()
        self._continuous_checkbox.setChecked(False)
        self._input_value_label.setText("--")

        device_id = self._device_combo.currentData()
        if device_id is None:
            self._status_label.setText("Status: No device selected")
            self._input_group.setEnabled(False)
            return

        device = self._core.hardware_manager.get_device(device_id)
        if device is None:
            self._status_label.setText("Status: Device not found")
            self._input_group.setEnabled(False)
            return

        device_type_str = getattr(device, "device_type", "unknown")
        board = getattr(device, "board", None)
        connected = board.is_connected if board else False
        initialized = getattr(device, "_initialized", False)

        status = "Connected" if connected else "Disconnected"
        if connected and initialized:
            status = "Ready"
        elif connected and not initialized:
            status = "Not initialized"

        self._status_label.setText(f"Status: {status} | Type: {device_type_str}")

        # Enable/disable input reading based on device type
        device_type = DeviceType.from_string_safe(device_type_str)
        is_input = device_type is not None and device_type.is_input
        self._input_group.setEnabled(is_input or device_type_str == "ADS1115")

        # Show/hide appropriate output controls based on device type
        is_digital_output = device_type_str == "DigitalOutput"
        is_pwm_output = device_type_str == "PWMOutput"
        is_servo = device_type_str == "Servo"
        is_output = is_digital_output or is_pwm_output or is_servo

        self._control_group.setVisible(is_output)
        self._digital_widget.setVisible(is_digital_output)
        self._pwm_widget.setVisible(is_pwm_output)
        self._servo_widget.setVisible(is_servo)

        # Emit signal
        self.device_selected.emit(device_id)

    def _get_selected_device(self):
        """Get the currently selected device."""
        device_id = self._device_combo.currentData()
        if device_id is None:
            return None
        return self._core.hardware_manager.get_device(device_id)

    def _set_digital_output(self, value: bool) -> None:
        """Set digital output to HIGH or LOW."""
        device = self._get_selected_device()
        if device is None:
            QMessageBox.warning(self, "No Device", "Please select a device first.")
            return

        async def set_output():
            try:
                if hasattr(device, "set_state"):
                    await device.set_state(value)
                elif hasattr(device, "turn_on") and hasattr(device, "turn_off"):
                    if value:
                        await device.turn_on()
                    else:
                        await device.turn_off()
                else:
                    pin = list(device.pins.values())[0] if device.pins else 0
                    await device.board.write_digital(pin, value)

                state = "ON" if value else "OFF"
                self._status_label.setText(f"Status: Output set to {state}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to set output: {e}")

        self._run_async(set_output())

    def _toggle_digital_output(self) -> None:
        """Toggle digital output."""
        device = self._get_selected_device()
        if device is None:
            QMessageBox.warning(self, "No Device", "Please select a device first.")
            return

        async def toggle():
            try:
                if hasattr(device, "toggle"):
                    await device.toggle()
                elif hasattr(device, "state"):
                    new_state = not device.state
                    if hasattr(device, "set_state"):
                        await device.set_state(new_state)
                self._status_label.setText("Status: Output toggled")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to toggle: {e}")

        self._run_async(toggle())

    def _on_pwm_changed(self, value: int) -> None:
        """Handle PWM value change (debounced to avoid overwhelming serial)."""
        device = self._get_selected_device()
        if device is None:
            return

        self._pending_pwm_value = value
        self._pending_pwm_device = device

        if not hasattr(self, "_pwm_debounce_timer") or self._pwm_debounce_timer is None:
            self._pwm_debounce_timer = QTimer()
            self._pwm_debounce_timer.setSingleShot(True)
            self._pwm_debounce_timer.timeout.connect(self._send_pending_pwm)

        self._pwm_debounce_timer.start(50)

    def _send_pending_pwm(self) -> None:
        """Send the most recent pending PWM value."""
        device = self._pending_pwm_device
        value = self._pending_pwm_value

        async def set_pwm():
            try:
                if hasattr(device, "set_value"):
                    await device.set_value(value)
                elif hasattr(device, "board"):
                    pin = list(device.pins.values())[0] if device.pins else 0
                    await device.board.write_analog(pin, value)
                self._status_label.setText(f"Status: PWM set to {value}")
            except Exception as e:
                logger.error(f"PWM error: {e}")

        self._run_async(set_pwm())

    def _on_servo_changed(self, angle: int) -> None:
        """Handle servo angle change."""
        device = self._get_selected_device()
        if device is None:
            return

        async def set_servo():
            try:
                if hasattr(device, "set_angle"):
                    await device.set_angle(angle)
                elif hasattr(device, "board"):
                    pin = list(device.pins.values())[0] if device.pins else 0
                    await device.board.write_servo(pin, angle)
                self._status_label.setText(f"Status: Servo set to {angle}°")
            except Exception as e:
                logger.error(f"Servo error: {e}")

        self._run_async(set_servo())

    def _read_input_once(self) -> None:
        """Read the input value once."""
        device = self._get_selected_device()
        if device is None:
            QMessageBox.warning(self, "No Device", "Please select a device first.")
            return

        device_type = getattr(device, "device_type", "")
        if device_type not in ("DigitalInput", "AnalogInput", "ADS1115"):
            QMessageBox.warning(self, "Invalid Device", "Please select an input device.")
            return

        async def read_value():
            try:
                # Auto-initialize if not initialized
                if not getattr(device, "_initialized", False):
                    self._status_label.setText("Status: Initializing device...")
                    await device.initialize()

                if device_type == "DigitalInput":
                    if hasattr(device, "read"):
                        value = await device.read()
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        value = await device.board.read_digital(pin)
                    display = "HIGH (1)" if value else "LOW (0)"
                    self._input_value_label.setText(display)
                    self._status_label.setText(f"Status: Digital = {display}")
                    self.value_read.emit(device.id, value)
                else:  # AnalogInput or ADS1115
                    if hasattr(device, "read"):
                        raw_value = await device.read()
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        raw_value = await device.board.read_analog(pin)

                    # Calculate voltage
                    voltage = (
                        raw_value / self._config.hardware.adc_resolution
                    ) * self._config.hardware.adc_reference_voltage
                    display = f"{raw_value}\n{voltage:.2f}V"
                    self._input_value_label.setText(display)
                    self._status_label.setText(f"Status: Analog = {raw_value} ({voltage:.2f}V)")
                    self.value_read.emit(device.id, raw_value)

            except Exception as e:
                QMessageBox.critical(self, "Read Error", str(e))

        self._run_async(read_value())

    def _on_continuous_changed(self, state: int) -> None:
        """Handle continuous reading checkbox change."""
        if state == Qt.CheckState.Checked.value:
            interval = self._poll_spinbox.value()
            self._input_poll_timer.setInterval(interval)
            self._input_poll_timer.start()
        else:
            self._input_poll_timer.stop()

    def _on_poll_interval_changed(self, value: int) -> None:
        """Handle poll interval change."""
        if self._input_poll_timer.isActive():
            self._input_poll_timer.setInterval(value)

    def _poll_input(self) -> None:
        """Poll input device (called by timer)."""
        self._read_input_once()
