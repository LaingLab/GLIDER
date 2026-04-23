"""
Device Control Panel - Dock widget for direct device I/O control.

Provides digital output (ON/OFF/Toggle), PWM control, servo control,
and analog/digital input reading with continuous polling support.
"""

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.core.hardware_manager import HardwareManager

logger = logging.getLogger(__name__)


class DeviceControlPanel(QWidget):
    """Panel for direct device I/O control."""

    status_message = pyqtSignal(str, int)  # message, timeout_ms
    analog_value_received = pyqtSignal(int, int)  # pin, value

    def __init__(self, hardware_manager: "HardwareManager", run_async_fn, parent=None):
        super().__init__(parent)
        self._hardware_manager = hardware_manager
        self._run_async = run_async_fn

        # Async task tracking
        self._pending_tasks: set = set()

        # Real-time callback tracking for analog inputs
        self._analog_callback_board = None
        self._analog_callback_pin = None
        self._analog_callback_func = None

        # PWM debounce
        self._pwm_debounce_timer = None
        self._pending_pwm_value = 0
        self._pending_pwm_device = None

        self._setup_ui()
        self.analog_value_received.connect(self._on_analog_value_received)

    def _setup_ui(self):
        """Build the device control panel UI."""
        # Wrap in scroll area for touch screens
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        control_widget = QWidget()
        control_widget.setMinimumWidth(240)
        self._control_layout = QVBoxLayout(control_widget)
        self._control_layout.setContentsMargins(6, 6, 6, 6)
        self._control_layout.setSpacing(8)

        # Device selector row
        device_layout = QHBoxLayout()
        device_layout.setSpacing(6)
        device_label = QLabel("Device:")
        device_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._device_combo = QComboBox()
        self._device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._device_combo.currentTextChanged.connect(self._on_device_selected)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self._device_combo, 1)
        self._control_layout.addLayout(device_layout)

        # Output Controls group
        self._control_group = QGroupBox("Output Controls")
        self._control_group_layout = QVBoxLayout(self._control_group)
        self._control_group_layout.setContentsMargins(8, 12, 8, 8)
        self._control_group_layout.setSpacing(8)

        # Digital output controls
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
        self._control_group_layout.addWidget(self._digital_widget)

        # PWM control row
        self._pwm_widget = QWidget()
        pwm_layout = QHBoxLayout(self._pwm_widget)
        pwm_layout.setContentsMargins(0, 0, 0, 0)
        pwm_layout.setSpacing(6)
        pwm_label = QLabel("PWM:")
        pwm_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._pwm_spinbox = QSpinBox()
        self._pwm_spinbox.setRange(0, 255)
        self._pwm_spinbox.setMinimumHeight(35)
        self._pwm_spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._pwm_spinbox.valueChanged.connect(self._on_pwm_changed)
        self._pwm_slider = QSlider(Qt.Orientation.Horizontal)
        self._pwm_slider.setRange(0, 255)
        self._pwm_slider.hide()
        self._pwm_spinbox.valueChanged.connect(self._pwm_slider.setValue)
        pwm_layout.addWidget(pwm_label)
        pwm_layout.addWidget(self._pwm_spinbox, 1)
        self._control_group_layout.addWidget(self._pwm_widget)
        self._pwm_widget.hide()

        self._control_layout.addWidget(self._control_group)

        # Input Reading group
        input_group = QGroupBox("Input Reading")
        input_group_layout = QVBoxLayout(input_group)
        input_group_layout.setContentsMargins(8, 12, 8, 8)
        input_group_layout.setSpacing(8)

        self._input_value_label = QLabel("--")
        self._input_value_label.setWordWrap(True)
        self._input_value_label.setProperty("inputValue", True)
        self._input_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._input_value_label.setMinimumHeight(48)
        self._input_value_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        input_group_layout.addWidget(self._input_value_label)

        self._read_btn = QPushButton("Read")
        self._read_btn.setMinimumHeight(32)
        self._read_btn.clicked.connect(self._read_input_once)
        input_group_layout.addWidget(self._read_btn)

        poll_row = QHBoxLayout()
        poll_row.setSpacing(8)
        self._continuous_checkbox = QCheckBox("Auto")
        self._continuous_checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Wire the checkbox to its handler. Without this, toggling "Auto" did
        # nothing at all — the handler existed only as dead code.
        self._continuous_checkbox.stateChanged.connect(self._on_continuous_changed)
        poll_row.addWidget(self._continuous_checkbox)

        poll_label = QLabel("Interval:")
        poll_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        poll_row.addWidget(poll_label)

        self._poll_spinbox = QSpinBox()
        self._poll_spinbox.setRange(50, 5000)
        self._poll_spinbox.setValue(100)
        self._poll_spinbox.setSuffix(" ms")
        self._poll_spinbox.setMinimumHeight(28)
        self._poll_spinbox.valueChanged.connect(self._on_poll_interval_changed)
        poll_row.addWidget(self._poll_spinbox)

        input_group_layout.addLayout(poll_row)
        self._input_group = input_group
        self._control_layout.addWidget(input_group)

        # Timer for continuous reading
        self._input_poll_timer = QTimer(self)
        self._input_poll_timer.timeout.connect(self._poll_input)

        # Status display
        self._device_status_label = QLabel("No device selected")
        self._device_status_label.setWordWrap(True)
        self._device_status_label.setProperty("textRole", "muted")
        self._device_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._control_layout.addWidget(self._device_status_label)

        self._control_layout.addStretch()

        control_scroll.setWidget(control_widget)
        outer_layout.addWidget(control_scroll)

    # --- Public API ---

    def refresh_devices(self):
        """Refresh the device selector combo box."""
        self._device_combo.clear()
        self._device_combo.addItem("-- Select Device --", None)

        for device_id, device in self._hardware_manager.devices.items():
            device_name = getattr(device, "name", device_id)
            device_type = getattr(device, "device_type", "unknown")
            self._device_combo.addItem(f"{device_name} ({device_type})", device_id)

    def stop_polling(self):
        """Stop all continuous input reading."""
        self._input_poll_timer.stop()
        self._stop_analog_callback()
        if self._continuous_checkbox.isChecked():
            self._continuous_checkbox.setChecked(False)

    # --- Internal methods ---

    def _on_device_selected(self, text: str) -> None:
        """Handle device selection change."""
        self._input_poll_timer.stop()
        self._continuous_checkbox.setChecked(False)
        self._input_value_label.setText("--")

        device_id = self._device_combo.currentData()
        if device_id is None:
            self._device_status_label.setText("Status: No device selected")
            self._input_group.setEnabled(False)
            return

        device = self._hardware_manager.get_device(device_id)
        if device is None:
            self._device_status_label.setText("Status: Device not found")
            self._input_group.setEnabled(False)
            return

        device_type = getattr(device, "device_type", "unknown")
        board = getattr(device, "board", None)
        connected = board.is_connected if board else False
        initialized = getattr(device, "_initialized", False)

        status = "Connected" if connected else "Disconnected"
        if connected and initialized:
            status = "Ready"
        elif connected and not initialized:
            status = "Not initialized"

        self._device_status_label.setText(f"Status: {status} | Type: {device_type}")

        # Enable/disable input reading based on device type
        is_input_device = device_type in ("DigitalInput", "AnalogInput", "ADS1115")
        self._input_group.setEnabled(is_input_device)

        # Show/hide appropriate output controls based on device type
        is_digital_output = device_type == "DigitalOutput"
        is_pwm_output = device_type == "PWMOutput"
        is_output = is_digital_output or is_pwm_output

        self._control_group.setVisible(is_output)
        self._digital_widget.setVisible(is_digital_output)
        self._pwm_widget.setVisible(is_pwm_output)
        if is_pwm_output:
            current_value = getattr(device, "_value", 0)
            self._pwm_spinbox.blockSignals(True)
            self._pwm_spinbox.setValue(current_value)
            self._pwm_spinbox.blockSignals(False)

    def _get_selected_device(self):
        """Get the currently selected device."""
        device_id = self._device_combo.currentData()
        if device_id is None:
            return None
        return self._hardware_manager.get_device(device_id)

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
                self._device_status_label.setText(f"Status: Output set to {state}")
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
                self._device_status_label.setText("Status: Output toggled")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to toggle: {e}")

        self._run_async(toggle())

    def _on_pwm_changed(self, value: int) -> None:
        """Handle PWM slider change (debounced)."""
        device = self._get_selected_device()
        if device is None:
            return

        self._pending_pwm_value = value
        self._pending_pwm_device = device

        if self._pwm_debounce_timer is None:
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
                self._device_status_label.setText(f"Status: PWM set to {value}")
            except Exception as e:
                logger.error(f"PWM error: {e}")
                self._device_status_label.setText(f"Status: PWM FAILED - {e}")

        self._run_async(set_pwm())

    def _on_servo_changed(self, angle: int) -> None:
        """Handle servo slider change."""
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
                self._device_status_label.setText(f"Status: Servo set to {angle}\u00b0")
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
            QMessageBox.warning(
                self,
                "Invalid Device",
                "Please select a DigitalInput, AnalogInput, or ADS1115 device.",
            )
            return

        async def read_value():
            try:
                # Auto-initialize if not initialized
                if not getattr(device, "_initialized", False):
                    self._device_status_label.setText("Status: Initializing device...")
                    await device.initialize()
                    logger.info(f"Auto-initialized device for reading: {device.id}")

                if device_type == "DigitalInput":
                    if hasattr(device, "read"):
                        value = await device.read()
                        display = "HIGH (1)" if value else "LOW (0)"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(f"Status: Digital input = {display}")
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        value = await device.board.read_digital(pin)
                        display = "HIGH (1)" if value else "LOW (0)"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(f"Status: Digital input = {display}")
                elif device_type == "AnalogInput":
                    if hasattr(device, "read") and hasattr(device, "read_voltage"):
                        raw_value = await device.read()
                        voltage = await device.read_voltage()
                        display = f"{raw_value}\n{voltage:.2f}V"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(
                            f"Status: Analog = {raw_value} ({voltage:.2f}V)"
                        )
                    elif hasattr(device, "read"):
                        raw_value = await device.read()
                        voltage = (raw_value / 1023.0) * 5.0
                        display = f"{raw_value}\n{voltage:.2f}V"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(
                            f"Status: Analog = {raw_value} ({voltage:.2f}V)"
                        )
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        raw_value = await device.board.read_analog(pin)
                        voltage = (raw_value / 1023.0) * 5.0
                        display = f"{raw_value}\n{voltage:.2f}V"
                        self._input_value_label.setText(display)
                        self._device_status_label.setText(
                            f"Status: Analog = {raw_value} ({voltage:.2f}V)"
                        )
                elif device_type == "ADS1115":
                    channel = device._config.settings.get("channel", 0)
                    raw_value = await device.read(channel)
                    voltage = await device.read_voltage(channel)
                    display = f"{raw_value}\n{voltage:.3f}V"
                    self._input_value_label.setText(display)
                    self._device_status_label.setText(
                        f"Status: ADS1115 Ch{channel} = {raw_value} ({voltage:.3f}V)"
                    )
            except Exception as e:
                logger.error(f"Read error: {e}")
                self._input_value_label.setText("ERROR")
                self._device_status_label.setText(f"Status: Read failed - {e}")

        self._run_async(read_value())

    def _on_continuous_changed(self, state: int) -> None:
        """Handle continuous checkbox state change."""
        if state == Qt.CheckState.Checked.value:
            device = self._get_selected_device()
            if device is None:
                self._continuous_checkbox.setChecked(False)
                QMessageBox.warning(self, "No Device", "Please select a device first.")
                return

            device_type = getattr(device, "device_type", "")
            if device_type not in ("DigitalInput", "AnalogInput", "ADS1115"):
                self._continuous_checkbox.setChecked(False)
                QMessageBox.warning(
                    self,
                    "Invalid Device",
                    "Please select a DigitalInput, AnalogInput, or ADS1115 device.",
                )
                return

            if device_type == "AnalogInput":
                self._start_analog_callback(device)
            elif device_type == "ADS1115":
                interval = self._poll_spinbox.value()
                self._input_poll_timer.start(interval)
                self._device_status_label.setText(f"Status: ADS1115 polling ({interval}ms)")
            else:
                interval = self._poll_spinbox.value()
                self._input_poll_timer.start(interval)
                self._device_status_label.setText(f"Status: Continuous reading ({interval}ms)")
        else:
            self._stop_analog_callback()
            self._input_poll_timer.stop()
            self._device_status_label.setText("Status: Continuous reading stopped")

    def _start_analog_callback(self, device) -> None:
        """Start real-time analog monitoring using board callbacks."""
        self._stop_analog_callback()

        pin = device.pins.get("input", list(device.pins.values())[0])
        board = device.board

        def analog_callback(callback_pin: int, value: int) -> None:
            self.analog_value_received.emit(callback_pin, value)

        board.register_callback(pin, analog_callback)
        logger.debug(f"Registered analog UI callback for pin {pin}")

        self._analog_callback_board = board
        self._analog_callback_pin = pin
        self._analog_callback_func = analog_callback

        self._device_status_label.setText("Status: Real-time monitoring (callback)")
        logger.info(f"Started real-time analog callback for pin {pin}")

    def _stop_analog_callback(self) -> None:
        """Stop real-time analog monitoring."""
        if self._analog_callback_board is not None and self._analog_callback_func is not None:
            try:
                self._analog_callback_board.unregister_callback(
                    self._analog_callback_pin, self._analog_callback_func
                )
                logger.info(f"Stopped analog callback for pin {self._analog_callback_pin}")
            except Exception as e:
                logger.debug(f"Error unregistering callback: {e}")

        self._analog_callback_board = None
        self._analog_callback_pin = None
        self._analog_callback_func = None

    @pyqtSlot(int, int)
    def _on_analog_value_received(self, pin: int, value: int) -> None:
        """Handle real-time analog value updates."""
        device = self._get_selected_device()
        if device is not None and hasattr(device, "_reference_voltage"):
            ref_voltage = device._reference_voltage
        else:
            ref_voltage = 5.0

        voltage = (value / 1023.0) * ref_voltage
        display = f"{value}\n{voltage:.2f}V"
        self._input_value_label.setText(display)

    def _on_poll_interval_changed(self, value: int) -> None:
        """Handle poll interval change."""
        if self._input_poll_timer.isActive():
            self._input_poll_timer.setInterval(value)
            self._device_status_label.setText(f"Status: Poll interval changed to {value}ms")

    def _poll_input(self) -> None:
        """Poll the input value (called by timer)."""
        device = self._get_selected_device()
        if device is None:
            self._input_poll_timer.stop()
            self._continuous_checkbox.setChecked(False)
            return

        device_type = getattr(device, "device_type", "")
        if device_type not in ("DigitalInput", "AnalogInput", "ADS1115"):
            self._input_poll_timer.stop()
            self._continuous_checkbox.setChecked(False)
            return

        async def read_value():
            try:
                if device_type == "DigitalInput":
                    if hasattr(device, "read"):
                        value = await device.read()
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        value = await device.board.read_digital(pin)
                    display = "HIGH (1)" if value else "LOW (0)"
                    self._input_value_label.setText(display)
                elif device_type == "AnalogInput":
                    if hasattr(device, "read"):
                        raw_value = await device.read()
                    else:
                        pin = device.pins.get("input", list(device.pins.values())[0])
                        raw_value = await device.board.read_analog(pin)

                    if hasattr(device, "_reference_voltage"):
                        ref_voltage = device._reference_voltage
                    else:
                        ref_voltage = 5.0
                    voltage = (raw_value / 1023.0) * ref_voltage
                    display = f"{raw_value}\n{voltage:.2f}V"
                    self._input_value_label.setText(display)
                elif device_type == "ADS1115":
                    channel = device._config.settings.get("channel", 0)
                    raw_value = await device.read(channel)
                    voltage = await device.read_voltage(channel)
                    display = f"{raw_value}\n{voltage:.3f}V"
                    self._input_value_label.setText(display)
            except Exception as e:
                logger.error(f"Poll read error: {e}")
                self._input_poll_timer.stop()
                self._continuous_checkbox.setChecked(False)
                self._input_value_label.setText("ERROR")

        self._run_async(read_value())

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Stop timers before teardown so they don't fire on a dead widget."""
        for attr in ("_input_poll_timer", "_pwm_debounce_timer"):
            timer = getattr(self, attr, None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
        super().closeEvent(event)
