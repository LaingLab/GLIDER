"""
Hardware Panel - Dock widget for hardware tree and board/device management.

Provides hardware tree widget, board/device add/edit/remove dialogs,
connect/disconnect logic, and board settings dialog.
"""

import glob
import logging
import sys
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import colors

if TYPE_CHECKING:
    from glider.core.hardware_manager import HardwareManager

logger = logging.getLogger(__name__)


class HardwarePanel(QWidget):
    """Panel for hardware tree and board/device management."""

    hardware_changed = pyqtSignal()
    status_message = pyqtSignal(str, int)  # message, timeout_ms

    def __init__(
        self,
        hardware_manager: "HardwareManager",
        session_fn,
        run_async_fn,
        parent=None,
    ):
        """
        Args:
            hardware_manager: HardwareManager instance
            session_fn: Callable that returns the current ExperimentSession (or None)
            run_async_fn: Callable to schedule async coroutines
        """
        super().__init__(parent)
        self._hardware_manager = hardware_manager
        self._session_fn = session_fn
        self._run_async = run_async_fn
        self._setup_ui()

    @property
    def _session(self):
        return self._session_fn()

    def _setup_ui(self):
        """Build the hardware panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Hardware tree
        self._hardware_tree = QTreeWidget()
        self._hardware_tree.setHeaderLabels(["Name", "Type", "Status"])
        self._hardware_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._hardware_tree.customContextMenuRequested.connect(self._on_hardware_context_menu)
        layout.addWidget(self._hardware_tree)

        # Hardware buttons
        hw_btn_layout = QHBoxLayout()
        add_board_btn = QPushButton("+ Board")
        add_board_btn.clicked.connect(self._on_add_board)
        hw_btn_layout.addWidget(add_board_btn)

        add_device_btn = QPushButton("+ Device")
        add_device_btn.clicked.connect(self._on_add_device)
        hw_btn_layout.addWidget(add_device_btn)

        layout.addLayout(hw_btn_layout)

    # --- Public API ---

    def refresh_tree(self):
        """Refresh the hardware tree widget."""
        self._hardware_tree.clear()

        for board_id, board in self._hardware_manager.boards.items():
            board_item = QTreeWidgetItem(
                [
                    board_id,
                    getattr(board, "name", type(board).__name__),
                    board.state.name if hasattr(board, "state") else "Unknown",
                ]
            )
            board_item.setData(0, Qt.ItemDataRole.UserRole, ("board", board_id))

            for device_id, device in self._hardware_manager.devices.items():
                if hasattr(device, "board") and device.board is board:
                    # Read pins from the authoritative source (device.pins, backed
                    # by device._config.pins) rather than a denormalized copy.
                    # Previously this used device._pins, a list populated once at
                    # creation time in HardwareManager — the Edit Device dialog
                    # updates device.config.pins but not that list, so the tree
                    # kept showing the old pin number after an edit.
                    pin_map = getattr(device, "pins", None)
                    if isinstance(pin_map, dict) and pin_map:
                        pin_values = list(pin_map.values())
                    else:
                        pin_values = getattr(device, "_pins", []) or []

                    device_type = getattr(device, "device_type", "unknown")
                    if device_type == "ADS1115":
                        cfg = getattr(device, "_config", None)
                        settings = getattr(cfg, "settings", {}) if cfg else {}
                        channel = settings.get("channel", 0)
                        pin_str = f"Ch {channel}"
                    elif device_type == "GenericI2C":
                        cfg = getattr(device, "_config", None)
                        settings = getattr(cfg, "settings", {}) if cfg else {}
                        address = settings.get("i2c_address", 0x40)
                        pin_str = f"0x{address:02X}"
                    elif pin_values:
                        pin_str = f"Pin {pin_values[0]}"
                    else:
                        pin_str = ""
                    device_item = QTreeWidgetItem(
                        [
                            getattr(device, "name", device_id),
                            f"{getattr(device, 'device_type', 'unknown')} ({pin_str})",
                            (
                                "Ready"
                                if getattr(device, "_initialized", False)
                                else "Not initialized"
                            ),
                        ]
                    )
                    device_item.setData(0, Qt.ItemDataRole.UserRole, ("device", device_id))
                    board_item.addChild(device_item)

            self._hardware_tree.addTopLevelItem(board_item)
            board_item.setExpanded(True)

        self._hardware_tree.resizeColumnToContents(0)
        self._hardware_tree.resizeColumnToContents(1)

        self.hardware_changed.emit()

    def show_board_settings_dialog(self) -> None:
        """Show a dialog to configure board settings (ports, etc.)."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Board Settings")
        dialog.setMinimumWidth(350)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        def get_available_ports():
            ports = []
            if sys.platform.startswith("linux"):
                patterns = ["/dev/ttyACM*", "/dev/ttyUSB*", "/dev/ttyAMA*"]
                for pattern in patterns:
                    ports.extend(glob.glob(pattern))
            elif sys.platform == "darwin":
                ports.extend(glob.glob("/dev/tty.usbmodem*"))
                ports.extend(glob.glob("/dev/tty.usbserial*"))
            else:
                for i in range(10):
                    ports.append(f"COM{i}")
            return sorted(ports)

        available_ports = get_available_ports()

        boards = self._hardware_manager.boards
        port_combos = {}

        if not boards:
            no_boards_label = QLabel("No boards configured.\nLoad an experiment first.")
            no_boards_label.setProperty("textRole", "muted")
            layout.addWidget(no_boards_label)
        else:
            for board_id, board in boards.items():
                group = QGroupBox(f"Board: {board_id}")
                group_layout = QVBoxLayout(group)

                board_type = getattr(board, "board_type", "unknown")
                type_label = QLabel(f"Type: {board_type}")
                group_layout.addWidget(type_label)

                port_layout = QHBoxLayout()
                port_label = QLabel("Port:")
                port_combo = QComboBox()
                port_combo.setEditable(True)

                current_port = getattr(board, "_port", "") or ""
                if available_ports:
                    port_combo.addItems(available_ports)
                if current_port and current_port not in available_ports:
                    port_combo.addItem(current_port)
                port_combo.setCurrentText(current_port)

                port_combos[board_id] = port_combo

                port_layout.addWidget(port_label)
                port_layout.addWidget(port_combo, 1)
                group_layout.addLayout(port_layout)

                connected = getattr(board, "is_connected", False)
                status_text = "Connected" if connected else "Disconnected"
                status_label = QLabel(f"Status: {status_text}")
                status_label.setStyleSheet(
                    f"color: {colors.SUCCESS if connected else colors.ERROR};"
                )
                group_layout.addWidget(status_label)

                layout.addWidget(group)

        if available_ports:
            ports_label = QLabel(f"Detected ports: {', '.join(available_ports)}")
            ports_label.setProperty("textRole", "muted")
            layout.addWidget(ports_label)
        else:
            ports_label = QLabel("No serial ports detected")
            ports_label.setStyleSheet(f"color: {colors.ERROR}; font-size: 12px;")
            layout.addWidget(ports_label)

        layout.addStretch()

        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("secondary", True)
        cancel_btn.clicked.connect(dialog.reject)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(dialog.accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted and port_combos:
            for board_id, combo in port_combos.items():
                new_port = combo.currentText()
                board = boards.get(board_id)
                if board and new_port:
                    old_port = getattr(board, "_port", "")
                    if new_port != old_port:
                        board._port = new_port
                        logger.info(f"Updated board '{board_id}' port: {old_port} -> {new_port}")

                        session = self._session
                        if session:
                            for board_config in session.boards:
                                if board_config.id == board_id:
                                    board_config.config["port"] = new_port
                                    session._dirty = True
                                    break

    # --- Internal methods ---

    def _on_add_board(self) -> None:
        """Show dialog to add a new board."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Board")
        dialog.setMinimumWidth(350)

        layout = QFormLayout(dialog)

        type_combo = QComboBox()
        type_combo.addItems(["telemetrix", "pigpio"])
        layout.addRow("Board Type:", type_combo)

        id_edit = QLineEdit()
        id_edit.setPlaceholderText("e.g., arduino_1")
        layout.addRow("Board ID:", id_edit)

        port_layout = QHBoxLayout()
        port_combo = QComboBox()
        port_combo.setMinimumWidth(200)

        def refresh_ports():
            port_combo.clear()
            port_combo.addItem("Auto-detect", None)
            try:
                import serial.tools.list_ports

                ports = serial.tools.list_ports.comports()
                for port in ports:
                    label = f"{port.device}"
                    if port.description and port.description != "n/a":
                        label += f" - {port.description}"
                    port_combo.addItem(label, port.device)
            except ImportError:
                pass

        refresh_ports()
        port_layout.addWidget(port_combo)

        refresh_btn = QPushButton("\u21bb")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(refresh_ports)
        port_layout.addWidget(refresh_btn)

        layout.addRow("Serial Port:", port_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            from glider.core.experiment_session import BoardConfig

            board_id = id_edit.text().strip() or f"board_{len(self._hardware_manager.boards)}"
            board_type = type_combo.currentText()
            port = port_combo.currentData()

            driver_type = "arduino" if board_type == "telemetrix" else "raspberry_pi"

            try:
                self._hardware_manager.add_board(board_id, board_type, port=port)

                session = self._session
                if session:
                    board_config = BoardConfig(
                        id=board_id,
                        driver_type=driver_type,
                        port=port,
                        board_type="uno",
                    )
                    session.add_board(board_config)

                self.refresh_tree()
                QMessageBox.information(self, "Success", f"Added board: {board_id}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add board: {e}")

    def _on_add_device(self) -> None:
        """Show dialog to add a new device."""
        if not self._hardware_manager.boards:
            QMessageBox.warning(self, "No Boards", "Please add a board first.")
            return

        device_type_map = {
            "Digital Output (LED, Relay)": ("DigitalOutput", ["output"]),
            "Digital Input (Button, Sensor)": ("DigitalInput", ["input"]),
            "Analog Input (Potentiometer)": ("AnalogInput", ["input"]),
            "PWM Output (Dimmable LED, Motor)": ("PWMOutput", ["output"]),
            "Servo Motor": ("Servo", ["signal"]),
            "Motor Governor": ("MotorGovernor", ["up", "down", "signal"]),
            "ADS1115 (I2C ADC)": ("ADS1115", []),
            "Generic I2C Device": ("GenericI2C", []),
        }

        dialog = QDialog(self)
        dialog.setWindowTitle("Add Device")
        dialog.setMinimumWidth(400)

        layout = QFormLayout(dialog)

        type_combo = QComboBox()
        type_combo.setMaxVisibleItems(10)
        type_combo.setMinimumWidth(280)
        type_combo.addItems(list(device_type_map.keys()))
        layout.addRow("Device Type:", type_combo)

        id_edit = QLineEdit()
        id_edit.setPlaceholderText("e.g., led_1")
        layout.addRow("Device ID:", id_edit)

        board_combo = QComboBox()
        board_combo.addItems(list(self._hardware_manager.boards.keys()))
        layout.addRow("Board:", board_combo)

        pin_container = QWidget()
        pin_layout = QFormLayout(pin_container)
        pin_layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow(pin_container)

        pin_spinboxes: dict[str, QSpinBox] = {}
        ads1115_settings: dict[str, QSpinBox] = {}
        i2c_settings: dict[str, QWidget] = {}

        def update_pin_inputs():
            while pin_layout.rowCount() > 0:
                pin_layout.removeRow(0)
            pin_spinboxes.clear()
            ads1115_settings.clear()
            i2c_settings.clear()

            ui_type = type_combo.currentText()
            device_type, pin_names = device_type_map[ui_type]

            is_analog = device_type == "AnalogInput"
            is_ads1115 = device_type == "ADS1115"
            is_generic_i2c = device_type == "GenericI2C"

            if is_generic_i2c:
                bus_spin = QSpinBox()
                bus_spin.setRange(0, 1)
                bus_spin.setValue(1)
                bus_spin.setToolTip("I2C bus number (1 = Pi primary bus)")
                i2c_settings["i2c_bus"] = bus_spin
                pin_layout.addRow("Bus:", bus_spin)

                addr_spin = QSpinBox()
                addr_spin.setDisplayIntegerBase(16)
                addr_spin.setPrefix("0x")
                addr_spin.setRange(0x03, 0x77)
                addr_spin.setValue(0x40)
                addr_spin.setToolTip("7-bit I2C address (0x03-0x77)")
                i2c_settings["i2c_address"] = addr_spin
                pin_layout.addRow("Address:", addr_spin)

                reg_spin = QSpinBox()
                reg_spin.setDisplayIntegerBase(16)
                reg_spin.setPrefix("0x")
                reg_spin.setRange(-1, 0xFF)
                reg_spin.setValue(-1)
                reg_spin.setSpecialValueText("None")
                reg_spin.setToolTip(
                    "Optional default register for the no-arg Read (None = raw byte)"
                )
                i2c_settings["register"] = reg_spin
                pin_layout.addRow("Register:", reg_spin)

                word_check = QCheckBox("16-bit big-endian (combine 2 registers)")
                word_check.setToolTip(
                    "Read 2 bytes MSB-first from the register (e.g. the AS5600 "
                    "12-bit angle at 0x0E/0x0F)"
                )
                i2c_settings["read_word"] = word_check
                pin_layout.addRow("Word read:", word_check)

                note = QLabel("Note: Uses I2C on GPIO2 (SDA) / GPIO3 (SCL)")
                note.setProperty("textRole", "muted")
                note.setWordWrap(True)
                pin_layout.addRow(note)
            elif is_ads1115:
                addr_spin = QSpinBox()
                addr_spin.setRange(72, 75)
                addr_spin.setValue(72)
                addr_spin.setToolTip("I2C address: 72=0x48, 73=0x49, 74=0x4A, 75=0x4B")
                ads1115_settings["i2c_address"] = addr_spin
                pin_layout.addRow("I2C Address:", addr_spin)

                chan_spin = QSpinBox()
                chan_spin.setRange(0, 3)
                chan_spin.setValue(0)
                chan_spin.setToolTip("ADC channel to read (0-3)")
                ads1115_settings["channel"] = chan_spin
                pin_layout.addRow("Channel:", chan_spin)

                gain_combo = QComboBox()
                gain_combo.addItems(
                    [
                        "1 (\u00b14.096V)",
                        "2 (\u00b12.048V)",
                        "4 (\u00b11.024V)",
                        "8 (\u00b10.512V)",
                        "16 (\u00b10.256V)",
                    ]
                )
                gain_combo.setCurrentIndex(0)
                ads1115_settings["gain_combo"] = gain_combo
                pin_layout.addRow("Gain:", gain_combo)

                note = QLabel("Note: Uses I2C on GPIO2 (SDA) and GPIO3 (SCL)")
                note.setProperty("textRole", "muted")
                note.setWordWrap(True)
                pin_layout.addRow(note)
            else:
                for pin_name in pin_names:
                    spin = QSpinBox()
                    spin.setRange(0, 53)

                    if is_analog:
                        spin.setValue(14)
                        spin.setSpecialValueText("Invalid")
                    else:
                        spin.setValue(0)

                    pin_spinboxes[pin_name] = spin
                    label = f"{pin_name.capitalize()} Pin:"
                    pin_layout.addRow(label, spin)

                if is_analog:
                    note = QLabel("Note: A0=14, A1=15, A2=16, A3=17, A4=18, A5=19")
                    note.setProperty("textRole", "muted")
                    note.setWordWrap(True)
                    pin_layout.addRow(note)

        type_combo.currentTextChanged.connect(lambda: update_pin_inputs())
        update_pin_inputs()

        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g., Status LED")
        layout.addRow("Name:", name_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            from glider.core.experiment_session import DeviceConfig

            device_id = id_edit.text().strip() or f"device_{len(self._hardware_manager.devices)}"
            ui_device_type = type_combo.currentText()
            board_id = board_combo.currentText()
            name = name_edit.text().strip() or device_id

            device_type, pin_names = device_type_map[ui_device_type]

            pins = {pin_name: pin_spinboxes[pin_name].value() for pin_name in pin_names}

            settings = {}
            if device_type == "ADS1115" and ads1115_settings:
                settings["i2c_address"] = ads1115_settings["i2c_address"].value()
                settings["channel"] = ads1115_settings["channel"].value()
                gain_text = ads1115_settings["gain_combo"].currentText()
                settings["gain"] = int(gain_text.split()[0])
            elif device_type == "GenericI2C" and i2c_settings:
                settings["i2c_bus"] = i2c_settings["i2c_bus"].value()
                settings["i2c_address"] = i2c_settings["i2c_address"].value()
                reg = i2c_settings["register"].value()
                if reg >= 0:  # -1 (special "None") ⇒ omit, leaving raw-byte read
                    settings["register"] = reg
                if i2c_settings["read_word"].isChecked():
                    settings["read_word"] = True

            try:
                self._hardware_manager.add_device_multi_pin(
                    device_id, device_type, board_id, pins, name=name, **settings
                )

                board = self._hardware_manager.get_board(board_id)
                if board and board.is_connected:

                    async def init_device():
                        try:
                            await self._hardware_manager.initialize_device(device_id)
                            logger.info(f"Auto-initialized device: {device_id}")
                        except Exception as e:
                            logger.error(f"Failed to auto-initialize device {device_id}: {e}")

                    self._run_async(init_device())

                session = self._session
                if session:
                    device_config = DeviceConfig(
                        id=device_id,
                        device_type=device_type,
                        name=name,
                        board_id=board_id,
                        pins=pins,
                        settings=settings,
                    )
                    session.add_device(device_config)

                self.refresh_tree()
                QMessageBox.information(self, "Success", f"Added device: {device_id}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add device: {e}")

    def _on_edit_board(self, board_id: str) -> None:
        """Show dialog to edit an existing board."""
        board = self._hardware_manager.get_board(board_id)
        if board is None:
            QMessageBox.warning(self, "Error", f"Board '{board_id}' not found.")
            return

        session = self._session
        board_config = session.get_board(board_id) if session else None
        current_port = board_config.port if board_config else getattr(board, "port", None)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Board: {board_id}")
        dialog.setMinimumWidth(350)

        layout = QFormLayout(dialog)

        id_label = QLabel(board_id)
        id_label.setProperty("textRole", "muted")
        layout.addRow("Board ID:", id_label)

        port_layout = QHBoxLayout()
        port_combo = QComboBox()
        port_combo.setMinimumWidth(200)

        def refresh_ports():
            port_combo.clear()
            port_combo.addItem("Auto-detect", None)
            try:
                import serial.tools.list_ports

                ports = serial.tools.list_ports.comports()
                for port in ports:
                    label = f"{port.device}"
                    if port.description and port.description != "n/a":
                        label += f" - {port.description}"
                    port_combo.addItem(label, port.device)
            except ImportError:
                pass

            if current_port:
                for i in range(port_combo.count()):
                    if port_combo.itemData(i) == current_port:
                        port_combo.setCurrentIndex(i)
                        break

        refresh_ports()
        port_layout.addWidget(port_combo)

        refresh_btn = QPushButton("\u21bb")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(refresh_ports)
        port_layout.addWidget(refresh_btn)

        layout.addRow("Serial Port:", port_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_port = port_combo.currentData()

            try:
                board.set_port(new_port)

                if session:
                    session.update_board(board_id, port=new_port)

                self.refresh_tree()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Updated board: {board_id}\n\n"
                    "Note: Port changes take effect after reconnecting.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update board: {e}")

    def _on_edit_device(self, device_id: str) -> None:
        """Show dialog to edit an existing device."""
        device = self._hardware_manager.get_device(device_id)
        if device is None:
            QMessageBox.warning(self, "Error", f"Device '{device_id}' not found.")
            return

        session = self._session
        device_config = session.get_device(device_id) if session else None

        current_name = device_config.name if device_config else getattr(device, "name", device_id)
        current_pins = device_config.pins if device_config else {}
        current_settings = device_config.settings if device_config else {}
        device_type = device_config.device_type if device_config else type(device).__name__

        if not current_pins and hasattr(device, "pin"):
            current_pins = {"output": device.pin} if hasattr(device, "pin") else {}
        if not current_pins and hasattr(device, "pins"):
            current_pins = device.pins if isinstance(device.pins, dict) else {}

        is_ads1115 = device_type == "ADS1115"
        is_generic_i2c = device_type == "GenericI2C"
        is_settings_device = is_ads1115 or is_generic_i2c

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Device: {device_id}")
        dialog.setMinimumWidth(380)

        layout = QFormLayout(dialog)

        id_label = QLabel(device_id)
        id_label.setProperty("textRole", "muted")
        layout.addRow("Device ID:", id_label)

        type_label = QLabel(device_type)
        type_label.setProperty("textRole", "muted")
        layout.addRow("Device Type:", type_label)

        name_edit = QLineEdit(current_name)
        layout.addRow("Name:", name_edit)

        pin_spinboxes: dict[str, QSpinBox] = {}
        ads1115_settings: dict[str, QSpinBox | QComboBox] = {}
        i2c_settings: dict[str, QWidget] = {}

        if is_ads1115:
            addr_spin = QSpinBox()
            addr_spin.setRange(72, 75)
            addr_spin.setValue(current_settings.get("i2c_address", 72))
            addr_spin.setToolTip("I2C address: 72=0x48, 73=0x49, 74=0x4A, 75=0x4B")
            ads1115_settings["i2c_address"] = addr_spin
            layout.addRow("I2C Address:", addr_spin)

            chan_spin = QSpinBox()
            chan_spin.setRange(0, 3)
            chan_spin.setValue(current_settings.get("channel", 0))
            chan_spin.setToolTip("ADC channel to read (0-3)")
            ads1115_settings["channel"] = chan_spin
            layout.addRow("Channel:", chan_spin)

            gain_combo = QComboBox()
            gain_options = [
                "1 (\u00b14.096V)",
                "2 (\u00b12.048V)",
                "4 (\u00b11.024V)",
                "8 (\u00b10.512V)",
                "16 (\u00b10.256V)",
            ]
            gain_combo.addItems(gain_options)
            current_gain = current_settings.get("gain", 1)
            gain_values = [1, 2, 4, 8, 16]
            if current_gain in gain_values:
                gain_combo.setCurrentIndex(gain_values.index(current_gain))
            ads1115_settings["gain_combo"] = gain_combo
            layout.addRow("Gain:", gain_combo)

            note = QLabel("Note: Uses I2C on GPIO2 (SDA) and GPIO3 (SCL)")
            note.setProperty("textRole", "muted")
            note.setWordWrap(True)
            layout.addRow(note)
        elif is_generic_i2c:
            bus_spin = QSpinBox()
            bus_spin.setRange(0, 1)
            bus_spin.setValue(current_settings.get("i2c_bus", 1))
            bus_spin.setToolTip("I2C bus number (1 = Pi primary bus)")
            i2c_settings["i2c_bus"] = bus_spin
            layout.addRow("Bus:", bus_spin)

            addr_spin = QSpinBox()
            addr_spin.setDisplayIntegerBase(16)
            addr_spin.setPrefix("0x")
            addr_spin.setRange(0x03, 0x77)
            addr_spin.setValue(current_settings.get("i2c_address", 0x40))
            addr_spin.setToolTip("7-bit I2C address (0x03-0x77)")
            i2c_settings["i2c_address"] = addr_spin
            layout.addRow("Address:", addr_spin)

            reg_spin = QSpinBox()
            reg_spin.setDisplayIntegerBase(16)
            reg_spin.setPrefix("0x")
            reg_spin.setRange(-1, 0xFF)
            reg_value = current_settings.get("register")
            reg_spin.setValue(reg_value if reg_value is not None else -1)
            reg_spin.setSpecialValueText("None")
            reg_spin.setToolTip("Optional default register for the no-arg Read (None = raw byte)")
            i2c_settings["register"] = reg_spin
            layout.addRow("Register:", reg_spin)

            word_check = QCheckBox("16-bit big-endian (combine 2 registers)")
            word_check.setChecked(bool(current_settings.get("read_word", False)))
            word_check.setToolTip(
                "Read 2 bytes MSB-first from the register (e.g. the AS5600 "
                "12-bit angle at 0x0E/0x0F)"
            )
            i2c_settings["read_word"] = word_check
            layout.addRow("Word read:", word_check)

            note = QLabel("Note: Uses I2C on GPIO2 (SDA) / GPIO3 (SCL)")
            note.setProperty("textRole", "muted")
            note.setWordWrap(True)
            layout.addRow(note)
        else:
            is_analog = "Analog" in device_type

            for pin_name, pin_value in current_pins.items():
                spin = QSpinBox()
                spin.setRange(0, 53)
                spin.setValue(pin_value)
                pin_spinboxes[pin_name] = spin
                label = f"{pin_name.capitalize()} Pin:"
                layout.addRow(label, spin)

            if is_analog:
                note = QLabel("Note: A0=14, A1=15, A2=16, A3=17, A4=18, A5=19")
                note.setProperty("textRole", "muted")
                note.setWordWrap(True)
                layout.addRow(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_name = name_edit.text().strip() or device_id
            new_pins = {pin_name: spin.value() for pin_name, spin in pin_spinboxes.items()}
            new_settings = None

            if is_ads1115 and ads1115_settings:
                new_settings = {
                    "i2c_address": ads1115_settings["i2c_address"].value(),
                    "channel": ads1115_settings["channel"].value(),
                    "gain": int(ads1115_settings["gain_combo"].currentText().split()[0]),
                }
            elif is_generic_i2c and i2c_settings:
                new_settings = {
                    "i2c_bus": i2c_settings["i2c_bus"].value(),
                    "i2c_address": i2c_settings["i2c_address"].value(),
                }
                reg = i2c_settings["register"].value()
                new_settings["register"] = reg if reg >= 0 else None
                new_settings["read_word"] = i2c_settings["read_word"].isChecked()

            try:
                device.name = new_name

                if is_settings_device and new_settings:
                    if hasattr(device, "config") and hasattr(device.config, "settings"):
                        device.config.settings.update(new_settings)
                    elif hasattr(device, "_config") and hasattr(device._config, "settings"):
                        device._config.settings.update(new_settings)
                else:
                    if hasattr(device, "config") and hasattr(device.config, "pins"):
                        device.config.pins.update(new_pins)
                    elif hasattr(device, "_config") and hasattr(device._config, "pins"):
                        device._config.pins.update(new_pins)

                    # Keep the legacy denormalized list (set by HardwareManager at
                    # creation time) in sync so anything still reading it sees the
                    # updated values.
                    if hasattr(device, "_pins"):
                        device._pins = list(new_pins.values())

                if session:
                    session.update_device(
                        device_id,
                        name=new_name,
                        pins=new_pins if not is_settings_device else None,
                        settings=new_settings,
                    )

                self.refresh_tree()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Updated device: {device_id}\n\n"
                    "Note: Changes take effect after reconnecting the board.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update device: {e}")

    def _on_hardware_context_menu(self, position) -> None:
        """Show context menu for hardware tree."""
        item = self._hardware_tree.itemAt(position)
        if item is None:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return

        item_type, item_id = data

        menu = QMenu(self)

        if item_type == "board":
            connect_action = menu.addAction("Connect")
            connect_action.triggered.connect(lambda: self._connect_board(item_id))

            disconnect_action = menu.addAction("Disconnect")
            disconnect_action.triggered.connect(lambda: self._disconnect_board(item_id))

            menu.addSeparator()

            edit_action = menu.addAction("Edit Board")
            edit_action.triggered.connect(lambda: self._on_edit_board(item_id))

            remove_action = menu.addAction("Remove Board")
            remove_action.triggered.connect(lambda: self._remove_board(item_id))

        elif item_type == "device":
            edit_action = menu.addAction("Edit Device")
            edit_action.triggered.connect(lambda: self._on_edit_device(item_id))

            remove_action = menu.addAction("Remove Device")
            remove_action.triggered.connect(lambda: self._remove_device(item_id))

        menu.exec(self._hardware_tree.viewport().mapToGlobal(position))

    def _connect_board(self, board_id: str) -> None:
        """Connect to a specific board and initialize its devices."""

        async def connect():
            try:
                success = await self._hardware_manager.connect_board(board_id)
                if success:
                    for device_id, device in self._hardware_manager.devices.items():
                        if hasattr(device, "board") and device.board is not None:
                            if device.board.id == board_id:
                                try:
                                    await self._hardware_manager.initialize_device(device_id)
                                except Exception as e:
                                    logger.warning(f"Failed to initialize device {device_id}: {e}")
                    self.status_message.emit(f"Connected to {board_id}", 3000)
                else:
                    QMessageBox.warning(
                        self, "Connection Failed", f"Could not connect to {board_id}"
                    )
                self.refresh_tree()
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", str(e))

        self._run_async(connect())

    def _disconnect_board(self, board_id: str) -> None:
        """Disconnect from a specific board."""

        async def disconnect():
            try:
                await self._hardware_manager.disconnect_board(board_id)
                self.refresh_tree()
            except Exception as e:
                QMessageBox.critical(self, "Disconnect Error", str(e))

        self._run_async(disconnect())

    def _remove_board(self, board_id: str) -> None:
        """Remove a board."""
        reply = QMessageBox.question(
            self,
            "Remove Board",
            f"Remove board '{board_id}' and all its devices?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:

            async def remove():
                await self._hardware_manager.remove_board(board_id)
                session = self._session
                if session:
                    session.remove_board(board_id)
                self.refresh_tree()

            self._run_async(remove())

    def _remove_device(self, device_id: str) -> None:
        """Remove a device."""
        reply = QMessageBox.question(
            self,
            "Remove Device",
            f"Remove device '{device_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:

            async def remove():
                await self._hardware_manager.remove_device(device_id)
                session = self._session
                if session:
                    session.remove_device(device_id)
                self.refresh_tree()

            self._run_async(remove())

    def on_connect_hardware(self) -> None:
        """Connect to all hardware."""
        self._run_async(self._connect_hardware_async())

    async def _connect_hardware_async(self) -> None:
        """Async hardware connection."""
        try:
            # setup_hardware needs to be called on core, not hardware_manager
            # This is delegated back to MainWindow via signal
            results = await self._hardware_manager.connect_all()
            self.refresh_tree()
            failed = [k for k, v in results.items() if not v]
            if failed:
                QMessageBox.warning(
                    self, "Connection Warning", f"Failed to connect: {', '.join(failed)}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))

    def on_disconnect_hardware(self) -> None:
        """Disconnect all hardware."""
        self._run_async(self._hardware_manager.disconnect_all())
