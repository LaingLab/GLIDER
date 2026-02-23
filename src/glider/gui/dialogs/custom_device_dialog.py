"""
Custom Device Editor Dialog.

Provides a GUI for creating and editing custom device definitions,
including pin configuration. Flow logic is created via the node graph.
"""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from glider.core.custom_device import CustomDeviceDefinition, PinDefinition, PinType

logger = logging.getLogger(__name__)


class PinEditorWidget(QWidget):
    """Widget for editing pin definitions."""

    pins_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Pin table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Name", "Pin #", "Type", "Default", "Description"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(1, 60)  # Pin #
        self._table.setColumnWidth(2, 120)  # Type
        self._table.setColumnWidth(3, 60)  # Default
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._table)

        # Buttons
        btn_layout = QHBoxLayout()
        self._add_btn = QPushButton("Add Pin")
        self._add_btn.clicked.connect(self._add_pin)
        btn_layout.addWidget(self._add_btn)

        self._remove_btn = QPushButton("Remove Pin")
        self._remove_btn.clicked.connect(self._remove_pin)
        btn_layout.addWidget(self._remove_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _add_pin(self):
        row = self._table.rowCount()
        self._table.insertRow(row)

        # Name
        name_item = QTableWidgetItem(f"pin{row + 1}")
        self._table.setItem(row, 0, name_item)

        # Pin number spinbox
        pin_spin = QSpinBox()
        pin_spin.setRange(0, 99)
        pin_spin.setValue(0)
        self._table.setCellWidget(row, 1, pin_spin)

        # Type combo
        type_combo = QComboBox()
        for pt in PinType:
            type_combo.addItem(pt.value, pt)
        self._table.setCellWidget(row, 2, type_combo)

        # Default value
        default_item = QTableWidgetItem("")
        self._table.setItem(row, 3, default_item)

        # Description
        desc_item = QTableWidgetItem("")
        self._table.setItem(row, 4, desc_item)

        self.pins_changed.emit()

    def _remove_pin(self):
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)
            self.pins_changed.emit()

    def get_pins(self) -> list[PinDefinition]:
        """Get the list of pin definitions."""
        pins = []
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            pin_spin = self._table.cellWidget(row, 1)
            type_combo = self._table.cellWidget(row, 2)
            default_item = self._table.item(row, 3)
            desc_item = self._table.item(row, 4)

            if name_item and type_combo:
                name = name_item.text()
                pin_number = pin_spin.value() if pin_spin else None
                pin_type = type_combo.currentData()
                default_text = default_item.text() if default_item else ""
                description = desc_item.text() if desc_item else ""

                # Parse default value
                default_value = None
                if default_text:
                    if pin_type in (PinType.DIGITAL_OUTPUT, PinType.DIGITAL_INPUT):
                        default_value = default_text.lower() in ("true", "1", "high")
                    elif pin_type in (PinType.ANALOG_INPUT, PinType.ANALOG_OUTPUT, PinType.PWM):
                        try:
                            default_value = int(default_text)
                        except ValueError:
                            pass

                pins.append(
                    PinDefinition(
                        name=name,
                        pin_type=pin_type,
                        pin_number=pin_number,
                        default_value=default_value,
                        description=description,
                    )
                )
        return pins

    def set_pins(self, pins: list[PinDefinition]):
        """Set the pin definitions."""
        self._table.setRowCount(0)
        for pin in pins:
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Name
            self._table.setItem(row, 0, QTableWidgetItem(pin.name))

            # Pin number
            pin_spin = QSpinBox()
            pin_spin.setRange(0, 99)
            pin_spin.setValue(pin.pin_number if pin.pin_number is not None else 0)
            self._table.setCellWidget(row, 1, pin_spin)

            # Type
            type_combo = QComboBox()
            for pt in PinType:
                type_combo.addItem(pt.value, pt)
            type_combo.setCurrentText(pin.pin_type.value)
            self._table.setCellWidget(row, 2, type_combo)

            # Default value
            default_str = ""
            if pin.default_value is not None:
                default_str = str(pin.default_value)
            self._table.setItem(row, 3, QTableWidgetItem(default_str))

            # Description
            self._table.setItem(row, 4, QTableWidgetItem(pin.description))


class CustomDeviceDialog(QDialog):
    """Dialog for creating/editing custom device definitions."""

    def __init__(self, definition: CustomDeviceDefinition | None = None, parent=None):
        super().__init__(parent)
        self._definition = definition or CustomDeviceDefinition()
        self._setup_ui()
        self._load_definition()

    def _setup_ui(self):
        self.setWindowTitle("Custom Device Editor")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)

        # Device info
        info_group = QGroupBox("Device Information")
        info_layout = QFormLayout(info_group)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g., Motor Governor")
        info_layout.addRow("Name:", self._name_edit)

        self._desc_edit = QTextEdit()
        self._desc_edit.setMaximumHeight(60)
        self._desc_edit.setPlaceholderText("Description of the device")
        info_layout.addRow("Description:", self._desc_edit)

        layout.addWidget(info_group)

        # Pins section
        pins_group = QGroupBox("Pin Definitions")
        pins_layout = QVBoxLayout(pins_group)
        self._pin_editor = PinEditorWidget()
        pins_layout.addWidget(self._pin_editor)
        layout.addWidget(pins_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_definition(self):
        """Load the definition into the UI."""
        self._name_edit.setText(self._definition.name)
        self._desc_edit.setPlainText(self._definition.description)
        self._pin_editor.set_pins(self._definition.pins)

    def _on_accept(self):
        """Handle OK button."""
        # Validate
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Device name is required.")
            return

        pins = self._pin_editor.get_pins()
        if not pins:
            QMessageBox.warning(self, "Validation Error", "At least one pin is required.")
            return

        # Build definition
        self._definition.name = name
        self._definition.description = self._desc_edit.toPlainText()
        self._definition.pins = pins

        self.accept()

    def get_definition(self) -> CustomDeviceDefinition:
        """Get the custom device definition."""
        return self._definition
