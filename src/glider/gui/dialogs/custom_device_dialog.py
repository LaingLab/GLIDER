"""
Custom Device Builder dialog.

A no-code GUI for authoring a declarative ``.gdevice`` device: pick a transport
(I2C or GPIO), name it, and define named actions that map to primitive ops. On
save the definition is written to the device library and registered, so it
appears in Add Device. See ``glider.hal.declarative_device``.
"""

import logging

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
)

from glider.hal.declarative_device import WRITE_VALUE_OPS, standard_settings

logger = logging.getLogger(__name__)

# (label, op) per transport.
_I2C_OPS = [
    ("Read byte", "read_byte"),
    ("Read word (16-bit)", "read_word"),
    ("Write byte", "write_byte"),
    ("Write word (16-bit)", "write_word"),
]
_GPIO_OPS = [
    ("Set HIGH", "set_high"),
    ("Set LOW", "set_low"),
    ("Read digital", "read_digital"),
    ("Read analog", "read_analog"),
    ("Write PWM", "write_pwm"),
]

_COL_NAME, _COL_OP, _COL_REG, _COL_PRIMARY = range(4)


class CustomDeviceDialog(QDialog):
    """Dialog to build and save a declarative custom device."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Custom Device")
        self.setMinimumWidth(560)
        self.device_name = ""

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g., MyTempSensor")
        form.addRow("Name:", self._name_edit)

        self._desc_edit = QLineEdit()
        self._desc_edit.setPlaceholderText("Optional description")
        form.addRow("Description:", self._desc_edit)

        self._transport_combo = QComboBox()
        self._transport_combo.addItem("I2C", "i2c")
        self._transport_combo.addItem("GPIO", "gpio")
        self._transport_combo.currentIndexChanged.connect(self._on_transport_changed)
        form.addRow("Transport:", self._transport_combo)
        layout.addLayout(form)

        layout.addWidget(QLabel("Actions:"))
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Action Name", "Operation", "Register", "Primary"])
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_NAME, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self._table)

        row_btns = QHBoxLayout()
        add_btn = QPushButton("Add Action")
        add_btn.clicked.connect(self._add_row)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        row_btns.addWidget(add_btn)
        row_btns.addWidget(remove_btn)
        row_btns.addStretch()
        layout.addLayout(row_btns)

        hint = QLabel(
            "I2C ops use the Register column; write ops take their value at run time "
            "(from a Device Action arg). GPIO ops act on the device's pin setting."
        )
        hint.setWordWrap(True)
        hint.setProperty("textRole", "muted")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._add_row()  # start with one action row

    # --- table helpers ---

    def _current_transport(self) -> str:
        return self._transport_combo.currentData()

    def _op_list(self) -> list[tuple[str, str]]:
        return _I2C_OPS if self._current_transport() == "i2c" else _GPIO_OPS

    def _add_row(self) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setCellWidget(row, _COL_NAME, QLineEdit())

        op_combo = QComboBox()
        for label, op in self._op_list():
            op_combo.addItem(label, op)
        op_combo.currentIndexChanged.connect(lambda _i, r=row: self._sync_register_enabled(r))
        self._table.setCellWidget(row, _COL_OP, op_combo)

        reg_spin = QSpinBox()
        reg_spin.setDisplayIntegerBase(16)
        reg_spin.setPrefix("0x")
        reg_spin.setRange(0x00, 0xFF)
        self._table.setCellWidget(row, _COL_REG, reg_spin)

        primary = QCheckBox()
        self._table.setCellWidget(row, _COL_PRIMARY, primary)

        self._sync_register_enabled(row)

    def _remove_selected(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)

    def _sync_register_enabled(self, row: int) -> None:
        """Register only applies to I2C ops."""
        reg = self._table.cellWidget(row, _COL_REG)
        if reg is not None:
            reg.setEnabled(self._current_transport() == "i2c")

    def _on_transport_changed(self) -> None:
        # Ops differ per transport, so reset the action rows.
        self._table.setRowCount(0)
        self._add_row()

    # --- build / save ---

    def _build_definition(self) -> dict:
        transport = self._current_transport()
        actions = []
        for row in range(self._table.rowCount()):
            name = self._table.cellWidget(row, _COL_NAME).text().strip()
            op = self._table.cellWidget(row, _COL_OP).currentData()
            primary = self._table.cellWidget(row, _COL_PRIMARY).isChecked()
            action = {"name": name, "op": op}
            if transport == "i2c":
                action["params"] = {"register": self._table.cellWidget(row, _COL_REG).value()}
            if op in WRITE_VALUE_OPS:
                action["runtime_args"] = ["value"]
            if primary:
                action["primary"] = True
            actions.append(action)

        return {
            "schema_version": "1.0",
            "name": self._name_edit.text().strip(),
            "description": self._desc_edit.text().strip(),
            "transport": transport,
            "settings": standard_settings(transport),
            "actions": actions,
        }

    def _on_accept(self) -> None:
        from glider.core.config import get_config
        from glider.core.device_library import register_definition, save_definition

        definition = self._build_definition()
        try:
            devices_dir = get_config().paths.devices_dir
            save_definition(definition, devices_dir)  # validates, then writes
            register_definition(definition)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid device", str(e))
            return
        except Exception as e:  # noqa: BLE001 - surfaced to user
            QMessageBox.critical(self, "Could not save device", str(e))
            return

        self.device_name = definition["name"]
        self.accept()
