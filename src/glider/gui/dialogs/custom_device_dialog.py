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
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
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


class CustomDeviceDialog(QDialog):
    """Dialog to build and save a declarative custom device."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Custom Device")
        self.setMinimumSize(720, 560)
        self.resize(760, 640)
        self.device_name = ""
        self._action_rows: list[dict] = []  # one dict of widgets per action

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

        # Column headers above the action rows.
        header = QHBoxLayout()
        for text, width in (("Name", None), ("Operation", 180), ("Register", 90), ("Primary", 70)):
            lbl = QLabel(text)
            lbl.setProperty("textRole", "muted")
            if width:
                lbl.setFixedWidth(width)
            header.addWidget(lbl, 1 if width is None else 0)
        header.addSpacing(34)  # space above the remove button
        layout.addLayout(header)

        # Scrollable list of action rows (each row is a real widget, so styled
        # editors size to their natural height with no clipping).
        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self._rows_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_container)
        layout.addWidget(scroll, stretch=1)

        add_btn = QPushButton("Add Action")
        add_btn.clicked.connect(self._add_row)
        add_row = QHBoxLayout()
        add_row.addWidget(add_btn)
        add_row.addStretch()
        layout.addLayout(add_row)

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

    # --- helpers ---

    def _current_transport(self) -> str:
        return self._transport_combo.currentData()

    def _op_list(self) -> list[tuple[str, str]]:
        return _I2C_OPS if self._current_transport() == "i2c" else _GPIO_OPS

    def _add_row(self) -> None:
        row_widget = QWidget()
        h = QHBoxLayout(row_widget)
        h.setContentsMargins(0, 0, 0, 0)

        name = QLineEdit()
        name.setPlaceholderText("Action name")

        op = QComboBox()
        for label, op_key in self._op_list():
            op.addItem(label, op_key)
        op.setFixedWidth(180)

        reg = QSpinBox()
        reg.setDisplayIntegerBase(16)
        reg.setPrefix("0x")
        reg.setRange(0x00, 0xFF)
        reg.setFixedWidth(90)

        primary = QCheckBox()
        primary.setFixedWidth(70)

        remove = QPushButton("✕")
        remove.setFixedWidth(34)
        remove.setToolTip("Remove this action")

        h.addWidget(name, 1)
        h.addWidget(op)
        h.addWidget(reg)
        h.addWidget(primary)
        h.addWidget(remove)

        entry = {"widget": row_widget, "name": name, "op": op, "reg": reg, "primary": primary}
        op.currentIndexChanged.connect(lambda _i, e=entry: self._sync_register_enabled(e))
        remove.clicked.connect(lambda _c, e=entry: self._remove_row(e))

        # Insert before the trailing stretch so rows stack from the top.
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row_widget)
        self._action_rows.append(entry)
        self._sync_register_enabled(entry)

    def _remove_row(self, entry: dict) -> None:
        if entry not in self._action_rows:
            return
        self._action_rows.remove(entry)
        entry["widget"].setParent(None)
        entry["widget"].deleteLater()

    def _sync_register_enabled(self, entry: dict) -> None:
        """Register only applies to I2C ops."""
        entry["reg"].setEnabled(self._current_transport() == "i2c")

    def _on_transport_changed(self) -> None:
        # Ops differ per transport, so reset the action rows.
        for entry in list(self._action_rows):
            self._remove_row(entry)
        self._add_row()

    # --- build / save ---

    def _build_definition(self) -> dict:
        transport = self._current_transport()
        actions = []
        for entry in self._action_rows:
            op = entry["op"].currentData()
            action = {"name": entry["name"].text().strip(), "op": op}
            if transport == "i2c":
                action["params"] = {"register": entry["reg"].value()}
            if op in WRITE_VALUE_OPS:
                action["runtime_args"] = ["value"]
            if entry["primary"].isChecked():
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
