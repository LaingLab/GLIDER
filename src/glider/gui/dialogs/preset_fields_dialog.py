"""Edit the fields a preset asks for before each run (and, for a new preset, its name)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from glider.core.experiment_session import RunField
from glider.core.presets import derive_keys, validate_labels

_KEY_ROLE = Qt.ItemDataRole.UserRole
# The run page shows these metadata tags beside run fields, so a field may not use them as a key.
_RESERVED_KEYS = {"experimenter", "notes", "protocol"}


class PresetFieldsDialog(QDialog):
    def __init__(self, fields: list[RunField], name: str = "", ask_name: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preset fields" if not ask_name else "New preset")
        self.setMinimumWidth(520)
        col = QVBoxLayout(self)

        self._name_edit: QLineEdit | None = None
        if ask_name:
            col.addWidget(QLabel("Preset name"))
            self._name_edit = QLineEdit(name)
            self._name_edit.setPlaceholderText("e.g. Pup retrieval")
            self._name_edit.textChanged.connect(self._validate)
            col.addWidget(self._name_edit)

        col.addWidget(QLabel("Fields to fill in before each run"))
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Label", "Required", "Default"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().hide()
        self._table.itemChanged.connect(self._validate)
        col.addWidget(self._table)

        buttons = QHBoxLayout()
        self._add_btn = QPushButton("Add field")
        self._remove_btn = QPushButton("Remove")
        self._up_btn = QPushButton("Move up")
        self._down_btn = QPushButton("Move down")
        self._add_btn.clicked.connect(lambda: self._append(RunField("", "")))
        self._remove_btn.clicked.connect(self._remove)
        self._up_btn.clicked.connect(lambda: self._move(-1))
        self._down_btn.clicked.connect(lambda: self._move(1))
        for b in (self._add_btn, self._remove_btn, self._up_btn, self._down_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        col.addLayout(buttons)

        self._error = QLabel()
        self._error.setProperty("textRole", "warning")
        col.addWidget(self._error)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        self._ok_btn = box.button(QDialogButtonBox.StandardButton.Ok)
        col.addWidget(box)

        for f in fields:
            self._append(f)
        self._validate()

    # --- rows ---

    def _append(self, f: RunField) -> None:
        self._table.blockSignals(True)
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._write_row(row, f)
        self._table.blockSignals(False)
        self._validate()

    def _write_row(self, row: int, f: RunField) -> None:
        label = QTableWidgetItem(f.label)
        label.setData(_KEY_ROLE, f.key)
        req = QTableWidgetItem()
        req.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        req.setCheckState(Qt.CheckState.Checked if f.required else Qt.CheckState.Unchecked)
        self._table.setItem(row, 0, label)
        self._table.setItem(row, 1, req)
        self._table.setItem(row, 2, QTableWidgetItem(f.default))

    def _read_row(self, row: int) -> RunField:
        label = self._table.item(row, 0)
        return RunField(
            key=label.data(_KEY_ROLE) or "",
            label=label.text().strip(),
            required=self._table.item(row, 1).checkState() == Qt.CheckState.Checked,
            default=self._table.item(row, 2).text().strip(),
        )

    def _remove(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)
            self._validate()

    def _move(self, delta: int) -> None:
        row = self._table.currentRow()
        target = row + delta
        if row < 0 or not 0 <= target < self._table.rowCount():
            return
        a, b = self._read_row(row), self._read_row(target)
        self._table.blockSignals(True)
        self._write_row(row, b)
        self._write_row(target, a)
        self._table.blockSignals(False)
        self._table.setCurrentCell(target, 0)

    # --- results ---

    def fields(self) -> list[RunField]:
        rows = [self._read_row(r) for r in range(self._table.rowCount())]
        fresh = iter(derive_keys([f.label for f in rows if not f.key]))
        taken = {f.key for f in rows if f.key}
        out = []
        for f in rows:
            key = f.key
            if not key:
                key = next(fresh)
                if key in _RESERVED_KEYS:
                    key += "_field"
                base, n = key, 2
                while key in taken:
                    key = f"{base}_{n}"
                    n += 1
                taken.add(key)
            out.append(RunField(key, f.label, f.required, f.default))
        return out

    def name(self) -> str:
        return self._name_edit.text().strip() if self._name_edit is not None else ""

    def _validate(self, *_args) -> None:
        if not hasattr(self, "_ok_btn"):
            return
        error = validate_labels([self._read_row(r).label for r in range(self._table.rowCount())])
        if error is None and self._name_edit is not None and not self.name():
            error = "Give the preset a name."
        self._error.setText(error or "")
        self._ok_btn.setEnabled(error is None)
