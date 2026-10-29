"""Choose how to cut the session into time bins for the binned export."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
)

__all__ = ["BIN_PRESETS_MIN", "SHEETS", "TimeBinsDialog"]

#: Offered in the bin box's drop-down; any other width can be typed.
BIN_PRESETS_MIN = (1.0, 3.0, 5.0, 10.0)

WHOLE, SELECTION = "whole", "selection"
#: The layout of a workbook with a sheet per measure, in place of tidy or wide.
SHEETS = "sheets"


class TimeBinsDialog(QDialog):
    """Bin width, the span to bin, and the file's shape."""

    def __init__(
        self, minutes: float, *, has_selection: bool, per_sheet: bool = False, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Export time bins")
        form = QFormLayout(self)
        self.preset = QComboBox()
        for value in BIN_PRESETS_MIN:
            self.preset.addItem(f"{value:g} min", value)
        self.preset.addItem("Custom", None)
        self.bin_width = QDoubleSpinBox()
        self.bin_width.setRange(0.05, 24 * 60.0)
        self.bin_width.setDecimals(2)
        self.bin_width.setSingleStep(1.0)
        self.bin_width.setSuffix(" min")
        self.bin_width.setValue(minutes)
        self.span = QComboBox()
        self.span.addItem("Whole session", WHOLE)
        self.span.addItem("Selected range", SELECTION)
        if has_selection:
            self.span.setCurrentIndex(1)
        else:
            # Disabled rather than hidden: the option is the hint that a
            # selection narrows the export.
            self.span.model().item(1).setEnabled(False)
        self.shape = QComboBox()
        self.shape.addItem("Tidy: a row per session per bin", "tidy")
        self.shape.addItem("Wide: a row per session", "wide")
        self.per_sheet = QCheckBox("One sheet per measure (Excel workbook)")
        self.per_sheet.setToolTip(
            "Each behaviour, zone and distance measure on its own sheet: "
            "a row per session, a column per bin."
        )
        self.per_sheet.toggled.connect(lambda on: self.shape.setEnabled(not on))
        self.per_sheet.setChecked(per_sheet)
        self._sync_preset(minutes)
        self.preset.activated.connect(self._preset_picked)
        self.bin_width.valueChanged.connect(self._sync_preset)
        form.addRow("Bin width", self.preset)
        form.addRow("", self.bin_width)
        form.addRow("Span", self.span)
        form.addRow("Layout", self.shape)
        form.addRow("", self.per_sheet)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _preset_picked(self, index: int) -> None:
        value = self.preset.itemData(index)
        if value is not None:
            self.bin_width.setValue(value)

    def _sync_preset(self, minutes: float) -> None:
        index = self.preset.findData(round(float(minutes), 2))
        self.preset.setCurrentIndex(index if index >= 0 else self.preset.count() - 1)

    def bin_seconds(self) -> float:
        return self.bin_width.value() * 60.0

    def whole_session(self) -> bool:
        return self.span.currentData() == WHOLE

    def layout_shape(self) -> str:
        return SHEETS if self.per_sheet.isChecked() else self.shape.currentData()
