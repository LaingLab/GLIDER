"""Edit the cohort's retrieval protocol: cutoff, nose keypoint, thresholds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
)

from glider.analysis.behavior.retrieval_setup import RetrievalProtocol

__all__ = ["ProtocolDialog"]


class ProtocolDialog(QDialog):
    def __init__(
        self, protocol: RetrievalProtocol, keypoints: list[str], folder: Path, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Retrieval thresholds — {Path(folder).name}")
        self._base = protocol
        form = QFormLayout(self)
        self.cutoff = QSpinBox()
        self.cutoff.setRange(1, 24 * 3600)
        self.cutoff.setSuffix(" s")
        self.cutoff.setValue(int(round(protocol.cutoff_s)))
        # Editable, and the saved name offered even when this session's poses
        # lack it: the protocol is the cohort's, not this video's.
        self.nose = QComboBox()
        self.nose.setEditable(True)
        names = list(keypoints)
        if protocol.nose_keypoint not in names:
            names.append(protocol.nose_keypoint)
        self.nose.addItems(names)
        self.nose.setCurrentText(protocol.nose_keypoint)
        self.in_nest = self._spin(protocol.in_nest_s, " s")
        self.contact = self._spin(protocol.contact_cm, " cm")
        self.carry = self._spin(protocol.carry_speed_cm_s, " cm/s")
        form.addRow("Cutoff after trial start", self.cutoff)
        form.addRow("Nose keypoint", self.nose)
        form.addRow("In nest for", self.in_nest)
        form.addRow("Contact within", self.contact)
        form.addRow("Carried faster than", self.carry)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _spin(value: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 1000.0)
        spin.setDecimals(2)
        spin.setSuffix(suffix)
        spin.setValue(value)
        return spin

    def protocol(self) -> RetrievalProtocol:
        """The edited protocol; unknown keys of the original ride along."""
        return replace(
            self._base,
            cutoff_s=float(self.cutoff.value()),
            # A cleared box is not a keypoint name; keep the one it had.
            nose_keypoint=self.nose.currentText().strip() or self._base.nose_keypoint,
            in_nest_s=self.in_nest.value(),
            contact_cm=self.contact.value(),
            carry_speed_cm_s=self.carry.value(),
        )
