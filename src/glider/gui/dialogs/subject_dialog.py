"""
Subject Dialog - Create and edit experiment subjects.

Provides a tabbed dialog for entering subject/animal information
including biological data and solution/drug details.
"""

import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.core.experiment_session import Subject

logger = logging.getLogger(__name__)

# Common values for dropdowns
SEX_OPTIONS = ["", "Male", "Female", "Unknown"]
ROUTE_OPTIONS = ["", "IP", "IV", "PO", "SC", "IM", "Topical", "Inhalation", "Other"]


class SubjectDialog(QDialog):
    """
    Dialog for creating and editing experiment subjects.

    Provides a tabbed interface with:
    - Basic Info (ID, name, group)
    - Biological (age, sex, weight, strain)
    - Solution/Drug (solution, concentration, dose, route)
    - Notes
    """

    def __init__(
        self,
        subject: Optional["Subject"] = None,
        parent: QWidget | None = None,
        is_touch_mode: bool = False,
    ):
        super().__init__(parent)
        self._subject = subject
        self._is_touch_mode = is_touch_mode
        self._is_new = subject is None

        self._setup_ui()

        if subject:
            self._load_subject(subject)

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        title = "Add Subject" if self._is_new else "Edit Subject"
        self.setWindowTitle(title)
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        if self._is_touch_mode:
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

        # Tab widget
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # Basic Info tab
        basic_tab = self._create_basic_tab()
        self._tabs.addTab(basic_tab, "Basic")

        # Biological tab
        bio_tab = self._create_bio_tab()
        self._tabs.addTab(bio_tab, "Biological")

        # Solution/Drug tab
        solution_tab = self._create_solution_tab()
        self._tabs.addTab(solution_tab, "Solution")

        # Notes tab
        notes_tab = self._create_notes_tab()
        self._tabs.addTab(notes_tab, "Notes")

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        if self._is_touch_mode:
            for button in button_box.buttons():
                button.setMinimumHeight(44)
                button.setStyleSheet("font-size: 14px; padding: 8px 16px;")

        layout.addWidget(button_box)

    def _create_basic_tab(self) -> QWidget:
        """Create the basic info tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

        # Basic info group
        group = QGroupBox("Basic Information")
        form = QFormLayout(group)

        if self._is_touch_mode:
            form.setSpacing(12)
            form.setContentsMargins(12, 20, 12, 12)

        # Subject ID (required)
        self._subject_id_edit = QLineEdit()
        self._subject_id_edit.setPlaceholderText("e.g., M001 (required)")
        form.addRow("Subject ID:", self._subject_id_edit)

        # Name
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g., Mouse 1")
        form.addRow("Name:", self._name_edit)

        # Group/Treatment
        self._group_edit = QLineEdit()
        self._group_edit.setPlaceholderText("e.g., Control, Drug A")
        form.addRow("Group:", self._group_edit)

        layout.addWidget(group)
        layout.addStretch()

        return widget

    def _create_bio_tab(self) -> QWidget:
        """Create the biological info tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

        # Biological info group
        group = QGroupBox("Biological Information")
        form = QFormLayout(group)

        if self._is_touch_mode:
            form.setSpacing(12)
            form.setContentsMargins(12, 20, 12, 12)

        # Age
        age_layout = QHBoxLayout()
        self._age_edit = QLineEdit()
        self._age_edit.setPlaceholderText("e.g., 8")
        age_layout.addWidget(self._age_edit)

        self._age_unit_combo = QComboBox()
        self._age_unit_combo.addItems(["weeks", "days", "months", "years"])
        self._age_unit_combo.setFixedWidth(80)
        age_layout.addWidget(self._age_unit_combo)

        form.addRow("Age:", age_layout)

        # Sex
        self._sex_combo = QComboBox()
        self._sex_combo.addItems(SEX_OPTIONS)
        form.addRow("Sex:", self._sex_combo)

        # Weight
        weight_layout = QHBoxLayout()
        self._weight_edit = QLineEdit()
        self._weight_edit.setPlaceholderText("e.g., 25.5")
        weight_layout.addWidget(self._weight_edit)

        self._weight_unit_combo = QComboBox()
        self._weight_unit_combo.addItems(["g", "kg", "mg", "lb", "oz"])
        self._weight_unit_combo.setFixedWidth(60)
        weight_layout.addWidget(self._weight_unit_combo)

        form.addRow("Weight:", weight_layout)

        # Strain/Genotype
        self._strain_edit = QLineEdit()
        self._strain_edit.setPlaceholderText("e.g., C57BL/6J")
        form.addRow("Strain:", self._strain_edit)

        layout.addWidget(group)
        layout.addStretch()

        return widget

    def _create_solution_tab(self) -> QWidget:
        """Create the solution/drug tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

        # Solution info group
        group = QGroupBox("Solution / Drug Information")
        form = QFormLayout(group)

        if self._is_touch_mode:
            form.setSpacing(12)
            form.setContentsMargins(12, 20, 12, 12)

        # Solution name
        self._solution_edit = QLineEdit()
        self._solution_edit.setPlaceholderText("e.g., Saline, Drug X")
        form.addRow("Solution:", self._solution_edit)

        # Concentration
        self._concentration_edit = QLineEdit()
        self._concentration_edit.setPlaceholderText("e.g., 10 mg/mL")
        form.addRow("Concentration:", self._concentration_edit)

        # Dose
        self._dose_edit = QLineEdit()
        self._dose_edit.setPlaceholderText("e.g., 5 mg/kg")
        form.addRow("Dose:", self._dose_edit)

        # Route of administration
        self._route_combo = QComboBox()
        self._route_combo.setEditable(True)
        self._route_combo.addItems(ROUTE_OPTIONS)
        form.addRow("Route:", self._route_combo)

        layout.addWidget(group)
        layout.addStretch()

        return widget

    def _create_notes_tab(self) -> QWidget:
        """Create the notes tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        if self._is_touch_mode:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

        # Notes group
        group = QGroupBox("Notes")
        group_layout = QVBoxLayout(group)

        if self._is_touch_mode:
            group_layout.setContentsMargins(12, 20, 12, 12)

        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText("Additional notes about this subject...")
        group_layout.addWidget(self._notes_edit)

        layout.addWidget(group)

        return widget

    def _load_subject(self, subject: "Subject") -> None:
        """Load subject data into the form."""
        self._subject_id_edit.setText(subject.subject_id)
        self._name_edit.setText(subject.name)
        self._group_edit.setText(subject.group)

        # Parse age (e.g., "8 weeks" -> "8", "weeks")
        if subject.age:
            parts = subject.age.split()
            if len(parts) >= 1:
                self._age_edit.setText(parts[0])
            if len(parts) >= 2:
                idx = self._age_unit_combo.findText(parts[1])
                if idx >= 0:
                    self._age_unit_combo.setCurrentIndex(idx)

        # Sex
        idx = self._sex_combo.findText(subject.sex)
        if idx >= 0:
            self._sex_combo.setCurrentIndex(idx)

        # Parse weight (e.g., "25.5 g" -> "25.5", "g")
        if subject.weight:
            parts = subject.weight.split()
            if len(parts) >= 1:
                self._weight_edit.setText(parts[0])
            if len(parts) >= 2:
                idx = self._weight_unit_combo.findText(parts[1])
                if idx >= 0:
                    self._weight_unit_combo.setCurrentIndex(idx)

        self._strain_edit.setText(subject.strain)
        self._solution_edit.setText(subject.solution)
        self._concentration_edit.setText(subject.concentration)
        self._dose_edit.setText(subject.dose)

        # Route
        idx = self._route_combo.findText(subject.route)
        if idx >= 0:
            self._route_combo.setCurrentIndex(idx)
        else:
            self._route_combo.setCurrentText(subject.route)

        self._notes_edit.setPlainText(subject.notes)

    def _on_accept(self) -> None:
        """Handle OK button."""
        # Validate required fields
        subject_id = self._subject_id_edit.text().strip()
        if not subject_id:
            QMessageBox.warning(
                self,
                "Required Field",
                "Subject ID is required.",
            )
            self._tabs.setCurrentIndex(0)
            self._subject_id_edit.setFocus()
            return

        self.accept()

    def get_subject(self) -> "Subject":
        """Get the subject data from the form."""
        from glider.core.experiment_session import Subject

        # Build age string
        age_value = self._age_edit.text().strip()
        age = ""
        if age_value:
            age = f"{age_value} {self._age_unit_combo.currentText()}"

        # Build weight string
        weight_value = self._weight_edit.text().strip()
        weight = ""
        if weight_value:
            weight = f"{weight_value} {self._weight_unit_combo.currentText()}"

        # Create or update subject
        if self._subject and not self._is_new:
            # Update existing subject
            self._subject.subject_id = self._subject_id_edit.text().strip()
            self._subject.name = self._name_edit.text().strip()
            self._subject.group = self._group_edit.text().strip()
            self._subject.age = age
            self._subject.sex = self._sex_combo.currentText()
            self._subject.weight = weight
            self._subject.strain = self._strain_edit.text().strip()
            self._subject.solution = self._solution_edit.text().strip()
            self._subject.concentration = self._concentration_edit.text().strip()
            self._subject.dose = self._dose_edit.text().strip()
            self._subject.route = self._route_combo.currentText()
            self._subject.notes = self._notes_edit.toPlainText()
            return self._subject
        else:
            # Create new subject
            return Subject(
                subject_id=self._subject_id_edit.text().strip(),
                name=self._name_edit.text().strip(),
                group=self._group_edit.text().strip(),
                age=age,
                sex=self._sex_combo.currentText(),
                weight=weight,
                strain=self._strain_edit.text().strip(),
                solution=self._solution_edit.text().strip(),
                concentration=self._concentration_edit.text().strip(),
                dose=self._dose_edit.text().strip(),
                route=self._route_combo.currentText(),
                notes=self._notes_edit.toPlainText(),
            )
