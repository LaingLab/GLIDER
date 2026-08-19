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

from glider.core.config import get_config
from glider.core.vocabulary import Vocabulary, load, save

if TYPE_CHECKING:
    from glider.core.experiment_session import Subject

logger = logging.getLogger(__name__)

# Common values for dropdowns. The leading "" is the "nothing chosen" row, a
# combo-box affordance rather than a term any lab uses, which is why
# glider.core.vocabulary's defaults omit it. A test pins the two together.
SEX_OPTIONS = ["", "Male", "Female", "Unknown"]
ROUTE_OPTIONS = ["", "IP", "IV", "PO", "SC", "IM", "Topical", "Inhalation", "Other"]

#: Which Subject attribute feeds which vocabulary list, for learn-on-save.
_LEARNED_FIELDS = (
    ("groups", "group"),
    ("strains", "strain"),
    ("solutions", "solution"),
    ("routes", "route"),
    ("sexes", "sex"),
)


class SubjectDialog(QDialog):
    """
    Dialog for creating and editing experiment subjects.

    Provides a tabbed interface with:
    - Basic Info (ID, name, group)
    - Biological (age, sex, weight, strain)
    - Solution/Drug (solution, concentration, dose, route)
    - Notes

    Group, strain, solution, route and sex are offered from the lab's
    vocabulary (:mod:`glider.core.vocabulary`) so that ``Control`` and
    ``control`` do not become two treatment groups. All five stay editable: a
    term the lab has not defined can always be typed, and is learned for the
    next subject.

    All five go through :meth:`_vocabulary_combo` deliberately. A field built
    inline instead once looked up its stored value with ``findText`` and showed
    a blank when it did not match, so opening and saving a subject recorded
    with a term the lab had since removed silently erased it.
    """

    def __init__(
        self,
        subject: Optional["Subject"] = None,
        parent: QWidget | None = None,
        is_touch_mode: bool = False,
        vocabulary: Vocabulary | None = None,
    ):
        super().__init__(parent)
        self._subject = subject
        self._is_touch_mode = is_touch_mode
        self._is_new = subject is None
        self._library_dir = get_config().paths.library_dir
        self._vocabulary = vocabulary if vocabulary is not None else load(self._library_dir)

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

        layout.addWidget(button_box)

    def _vocabulary_combo(self, list_name: str, placeholder: str) -> QComboBox:
        """An editable combo offering ``list_name``, starting on nothing chosen.

        The leading blank matters: without it the combo would open showing the
        lab's first group and every new animal would silently be labelled with
        it.
        """
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(["", *self._vocabulary.get(list_name)])
        combo.setCurrentIndex(0)
        if line_edit := combo.lineEdit():
            line_edit.setPlaceholderText(placeholder)
        return combo

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
        self._group_combo = self._vocabulary_combo("groups", "e.g., Control, Drug A")
        form.addRow("Group:", self._group_combo)

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
        self._sex_combo = self._vocabulary_combo("sexes", "e.g., Male, Female")
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
        self._strain_combo = self._vocabulary_combo("strains", "e.g., C57BL/6J")
        form.addRow("Strain:", self._strain_combo)

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
        self._solution_combo = self._vocabulary_combo("solutions", "e.g., Saline, Drug X")
        form.addRow("Solution:", self._solution_combo)

        # Concentration
        self._concentration_edit = QLineEdit()
        self._concentration_edit.setPlaceholderText("e.g., 10 mg/mL")
        form.addRow("Concentration:", self._concentration_edit)

        # Dose
        self._dose_edit = QLineEdit()
        self._dose_edit.setPlaceholderText("e.g., 5 mg/kg")
        form.addRow("Dose:", self._dose_edit)

        # Route of administration
        self._route_combo = self._vocabulary_combo("routes", "e.g., IP, SC")
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
        self._group_combo.setCurrentText(subject.group)

        # Parse age (e.g., "8 weeks" -> "8", "weeks")
        if subject.age:
            parts = subject.age.split()
            if len(parts) >= 1:
                self._age_edit.setText(parts[0])
            if len(parts) >= 2:
                idx = self._age_unit_combo.findText(parts[1])
                if idx >= 0:
                    self._age_unit_combo.setCurrentIndex(idx)

        self._sex_combo.setCurrentText(subject.sex)

        # Parse weight (e.g., "25.5 g" -> "25.5", "g")
        if subject.weight:
            parts = subject.weight.split()
            if len(parts) >= 1:
                self._weight_edit.setText(parts[0])
            if len(parts) >= 2:
                idx = self._weight_unit_combo.findText(parts[1])
                if idx >= 0:
                    self._weight_unit_combo.setCurrentIndex(idx)

        # Editable combos: setCurrentText selects a matching entry, and falls
        # back to putting the raw text in the line edit. That fallback is what
        # keeps a .glider file written before the vocabulary existed loading
        # unchanged.
        self._strain_combo.setCurrentText(subject.strain)
        self._solution_combo.setCurrentText(subject.solution)
        self._concentration_edit.setText(subject.concentration)
        self._dose_edit.setText(subject.dose)
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

        # Every field is read once, here, so the update and create branches
        # below cannot drift apart. They did once: a field converted to a combo
        # in only one branch silently stops persisting in the other, and only
        # for the half of users who edit rather than create.
        values = {
            "subject_id": self._subject_id_edit.text().strip(),
            "name": self._name_edit.text().strip(),
            "group": self._group_combo.currentText().strip(),
            "age": age,
            "sex": self._sex_combo.currentText().strip(),
            "weight": weight,
            "strain": self._strain_combo.currentText().strip(),
            "solution": self._solution_combo.currentText().strip(),
            "concentration": self._concentration_edit.text().strip(),
            "dose": self._dose_edit.text().strip(),
            "route": self._route_combo.currentText().strip(),
            "notes": self._notes_edit.toPlainText(),
        }

        # Create or update subject
        if self._subject and not self._is_new:
            subject = self._subject
            for attribute, value in values.items():
                setattr(subject, attribute, value)
        else:
            subject = Subject(**values)

        self._learn_vocabulary(subject)
        return subject

    def _learn_vocabulary(self, subject: "Subject") -> None:
        """Teach the lab's vocabulary any term this subject introduced.

        The file is rewritten only when something was actually new -- that is
        what ``Vocabulary.add``'s bool return is for. A failed write is logged
        and dropped: losing a vocabulary entry must never cost the user the
        subject they just typed.
        """
        learned = False
        for list_name, attribute in _LEARNED_FIELDS:
            if self._vocabulary.add(list_name, getattr(subject, attribute)):
                learned = True

        if not learned:
            return

        if not save(self._vocabulary, self._library_dir):
            logger.warning("Could not record new vocabulary terms from subject %s", subject.id)
