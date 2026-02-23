"""
Experiment Dialog - Manage experiment metadata and subjects.

Provides a dialog for configuring experiment information
and managing subjects/animals for recording sessions.
"""

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.core.experiment_session import ExperimentSession, Subject

logger = logging.getLogger(__name__)

# Common experiment types
EXPERIMENT_TYPES = [
    "",
    "Open Field",
    "Morris Water Maze",
    "Elevated Plus Maze",
    "Rotarod",
    "Novel Object Recognition",
    "Fear Conditioning",
    "Forced Swim Test",
    "Tail Suspension Test",
    "Y-Maze",
    "Barnes Maze",
    "Radial Arm Maze",
    "Social Interaction",
    "Light-Dark Box",
    "Other",
]


class ExperimentDialog(QDialog):
    """
    Dialog for managing experiment metadata and subjects.

    Provides:
    - Experiment info editing (name, protocol, type, experimenter, etc.)
    - Subject table with add/edit/remove functionality
    - Active subject selection for recordings

    Signals:
        metadata_changed: Emitted when any experiment metadata changes
        active_subject_changed: Emitted when the active subject changes
        subject_added: Emitted when a subject is added
        subject_removed: Emitted when a subject is removed
        edit_subject_requested: Emitted when user wants to edit a subject
    """

    metadata_changed = pyqtSignal()
    active_subject_changed = pyqtSignal(str)  # subject_id
    subject_added = pyqtSignal(object)  # Subject
    subject_removed = pyqtSignal(str)  # subject_id
    edit_subject_requested = pyqtSignal(str)  # subject_id
    recording_directory_changed = pyqtSignal(str)  # directory path

    def __init__(
        self,
        session: "ExperimentSession",
        parent: QWidget | None = None,
        is_touch_mode: bool = False,
    ):
        super().__init__(parent)
        self._session = session
        self._is_touch_mode = is_touch_mode
        self._updating = False  # Prevent recursive updates

        self._setup_ui()
        self._load_from_session()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle("Experiment Settings")
        self.setMinimumSize(550, 500)
        self.resize(600, 600)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(12)

        # Experiment Info group
        self._info_group = QGroupBox("Experiment Info")
        info_layout = QFormLayout(self._info_group)
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(12, 20, 12, 12)

        if self._is_touch_mode:
            info_layout.setSpacing(12)

        # Name
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Experiment name")
        info_layout.addRow("Name:", self._name_edit)

        # Protocol
        self._protocol_edit = QLineEdit()
        self._protocol_edit.setPlaceholderText("e.g., OFT-001")
        info_layout.addRow("Protocol:", self._protocol_edit)

        # Experiment Type
        self._type_combo = QComboBox()
        self._type_combo.setEditable(True)
        self._type_combo.addItems(EXPERIMENT_TYPES)
        info_layout.addRow("Type:", self._type_combo)

        # Experimenter
        self._experimenter_edit = QLineEdit()
        self._experimenter_edit.setPlaceholderText("Experimenter name")
        info_layout.addRow("Experimenter:", self._experimenter_edit)

        # Lab
        self._lab_edit = QLineEdit()
        self._lab_edit.setPlaceholderText("Lab name")
        info_layout.addRow("Lab:", self._lab_edit)

        # Project
        self._project_edit = QLineEdit()
        self._project_edit.setPlaceholderText("Project name")
        info_layout.addRow("Project:", self._project_edit)

        # Notes
        self._notes_edit = QTextEdit()
        self._notes_edit.setMaximumHeight(80)
        self._notes_edit.setPlaceholderText("Experiment notes...")
        info_layout.addRow("Notes:", self._notes_edit)

        # Save Directory
        save_dir_layout = QHBoxLayout()
        self._save_dir_edit = QLineEdit()
        self._save_dir_edit.setReadOnly(True)
        self._save_dir_edit.setPlaceholderText("Default (current directory)")
        save_dir_layout.addWidget(self._save_dir_edit)
        self._browse_dir_btn = QPushButton("Browse")
        self._browse_dir_btn.setFixedWidth(80)
        save_dir_layout.addWidget(self._browse_dir_btn)
        info_layout.addRow("Save Dir:", save_dir_layout)

        layout.addWidget(self._info_group)

        # Subjects group
        self._subjects_group = QGroupBox("Subjects")
        subjects_layout = QVBoxLayout(self._subjects_group)
        subjects_layout.setSpacing(8)
        subjects_layout.setContentsMargins(12, 20, 12, 12)

        # Subject table
        self._subject_table = QTableWidget()
        self._subject_table.setColumnCount(5)
        self._subject_table.setHorizontalHeaderLabels(["ID", "Name", "Group", "Sex", "Solution"])
        self._subject_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._subject_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._subject_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._subject_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self._subject_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self._subject_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._subject_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._subject_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._subject_table.setMinimumHeight(150)
        self._subject_table.verticalHeader().setVisible(False)
        subjects_layout.addWidget(self._subject_table)

        # Subject buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self._add_btn = QPushButton("Add")
        self._add_btn.setFixedWidth(70)
        btn_layout.addWidget(self._add_btn)

        self._edit_btn = QPushButton("Edit")
        self._edit_btn.setFixedWidth(70)
        self._edit_btn.setEnabled(False)
        btn_layout.addWidget(self._edit_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setFixedWidth(80)
        self._remove_btn.setEnabled(False)
        btn_layout.addWidget(self._remove_btn)

        btn_layout.addStretch()

        self._set_active_btn = QPushButton("Set Active")
        self._set_active_btn.setEnabled(False)
        btn_layout.addWidget(self._set_active_btn)

        subjects_layout.addLayout(btn_layout)

        # Active subject display
        self._active_frame = QFrame()
        self._active_frame.setStyleSheet("""
            QFrame {
                background-color: #1a3a1a;
                border: 1px solid #2a5a2a;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        active_layout = QHBoxLayout(self._active_frame)
        active_layout.setContentsMargins(10, 6, 10, 6)

        active_label = QLabel("Active:")
        active_label.setStyleSheet("font-weight: bold; color: #8f8;")
        active_layout.addWidget(active_label)

        self._active_subject_label = QLabel("None")
        self._active_subject_label.setStyleSheet("color: #afa;")
        active_layout.addWidget(self._active_subject_label)
        active_layout.addStretch()

        subjects_layout.addWidget(self._active_frame)

        layout.addWidget(self._subjects_group)
        layout.addStretch()

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)

        if self._is_touch_mode:
            for button in button_box.buttons():
                button.setMinimumHeight(44)
                button.setStyleSheet("font-size: 14px; padding: 8px 16px;")

        main_layout.addWidget(button_box)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Experiment info changes
        self._name_edit.textChanged.connect(self._on_info_changed)
        self._protocol_edit.textChanged.connect(self._on_info_changed)
        self._type_combo.currentTextChanged.connect(self._on_info_changed)
        self._experimenter_edit.textChanged.connect(self._on_info_changed)
        self._lab_edit.textChanged.connect(self._on_info_changed)
        self._project_edit.textChanged.connect(self._on_info_changed)
        self._notes_edit.textChanged.connect(self._on_info_changed)

        # Save directory
        self._browse_dir_btn.clicked.connect(self._on_browse_directory)

        # Subject table
        self._subject_table.itemSelectionChanged.connect(self._on_selection_changed)
        self._subject_table.doubleClicked.connect(self._on_edit_subject)

        # Buttons
        self._add_btn.clicked.connect(self._on_add_subject)
        self._edit_btn.clicked.connect(self._on_edit_subject)
        self._remove_btn.clicked.connect(self._on_remove_subject)
        self._set_active_btn.clicked.connect(self._on_set_active)

    def _load_from_session(self) -> None:
        """Load data from the session metadata."""
        self._updating = True
        try:
            metadata = self._session.metadata

            self._name_edit.setText(metadata.name)
            self._protocol_edit.setText(metadata.protocol)

            # Set experiment type in combo
            idx = self._type_combo.findText(metadata.experiment_type)
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)
            else:
                self._type_combo.setCurrentText(metadata.experiment_type)

            self._experimenter_edit.setText(metadata.experimenter)
            self._lab_edit.setText(metadata.lab)
            self._project_edit.setText(metadata.project)
            self._notes_edit.setPlainText(metadata.notes)
            self._save_dir_edit.setText(metadata.recording_directory)

            self._refresh_subject_table()
            self._update_active_display()
        finally:
            self._updating = False

    def _refresh_subject_table(self) -> None:
        """Refresh the subject table from session data."""
        metadata = self._session.metadata
        self._subject_table.setRowCount(len(metadata.subjects))

        for row, subject in enumerate(metadata.subjects):
            # Store subject ID in first column
            id_item = QTableWidgetItem(subject.subject_id)
            id_item.setData(Qt.ItemDataRole.UserRole, subject.id)

            self._subject_table.setItem(row, 0, id_item)
            self._subject_table.setItem(row, 1, QTableWidgetItem(subject.name))
            self._subject_table.setItem(row, 2, QTableWidgetItem(subject.group))
            self._subject_table.setItem(row, 3, QTableWidgetItem(subject.sex))
            self._subject_table.setItem(row, 4, QTableWidgetItem(subject.solution))

            # Highlight active row
            if subject.id == metadata.active_subject_id:
                for col in range(5):
                    item = self._subject_table.item(row, col)
                    if item:
                        item.setBackground(Qt.GlobalColor.darkGreen)

    def _update_active_display(self) -> None:
        """Update the active subject display."""
        subject = self._session.metadata.get_active_subject()
        if subject:
            text = f"{subject.subject_id}"
            if subject.name:
                text += f" - {subject.name}"
            if subject.group:
                text += f" ({subject.group})"
            self._active_subject_label.setText(text)
        else:
            self._active_subject_label.setText("None")

    def _on_info_changed(self) -> None:
        """Handle experiment info changes."""
        if self._updating:
            return

        metadata = self._session.metadata
        metadata.name = self._name_edit.text()
        metadata.protocol = self._protocol_edit.text()
        metadata.experiment_type = self._type_combo.currentText()
        metadata.experimenter = self._experimenter_edit.text()
        metadata.lab = self._lab_edit.text()
        metadata.project = self._project_edit.text()
        metadata.notes = self._notes_edit.toPlainText()

        self._session._dirty = True
        self.metadata_changed.emit()

    def _on_browse_directory(self) -> None:
        """Handle browse directory button."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory",
            self._save_dir_edit.text(),
        )
        if directory:
            self._save_dir_edit.setText(directory)
            self._session.metadata.recording_directory = directory
            self._session._dirty = True
            self.recording_directory_changed.emit(directory)

    def _on_selection_changed(self) -> None:
        """Handle subject table selection changes."""
        selected = self._subject_table.selectedItems()
        has_selection = len(selected) > 0

        self._edit_btn.setEnabled(has_selection)
        self._remove_btn.setEnabled(has_selection)
        self._set_active_btn.setEnabled(has_selection)

    def _on_add_subject(self) -> None:
        """Handle add subject button."""
        # Emit signal to let parent open the dialog
        self.edit_subject_requested.emit("")

    def _on_edit_subject(self) -> None:
        """Handle edit subject button or double-click."""
        selected_row = self._subject_table.currentRow()
        if selected_row < 0:
            return

        item = self._subject_table.item(selected_row, 0)
        if item:
            subject_id = item.data(Qt.ItemDataRole.UserRole)
            self.edit_subject_requested.emit(subject_id)

    def _on_remove_subject(self) -> None:
        """Handle remove subject button."""
        selected_row = self._subject_table.currentRow()
        if selected_row < 0:
            return

        item = self._subject_table.item(selected_row, 0)
        if not item:
            return

        subject_id = item.data(Qt.ItemDataRole.UserRole)
        subject = self._session.metadata.get_subject(subject_id)

        if subject:
            reply = QMessageBox.question(
                self,
                "Remove Subject",
                f"Remove subject '{subject.subject_id}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._session.metadata.remove_subject(subject_id)
                self._session._dirty = True
                self._refresh_subject_table()
                self._update_active_display()
                self.subject_removed.emit(subject_id)

    def _on_set_active(self) -> None:
        """Handle set active subject button."""
        selected_row = self._subject_table.currentRow()
        if selected_row < 0:
            return

        item = self._subject_table.item(selected_row, 0)
        if item:
            subject_id = item.data(Qt.ItemDataRole.UserRole)
            if self._session.metadata.set_active_subject(subject_id):
                self._session._dirty = True
                self._refresh_subject_table()
                self._update_active_display()
                self.active_subject_changed.emit(subject_id)

    def add_subject(self, subject: "Subject") -> None:
        """Add a subject to the session."""
        self._session.metadata.add_subject(subject)
        self._session._dirty = True
        self._refresh_subject_table()
        self._update_active_display()
        self.subject_added.emit(subject)

    def update_subject(self, subject: "Subject") -> None:
        """Update an existing subject."""
        # Find and replace the subject
        metadata = self._session.metadata
        for i, s in enumerate(metadata.subjects):
            if s.id == subject.id:
                metadata.subjects[i] = subject
                self._session._dirty = True
                self._refresh_subject_table()
                self._update_active_display()
                return

    def refresh(self) -> None:
        """Refresh the dialog from session data."""
        self._load_from_session()

    def set_session(self, session: "ExperimentSession") -> None:
        """Set a new session."""
        self._session = session
        self._load_from_session()
