"""The run page's title block: experiment title, run label, state pill, field tags.

Owns the binding between the tags and ``session.metadata`` that
RunControlPanel used to hold. Read-only while a run is live.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from glider.gui.dashboard.panels.field_tags import FieldTags, TagField

_LIVE = ("RUNNING", "PAUSED")
# Part 1 tags: (metadata attribute, label).
_META_TAGS = (("experimenter", "Experimenter"), ("protocol", "Protocol"), ("notes", "Notes"))


class RunHeader(QWidget):
    experiment_name_changed = pyqtSignal(str)

    def __init__(self, core, parent=None):
        super().__init__(parent)
        self._core = core
        self._state_name = "IDLE"
        self._missing: str | None = None

        col = QVBoxLayout(self)
        col.setContentsMargins(4, 4, 4, 0)
        col.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(10)
        self._title_row = top
        self._title = QLineEdit()
        self._title.setProperty("runTitle", True)
        self._title.setPlaceholderText("Name this experiment")
        self._title.editingFinished.connect(self._on_title_edited)
        top.addWidget(self._title, 1)
        self._run_label = QLabel()
        self._run_label.setProperty("textRole", "muted")
        top.addWidget(self._run_label)
        self._pill = QLabel()
        self._pill.setProperty("runnerStatus", True)
        top.addWidget(self._pill)
        col.addLayout(top)

        self._tags = FieldTags()
        self._tags.value_edited.connect(self._on_tag_edited)
        col.addWidget(self._tags)

        self.refresh()

    # --- public API ---

    def refresh(self) -> None:
        md = self._metadata()
        self._title.blockSignals(True)
        self._title.setText(md.name if md else "")
        self._title.blockSignals(False)
        self._tags.set_fields(self.tag_fields())
        self._apply_state()

    def tag_fields(self) -> list[TagField]:
        md = self._metadata()
        return [
            TagField(attr, label, str(getattr(md, attr, "") or "") if md else "")
            for attr, label in _META_TAGS
        ]

    def update_state(self, state_name: str) -> None:
        self._state_name = state_name
        self._apply_state()

    def set_needs_details(self, missing_label: str | None) -> None:
        self._missing = missing_label
        self._apply_state()

    # --- internals ---

    def _metadata(self):
        session = self._core.session
        return getattr(session, "metadata", None) if session is not None else None

    def _live(self) -> bool:
        return self._state_name in _LIVE

    def _apply_state(self) -> None:
        live = self._live()
        if live:
            text = self._state_name
        elif self._missing:
            text = "NEEDS DETAILS"
        else:
            text = "READY"
        self._pill.setText(text)
        self._pill.setProperty("statusState", text.replace(" ", "_"))
        self._pill.style().unpolish(self._pill)
        self._pill.style().polish(self._pill)
        self._title.setReadOnly(live)
        self._tags.set_locked(live)

    def _set_metadata(self, attr: str, value: str) -> None:
        md = self._metadata()
        if md is None or self._live():
            return
        setattr(md, attr, value)
        self._core.session.mark_dirty()

    def _on_title_edited(self) -> None:
        name = self._title.text().strip()
        md = self._metadata()
        if md is not None and name != md.name:
            self._set_metadata("name", name)
            self.experiment_name_changed.emit(name)

    def _on_tag_edited(self, key: str, value: str) -> None:
        self._set_metadata(key, value)
        self._tags.set_fields(self.tag_fields())
