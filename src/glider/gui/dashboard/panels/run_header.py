"""The run page's title block: experiment title, run label, state pill, field tags.

Owns the binding between the tags and ``session.metadata`` that
RunControlPanel used to hold. Read-only while a run is live.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.core import presets
from glider.gui.dashboard.panels.field_tags import FieldTags, TagField

_LIVE = ("RUNNING", "PAUSED")
# Non-live states the pill names as they are instead of READY / NEEDS DETAILS.
_SHOWN_AS_IS = ("STOPPING", "ERROR", "INITIALIZING")
# Part 1 tags: (metadata attribute, label).
_META_TAGS = (("experimenter", "Experimenter"), ("protocol", "Protocol"), ("notes", "Notes"))


def _fields(md) -> list:
    """The preset's run fields; [] for no metadata or a non-list (mock) value."""
    fields = getattr(md, "run_fields", None)
    return fields if isinstance(fields, list) else []


def _type(md) -> str:
    t = getattr(md, "experiment_type", "")
    return t if isinstance(t, str) else ""


class RunHeader(QWidget):
    experiment_name_changed = pyqtSignal(str)
    preset_chosen = pyqtSignal(str)
    new_preset_requested = pyqtSignal()
    save_as_preset_requested = pyqtSignal()
    edit_fields_requested = pyqtSignal()
    open_file_requested = pyqtSignal()
    show_presets_folder_requested = pyqtSignal()
    run_values_changed = pyqtSignal()

    def __init__(self, core, parent=None):
        super().__init__(parent)
        self._core = core
        self._state_name = "IDLE"
        self._missing: str | None = None
        self._ready = True  # board + flow, fed by RunControlPanel.readiness_changed

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
        # Sized to its text, so the ▾ beside it reads as part of the title.
        self._title.textChanged.connect(self._fit_title)
        self._picker = QToolButton()
        self._picker.setProperty("presetPicker", True)
        self._picker.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._menu = QMenu(self._picker)
        self._menu.setToolTipsVisible(True)  # field labels as each preset's subtitle
        self._menu.aboutToShow.connect(self._rebuild_menu)
        self._picker.setMenu(self._menu)
        top.addWidget(self._title)
        top.addWidget(self._picker)
        # Takes the slack, so the title, ▾ and pill keep their size hints.
        top.addStretch()
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
        self._tags.commit_pending()
        md = self._metadata()
        self._title.blockSignals(True)
        self._title.setText(md.name if md else "")
        self._title.blockSignals(False)
        self._fit_title()
        preset = self._is_preset()
        self._title.setVisible(not preset)
        self._picker.setText(f"{_type(md) or md.name} ▾" if preset else "▾")
        self._tags.set_fields(self.tag_fields())
        self._apply_state()

    def tag_fields(self) -> list[TagField]:
        md = self._metadata()
        if md is None:
            return [TagField(a, lab, "") for a, lab in _META_TAGS]
        if not _fields(md):
            return [TagField(a, lab, str(getattr(md, a, "") or "")) for a, lab in _META_TAGS]
        tags = [
            TagField(f.key, f.label, md.run_values.get(f.key, ""), f.required) for f in _fields(md)
        ]
        tags += [
            TagField("experimenter", "Experimenter", md.experimenter),
            TagField("notes", "Notes", md.notes),
        ]
        return tags

    def commit_pending(self) -> None:
        """Commit a tag the operator is still typing in (e.g. before a start)."""
        self._tags.commit_pending()

    def set_run_info(self, run_number: int | None, folder: str) -> None:
        self._run_label.setText(f"Run {run_number} today" if run_number else "")
        self._tags.set_trailing_text(f"→ {folder}" if folder else "")

    def update_state(self, state_name: str) -> None:
        self._state_name = state_name
        self._apply_state()

    def set_ready(self, ready: bool) -> None:
        """Whether a board and a runnable flow are in place (not the run fields)."""
        self._ready = bool(ready)
        self._apply_state()

    def set_needs_details(self, missing_label: str | None) -> None:
        self._missing = missing_label
        self._apply_state()

    # --- internals ---

    def _metadata(self):
        session = self._core.session
        return getattr(session, "metadata", None) if session is not None else None

    def _is_preset(self) -> bool:
        # Run fields only: an old file with just an Experiment Type keeps
        # its editable title.
        md = self._metadata()
        return bool(md and _fields(md))

    def _rebuild_menu(self) -> None:
        self._menu.clear()
        current = str(getattr(self._core.session, "file_path", None) or "")
        for info in presets.list_presets():
            action = self._menu.addAction(info.name)
            action.setToolTip(" · ".join(info.field_labels))
            action.setCheckable(True)
            action.setChecked(current == str(info.path))
            action.triggered.connect(lambda _=False, p=str(info.path): self.preset_chosen.emit(p))
        if not self._menu.actions():
            self._menu.addAction("No presets yet").setEnabled(False)
        self._menu.addSeparator()
        for text, signal in (
            ("New preset…", self.new_preset_requested),
            ("Save current setup as preset…", self.save_as_preset_requested),
            ("Edit preset fields…", self.edit_fields_requested),
            ("Open .glider file…", self.open_file_requested),
        ):
            self._menu.addAction(text).triggered.connect(lambda _=False, s=signal: s.emit())
        self._menu.addSeparator()
        self._menu.addAction("Show presets folder").triggered.connect(
            lambda _=False: self.show_presets_folder_requested.emit()
        )

    def _live(self) -> bool:
        return self._state_name in _LIVE

    def _apply_state(self) -> None:
        live = self._live()
        md = self._metadata()
        missing = self._missing or (
            presets.missing_required(_fields(md), md.run_values) if _fields(md) else None
        )
        if live or self._state_name in _SHOWN_AS_IS:
            text = self._state_name
        elif missing:
            text = "NEEDS DETAILS"
        elif not self._ready:
            text = "NOT READY"
        else:
            text = "READY"
        self._pill.setText(text)
        self._pill.setProperty("statusState", text.replace(" ", "_"))
        self._pill.style().unpolish(self._pill)
        self._pill.style().polish(self._pill)
        self._picker.setEnabled(not live)
        self._title.setReadOnly(live)
        self._tags.set_locked(live)

    def _fit_title(self, text: str = "") -> None:
        self._title.ensurePolished()  # the stylesheet sets the 18 px title font
        metrics = self._title.fontMetrics()
        shown = self._title.text() or self._title.placeholderText()
        self._title.setFixedWidth(min(metrics.horizontalAdvance(shown) + 24, 720))

    def showEvent(self, event):  # noqa: N802 (Qt override)
        # The app stylesheet can land after construction; measure again.
        super().showEvent(event)
        self._fit_title()

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
        md = self._metadata()
        if md is not None and any(f.key == key for f in _fields(md)):
            if self._live():
                return
            md.run_values[key] = value
            self.run_values_changed.emit()
        else:
            self._set_metadata(key, value)
        self._tags.set_fields(self.tag_fields())
        self._apply_state()
