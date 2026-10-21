"""A row of click-to-edit field tags under the run-page title.

Each tag reads "Label  value". Clicking one swaps it for a one-line editor;
Return or focus loss commits (``value_edited``), Escape cancels. Locking, a
field change and ``commit_pending`` commit an open edit first, so typed text
is never lost. Locked while a run is live.

``columns=n`` lays the tags in an n-column grid instead of one row (the Pi's
480 px screen), with the trailing label on its own full-width row.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

_MAX_VALUE_CHARS = 40


@dataclass(frozen=True)
class TagField:
    key: str
    label: str
    value: str = ""
    required: bool = False


class _Editor(QLineEdit):
    cancelled = pyqtSignal()

    def keyPressEvent(self, event):  # noqa: N802 (Qt override)
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(event)


class FieldTags(QWidget):
    value_edited = pyqtSignal(str, str)

    def __init__(self, parent=None, *, columns: int = 0):
        super().__init__(parent)
        self._columns = columns
        self._fields: list[TagField] = []
        self._chips: dict[str, QPushButton] = {}
        self._editor: _Editor | None = None
        self._editing_key = ""
        self._locked = False
        self._row: QGridLayout | QHBoxLayout = QGridLayout(self) if columns else QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(6)
        self._trailing = QLabel()
        self._trailing.setProperty("textRole", "muted")
        self._trailing.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if columns:
            # A run-folder path has no spaces to wrap at; let it clip (the
            # tooltip has the whole text) rather than widen the 480 px page.
            self._trailing.setWordWrap(True)
            self._trailing.setMinimumWidth(1)

    # --- public API ---

    def set_fields(self, fields: list[TagField]) -> None:
        self.commit_pending()
        self._fields = list(fields)
        self._rebuild()

    def set_locked(self, locked: bool) -> None:
        if locked:
            self.commit_pending()
        self._locked = locked
        for chip in self._chips.values():
            chip.setProperty("locked", locked)
            chip.style().unpolish(chip)
            chip.style().polish(chip)

    def commit_pending(self) -> None:
        """Commit the open editor, if any (as Return would)."""
        if self._editor is not None:
            self._commit(self._editing_key)

    def set_trailing_text(self, text: str) -> None:
        self._trailing.setText(text)
        self._trailing.setToolTip(text)

    def chip(self, key: str) -> QPushButton:
        return self._chips[key]

    def editor(self) -> QLineEdit | None:
        return self._editor

    # --- internals ---

    def _rebuild(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._trailing:
                # Hide and detach now; deleteLater alone leaves it painted
                # until the event loop runs.
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._chips.clear()
        for i, f in enumerate(self._fields):
            chip = QPushButton(self._chip_text(f))
            chip.setProperty("fieldTag", True)
            chip.setProperty("need", f.required and not f.value)
            chip.setProperty("locked", self._locked)
            chip.setToolTip(f.value)
            chip.setCursor(Qt.CursorShape.IBeamCursor)
            chip.clicked.connect(lambda _=False, key=f.key: self._open_editor(key))
            self._chips[f.key] = chip
            if isinstance(self._row, QGridLayout):
                self._row.addWidget(chip, *divmod(i, self._columns))
            else:
                self._row.addWidget(chip)
        if isinstance(self._row, QGridLayout):
            rows = -(-len(self._fields) // self._columns)
            self._row.addWidget(self._trailing, rows, 0, 1, self._columns)
        else:
            self._row.addWidget(self._trailing, 1)

    @staticmethod
    def _chip_text(f: TagField) -> str:
        if not f.value:
            return f"{f.label}  {'required' if f.required else '—'}"
        value = f.value.replace("\n", " ")
        if len(value) > _MAX_VALUE_CHARS:
            value = value[: _MAX_VALUE_CHARS - 1] + "…"
        return f"{f.label}  {value}"

    def _open_editor(self, key: str) -> None:
        if self._locked:
            return
        self._close_editor()
        field = next(f for f in self._fields if f.key == key)
        chip = self._chips[key]
        editor = _Editor(field.value)
        editor.setPlaceholderText(field.label)
        editor.setProperty("fieldTagEditor", True)
        # editingFinished covers Return and focus loss.
        editor.editingFinished.connect(lambda: self._commit(key))
        editor.cancelled.connect(self._close_editor)
        chip.hide()
        if isinstance(self._row, QGridLayout):
            # The editor takes the chip's grid cell.
            row, col = self._row.getItemPosition(self._row.indexOf(chip))[:2]
            self._row.addWidget(editor, row, col)
        else:
            self._row.insertWidget(self._row.indexOf(chip), editor)
        self._editor = editor
        self._editing_key = key
        editor.setFocus()
        editor.selectAll()

    def _commit(self, key: str) -> None:
        if self._editor is None:
            return
        text = self._editor.text().strip()
        self._close_editor()
        self.value_edited.emit(key, text)

    def _close_editor(self) -> None:
        if self._editor is None:
            return
        editor, self._editor = self._editor, None
        # Disconnect first: hiding a focused editor loses focus, and the
        # resulting editingFinished must not commit a cancelled edit.
        editor.editingFinished.disconnect()
        editor.hide()
        editor.deleteLater()
        chip = self._chips.get(self._editing_key)
        if chip is not None:
            chip.show()
