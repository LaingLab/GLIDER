"""
Lab Setup Dialog - define the lab's vocabulary once, up front.

Subjects carry treatment metadata -- group, strain, solution, route, sex. Left
as free text it drifts: ``Control``, ``control`` and ``Ctrl`` become three
treatment groups no analysis can reconcile. :mod:`glider.core.vocabulary` is
the store that prevents that; this dialog is where a lab fills it in, and the
answer to the complaint that nobody ever found these fields.

Four behaviours here are deliberate, and each is a way this could fail quietly:

* **Neither button is the window default.** Return in a ``QLineEdit`` reaches
  the dialog's default button as well as emitting ``returnPressed``, and
  ``QDialogButtonBox`` promotes Done to default on its own Show event. Left
  alone, the documented "type a term and press Enter" also saved the file and
  closed the form on the first term given -- and the caller records the offer
  as seen before opening this, so there was no second chance. See
  :meth:`LabSetupDialog.showEvent`.
* **Skip is a first-class exit.** The person doing first launch is often not the
  person who knows the lab's strains. Skip closes the dialog and writes nothing
  -- no file, no half-vocabulary. The caller marks the setup seen either way, so
  skipping is genuinely free rather than a trap that re-asks every launch.
* **Done surfaces a failed write.** ``vocabulary.save`` returns ``False`` rather
  than raising on an unwritable library directory. Closing anyway would discard
  the entire vocabulary just typed, silently; instead the reason is shown inline
  and the dialog stays open so the user can fix it and retry.
* **Edits go through** :meth:`~glider.core.vocabulary.Vocabulary.add` **and**
  :meth:`~glider.core.vocabulary.Vocabulary.remove`, never through the list that
  ``get`` returns. That list is live: appending to it directly would bypass the
  case- and Unicode-folding this whole feature exists to provide, re-creating
  the duplicate groups through the very form meant to prevent them.

The dialog edits a *copy* of the vocabulary it is given, so an abandoned session
leaves the caller's object as untouched as the file.
"""

import logging
from pathlib import Path

from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.core.config import get_config
from glider.core.vocabulary import LISTS, Vocabulary, load, save

logger = logging.getLogger(__name__)

#: Heading and example placeholder for each vocabulary list. Looked up with a
#: fallback so a list added to LISTS still gets an editor, just a plainer one.
LIST_PRESENTATION = {
    "groups": ("Treatment groups", "e.g. Control"),
    "strains": ("Strains", "e.g. C57BL/6J"),
    "solutions": ("Solutions and drugs", "e.g. Saline"),
    "routes": ("Routes of administration", "e.g. IP"),
    "sexes": ("Sexes", "e.g. Male"),
}

INTRO_TEXT = (
    "Set up the terms your lab uses. They become the choices offered on every "
    "subject form, so the same treatment group is spelled the same way every "
    "time. You can skip this and fill it in later from Experiment → Lab "
    "Setup… — new terms typed on a subject form are learned automatically."
)


class VocabularyListEditor(QGroupBox):
    """One editable list: type a value and press Enter, remove per row.

    Every change goes through the vocabulary's own ``add``/``remove``, so a
    duplicate folds into the existing spelling and a whitespace-only value is
    refused -- both silently. Being told off for typing ``Control`` twice
    teaches nothing; showing one row does.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        name: str,
        title: str,
        placeholder: str,
        parent: QWidget | None = None,
    ):
        super().__init__(title, parent)
        self._vocabulary = vocabulary
        self._name = name
        self._remove_buttons: dict[str, QToolButton] = {}

        layout = QVBoxLayout(self)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText(placeholder)
        self.entry.setToolTip("Type a value and press Enter to add it")
        self.entry.returnPressed.connect(self._on_entry)
        layout.addWidget(self.entry)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(90)
        layout.addWidget(self.list_widget)

        self._rebuild()

    # -- State -----------------------------------------------------------------

    def values(self) -> list[str]:
        """The values currently listed. A copy: the stored list is live."""
        return list(self._vocabulary.get(self._name))

    def row_count(self) -> int:
        return self.list_widget.count()

    def remove_button(self, value: str) -> QToolButton | None:
        """The remove control for ``value``'s row, if it has one."""
        return self._remove_buttons.get(value)

    # -- Editing ---------------------------------------------------------------

    def _on_entry(self) -> None:
        """Add whatever was typed, if the vocabulary accepts it.

        ``add`` refuses duplicates (case- and Unicode-folded) and blank values
        by returning False. Refusals are silent by design; the entry clears
        either way, because the value the user wanted is on screen already.
        """
        if self._vocabulary.add(self._name, self.entry.text()):
            self._rebuild()
        self.entry.clear()

    def _on_remove(self, value: str) -> None:
        if self._vocabulary.remove(self._name, value):
            self._rebuild()

    def _rebuild(self) -> None:
        """Redraw every row from the vocabulary.

        Rebuilding wholesale rather than deleting one row keeps the rows and
        their remove buttons in step with the stored list -- a stale button
        bound to a shifted index removes the wrong entry, which nobody notices
        until the second click.
        """
        self.list_widget.clear()
        self._remove_buttons.clear()
        for value in self._vocabulary.get(self._name):
            self._append_row(value)

    def _append_row(self, value: str) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.addWidget(QLabel(value))
        row_layout.addStretch()

        button = QToolButton()
        button.setText("✕")
        button.setToolTip(f"Remove {value}")
        button.clicked.connect(lambda _checked=False, v=value: self._on_remove(v))
        row_layout.addWidget(button)

        item = QListWidgetItem(self.list_widget)
        item.setSizeHint(row.sizeHint())
        self.list_widget.setItemWidget(item, row)
        self._remove_buttons[value] = button


class LabSetupDialog(QDialog):
    """The setup form: five editable lists, Skip and Done.

    Both buttons close the dialog and both mean "seen" -- the caller records
    that regardless of the result code, so a skipped setup is not re-asked at
    every launch. Only Done writes the file.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        vocabulary: Vocabulary | None = None,
        is_touch_mode: bool = False,
    ):
        super().__init__(parent)
        self._is_touch_mode = is_touch_mode
        source = vocabulary if vocabulary is not None else load(self._library_dir())

        # Edit a copy. Skip must cost nothing, in memory as well as on disk,
        # and the caller may still be holding the vocabulary it passed in.
        # LISTS names are the dataclass field names, as `to_dict` also relies on.
        self.vocabulary = Vocabulary(**{name: list(source.get(name)) for name in LISTS})

        self.editors: dict[str, VocabularyListEditor] = {}
        self._setup_ui()

    @staticmethod
    def _library_dir() -> Path:
        """Read afresh each time: a retry after a fixed path must see it."""
        return Path(get_config().paths.library_dir)

    def _setup_ui(self) -> None:
        self.setWindowTitle("Lab Setup")
        self.setMinimumSize(480, 560)

        layout = QVBoxLayout(self)
        if self._is_touch_mode:
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

        intro = QLabel(INTRO_TEXT)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Five lists do not fit a small screen, and the Pi runs at 480px high.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QWidget()
        contents_layout = QVBoxLayout(contents)

        for name in LISTS:
            title, placeholder = LIST_PRESENTATION.get(name, (name.replace("_", " ").title(), ""))
            editor = VocabularyListEditor(self.vocabulary, name, title, placeholder)
            self.editors[name] = editor
            contents_layout.addWidget(editor)

        contents_layout.addStretch()
        scroll.setWidget(contents)
        layout.addWidget(scroll)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setObjectName("errorLabel")
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox()
        self.done_button = buttons.addButton("Done", QDialogButtonBox.ButtonRole.AcceptRole)
        self.skip_button = buttons.addButton("Skip", QDialogButtonBox.ButtonRole.RejectRole)
        self.skip_button.setToolTip("Close without saving; you can set this up later")
        self._clear_default_buttons()
        self.done_button.clicked.connect(self._on_done)
        self.skip_button.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _clear_default_buttons(self) -> None:
        """Make sure neither button is the window's default.

        Return in a ``QLineEdit`` reaches the dialog's default button as well
        as emitting ``returnPressed``, so a default button here turns the "type
        a term and press Enter" this whole form is built around into "add the
        term, save the file and close the form". On a first launch that is the
        entire vocabulary -- one entry -- because the caller records the offer
        as seen *before* opening this dialog and never asks again.
        """
        for button in (self.done_button, self.skip_button):
            button.setAutoDefault(False)
            button.setDefault(False)

    def showEvent(self, event: QShowEvent) -> None:
        """Undo Qt's promotion of Done to default button.

        ``QDialogButtonBox`` promotes its first AcceptRole button to default on
        its own Show event, overriding whatever was set at construction: Done
        reports ``isDefault() is False`` before ``show()`` and True after. That
        gap is why the Enter-saves-and-closes bug survived a full test file --
        every test in it ran against an unshown dialog, a state no user is ever
        in. Children are shown before the dialog's own show event, so clearing
        here lands after the promotion, on every show.
        """
        super().showEvent(event)
        self._clear_default_buttons()

    def _on_done(self) -> None:
        """Write the vocabulary, or explain why it could not be written.

        Staying open on failure is the point: the alternative is closing with
        nothing saved and nothing said, losing everything just typed.
        """
        library_dir = self._library_dir()
        if save(self.vocabulary, library_dir):
            self.error_label.setVisible(False)
            self.error_label.clear()
            self.accept()
            return

        logger.warning("Lab setup could not save the vocabulary to %s", library_dir)
        self.error_label.setText(
            f"Could not save the lab vocabulary to {library_dir}. "
            "Check that the folder exists and is writable, then press Done again. "
            "Nothing you have typed here has been lost."
        )
        self.error_label.setVisible(True)
