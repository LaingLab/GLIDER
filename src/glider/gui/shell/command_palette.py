"""One keystroke to every action in the Builder.

``Ctrl+K`` puts a centred overlay over the window: type, and the list narrows;
Enter runs what is selected; Escape closes it. Once the menu bar is down to four
menus this is not a power-user shortcut sitting on top of a discoverable UI --
it *is* the discoverable UI, and it is read by someone who has never seen it,
because GLIDER has no existing users and for the foreseeable future every user
is a first-time user.

Four decisions carry meaning rather than convenience:

* **The commands are the existing ``QAction``s.** There are about thirty of
  them, and they already carry their text, their shortcut and -- the part that
  matters -- whether they are enabled *right now*. A parallel registry would be
  a second description of the same fact, and this project has been bitten
  repeatedly by two such descriptions drifting apart. :func:`commands_from_menu_bar`
  walks the real menu bar, and :meth:`CommandPalette.open` walks it again on
  every open so that "Undo" is greyed exactly when the Edit menu would grey it.

* **A disabled command is greyed, never hidden.** The palette has to answer
  "can I do this yet?" as well as "where is it?". A command that disappears
  when it is unavailable answers neither: absence reads as "this app cannot do
  that", which is a different and false statement. It is still not a trap --
  a disabled row cannot be selected, so Enter cannot reach it.

* **Each row names the menu it came from.** After Task 6 the category is the
  only thing left that says where a command used to live, and the shortcut
  beside it is how the palette teaches itself out of a job.

* **Matching and ranking are module-level functions over ``(text, payload)``
  pairs.** They touch no Qt at all, which is what makes them testable without
  showing a widget -- every GUI test in this project has to show its widget,
  so logic that escapes that is worth arranging for. Subsequence matching means
  ``sbj`` finds "Add Subject…"; ranking is by where the match starts and then
  by the text, so a command whose name *begins* with what was typed wins. Thirty
  items do not need a fuzzy-matching dependency.

**No colour is set from Python here**, following ``side_panel.py`` and
``status_strip.py``: every part carries an ``objectName`` -- and, where it
varies, a dynamic ``state`` property -- and ``desktop.qss`` owns the rest. A
widget-level ``setStyleSheet`` out-prioritises the application stylesheet, so
one stray call would make that widget the single thing in the shell the theme
cannot reach. Property changes go through :func:`glider.gui.styles.restyle`,
which is the shared helper this module's arrival was the occasion to extract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenuBar,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import restyle

__all__ = [
    "CARD_MARGIN",
    "CARD_WIDTH",
    "LIST_HEIGHT",
    "MIN_CARD_WIDTH",
    "Command",
    "CommandPalette",
    "commands_from_menu_bar",
    "match_positions",
    "rank",
]

#: The card's preferred width, in pixels. Wide enough for the longest action
#: text in the menu bar plus its shortcut without eliding.
CARD_WIDTH = 560

#: The narrowest the card may be squeezed to before it stops being readable.
MIN_CARD_WIDTH = 320

#: Space left either side of the card on a host too narrow for
#: :data:`CARD_WIDTH`, so the overlay never reads as a full-screen takeover.
CARD_MARGIN = 40

#: The tallest the results list may grow, in pixels -- about eight rows. Past
#: that the palette stops looking like an overlay and starts looking like a
#: window, and nobody reads the ninth result anyway.
LIST_HEIGHT = 320

#: Shown when there is nothing to list at all. The runner window reaches this
#: honestly: it has no menu bar. A blank box there would read as broken.
_NOTHING_AT_ALL = "No commands available"

_T = TypeVar("_T")


# --------------------------------------------------------------------- matching


def match_positions(query: str, text: str) -> tuple[int, ...] | None:
    """Where each character of *query* lands in *text*, or ``None``.

    A subsequence match: the characters must appear in order but need not be
    adjacent, which is what lets ``sbj`` find "Add Subject…". Case is ignored in
    both directions -- nobody types capitals into a palette.

    An empty query returns ``()``, which is a match at position zero rather than
    a failure: an empty query is the *unfiltered* list, not the empty one.
    """
    needle = query.strip().lower()
    if not needle:
        return ()
    haystack = text.lower()
    positions: list[int] = []
    start = 0
    for character in needle:
        found = haystack.find(character, start)
        if found < 0:
            return None
        positions.append(found)
        start = found + 1
    return tuple(positions)


def rank(query: str, entries: Sequence[tuple[str, _T]]) -> list[tuple[str, _T]]:
    """The entries matching *query*, best first.

    Args:
        entries: ``(text, payload)`` pairs. The payload is never inspected --
            that is what keeps this function free of Qt, and of the widget's
            needs.

    Ranked by where the match begins, then by the text itself. First key: a
    command whose name *starts* with what was typed is overwhelmingly what the
    typist meant. Second key: without it the order would depend on the order the
    corpus happened to be built in, and two runs could disagree.

    An empty query returns the entries **untouched, in the order given**. That
    is deliberate and not an optimisation: the corpus arrives in menu order, so
    an unfiltered palette reads File, then Edit, then Run -- the same grouping
    the menus had. Sorting it alphabetically would throw away the only structure
    a first-time user has to hold on to.
    """
    entries = list(entries)
    if not query.strip():
        return entries

    scored: list[tuple[int, str, str, _T]] = []
    for text, payload in entries:
        positions = match_positions(query, text)
        if positions is None:
            continue
        scored.append((positions[0], text.lower(), text, payload))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [(text, payload) for _, _, text, payload in scored]


# ---------------------------------------------------------------- the commands


@dataclass(frozen=True)
class Command:
    """One action, as the palette needs to show it.

    Everything here is *read from* the action rather than declared beside it.
    :attr:`enabled` in particular is a snapshot: it is true at the moment the
    palette was opened, which is why the palette re-reads its source on every
    open instead of caching a list.
    """

    text: str
    category: str
    shortcut: str
    enabled: bool
    action: QAction


def _strip_mnemonic(text: str) -> str:
    """``"Save &As..."`` as the user reads it: ``"Save As..."``.

    A doubled ``&&`` is a literal ampersand and survives.
    """
    return text.replace("&&", "\x00").replace("&", "").replace("\x00", "&").strip()


def commands_from_menu_bar(menu_bar: QMenuBar | None) -> list[Command]:
    """Every actionable item in *menu_bar*, in menu order.

    Separators and menu titles are skipped; submenus are walked, and their
    contents keep the *top-level* menu's name as their category -- a user
    looking for where something lived remembers "it was under Tools", not the
    submenu it was nested in.

    ``None`` yields an empty list rather than raising. A window with no menu bar
    is a real state (the Pi runner has none), and the palette shows its named
    empty state for it.

    Shortcuts are rendered as portable text (``"Ctrl+N"``) rather than native
    text, so the string a test compares is the same on all three platforms CI
    runs.
    """
    if menu_bar is None:
        return []
    commands: list[Command] = []
    for action in menu_bar.actions():
        menu = action.menu()
        if menu is None:
            continue
        _collect(menu, _strip_mnemonic(action.text()), commands)
    return commands


def _collect(menu, category: str, into: list[Command]) -> None:
    for action in menu.actions():
        if action.isSeparator():
            continue
        submenu = action.menu()
        if submenu is not None:
            _collect(submenu, category, into)
            continue
        text = _strip_mnemonic(action.text())
        if not text:
            continue
        into.append(
            Command(
                text=text,
                category=category,
                shortcut=action.shortcut().toString(QKeySequence.SequenceFormat.PortableText),
                enabled=action.isEnabled(),
                action=action,
            )
        )


# ----------------------------------------------------------------- the overlay


class CommandPalette(QFrame):
    """A centred, filterable list of every action, over its parent.

    Args:
        parent: The widget the overlay covers. The palette sizes itself to the
            parent's rectangle and follows it when it resizes, so the parent
            should be the window (or the page) the palette is meant to dim.

    Signals:
        command_triggered: The text of the command that was just run. The
            palette runs the action itself -- this is for anything that wants to
            *observe* the fact, such as a status line.
    """

    command_triggered = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("commandPalette")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._source: Callable[[], Iterable[Command]] = list
        self._commands: list[Command] = []
        self._shown: list[Command] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch(1)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addStretch(1)
        self._card = self._build_card()
        row.addWidget(self._card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        self._search.installEventFilter(self)
        self._list.installEventFilter(self)
        if parent is not None:
            parent.installEventFilter(self)
        self.hide()

    # ----------------------------------------------------------------- build

    def _build_card(self) -> QFrame:
        card = QFrame(self)
        card.setObjectName("commandPaletteCard")
        card.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._search = QLineEdit(card)
        self._search.setObjectName("commandPaletteSearch")
        self._search.setPlaceholderText("Type a command…")
        self._search.setClearButtonEnabled(False)
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        self._list = QListWidget(card)
        self._list.setObjectName("commandPaletteList")
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setUniformItemSizes(False)
        self._list.setMaximumHeight(LIST_HEIGHT)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        self._empty = QLabel(_NOTHING_AT_ALL, card)
        self._empty.setObjectName("commandPaletteEmpty")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        layout.addWidget(self._empty)

        return card

    # ---------------------------------------------------------------- sources

    def set_command_source(self, source: Callable[[], Iterable[Command]]) -> None:
        """Read the commands from *source* on every open.

        This is the form the window uses. The point is the re-read: whether
        Undo is available changes constantly, and a list captured once would go
        stale the moment the user did anything.
        """
        self._source = source

    def set_commands(self, commands: Iterable[Command]) -> None:
        """Use a fixed list of commands.

        Convenient for a caller whose corpus genuinely does not change, and for
        tests. Prefer :meth:`set_command_source` anywhere the enabled state can
        move underneath the palette.
        """
        fixed = list(commands)
        self._source = lambda: fixed

    def commands(self) -> list[Command]:
        """The whole corpus as of the last open, in source order."""
        return list(self._commands)

    # ------------------------------------------------------------ open, close

    def open(self) -> None:
        """Cover the parent, re-read the commands and take the keyboard.

        The query is cleared first: reopening onto the last search's leftovers
        would show a filtered list nobody asked for, and the filter that
        produced it is no longer on screen to explain it.
        """
        self._cover()
        self._commands = list(self._source())
        blocked = self._search.blockSignals(True)
        self._search.clear()
        self._search.blockSignals(blocked)
        self._apply_filter("")
        self.show()
        self.raise_()
        self._search.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def dismiss(self) -> None:
        """Close the overlay, running nothing."""
        self.hide()

    def _cover(self) -> None:
        """Match the parent's rectangle, and size the card inside it."""
        parent = self.parentWidget()
        if parent is None:
            return
        self.setGeometry(parent.rect())
        width = max(MIN_CARD_WIDTH, min(CARD_WIDTH, parent.width() - CARD_MARGIN * 2))
        self._card.setFixedWidth(width)

    # ------------------------------------------------------------- filtering

    def _apply_filter(self, query: str) -> None:
        ranked = rank(query, [(command.text, command) for command in self._commands])
        self._shown = [command for _, command in ranked]
        self._rebuild_rows()
        self._empty.setText(self._empty_text(query))
        self._empty.setVisible(not self._shown)
        self._list.setVisible(bool(self._shown))

    @staticmethod
    def _empty_text(query: str) -> str:
        """Why the list is empty, in words, rather than nothing at all."""
        typed = query.strip()
        if not typed:
            return _NOTHING_AT_ALL
        return f"No commands match '{typed}'"

    def _rebuild_rows(self) -> None:
        self._list.clear()
        for command in self._shown:
            item = QListWidgetItem(self._list)
            item.setData(Qt.ItemDataRole.UserRole, command)
            if not command.enabled:
                # Visible, and still not a trap: an unselectable row cannot be
                # reached by the arrow keys, so Enter cannot run something the
                # app has already said it cannot do.
                item.setFlags(Qt.ItemFlag.NoItemFlags)
            widget = self._build_row(command)
            item.setSizeHint(widget.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)
        self._select_first_enabled()

    def _build_row(self, command: Command) -> QFrame:
        row = QFrame(self._list)
        row.setObjectName("commandPaletteRow")
        row.setFrameShape(QFrame.Shape.NoFrame)
        row.setProperty("state", "enabled" if command.enabled else "disabled")

        layout = QHBoxLayout(row)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        text = QLabel(command.text, row)
        text.setObjectName("commandPaletteRowText")
        layout.addWidget(text)

        category = QLabel(command.category, row)
        category.setObjectName("commandPaletteRowCategory")
        layout.addWidget(category)

        layout.addStretch(1)

        shortcut = QLabel(command.shortcut, row)
        shortcut.setObjectName("commandPaletteRowShortcut")
        shortcut.setVisible(bool(command.shortcut))
        layout.addWidget(shortcut)

        restyle(row)
        return row

    # ------------------------------------------------------------- selection

    def _select_first_enabled(self) -> None:
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsSelectable:
                self._list.setCurrentRow(index)
                return
        self._list.setCurrentRow(-1)

    def _step(self, delta: int) -> None:
        """Move to the next selectable row, wrapping at either end."""
        count = self._list.count()
        if count == 0:
            return
        start = self._list.currentRow()
        index = 0 if start < 0 else start
        for _ in range(count):
            index = (index + delta) % count
            item = self._list.item(index)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsSelectable:
                self._list.setCurrentRow(index)
                return

    def current_command(self) -> Command | None:
        """The command Enter would run, or ``None`` if nothing is selectable."""
        item = self._list.currentItem()
        if item is None or not item.flags() & Qt.ItemFlag.ItemIsSelectable:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    # --------------------------------------------------------------- running

    def _run_current(self) -> None:
        command = self.current_command()
        if command is None:
            return
        self.dismiss()
        # Dismissed first, so an action that opens a modal dialog does not do it
        # underneath an overlay that is still on top of the window.
        command.action.trigger()
        self.command_triggered.emit(command.text)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if not item.flags() & Qt.ItemFlag.ItemIsSelectable:
            return
        self._list.setCurrentItem(item)
        self._run_current()

    # ------------------------------------------------------------------ keys

    def _handle_key(self, event) -> bool:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.dismiss()
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._run_current()
            return True
        if key == Qt.Key.Key_Down:
            self._step(1)
            return True
        if key == Qt.Key.Key_Up:
            self._step(-1)
            return True
        return False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        """Take the palette's keys before the search field eats them, and keep
        the overlay on top of a parent that resized underneath it.

        Filtering rather than relying on key propagation: a ``QLineEdit``
        ignores Escape and the arrows *today*, and the palette should not depend
        on that staying true.
        """
        if event.type() == QEvent.Type.KeyPress and obj in (self._search, self._list):
            if self._handle_key(event):
                return True
        elif event.type() == QEvent.Type.Resize and obj is self.parentWidget():
            if self.isVisible():
                self._cover()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self._handle_key(event):
            super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        """A click on the backdrop closes; a click on the card does not.

        The overlay swallows it either way, which is the other half of the job:
        a stray click must not reach the Builder underneath.
        """
        if self._card.geometry().contains(event.position().toPoint()):
            super().mousePressEvent(event)
            return
        event.accept()
        self.dismiss()

    # -------------------------------------------------------------- accessors

    def card(self) -> QFrame:
        """The centred card the overlay is dimming the rest of the window for."""
        return self._card

    def search_field(self) -> QLineEdit:
        """The query field, which holds the keyboard while the palette is open."""
        return self._search

    def list_widget(self) -> QListWidget:
        """The results list. Hidden when there is nothing to list."""
        return self._list

    def empty_label(self) -> QLabel:
        """The named empty state. Shown instead of the list, never beside it."""
        return self._empty

    def shown_texts(self) -> list[str]:
        """The command names currently listed, in display order."""
        return [command.text for command in self._shown]

    def rows(self) -> list[QFrame]:
        """The row widgets currently listed, in display order."""
        widgets = []
        for index in range(self._list.count()):
            item = self._list.item(index)
            widget = self._list.itemWidget(item) if item is not None else None
            if widget is not None:
                widgets.append(widget)
        return widgets

    @staticmethod
    def row_text(row: QFrame) -> str:
        """The command name shown on *row*."""
        return _row_label(row, "commandPaletteRowText")

    @staticmethod
    def row_category(row: QFrame) -> str:
        """The menu *row* came from -- the only trace of it left after Task 6."""
        return _row_label(row, "commandPaletteRowCategory")

    @staticmethod
    def row_shortcut(row: QFrame) -> str:
        """The shortcut shown on *row*, or ``""`` if it has none."""
        return _row_label(row, "commandPaletteRowShortcut")


def _row_label(row: QFrame, name: str) -> str:
    label = row.findChild(QLabel, name)
    return label.text() if label is not None else ""
