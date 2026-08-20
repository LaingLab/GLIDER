"""What one keystroke to every action has to guarantee.

Task 6 takes four menus away. From that point the palette is not a shortcut for
people who already know where things are -- it is the *navigation*. And GLIDER
has no existing users, so for the foreseeable future every user is a first-time
user: someone who has never seen this overlay has to be able to read it. Three
of the tests below exist only for that reader.

* :func:`test_a_disabled_command_is_greyed_rather_than_hidden` -- an action that
  vanishes when it is unavailable teaches nothing. Greyed answers "not yet";
  absent answers "this app cannot do that", which is a lie.
* :func:`test_an_empty_corpus_names_itself_rather_than_going_blank` and
  :func:`test_a_query_that_matches_nothing_says_so` -- a blank box is
  indistinguishable from a broken one.
* :func:`test_a_row_carries_its_source_menu_and_its_shortcut` -- the category is
  the only thing left saying *where* a command used to live once the menus are
  gone.

The first half of the file touches no Qt at all. Matching and ranking are plain
functions over ``(text, payload)`` pairs, and the payload is deliberately a
string in those tests: if they could only be written against ``QAction`` the
logic would not actually be separable from the widget.

**Every widget test runs against a shown palette**, mirroring
``test_side_panel.py`` and ``test_status_strip.py``. Qt settles geometry,
visibility and style resolution on the *show* event, and the host is returned
alongside the palette because ``qtbot.addWidget`` keeps only a weak reference --
drop the host and Qt destroys it and, with it, the widget under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction, QKeySequence, QPalette
from PyQt6.QtWidgets import QLabel, QMenuBar, QVBoxLayout, QWidget

import glider.gui.styles
from glider.gui.shell.command_palette import (
    Command,
    CommandPalette,
    commands_from_menu_bar,
    match_positions,
    rank,
)
from glider.gui.styles.colors import TEXT_DISABLED

#: The host the overlay covers. Big enough that "centred" is a real claim rather
#: than an artefact of the card filling everything it was given.
_HOST_WIDTH = 1000
_HOST_HEIGHT = 700


# --------------------------------------------------------------- pure matching


def test_a_subsequence_matches_out_of_order_characters():
    """The whole reason for subsequence matching: `sbj` has to find the Add
    Subject action, which is the spec's own example."""
    assert match_positions("sbj", "Add Subject…") == (4, 6, 7)


def test_a_non_subsequence_does_not_match():
    """The characters are all present but not in order, which is not a match."""
    assert match_positions("jbs", "Add Subject…") is None


def test_matching_is_case_insensitive():
    """Nobody types capitals into a palette."""
    assert match_positions("ADD", "Add Subject…") == (0, 1, 2)
    assert match_positions("add", "Add Subject…") == (0, 1, 2)


def test_an_empty_query_matches_everything_at_the_start():
    """An empty query is not "no match" -- it is the unfiltered list."""
    assert match_positions("", "Add Subject…") == ()


def test_an_earlier_match_ranks_first():
    """Ranking is by where the match starts: a command whose name *begins* with
    what was typed is what the typist meant."""
    entries = [("Add Subject…", "a"), ("Save As…", "b")]

    assert [text for text, _ in rank("s", entries)] == ["Save As…", "Add Subject…"]


def test_an_equal_match_position_falls_back_to_the_text():
    """Two commands matching at the same place sort by name, so the order is a
    property of the corpus rather than of the order it happened to be built."""
    entries = [("Stop", "b"), ("Start", "a")]

    assert [text for text, _ in rank("st", entries)] == ["Start", "Stop"]


def test_no_match_returns_nothing():
    assert rank("zzz", [("Add Subject…", "a"), ("Save As…", "b")]) == []


def test_an_empty_query_keeps_the_source_order():
    """Unfiltered, the palette reads like the menus it replaces -- File before
    Edit before Run. Sorting it alphabetically would throw away the only
    grouping a first-time user has."""
    entries = [("Save As…", "b"), ("New", "a"), ("Open…", "c")]

    assert rank("", entries) == entries


def test_ranking_never_looks_at_the_payload():
    """The payload is opaque on purpose: that is what makes this logic testable
    with no Qt, and what stops the widget's needs leaking into it."""
    marker = object()

    assert rank("n", [("New", marker)]) == [("New", marker)]


# ------------------------------------------------------------ sourcing actions


def _menu_bar(parent: QWidget) -> QMenuBar:
    """A menu bar shaped like the Builder's: two menus, a separator, one
    disabled action, one with a shortcut."""
    bar = QMenuBar(parent)

    file_menu = bar.addMenu("&File")
    new_action = QAction("&New", parent)
    new_action.setShortcut(QKeySequence("Ctrl+N"))
    file_menu.addAction(new_action)
    file_menu.addSeparator()
    file_menu.addAction(QAction("Save &As...", parent))

    edit_menu = bar.addMenu("&Edit")
    undo = QAction("&Undo", parent)
    undo.setEnabled(False)
    edit_menu.addAction(undo)
    edit_menu.addAction(QAction("&Add Subject…", parent))

    return bar


def test_commands_come_from_the_menu_actions(qtbot):
    """Sourced from the existing actions, never a parallel list: a second list
    would drift from the first, and this project has been bitten by two things
    that describe the same fact disagreeing."""
    host = QWidget()
    qtbot.addWidget(host)
    bar = _menu_bar(host)

    commands = commands_from_menu_bar(bar)

    assert [c.text for c in commands] == ["New", "Save As...", "Undo", "Add Subject…"]
    assert [c.category for c in commands] == ["File", "File", "Edit", "Edit"]
    del host


def test_a_commands_shortcut_and_enabled_state_come_along(qtbot):
    """The action already carries both. Re-deriving either here is how the two
    lists start to disagree."""
    host = QWidget()
    qtbot.addWidget(host)
    commands = commands_from_menu_bar(_menu_bar(host))
    by_text = {c.text: c for c in commands}

    assert by_text["New"].shortcut == "Ctrl+N"
    assert by_text["Save As..."].shortcut == ""
    assert by_text["Undo"].enabled is False
    assert by_text["New"].enabled is True
    del host


def test_separators_do_not_become_commands(qtbot):
    host = QWidget()
    qtbot.addWidget(host)

    commands = commands_from_menu_bar(_menu_bar(host))

    assert all(c.text for c in commands)
    assert len(commands) == 4
    del host


def test_ctrl_k_is_not_already_taken(qtbot, main_window_factory):
    """The palette's own key must not shadow something a user already relies
    on. Asserted against the real window rather than a grep, so it stays true
    as the menus change.

    Sourced from ``window.commands()`` rather than the menu bar, and that is the
    whole point of the check: since Task 6 the bar shows four of the window's
    eight menus, so asking the bar would quietly stop looking at Experiment,
    Hardware, Run and Tools -- sixteen actions, and exactly where a new shortcut
    is most likely to be added. It would keep passing while covering half of
    what it claims to.
    """
    window = main_window_factory(desktop_mode=True)
    window.show()

    taken = {c.text for c in window.commands() if c.shortcut.lower() == "ctrl+k"}

    assert taken == set()
    del window


# ------------------------------------------------------------------- the widget


def _text_colour(widget: QWidget) -> str:
    """The colour Qt resolved for *widget*'s text, after the stylesheet ran."""
    return widget.palette().color(QPalette.ColorRole.WindowText).name()


def _commands(parent: QWidget) -> list[Command]:
    """The corpus the widget tests run against, including a disabled action."""
    return commands_from_menu_bar(_menu_bar(parent))


def _shown(qtbot, commands=None, *, themed: bool = False) -> tuple[QWidget, CommandPalette]:
    """A palette over a host, opened, shown and exposed.

    Both are returned because ``qtbot.addWidget`` keeps only a weak reference:
    drop the host and Qt destroys it and, with it, the palette under test. Task
    2 lost a debugging cycle to exactly that.

    The host carries the stylesheet when *themed*, never the palette itself --
    which is the very thing
    :func:`test_nothing_in_the_palette_sets_its_own_stylesheet` forbids, and is
    also how ``MainWindow`` applies the theme in the running app.
    """
    host = QWidget()
    qtbot.addWidget(host)
    if themed:
        qss = Path(glider.gui.styles.__file__).with_name("desktop.qss")
        host.setStyleSheet(qss.read_text(encoding="utf-8"))
    layout = QVBoxLayout(host)
    layout.addWidget(QLabel("the builder is under here"))
    host.resize(_HOST_WIDTH, _HOST_HEIGHT)

    palette = CommandPalette(host)
    palette.set_commands(_commands(host) if commands is None else commands)

    with qtbot.waitExposed(host):
        host.show()
    palette.open()
    qtbot.waitUntil(palette.isVisible)
    return host, palette


def _settle(qtbot, host: QWidget) -> None:
    """Force the layout pass a geometry assertion must survive.

    ``waitUntil`` alone is satisfied by a transient size, so every geometry
    claim below is made after the host has actually laid out again. This is
    Task 3's pattern.
    """
    qtbot.wait(1)
    host.resize(host.width() + 1, host.height())
    qtbot.wait(1)
    host.resize(host.width() - 1, host.height())
    qtbot.wait(1)


@pytest.fixture
def palette(qtbot):
    host, palette = _shown(qtbot)
    yield palette
    del host


def test_the_palette_lists_every_command_when_nothing_is_typed(palette):
    assert palette.shown_texts() == ["New", "Save As...", "Undo", "Add Subject…"]


def test_typing_narrows_the_list(qtbot, palette):
    qtbot.keyClicks(palette.search_field(), "sbj")

    assert palette.shown_texts() == ["Add Subject…"]


def test_a_row_carries_its_source_menu_and_its_shortcut(palette):
    """With the menus gone the category is the only thing left saying where a
    command came from, and the shortcut is how the palette teaches itself out
    of a job."""
    row = palette.rows()[0]

    assert palette.row_text(row) == "New"
    assert palette.row_category(row) == "File"
    assert palette.row_shortcut(row) == "Ctrl+N"


def test_enter_triggers_the_selected_command_exactly_once(qtbot, palette):
    fired = []
    palette.commands()[0].action.triggered.connect(lambda: fired.append(1))

    qtbot.keyClick(palette.search_field(), Qt.Key.Key_Return)

    assert fired == [1]


def test_enter_closes_the_palette_behind_it(qtbot, palette):
    qtbot.keyClick(palette.search_field(), Qt.Key.Key_Return)

    assert not palette.isVisible()


def test_the_arrow_keys_move_the_selection(qtbot, palette):
    qtbot.keyClick(palette.search_field(), Qt.Key.Key_Down)

    assert palette.current_command().text == "Save As..."


def test_escape_closes_without_triggering_anything(qtbot, palette):
    fired = []
    for command in palette.commands():
        command.action.triggered.connect(lambda: fired.append(1))

    qtbot.keyClick(palette.search_field(), Qt.Key.Key_Escape)

    assert not palette.isVisible()
    assert fired == []


def test_a_disabled_command_is_greyed_rather_than_hidden(palette):
    """The palette has to answer "can I do this yet?" as well as "where is
    it?". A command that disappears when it is unavailable answers neither."""
    disabled = [palette.row_text(r) for r in palette.rows() if r.property("state") == "disabled"]

    assert "Undo" in palette.shown_texts()
    assert disabled == ["Undo"]


def test_a_disabled_command_cannot_be_triggered(qtbot, palette):
    """Visible, and still not a trap: selecting it is refused, so Enter cannot
    run something the app has said it cannot do."""
    undo = next(c for c in palette.commands() if c.text == "Undo")
    fired = []
    undo.action.triggered.connect(lambda: fired.append(1))

    qtbot.keyClicks(palette.search_field(), "undo")
    qtbot.keyClick(palette.search_field(), Qt.Key.Key_Return)

    assert palette.shown_texts() == ["Undo"]
    assert palette.current_command() is None
    assert fired == []


def test_an_empty_corpus_names_itself_rather_than_going_blank(qtbot):
    """A palette with nothing in it is the runner window's normal state -- it
    has no menu bar. A blank box there reads as broken."""
    host, palette = _shown(qtbot, commands=[])

    assert palette.empty_label().isVisible()
    assert palette.empty_label().text().strip() != ""
    assert not palette.list_widget().isVisible()
    del host


def test_a_query_that_matches_nothing_says_so(qtbot, palette):
    qtbot.keyClicks(palette.search_field(), "zzzz")

    assert palette.shown_texts() == []
    assert palette.empty_label().isVisible()
    assert "zzzz" in palette.empty_label().text()


def test_reopening_starts_from_a_clear_query(qtbot, palette):
    """Last search's leftovers would make the palette open on a filtered list
    nobody asked for."""
    qtbot.keyClicks(palette.search_field(), "sbj")
    qtbot.keyClick(palette.search_field(), Qt.Key.Key_Escape)

    palette.open()

    assert palette.search_field().text() == ""
    assert len(palette.shown_texts()) == 4


def test_the_overlay_covers_its_host_and_the_card_is_centred(qtbot):
    host, palette = _shown(qtbot)
    _settle(qtbot, host)

    assert palette.size() == host.size()
    card = palette.card()
    assert abs(card.geometry().center().x() - palette.rect().center().x()) <= 1
    assert abs(card.geometry().center().y() - palette.rect().center().y()) <= 1
    del host


def test_clicking_the_backdrop_dismisses_the_palette(qtbot, palette):
    """The overlay swallows the click that lands outside the card, so a stray
    click cannot reach the Builder underneath it."""
    # A non-null point on purpose: QTest reads QPoint(0, 0) as "no position
    # given" and clicks the widget's centre, which is the card.
    qtbot.mouseClick(palette, Qt.MouseButton.LeftButton, pos=QPoint(8, 8))

    assert not palette.isVisible()


def test_a_disabled_row_is_actually_painted_grey(qtbot):
    """Under the **real** stylesheet, and reading the colour Qt resolved.

    This is the only way to tell three failure modes apart: property set but
    never re-polished, re-polished but no matching rule in ``desktop.qss``, and
    a rule that fires. A row whose ``state`` property is correct and whose
    colour never changed is exactly the bug that would make "greyed rather than
    hidden" a claim with nothing behind it.
    """
    host, palette = _shown(qtbot, themed=True)
    by_text = {palette.row_text(row): row for row in palette.rows()}

    disabled = by_text["Undo"].findChild(QLabel, "commandPaletteRowText")
    enabled = by_text["New"].findChild(QLabel, "commandPaletteRowText")

    assert _text_colour(disabled) == TEXT_DISABLED
    assert _text_colour(enabled) != TEXT_DISABLED
    del host


def test_every_row_is_given_the_height_its_styled_text_needs(qtbot):
    """Under the **real** stylesheet, and asserted on metrics rather than on
    pixels: this machine renders no fonts offscreen, so "does it look cut?" is
    a question a screenshot here cannot answer.

    Two claims, and the bug needed both to be false. The *hint* has to be the
    styled row's, not an unpolished widget's -- ``desktop.qss`` puts the command
    name at 13 px where the global rule says 12. And the row has to actually
    *get* that height: ``setItemWidget`` positions the widget through the
    delegate, whose default is the item's **text** sub-rectangle, which under
    ``QListView::item { padding: 6px }`` is 12 px shorter than the hint. The row
    was handed 13 px for a 25 px layout, and its labels were what gave way.
    """
    host, palette = _shown(qtbot, themed=True)
    _settle(qtbot, host)
    rows = palette.list_widget()

    assert rows.count() > 0, "an empty list would make every claim below vacuous"
    for index in range(rows.count()):
        item = rows.item(index)
        row = rows.itemWidget(item)
        row.ensurePolished()
        margins = row.layout().contentsMargins()
        labels = [label for label in row.findChildren(QLabel) if label.isVisible()]
        needed = max(label.fontMetrics().height() for label in labels)
        needed += margins.top() + margins.bottom()

        assert item.sizeHint().height() >= needed
        assert row.height() >= needed
        for label in labels:
            assert label.height() >= label.fontMetrics().height()
    del host


def test_nothing_in_the_palette_sets_its_own_stylesheet(palette):
    """A widget-level stylesheet out-prioritises the application stylesheet and
    silently blocks all future theming. That trap has cost this project once."""
    offenders = [
        w.objectName() or type(w).__name__ for w in palette.findChildren(QWidget) if w.styleSheet()
    ]

    assert palette.styleSheet() == ""
    assert offenders == []


# ------------------------------------------------------------ from the window


def _active_builder(qtbot, main_window_factory):
    """A shown, **activated** Builder window.

    The activation is load-bearing rather than defensive. ``Ctrl+K`` is bound
    with ``WindowShortcut`` context, so Qt only matches it while the window is
    the active one; without ``waitActive`` the key press is delivered, matches
    nothing, and the test fails claiming the shortcut is unwired when it is
    merely unfocused.
    """
    window = main_window_factory(desktop_mode=True)
    with qtbot.waitExposed(window):
        window.show()
    window.switch_to_builder()
    window.activateWindow()
    qtbot.waitActive(window)
    return window


def _press_ctrl_k(qtbot, window):
    qtbot.keyClick(window, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
    palette = window.command_palette()
    assert palette is not None
    qtbot.waitUntil(palette.isVisible)
    return palette


def test_ctrl_k_opens_the_palette_from_the_main_window(qtbot, main_window_factory):
    window = _active_builder(qtbot, main_window_factory)

    palette = _press_ctrl_k(qtbot, window)

    assert palette.isVisible()
    del window


def test_the_window_palette_is_full_of_the_windows_commands(qtbot, main_window_factory):
    """The real corpus, and not the palette's own key: an entry that opens the
    thing you are already looking at is the one row that teaches nothing.

    "Add Subject..." is the sample on purpose: it is on the Experiment menu,
    which since Task 6 is not on the menu bar at all. Finding it here is the
    palette proving it sources from the window's menus rather than the bar.
    """
    window = _active_builder(qtbot, main_window_factory)

    palette = _press_ctrl_k(qtbot, window)

    assert "Add Subject..." in palette.shown_texts()
    assert "Command Palette" not in palette.shown_texts()
    del window


def test_the_window_palette_greys_undo_until_there_is_something_to_undo(qtbot, main_window_factory):
    """The point of sourcing from the actions rather than a list beside them:
    the palette is right about availability without being told."""
    window = _active_builder(qtbot, main_window_factory)

    palette = _press_ctrl_k(qtbot, window)
    undo = next(c for c in palette.commands() if c.text == "Undo")

    assert undo.enabled is False
    del window


def test_escape_closes_the_window_palette_cleanly(qtbot, main_window_factory):
    window = _active_builder(qtbot, main_window_factory)
    palette = _press_ctrl_k(qtbot, window)

    qtbot.keyClick(palette.search_field(), Qt.Key.Key_Escape)

    assert not palette.isVisible()
    assert window.command_palette() is palette
    del window
