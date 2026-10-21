import pytest
from PyQt6.QtCore import Qt

from glider.gui.dashboard.panels.field_tags import FieldTags, TagField

pytestmark = pytest.mark.usefixtures("qtbot")


@pytest.fixture
def tags(qtbot):
    w = FieldTags()
    qtbot.addWidget(w)
    w.set_fields(
        [TagField("dam_id", "Dam ID", "", True), TagField("notes", "Notes", "Nest at 09:10")]
    )
    return w


def test_chip_shows_label_and_value(tags):
    assert tags.chip("notes").text() == "Notes  Nest at 09:10"


def test_empty_required_chip_is_flagged(tags):
    chip = tags.chip("dam_id")
    assert chip.text() == "Dam ID  required"
    assert chip.property("need") is True


def test_click_edit_commit_emits(qtbot, tags):
    tags.chip("dam_id").click()
    editor = tags.editor()
    assert editor is not None and editor.text() == ""
    editor.setText("D14")
    with qtbot.waitSignal(tags.value_edited) as sig:
        qtbot.keyClick(editor, Qt.Key.Key_Return)
    assert sig.args == ["dam_id", "D14"]
    assert tags.editor() is None


def test_escape_cancels(qtbot, tags):
    tags.chip("notes").click()
    tags.editor().setText("changed")
    with qtbot.assertNotEmitted(tags.value_edited):
        qtbot.keyClick(tags.editor(), Qt.Key.Key_Escape)
    assert tags.editor() is None


def test_locked_tags_do_not_edit(tags):
    tags.set_locked(True)
    tags.chip("notes").click()
    assert tags.editor() is None


def test_long_value_is_elided_with_full_tooltip(tags):
    long = "x" * 200
    tags.set_fields([TagField("notes", "Notes", long)])
    assert len(tags.chip("notes").text()) < 60
    assert tags.chip("notes").toolTip() == long


def test_commit_pending_commits_typed_text(qtbot, tags):
    tags.chip("dam_id").click()
    tags.editor().setText("D14")
    with qtbot.waitSignal(tags.value_edited) as sig:
        tags.commit_pending()
    assert sig.args == ["dam_id", "D14"]
    assert tags.editor() is None


def test_focus_loss_commits(qtbot, tags):
    tags.chip("dam_id").click()
    tags.editor().setText("D14")
    with qtbot.waitSignal(tags.value_edited) as sig:
        tags.editor().editingFinished.emit()
    assert sig.args == ["dam_id", "D14"]


def test_locking_commits_an_open_edit(qtbot, tags):
    tags.chip("dam_id").click()
    tags.editor().setText("D14")
    with qtbot.waitSignal(tags.value_edited) as sig:
        tags.set_locked(True)
    assert sig.args == ["dam_id", "D14"]


def test_escape_then_focus_loss_does_not_commit(qtbot, tags):
    tags.chip("notes").click()
    editor = tags.editor()
    editor.setText("changed")
    with qtbot.assertNotEmitted(tags.value_edited):
        qtbot.keyClick(editor, Qt.Key.Key_Escape)
        editor.editingFinished.emit()
        tags.commit_pending()


def test_rebuild_leaves_only_current_chips(tags):
    from PyQt6.QtWidgets import QPushButton

    tags.set_fields([TagField("a", "A"), TagField("b", "B"), TagField("c", "C")])
    tags.set_fields([TagField("a", "A")])
    assert len(tags.findChildren(QPushButton)) == 1


def _three(qtbot):
    w = FieldTags(columns=2)
    qtbot.addWidget(w)
    w.set_fields(
        [
            TagField("dam_id", "Dam ID", "", True),
            TagField("pnd", "Postnatal day"),
            TagField("litter", "Litter size"),
        ]
    )
    return w


def test_two_column_grid_keeps_editing(qtbot):
    w = _three(qtbot)
    grid = w.layout()
    assert grid.getItemPosition(grid.indexOf(w.chip("litter")))[:2] == (1, 0)
    w.chip("dam_id").click()
    assert w.editor() is not None
    assert grid.getItemPosition(grid.indexOf(w.editor()))[:2] == (0, 0)


@pytest.mark.parametrize("key, cell", [("pnd", (0, 1)), ("litter", (1, 0))])
def test_the_editor_takes_the_clicked_chips_cell(qtbot, key, cell):
    w = _three(qtbot)
    w.chip(key).click()
    grid = w.layout()
    assert grid.getItemPosition(grid.indexOf(w.editor()))[:2] == cell


def test_two_column_grid_puts_the_trailing_label_on_its_own_row(qtbot):
    w = _three(qtbot)
    grid = w.layout()
    assert grid.getItemPosition(grid.indexOf(w._trailing)) == (2, 0, 1, 2)
    assert grid.getItemPosition(grid.indexOf(w.chip("pnd")))[:2] == (0, 1)


def test_grid_survives_a_rebuild_and_a_cancelled_edit(qtbot):
    w = _three(qtbot)
    w.chip("pnd").click()
    qtbot.keyClick(w.editor(), Qt.Key.Key_Escape)
    assert w.editor() is None and w.chip("pnd").isVisibleTo(w)
    w.set_fields([TagField("only", "Only one")])
    grid = w.layout()
    assert grid.getItemPosition(grid.indexOf(w.chip("only")))[:2] == (0, 0)
    assert grid.getItemPosition(grid.indexOf(w._trailing)) == (1, 0, 1, 2)


def test_default_is_still_a_single_row(qtbot, tags):
    from PyQt6.QtWidgets import QHBoxLayout

    assert isinstance(tags.layout(), QHBoxLayout)


_LABEL = {"a": "Experimenter", "b": "Postnatal day"}
_FULL = {"a": "Experimenter  " + "A" * 40, "b": "Postnatal day  " + "B" * 40}


def _long(qtbot, columns=2):
    w = FieldTags(columns=columns)
    qtbot.addWidget(w)
    w.set_fields(
        [
            TagField("a", "Experimenter", "A" * 40),
            TagField("b", "Postnatal day", "B" * 40),
        ]
    )
    return w


def test_long_values_do_not_widen_the_two_column_grid(qtbot):
    from glider.gui.styles import load_stylesheet

    w = _long(qtbot)
    assert w.minimumSizeHint().width() <= 480
    w.setStyleSheet(load_stylesheet("touch"))
    w.ensurePolished()
    for chip in w.findChildren(type(w.chip("a"))):
        chip.ensurePolished()
    assert w.minimumSizeHint().width() <= 480


def test_grid_chips_elide_to_their_column_and_keep_the_full_value_in_the_tooltip(qtbot):
    w = _long(qtbot)
    w.resize(300, 120)
    w.show()
    qtbot.waitExposed(w)
    for key, ch in (("a", "A"), ("b", "B")):
        chip = w.chip(key)
        assert chip.width() < 160  # two columns share 300 px
        assert chip.text().endswith("…") and chip.text() != _FULL[key]
        assert chip.text().startswith(_LABEL[key])  # the label stays readable
        assert chip.fontMetrics().horizontalAdvance(chip.text()) <= chip.width()
        assert chip.toolTip() == ch * 40
    # Plenty of room: nothing is elided.
    w.resize(1400, 120)
    assert w.chip("a").text() == _FULL["a"]


def test_the_two_columns_split_the_width_evenly(qtbot):
    w = _long(qtbot)
    w.resize(400, 120)
    w.show()
    qtbot.waitExposed(w)
    assert abs(w.chip("a").width() - w.chip("b").width()) <= 1
    # Still even with an editor open in one of them.
    w.chip("a").click()
    qtbot.waitUntil(lambda: w.editor().isVisible(), timeout=1000)
    qtbot.waitUntil(lambda: w.editor().width() < 400, timeout=1000)
    assert abs(w.editor().width() - w.chip("b").width()) <= 1
    assert w.width() == 400


def test_row_mode_chips_are_not_elided(qtbot):
    w = _long(qtbot, columns=0)
    w.resize(200, 60)
    w.show()
    qtbot.waitExposed(w)
    assert w.chip("a").text() == _FULL["a"]
