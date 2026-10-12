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
