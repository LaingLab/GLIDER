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
