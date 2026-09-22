"""The marker editor: name, colour, note, scope, and turning a point into a range."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, Qt  # noqa: E402

from glider.analysis import markers as mk  # noqa: E402
from glider.gui.review.marker_editor import COHORT_NEEDS_A_FOLDER, MarkerEditor  # noqa: E402


@pytest.fixture
def point():
    return mk.Marker("point", 12.5, name="door", color="red")


def _editor(qtbot, marker, *, cohort_problem=None, in_out=None):
    editor = MarkerEditor(marker, cohort_problem=cohort_problem, in_out=in_out)
    qtbot.addWidget(editor)
    return editor


def _swatch(editor, name):
    return next(b for b in editor.swatches.buttons() if b.property("swatch") == name)


def test_done_hands_back_an_edited_copy(qtbot, point):
    editor = _editor(qtbot, point)
    editor.name.setText("door stuck")
    editor.note.setPlainText("lever jammed")
    _swatch(editor, "green").setChecked(True)
    editor.whole_cohort.setChecked(True)
    seen = []
    editor.done.connect(seen.append)
    editor.done_btn.click()
    (edited,) = seen
    assert (edited.name, edited.note, edited.color, edited.scope) == (
        "door stuck",
        "lever jammed",
        "green",
        "cohort",
    )
    assert edited.id == point.id
    assert point.name == "door"  # the marker it was given is untouched


def test_the_current_colour_and_scope_start_checked(qtbot, point):
    editor = _editor(qtbot, point)
    assert editor.swatches.checkedButton().property("swatch") == "red"
    assert editor.this_session.isChecked()


def test_every_swatch_is_offered(qtbot, point):
    editor = _editor(qtbot, point)
    assert [b.property("swatch") for b in editor.swatches.buttons()] == list(mk.SWATCHES)


def test_delete_reports_the_id(qtbot, point):
    editor = _editor(qtbot, point)
    seen = []
    editor.deleted.connect(seen.append)
    editor.delete_btn.click()
    assert seen == [point.id]


def test_escape_changes_nothing(qtbot, point):
    editor = _editor(qtbot, point)
    seen = []
    editor.done.connect(seen.append)
    editor.deleted.connect(seen.append)
    qtbot.keyClick(editor, Qt.Key.Key_Escape)
    assert seen == []


def test_enter_in_the_name_is_done(qtbot, point):
    editor = _editor(qtbot, point)
    seen = []
    editor.done.connect(seen.append)
    qtbot.keyClick(editor.name, Qt.Key.Key_Return)
    assert len(seen) == 1


def test_the_cohort_scope_needs_a_cohort(qtbot, point):
    editor = _editor(qtbot, point, cohort_problem=COHORT_NEEDS_A_FOLDER)
    assert not editor.whole_cohort.isEnabled()
    assert editor.whole_cohort.toolTip() == COHORT_NEEDS_A_FOLDER


def test_extend_turns_a_point_into_the_in_out_range(qtbot, point):
    editor = _editor(qtbot, point, in_out=(60.0, 120.0))
    seen = []
    editor.done.connect(seen.append)
    editor.extend.click()
    editor.done_btn.click()
    (edited,) = seen
    assert (edited.kind, edited.start_s, edited.end_s) == ("range", 60.0, 120.0)


def test_extend_needs_a_range(qtbot, point):
    assert not _editor(qtbot, point).extend.isEnabled()


def test_a_range_marker_has_nothing_to_extend(qtbot):
    editor = _editor(qtbot, mk.Marker("range", 0.0, 60.0, name="Baseline"))
    assert editor.extend.isHidden()


def test_it_opens_on_screen(qtbot, point):
    editor = _editor(qtbot, point)
    editor.popup(QPoint(10_000, 10_000))
    screen = editor.screen().availableGeometry()
    assert screen.contains(editor.frameGeometry().topLeft())
    editor.close()
