import pytest
from PyQt6.QtCore import Qt

from glider.core.experiment_session import RunField
from glider.gui.dialogs.preset_fields_dialog import PresetFieldsDialog

pytestmark = pytest.mark.usefixtures("qtbot")


def _dlg(qtbot, fields=(), ask_name=False, name=""):
    d = PresetFieldsDialog(list(fields), name=name, ask_name=ask_name)
    qtbot.addWidget(d)
    return d


def _set(d, row, label=None, required=None, default=None):
    if label is not None:
        d._table.item(row, 0).setText(label)
    if required is not None:
        d._table.item(row, 1).setCheckState(
            Qt.CheckState.Checked if required else Qt.CheckState.Unchecked
        )
    if default is not None:
        d._table.item(row, 2).setText(default)


def test_add_rows_and_read_fields(qtbot):
    d = _dlg(qtbot)
    d._add_btn.click()
    d._add_btn.click()
    _set(d, 0, "Dam ID", True)
    _set(d, 1, "Litter size", False, "6")
    assert d.fields() == [
        RunField("dam_id", "Dam ID", True, ""),
        RunField("litter_size", "Litter size", False, "6"),
    ]


def test_existing_keys_survive(qtbot):
    d = _dlg(qtbot, [RunField("dam", "Dam ID", True)])
    assert d.fields()[0].key == "dam"


def test_blank_label_blocks_ok(qtbot):
    d = _dlg(qtbot)
    d._add_btn.click()
    d._validate()
    assert not d._ok_btn.isEnabled()
    assert d._error.text() == "Every field needs a label."


def test_name_required_when_asked(qtbot):
    d = _dlg(qtbot, ask_name=True)
    d._validate()
    assert not d._ok_btn.isEnabled()
    d._name_edit.setText("Open field")
    d._validate()
    assert d._ok_btn.isEnabled() and d.name() == "Open field"


def test_reorder_and_remove(qtbot):
    d = _dlg(qtbot, [RunField("a", "A"), RunField("b", "B")])
    d._table.setCurrentCell(1, 0)
    d._up_btn.click()
    assert [f.label for f in d.fields()] == ["B", "A"]
    d._table.setCurrentCell(0, 0)
    d._remove_btn.click()
    assert [f.label for f in d.fields()] == ["A"]


def test_reserved_keys_are_renamed(qtbot):
    d = _dlg(qtbot)
    d._add_btn.click()
    _set(d, 0, "Notes")
    assert d.fields()[0].key == "notes_field"
