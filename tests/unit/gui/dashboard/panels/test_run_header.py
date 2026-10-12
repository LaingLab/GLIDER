import pytest

from glider.core.experiment_session import ExperimentSession
from glider.gui.dashboard.panels.run_header import RunHeader

pytestmark = pytest.mark.usefixtures("qtbot")


def _header(qtbot, mock_core, **meta):
    session = ExperimentSession()
    for k, v in meta.items():
        setattr(session.metadata, k, v)
    session._mark_clean()
    mock_core.session = session
    h = RunHeader(mock_core)
    qtbot.addWidget(h)
    return h


def test_shows_name_and_metadata_tags(qtbot, mock_core):
    h = _header(qtbot, mock_core, name="OF 3", experimenter="GB", protocol="P-17")
    assert h._title.text() == "OF 3"
    assert h._tags.chip("experimenter").text() == "Experimenter  GB"
    assert h._tags.chip("protocol").text() == "Protocol  P-17"


def test_tag_edit_writes_metadata_and_marks_dirty(qtbot, mock_core):
    h = _header(qtbot, mock_core)
    h._tags.value_edited.emit("experimenter", "AB")
    assert mock_core.session.metadata.experimenter == "AB"
    assert mock_core.session.is_dirty
    assert h._tags.chip("experimenter").text() == "Experimenter  AB"


def test_title_edit_writes_name(qtbot, mock_core):
    h = _header(qtbot, mock_core)
    h._title.setText("Pup retrieval")
    h._title.editingFinished.emit()
    assert mock_core.session.metadata.name == "Pup retrieval"


def test_pill_follows_state(qtbot, mock_core):
    h = _header(qtbot, mock_core)
    for state, text in (
        ("IDLE", "READY"),
        ("RUNNING", "RUNNING"),
        ("PAUSED", "PAUSED"),
        ("STOPPED", "READY"),
    ):
        h.update_state(state)
        assert h._pill.text() == text, state


def test_needs_details_overrides_ready_only(qtbot, mock_core):
    h = _header(qtbot, mock_core)
    h.set_needs_details("Dam ID")
    assert h._pill.text() == "NEEDS DETAILS"
    h.update_state("RUNNING")
    assert h._pill.text() == "RUNNING"


def test_locked_while_live(qtbot, mock_core):
    h = _header(qtbot, mock_core)
    h.update_state("RUNNING")
    assert h._title.isReadOnly()
    h._tags.chip("notes").click()
    assert h._tags.editor() is None
    h.update_state("STOPPED")
    assert not h._title.isReadOnly()


def test_refresh_is_not_an_edit(qtbot, mock_core):
    h = _header(qtbot, mock_core)
    new = ExperimentSession()
    new.metadata.name = "Second"
    mock_core.session = new
    h.refresh()
    assert h._title.text() == "Second"
    assert not new.is_dirty


from glider.core import presets  # noqa: E402
from glider.core.experiment_session import RunField  # noqa: E402


def _preset_header(qtbot, mock_core):
    h = _header(qtbot, mock_core, name="Pup run", experiment_type="Pup retrieval")
    md = mock_core.session.metadata
    md.run_fields = [
        RunField("dam_id", "Dam ID", True),
        RunField("litter", "Litter size", False, "6"),
    ]
    md.reset_run_values()
    h.refresh()
    return h


def test_preset_tags_are_run_fields_then_experimenter_notes(qtbot, mock_core):
    h = _preset_header(qtbot, mock_core)
    keys = [f.key for f in h.tag_fields()]
    assert keys == ["dam_id", "litter", "experimenter", "notes"]
    assert h._tags.chip("litter").text() == "Litter size  6"


def test_run_value_edit_is_not_dirty(qtbot, mock_core):
    h = _preset_header(qtbot, mock_core)
    mock_core.session._mark_clean()
    with qtbot.waitSignal(h.run_values_changed):
        h._tags.value_edited.emit("dam_id", "D14")
    assert mock_core.session.metadata.run_values["dam_id"] == "D14"
    assert not mock_core.session.is_dirty


def test_pill_needs_details_until_required_field_filled(qtbot, mock_core):
    h = _preset_header(qtbot, mock_core)
    assert h._pill.text() == "NEEDS DETAILS"
    h._tags.value_edited.emit("dam_id", "D14")
    assert h._pill.text() == "READY"


def test_picker_lists_presets(qtbot, mock_core, tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "presets_dir", lambda: tmp_path)
    from glider.core.experiment_session import ExperimentSession

    path = presets.save_preset(ExperimentSession(), "Open field", tmp_path)
    h = _preset_header(qtbot, mock_core)
    h._menu.aboutToShow.emit()
    texts = [a.text() for a in h._menu.actions() if a.text()]
    assert "Open field" in texts
    assert "New preset…" in texts and "Show presets folder" in texts
    action = next(a for a in h._menu.actions() if a.text() == "Open field")
    with qtbot.waitSignal(h.preset_chosen) as sig:
        action.trigger()
    assert sig.args == [str(path)]


def test_picker_disabled_while_live(qtbot, mock_core):
    h = _preset_header(qtbot, mock_core)
    h.update_state("RUNNING")
    assert not h._picker.isEnabled()
    h.update_state("STOPPED")
    assert h._picker.isEnabled()


def test_run_info(qtbot, mock_core):
    h = _preset_header(qtbot, mock_core)
    h.set_run_info(4, "~/GLIDER/pup-retrieval/d14_20261011_run04")
    assert h._run_label.text() == "Run 4 today"
    assert h._tags._trailing.text().endswith("run04")
