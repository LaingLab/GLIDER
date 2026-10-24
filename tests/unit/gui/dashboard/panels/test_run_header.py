import pytest

from glider.core.experiment_session import ExperimentSession
from glider.gui.dashboard.panels.run_header import RunHeader

pytestmark = pytest.mark.usefixtures("qtbot")


def _header(qtbot, mock_core, touch=False, **meta):
    session = ExperimentSession()
    for k, v in meta.items():
        setattr(session.metadata, k, v)
    session._mark_clean()
    mock_core.session = session
    h = RunHeader(mock_core, touch=touch)
    qtbot.addWidget(h)
    return h


def test_shows_name_and_metadata_tags(qtbot, mock_core):
    h = _header(qtbot, mock_core, name="OF 3", experimenter="GB", protocol="P-17")
    assert h._title.text() == "OF 3"
    assert h._tags.chip("experimenter").text() == "Experimenter  GB"
    # The old free-text protocol is kept in the file but no longer shown.
    assert [f.key for f in h.tag_fields()] == ["experimenter", "notes"]


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


@pytest.mark.parametrize("state", ["STOPPING", "ERROR", "INITIALIZING"])
def test_pill_shows_non_ready_states(qtbot, mock_core, state):
    h = _header(qtbot, mock_core)
    h.update_state(state)
    assert h._pill.text() == state
    assert h._pill.property("statusState") == state
    assert not h._title.isReadOnly()  # only RUNNING/PAUSED lock editing


def test_pill_keeps_its_size_in_preset_mode(qtbot, mock_core):
    h = _preset_header(qtbot, mock_core)
    h.resize(1200, 80)
    h.show()
    qtbot.waitExposed(h)
    assert h._pill.width() < 300


def test_preset_subtitles_show_as_tooltips(qtbot, mock_core):
    h = _header(qtbot, mock_core)
    assert h._menu.toolTipsVisible()


def test_experiment_type_alone_keeps_the_editable_title(qtbot, mock_core):
    h = _header(qtbot, mock_core, name="Old file", experiment_type="Open field")
    assert h._title.isVisibleTo(h)


def test_header_commit_pending_commits_an_open_tag(qtbot, mock_core):
    h = _header(qtbot, mock_core)
    h._tags.chip("notes").click()
    h._tags.editor().setText("nest at 09:10")
    h.commit_pending()
    assert mock_core.session.metadata.notes == "nest at 09:10"


def test_pill_says_not_ready_when_board_or_flow_missing(qtbot, mock_core):
    h = _header(qtbot, mock_core)
    h.set_ready(False)
    assert h._pill.text() == "NOT READY"
    h.set_ready(True)
    assert h._pill.text() == "READY"


def test_missing_field_outranks_not_ready(qtbot, mock_core):
    h = _preset_header(qtbot, mock_core)
    h.set_ready(False)
    assert h._pill.text() == "NEEDS DETAILS"


def test_non_preset_picker_sits_right_after_the_title(qtbot, mock_core):
    h = _header(qtbot, mock_core, name="Camera Recording 5 min + LED Blink")
    h.resize(1400, 120)
    h.show()
    qtbot.waitExposed(h)
    row = h._title_row
    assert row.indexOf(h._title) < row.indexOf(h._picker)
    gap = h._picker.geometry().left() - h._title.geometry().right()
    assert gap < 40


def _menu_texts(h):
    h._menu.aboutToShow.emit()
    return [a.text() for a in h._menu.actions() if a.text()]


def test_menu_lists_the_built_ins_first(qtbot, mock_core, tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "presets_dir", lambda: tmp_path)
    h = _header(qtbot, mock_core)
    texts = _menu_texts(h)
    assert texts[:4] == ["Built-in", "Pup Retrieval", "Open Field", "No presets yet"]
    header = h._menu.actions()[0]
    assert not header.isEnabled()


def test_a_built_in_announces_its_name(qtbot, mock_core, tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "presets_dir", lambda: tmp_path)
    h = _header(qtbot, mock_core)
    h._menu.aboutToShow.emit()
    action = next(a for a in h._menu.actions() if a.text() == "Open Field")
    with qtbot.waitSignal(h.builtin_preset_chosen) as sig:
        action.trigger()
    assert sig.args == ["Open Field"]


def _checked(h):
    h._menu.aboutToShow.emit()
    return [a.text() for a in h._menu.actions() if a.isChecked()]


def test_the_built_in_matching_the_protocol_is_checked(qtbot, mock_core, tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "presets_dir", lambda: tmp_path)
    h = _header(qtbot, mock_core, experiment_type="Pup Retrieval")
    assert _checked(h) == ["Pup Retrieval"]


def test_a_user_preset_file_is_checked_not_the_built_in(qtbot, mock_core, tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "presets_dir", lambda: tmp_path)
    path = presets.save_preset(ExperimentSession(), "Pup Retrieval", tmp_path)
    h = _header(qtbot, mock_core, experiment_type="Pup Retrieval")
    mock_core.session._file_path = str(path)
    texts = _menu_texts(h)
    assert texts.count("Pup Retrieval") == 2  # the built-in and the user's own
    assert "No presets yet" not in texts
    checked = [a for a in h._menu.actions() if a.isChecked()]
    assert len(checked) == 1
    assert h._menu.actions().index(checked[0]) > texts.index("Open Field")


def test_touch_header_hides_title_and_picker(qtbot, mock_core):
    h = _header(qtbot, mock_core, touch=True)
    assert not h._title.isVisibleTo(h) and not h._picker.isVisibleTo(h)


def test_touch_header_stays_without_title_after_a_refresh(qtbot, mock_core):
    h = _header(qtbot, mock_core, touch=True, name="OF 3")
    h.refresh()
    assert not h._title.isVisibleTo(h)


def test_touch_header_lays_tags_out_in_two_columns(qtbot, mock_core):
    from PyQt6.QtWidgets import QGridLayout

    h = _header(qtbot, mock_core, touch=True)
    assert isinstance(h._tags.layout(), QGridLayout)


def test_desktop_header_keeps_title_picker_and_one_row(qtbot, mock_core):
    from PyQt6.QtWidgets import QHBoxLayout

    h = _header(qtbot, mock_core)
    assert h._title.isVisibleTo(h) and h._picker.isVisibleTo(h)
    assert isinstance(h._tags.layout(), QHBoxLayout)


def test_the_run_can_be_assigned_an_animal_from_the_animals_list(qtbot, mock_core):
    """#215: the animal picked here is the session's active subject, which the
    recorders already write into every CSV header."""
    from glider.core.experiment_session import Subject

    h = _header(qtbot, mock_core)
    md = mock_core.session.metadata
    first, second = Subject(subject_id="M001", group="Control"), Subject(subject_id="M002")
    md.add_subject(first)
    md.add_subject(second)
    h.refresh()

    assert [h.animal.itemText(i) for i in range(h.animal.count())] == [
        "No animal",
        "M001 (Control)",
        "M002",
    ]
    assert h.animal.currentData() == first.id  # the first animal starts active

    h.animal.setCurrentIndex(2)
    assert md.get_active_subject() is second
    assert mock_core.session.is_dirty

    h.update_state("RUNNING")
    assert not h.animal.isEnabled()  # a live run's animal is fixed
