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
