import pytest

from glider.core import presets
from glider.core.experiment_session import RunField

pytestmark = pytest.mark.usefixtures("qtbot")

FIELDS = [RunField("animal_id", "Animal ID", True)]


@pytest.fixture
def window(main_window_factory, monkeypatch, tmp_path):
    monkeypatch.setattr(presets, "presets_dir", lambda: tmp_path / "presets")
    w = main_window_factory()
    monkeypatch.setattr(
        w, "_ask_preset_fields", lambda fields, name="", ask_name=False: ("Open field", FIELDS)
    )
    return w


def test_new_preset_from_scratch(window, tmp_path):
    window._on_new_preset()
    path = tmp_path / "presets" / "open-field.glider"
    assert path.exists()
    md = window._core.session.metadata
    assert md.experiment_type == "Open field" and md.run_fields == FIELDS
    assert str(window._core.session.file_path) == str(path)
    assert window._stack.currentIndex() == 0  # Design tab, to build the flow


def test_save_current_as_preset_copies_setup(window, tmp_path):
    window._core.session.metadata.name = "My OFT rig"
    window._on_save_as_preset()
    [info] = presets.list_presets(tmp_path / "presets")
    assert info.name == "Open field" and info.field_labels == ("Animal ID",)


def test_choose_preset_loads_it_with_default_values(window, tmp_path):
    window._on_new_preset()
    path = str(tmp_path / "presets" / "open-field.glider")
    window._core.session.metadata.run_values["animal_id"] = "M031"
    window._core.session._mark_clean()
    window._on_preset_chosen(path)
    assert window._core.session.metadata.run_values == {"animal_id": ""}
    assert window._run_header._tags.chip("animal_id").property("need") is True


def test_edit_fields_writes_back(window, tmp_path, monkeypatch):
    window._on_new_preset()
    new = [RunField("animal_id", "Animal ID", True), RunField("group", "Group")]
    monkeypatch.setattr(
        window, "_ask_preset_fields", lambda fields, name="", ask_name=False: ("Open field", new)
    )
    window._on_edit_preset_fields()
    [info] = presets.list_presets(tmp_path / "presets")
    assert info.field_labels == ("Animal ID", "Group")


def test_save_as_preset_keeps_working_file_and_unsaved_edits(window, tmp_path):
    orig = str(tmp_path / "mine.glider")
    window._core.session.save(orig)
    window._core.session.mark_dirty()
    window._on_save_as_preset()
    s = window._core.session
    assert s.file_path == orig and s.is_dirty
    assert s.metadata.run_fields == FIELDS
    assert (tmp_path / "presets" / "open-field.glider").exists()


def test_edit_fields_keeps_filled_values(window, monkeypatch):
    window._on_new_preset()
    window._core.session.metadata.run_values["animal_id"] = "M031"
    new = [RunField("animal_id", "Animal ID", True), RunField("group", "Group")]
    monkeypatch.setattr(
        window, "_ask_preset_fields", lambda fields, name="", ask_name=False: ("Open field", new)
    )
    window._on_edit_preset_fields()
    assert window._core.session.metadata.run_values == {"animal_id": "M031", "group": ""}


def test_new_preset_cancelled_at_save_prompt_leaves_session_alone(window, tmp_path, monkeypatch):
    s = window._core.session
    s.metadata.name = "Mine"
    orig = str(tmp_path / "mine.glider")
    s.save(orig)
    asked = []
    monkeypatch.setattr(window, "_check_save", lambda: asked.append(1) or False)
    window._on_new_preset()
    monkeypatch.setattr(window, "_check_save", lambda: True)  # let the window close
    assert window._core.session is s
    assert s.metadata.name == "Mine" and s.metadata.run_fields == []
    assert str(s.file_path) == orig
    assert not (tmp_path / "presets").exists()
    assert asked == [1]  # one prompt, not two


def test_save_as_preset_syncs_live_settings_first(window, monkeypatch):
    synced = []
    monkeypatch.setattr(window._core, "sync_session_for_save", lambda: synced.append(1))
    window._on_save_as_preset()
    assert synced == [1]


def test_new_preset_prompts_to_save_once(window, tmp_path, monkeypatch):
    answers = iter([True, False])  # a second prompt would be cancelled
    asked = []
    monkeypatch.setattr(window, "_check_save", lambda: asked.append(1) or next(answers))
    window._on_new_preset()
    monkeypatch.setattr(window, "_check_save", lambda: True)  # let the window close
    assert asked == [1]
    assert window._core.session.metadata.run_fields == FIELDS
    assert (tmp_path / "presets" / "open-field.glider").exists()


def test_a_built_in_preset_applies_to_the_open_experiment(window, qtbot, monkeypatch):
    from glider.core.experiment_session import NodeConfig

    s = window._core.session
    s.metadata.name = "My rig"
    s.metadata.protocol = "OFT-001"
    s.add_node(NodeConfig("n1", "Delay"))
    s._mark_clean()
    window._open_session_review()
    review = window._session_review_window
    qtbot.addWidget(review)
    window._experiment_detail_pages()
    prompts = []
    monkeypatch.setattr(window, "_check_save", lambda: prompts.append(1) or True)

    window._run_header.builtin_preset_chosen.emit("Pup Retrieval")

    assert window._core.session is s and prompts == []
    md = s.metadata
    assert md.experiment_type == "Pup Retrieval"
    assert [f.label for f in md.run_fields] == ["Dam ID", "Postnatal day", "Litter size"]
    assert md.run_values == {"dam_id": "", "postnatal_day": "", "litter_size": ""}
    assert [n.id for n in s.flow.nodes] == ["n1"]
    assert (md.name, md.protocol) == ("My rig", "OFT-001")
    assert s.is_dirty
    assert review._type_combo.currentText() == "Pup Retrieval"
    assert window._experiment_details._protocol_combo.currentText() == "Pup Retrieval"
    assert window._run_header._picker.text() == "Pup Retrieval ▾"
    assert window._run_header._tags.chip("dam_id").property("need") is True


def test_a_built_in_preset_is_refused_while_running(window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from glider.core.experiment_session import SessionState

    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: told.append(a[2]))
    s = window._core.session
    s._state = SessionState.RUNNING
    window._run_header.builtin_preset_chosen.emit("Open Field")
    s._state = SessionState.IDLE  # let the window close
    assert s.metadata.experiment_type == "" and s.metadata.run_fields == []
    assert told == ["Stop the experiment before you switch presets."]


def test_a_built_in_over_one_of_my_presets_detaches_it(window, tmp_path, monkeypatch):
    """The open file is my own preset: the built-in must not end up saved into
    it -- not by Save, and not by Edit preset fields, which saves silently."""
    window._on_new_preset()  # my "Open field" preset, now the open file
    mine = tmp_path / "presets" / "open-field.glider"
    before = mine.read_bytes()

    window._run_header.builtin_preset_chosen.emit("Pup Retrieval")

    s = window._core.session
    assert s.file_path is None  # Save now asks where
    assert s.metadata.experiment_type == "Pup Retrieval"
    new = [RunField("dam_id", "Dam ID", True)]
    monkeypatch.setattr(
        window, "_ask_preset_fields", lambda fields, name="", ask_name=False: ("x", new)
    )
    window._on_edit_preset_fields()
    assert mine.read_bytes() == before
    window._run_header._rebuild_menu()
    ticked = [a.text() for a in window._run_header._menu.actions() if a.isChecked()]
    assert ticked == ["Pup Retrieval"]
    monkeypatch.setattr(window, "_check_save", lambda: True)  # let the window close


def test_a_built_in_keeps_an_ordinary_experiment_file_attached(window, tmp_path):
    s = window._core.session
    s.save(str(tmp_path / "rig.glider"))
    window._run_header.builtin_preset_chosen.emit("Open Field")
    assert s.file_path == str(tmp_path / "rig.glider")
