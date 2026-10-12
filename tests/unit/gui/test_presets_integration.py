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
