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
