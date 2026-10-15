# tests/unit/core/test_presets.py
import json
from datetime import date

import pytest

from glider.core import presets
from glider.core.experiment_session import ExperimentSession, RunField


@pytest.mark.parametrize(
    "text, slug",
    [
        ("Pup retrieval", "pup-retrieval"),
        ("Pup retrieval!!", "pup-retrieval"),
        ("  ", "preset"),
        ("Ünïcode Test", "unicode-test"),
        ("D14/A", "d14-a"),
    ],
)
def test_slugify_cases(text, slug):
    assert presets.slugify(text) == slug


def test_save_and_list(tmp_path):
    s = ExperimentSession()
    s.metadata.run_fields = [RunField("dam_id", "Dam ID", True)]
    path = presets.save_preset(s, "Pup retrieval", tmp_path)
    assert path == tmp_path / "pup-retrieval.glider"
    [info] = presets.list_presets(tmp_path)
    assert (info.name, info.field_labels) == ("Pup retrieval", ("Dam ID",))


def test_save_preset_never_overwrites(tmp_path):
    a = presets.save_preset(ExperimentSession(), "OFT", tmp_path)
    b = presets.save_preset(ExperimentSession(), "OFT", tmp_path)
    assert a != b and b.name == "oft-2.glider" and a.exists()


def test_saved_preset_has_no_run_values(tmp_path):
    s = ExperimentSession()
    s.metadata.run_values = {"dam_id": "D14"}
    path = presets.save_preset(s, "X", tmp_path)
    assert "run_values" not in json.loads(path.read_text())["metadata"]


def test_list_skips_unreadable(tmp_path):
    (tmp_path / "broken.glider").write_text("{nope")
    presets.save_preset(ExperimentSession(), "Good", tmp_path)
    assert [p.name for p in presets.list_presets(tmp_path)] == ["Good"]


def test_list_tolerates_malformed_run_fields(tmp_path):
    (tmp_path / "odd.glider").write_text(
        json.dumps(
            {"metadata": {"experiment_type": "Odd", "run_fields": ["junk", {"label": "Dam ID"}]}}
        )
    )
    (tmp_path / "bad.glider").write_text(
        json.dumps({"metadata": {"experiment_type": "Bad", "run_fields": "nope"}})
    )
    by_name = {p.name: p.field_labels for p in presets.list_presets(tmp_path)}
    assert by_name == {"Odd": ("Dam ID",), "Bad": ()}


def test_list_missing_folder_is_empty(tmp_path):
    assert presets.list_presets(tmp_path / "nope") == []


def test_run_folder_numbers_up(tmp_path):
    day = date(2026, 10, 11)
    first = presets.run_folder(tmp_path, "pup-retrieval", ["D14", "P5"], day)
    assert first == tmp_path / "pup-retrieval" / "d14_p5_20261011_run01"
    first.mkdir(parents=True)
    assert presets.next_run_number(tmp_path, "pup-retrieval", day) == 2
    assert presets.run_folder(tmp_path, "pup-retrieval", ["D15"], day).name == "d15_20261011_run02"


def test_run_folder_slugs_values(tmp_path):
    p = presets.run_folder(tmp_path, "pup", ["D14/A", "my dam"], date(2026, 10, 11))
    assert p.parent == tmp_path / "pup"
    assert p.name == "d14-a_my-dam_20261011_run01"


def test_run_folder_without_values(tmp_path):
    assert presets.run_folder(tmp_path, "oft", [], date(2026, 10, 11)).name == "20261011_run01"


def test_missing_required():
    fields = [RunField("a", "Dam ID", True), RunField("b", "Notes"), RunField("c", "PND", True)]
    assert presets.missing_required(fields, {"a": "D14", "c": " "}) == "PND"
    assert presets.missing_required(fields, {"a": "D14", "c": "P5"}) is None


def test_derive_keys_dedupes():
    assert presets.derive_keys(["Dam ID", "Dam-ID", "PND"]) == ["dam_id", "dam_id_2", "pnd"]


def test_validate_labels():
    assert presets.validate_labels(["Dam ID", ""]) == "Every field needs a label."
    assert presets.validate_labels(["Dam ID", "dam id"]) == "Two fields are both called “dam id”."
    assert presets.validate_labels(["Dam ID", "PND"]) is None


def test_list_skips_non_object_json(tmp_path):
    (tmp_path / "array.glider").write_text("[1, 2]")
    presets.save_preset(ExperimentSession(), "Good", tmp_path)
    assert [p.name for p in presets.list_presets(tmp_path)] == ["Good"]


def test_builtin_presets_name_the_protocol_and_ask_for_its_fields():
    shown = {
        p.name: [(f.key, f.label, f.required) for f in p.fields] for p in presets.BUILTIN_PRESETS
    }
    assert shown == {
        "Pup Retrieval": [
            ("dam_id", "Dam ID", True),
            ("postnatal_day", "Postnatal day", False),
            ("litter_size", "Litter size", False),
        ],
        "Open Field": [("animal_id", "Animal ID", True), ("group", "Group", False)],
    }
    for p in presets.BUILTIN_PRESETS:
        assert [f.key for f in p.fields] == presets.derive_keys([f.label for f in p.fields])


def test_apply_builtin_sets_protocol_fields_and_fresh_values():
    s = ExperimentSession()
    s.metadata.protocol = "OFT-001"
    s.metadata.run_fields = [RunField("old", "Old")]
    s.metadata.run_values = {"old": "x"}
    s._mark_clean()
    presets.apply_builtin(s, "Pup Retrieval")
    md = s.metadata
    assert md.experiment_type == "Pup Retrieval"
    assert [f.label for f in md.run_fields] == ["Dam ID", "Postnatal day", "Litter size"]
    assert md.run_values == {"dam_id": "", "postnatal_day": "", "litter_size": ""}
    assert md.protocol == "OFT-001"
    assert s.is_dirty
    # Copies: editing the session's fields never edits the built-in.
    md.run_fields[0].label = "Mother"
    assert presets.BUILTIN_PRESETS[0].fields[0].label == "Dam ID"


def test_apply_builtin_refuses_an_unknown_name():
    with pytest.raises(KeyError):
        presets.apply_builtin(ExperimentSession(), "Rotarod")
