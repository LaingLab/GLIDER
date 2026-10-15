from glider.core.experiment_types import (
    ANALYSIS_PUP_RETRIEVAL,
    EXPERIMENT_TYPES,
    PRESETS,
    has_analysis,
    preset_for,
    type_label,
)


def test_the_list_starts_with_none_open_field_and_pup_retrieval():
    assert EXPERIMENT_TYPES[:3] == ("", "Open Field", "Pup Retrieval")


def test_pup_retrieval_is_unlocked_by_its_preset_and_presets_named_after_it():
    assert has_analysis("Pup Retrieval", ANALYSIS_PUP_RETRIEVAL)
    assert has_analysis("pup retrieval", ANALYSIS_PUP_RETRIEVAL)
    assert has_analysis("Pup Retrieval (P4, 5 min)", ANALYSIS_PUP_RETRIEVAL)
    assert not has_analysis("Open Field", ANALYSIS_PUP_RETRIEVAL)
    assert not has_analysis("Something new", ANALYSIS_PUP_RETRIEVAL)
    assert not has_analysis("", ANALYSIS_PUP_RETRIEVAL)
    assert not has_analysis(None, ANALYSIS_PUP_RETRIEVAL)


def test_preset_for_matches_exactly_then_the_longest_contained_name():
    assert preset_for("open field").name == "Open Field"
    assert preset_for("Open Field 2").name == "Open Field"
    assert preset_for("Something new") is None
    assert preset_for("").name == ""
    assert preset_for(None).name == ""


def test_an_analysis_belongs_only_to_the_presets_that_list_it():
    assert {p.name for p in PRESETS if ANALYSIS_PUP_RETRIEVAL in p.analyses} == {"Pup Retrieval"}


def test_labels():
    assert type_label("") == "None"
    assert type_label("Open Field") == "Open Field"
