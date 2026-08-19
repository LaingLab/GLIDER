"""The lab's vocabulary: what it stores, and what it refuses to lose.

Case-insensitive de-duplication is the whole point of this module, not a
nicety -- "Control" and "control" splitting one treatment group into two is
the failure that makes a cohort comparison silently wrong, and no downstream
code can tell the two apart afterwards.
"""

import json

from glider.core.vocabulary import LISTS, Vocabulary, load, save


def test_a_missing_file_yields_defaults(tmp_path):
    vocab = load(tmp_path)

    assert isinstance(vocab, Vocabulary)
    assert vocab.groups == []
    assert vocab.strains == []
    assert "IP" in vocab.routes
    assert "Male" in vocab.sexes


def test_round_trip(tmp_path):
    vocab = load(tmp_path)
    vocab.add("groups", "Drug A")
    save(vocab, tmp_path)

    assert load(tmp_path).groups == ["Drug A"]


def test_a_malformed_file_yields_defaults_rather_than_raising(tmp_path):
    """A broken vocabulary must never stop the app starting."""
    (tmp_path / "vocabulary.json").write_text("{ not json", encoding="utf-8")

    vocab = load(tmp_path)

    assert vocab.groups == []
    assert "IP" in vocab.routes


def test_a_file_that_is_not_an_object_yields_defaults(tmp_path):
    (tmp_path / "vocabulary.json").write_text("[1, 2, 3]", encoding="utf-8")

    assert load(tmp_path).groups == []


def test_case_insensitive_dedup(tmp_path):
    """The failure this module exists to prevent."""
    vocab = load(tmp_path)
    vocab.add("groups", "Control")
    vocab.add("groups", "control")
    vocab.add("groups", "  CONTROL  ")

    assert vocab.groups == ["Control"]


def test_first_spelling_wins(tmp_path):
    vocab = load(tmp_path)
    vocab.add("groups", "Drug A")
    vocab.add("groups", "drug a")

    assert vocab.groups == ["Drug A"]


def test_entry_order_is_preserved(tmp_path):
    vocab = load(tmp_path)
    for name in ("Vehicle", "Drug A", "Control"):
        vocab.add("groups", name)

    assert vocab.groups == ["Vehicle", "Drug A", "Control"]


def test_whitespace_only_is_rejected(tmp_path):
    vocab = load(tmp_path)

    assert vocab.add("groups", "   ") is False
    assert vocab.groups == []


def test_values_are_stored_trimmed(tmp_path):
    vocab = load(tmp_path)
    vocab.add("strains", "  C57BL/6J  ")

    assert vocab.strains == ["C57BL/6J"]


def test_add_reports_whether_it_changed_anything(tmp_path):
    """The subject dialog only re-saves when something was learned."""
    vocab = load(tmp_path)

    assert vocab.add("groups", "Control") is True
    assert vocab.add("groups", "Control") is False


def test_remove(tmp_path):
    vocab = load(tmp_path)
    vocab.add("groups", "Control")

    assert vocab.remove("groups", "control") is True
    assert vocab.groups == []


def test_unknown_list_name_is_refused(tmp_path):
    vocab = load(tmp_path)

    try:
        vocab.add("colours", "red")
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown list")


def test_an_unwritable_directory_reports_rather_than_raising(tmp_path, monkeypatch):
    """A read-only home must not take the app down; the caller shows the failure."""
    vocab = load(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr("pathlib.Path.write_text", boom)

    assert save(vocab, tmp_path) is False


def test_saved_file_is_readable_json(tmp_path):
    vocab = load(tmp_path)
    vocab.add("groups", "Control")
    save(vocab, tmp_path)

    data = json.loads((tmp_path / "vocabulary.json").read_text(encoding="utf-8"))
    assert data["groups"] == ["Control"]
    assert data["schema_version"] == "1.0"


def test_every_declared_list_round_trips(tmp_path):
    """Guards against a list being added to LISTS but forgotten in save/load."""
    vocab = load(tmp_path)
    for name in LISTS:
        vocab.add(name, f"value-for-{name}")
    save(vocab, tmp_path)

    reloaded = load(tmp_path)
    for name in LISTS:
        assert f"value-for-{name}" in reloaded.get(name)
