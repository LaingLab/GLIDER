"""The lab's vocabulary: what it stores, and what it refuses to lose.

Case-insensitive de-duplication is the whole point of this module, not a
nicety -- "Control" and "control" splitting one treatment group into two is
the failure that makes a cohort comparison silently wrong, and no downstream
code can tell the two apart afterwards.
"""

import json
import logging
import time

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


def test_a_partially_valid_file_keeps_what_it_can(tmp_path):
    """One bad key must not cost the lab the rest of its vocabulary."""
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "groups": ["Control", 7, None, "Drug A"],
                "strains": "not a list",
            }
        ),
        encoding="utf-8",
    )

    vocab = load(tmp_path)

    assert vocab.groups == ["Control", "Drug A"]  # non-strings dropped
    assert vocab.strains == []  # wrong type falls back to the default


def test_an_explicit_empty_list_clears_the_defaults(tmp_path):
    """A lab that deliberately emptied its routes must not find them back."""
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "1.0", "routes": []}), encoding="utf-8"
    )

    assert load(tmp_path).routes == []


def test_a_missing_key_keeps_its_defaults(tmp_path):
    """Absent is not the same as emptied -- only an explicit [] clears."""
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "1.0", "groups": ["Control"]}), encoding="utf-8"
    )

    vocab = load(tmp_path)

    assert vocab.groups == ["Control"]
    assert "IP" in vocab.routes
    assert "Male" in vocab.sexes


def test_values_in_a_hand_edited_file_are_de_duplicated_on_read(tmp_path):
    """The fold applies to what is read, not only to what is added."""
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "1.0", "groups": ["Control", "control", " CONTROL "]}),
        encoding="utf-8",
    )

    assert load(tmp_path).groups == ["Control"]


def test_a_hand_edited_file_keeps_the_first_spelling_of_a_duplicate(tmp_path):
    """First spelling wins on read, not just on add.

    Pinned separately because ``load`` builds its own de-duplication rather
    than deferring to ``add`` for every entry, and "keeps a duplicate's later
    spelling" is a silent way for that to be wrong.
    """
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "1.0", "groups": ["Drug A", "drug a", "Vehicle"]}),
        encoding="utf-8",
    )

    assert load(tmp_path).groups == ["Drug A", "Vehicle"]


def test_a_large_file_loads_in_linear_time(tmp_path):
    """``load`` must not be quadratic in the number of terms.

    It used to call ``add`` once per entry, and ``add`` re-derives the
    comparison key of every entry already read. The key normalises Unicode,
    case-folds and strips invisible characters, so the constant is not small
    either: 5 000 terms took about three seconds, and both ``SubjectDialog``
    and ``LabSetupDialog`` call ``load`` on the GUI thread in ``__init__``, so
    that was three seconds of frozen window before Add Subject painted. No lab
    hand-types 5 000 terms, but a merge between rigs or a scripted import
    reaches that easily.

    The bound is deliberately loose. This is a guard against a return to
    O(n^2) -- which at this size is orders of magnitude over the line -- not a
    measurement of the machine it runs on.
    """
    terms = [f"Group {index}" for index in range(5000)]
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "1.0", "groups": terms}), encoding="utf-8"
    )

    start = time.perf_counter()
    vocab = load(tmp_path)
    elapsed = time.perf_counter() - start

    assert vocab.groups == terms
    assert elapsed < 1.0, f"loading {len(terms)} terms took {elapsed:.2f}s"


def test_an_unknown_schema_version_still_loads(tmp_path):
    """A future v2 file must yield a usable vocabulary, not an empty form."""
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "99.0", "groups": ["Control"]}), encoding="utf-8"
    )

    assert load(tmp_path).groups == ["Control"]


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


def test_unicode_spellings_of_one_name_are_one_entry(tmp_path):
    """Same class of bug as Control/control, through a different door.

    macOS favours decomposed forms, Windows composed ones, so one lab typing
    the same strain on two machines produces two strings that a dropdown
    renders identically. Written as explicit escapes so this file's own
    encoding cannot quietly make the test vacuous.
    """
    composed = "Br\u00e9gy"  # precomposed e-acute
    decomposed = "Bre\u0301gy"  # e + combining acute
    assert composed != decomposed, "the two spellings must differ as raw strings"

    vocab = load(tmp_path)
    assert vocab.add("strains", composed) is True
    assert vocab.add("strains", decomposed) is False

    assert vocab.strains == [composed]
    assert vocab.remove("strains", decomposed) is True


def test_remove(tmp_path):
    vocab = load(tmp_path)
    vocab.add("groups", "Control")

    assert vocab.remove("groups", "control") is True
    assert vocab.groups == []


def test_remove_reports_when_there_was_nothing_to_remove(tmp_path):
    """Task 3's remove control reads this return value."""
    vocab = load(tmp_path)

    assert vocab.remove("groups", "Never added") is False


def test_unknown_list_name_is_refused(tmp_path):
    vocab = load(tmp_path)

    try:
        vocab.add("colours", "red")
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown list")


def test_an_unwritable_directory_reports_rather_than_raising(tmp_path):
    """A read-only home must not take the app down; the caller shows the failure.

    The failure is a real one -- a library path whose parent is a file, so
    ``mkdir`` genuinely fails -- rather than a patched write method, so this
    stays honest if the write mechanism is ever rewritten.
    """
    (tmp_path / "blocker").write_text("not a directory", encoding="utf-8")
    vocab = load(tmp_path)

    assert save(vocab, tmp_path / "blocker" / "lib") is False


def test_save_reports_success(tmp_path):
    """Without this, a save that always returned False would pass every test."""
    assert save(load(tmp_path), tmp_path) is True


def test_a_failed_save_leaves_the_previous_vocabulary_intact(tmp_path):
    """The whole vocabulary is one file; a broken write must not eat it."""
    vocab = load(tmp_path)
    vocab.add("groups", "Control")
    save(vocab, tmp_path)

    # A directory where the temporary file wants to be: the write fails, but
    # the vocabulary already on disk is never truncated.
    (tmp_path / "vocabulary.json.tmp").mkdir()
    vocab.add("groups", "Drug A")

    assert save(vocab, tmp_path) is False
    assert load(tmp_path).groups == ["Control"]


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


def test_zero_width_characters_do_not_fork_an_entry(tmp_path):
    """The commonest invisible passenger on a paste out of Excel or a PDF.

    ``str.strip()`` does not touch U+200B/U+200D/U+FEFF, and NFKC does not
    remove them either, so two rows that are pixel-identical on screen could
    sit in the same list -- the exact duplicate-cohort failure this module
    exists to prevent, arriving through a character nobody can see to delete.
    """
    for invisible in ("\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff"):
        vocab = load(tmp_path)
        assert vocab.add("groups", "Control") is True
        assert vocab.add("groups", f"{invisible}Control{invisible}") is False, repr(invisible)
        assert vocab.groups == ["Control"]
        assert vocab.remove("groups", f"Control{invisible}") is True


def test_a_value_that_is_only_zero_width_characters_is_refused(tmp_path):
    """Otherwise an invisible row appears in Lab Setup with nothing to click."""
    vocab = load(tmp_path)

    assert vocab.add("groups", "\u200b\ufeff") is False
    assert vocab.groups == []


def test_a_dropped_non_string_entry_is_logged(tmp_path, caplog):
    """The docs tell the reader a hand-edited file that GLIDER cannot use gets
    a warning. Dropping entries in silence means terms vanish from the subject
    form with nothing to explain it."""
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "1.0", "groups": ["Control", 7]}), encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="glider.core.vocabulary"):
        assert load(tmp_path).groups == ["Control"]

    assert any("groups" in r.getMessage() for r in caplog.records), caplog.text


def test_a_wrong_typed_list_is_logged(tmp_path, caplog):
    """Silently restoring the defaults for a list the user thought they had
    edited is the same vanishing act, one level up."""
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "1.0", "strains": "not a list"}), encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="glider.core.vocabulary"):
        assert load(tmp_path).strains == []

    assert any("strains" in r.getMessage() for r in caplog.records), caplog.text


def test_an_explicit_null_list_is_logged(tmp_path, caplog):
    """``"routes": null`` is a hand-edit mistake, not an omission.

    Absent means "I never touched this list", and restoring its defaults in
    silence is right. ``null`` means someone edited the file and got it wrong,
    and it takes the same route as ``"routes": "IP"`` -- the terms revert --
    so it deserves the same warning rather than the silent path.
    """
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "1.0", "routes": None}), encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="glider.core.vocabulary"):
        assert "IP" in load(tmp_path).routes

    assert any("routes" in r.getMessage() for r in caplog.records), caplog.text


def test_an_absent_list_is_not_logged(tmp_path, caplog):
    """The other side of it: absent stays silent, or every load warns."""
    (tmp_path / "vocabulary.json").write_text(
        json.dumps({"schema_version": "1.0", "groups": ["Control"]}), encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="glider.core.vocabulary"):
        assert "IP" in load(tmp_path).routes

    assert caplog.records == []


def test_a_well_formed_file_logs_nothing(tmp_path, caplog):
    """A warning on every normal load teaches people to ignore warnings."""
    stored = Vocabulary(groups=["Control"])
    assert save(stored, tmp_path) is True

    with caplog.at_level(logging.WARNING, logger="glider.core.vocabulary"):
        assert load(tmp_path).groups == ["Control"]

    assert caplog.records == []
