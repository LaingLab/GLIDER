"""The subject form reads the lab's vocabulary, and learns from it.

Two things here are load-bearing and easy to get wrong.

First, ``SubjectDialog.get_subject`` has two branches -- it mutates an existing
Subject when editing and constructs a new one when creating -- and each branch
reads every field separately. A field converted from ``QLineEdit`` to
``QComboBox`` in only one branch leaves the other reading ``.text()`` off a
combo, which silently returns nothing. Tests below drive *both* branches on
purpose; a suite that only ever creates subjects passes against a dialog whose
edit path has stopped saving the group.

Second, learn-on-save must re-write the vocabulary file only when something was
actually learned, and must never let a failed write cost the user the subject
they just typed.
"""

import pytest

from glider.core.experiment_session import Subject
from glider.core.vocabulary import DEFAULT_ROUTES, DEFAULT_SEXES, Vocabulary, load, save
from glider.gui.dialogs import subject_dialog as subject_dialog_module
from glider.gui.dialogs.subject_dialog import ROUTE_OPTIONS, SEX_OPTIONS, SubjectDialog


def items(combo) -> list[str]:
    """Every entry a combo box currently offers."""
    return [combo.itemText(i) for i in range(combo.count())]


#: Every vocabulary-backed field: Subject attribute, widget, vocabulary list.
#: Tests parametrise over this rather than naming fields by hand, so a field
#: that behaves unlike the others cannot escape by being the one nobody
#: remembered to write a case for.
VOCABULARY_FIELDS = [
    ("group", "_group_combo", "groups"),
    ("strain", "_strain_combo", "strains"),
    ("solution", "_solution_combo", "solutions"),
    ("route", "_route_combo", "routes"),
    ("sex", "_sex_combo", "sexes"),
]
FIELD_IDS = [field[0] for field in VOCABULARY_FIELDS]

#: Values absent from the ``vocab`` fixture below.
UNRECOGNISED_VALUES = {
    "group": "Legacy Cohort",
    "strain": "Wistar",
    "solution": "DMSO",
    "route": "ICV",
    "sex": "M",
}

#: Values present in the ``vocab`` fixture below.
RECOGNISED_VALUES = {
    "group": "Control",
    "strain": "C57BL/6J",
    "solution": "Saline",
    "route": "IP",
    "sex": "Male",
}


@pytest.fixture(autouse=True)
def library_dir(tmp_path, monkeypatch):
    """Point the vocabulary file at a tmp dir, never the developer's ~/.glider."""
    from glider.core.config import get_config

    path = tmp_path / "library"
    path.mkdir()
    monkeypatch.setattr(get_config().paths, "library_dir", path)
    return path


@pytest.fixture
def vocab():
    """A lab that has been through setup."""
    return Vocabulary(
        groups=["Control", "Drug A"],
        strains=["C57BL/6J"],
        solutions=["Saline"],
        routes=["IP", "SC"],
        sexes=["Male", "Female"],
    )


@pytest.fixture
def dialog(qtbot):
    def _make(**kwargs):
        d = SubjectDialog(**kwargs)
        qtbot.addWidget(d)
        return d

    return _make


def press_ok(d) -> None:
    """Press OK the way a user does.

    Learning happens here and nowhere else. ``get_subject`` is a getter and
    must stay one: it used to write ``vocabulary.json`` as a side effect, so
    Cancel's safety depended on every caller remembering to check the result
    code rather than on the dialog.
    """
    d._on_accept()


# --- The defaults the two modules must agree on -------------------------------


def test_the_dialog_options_match_the_vocabulary_defaults():
    """The vocabulary module cannot import these (Qt), so pin them instead.

    Compared against the *stripped* form: the leading "" is a combo affordance
    meaning "nothing chosen", not a term any lab uses.
    """
    assert ROUTE_OPTIONS == ["", *DEFAULT_ROUTES]
    assert SEX_OPTIONS == ["", *DEFAULT_SEXES]


# --- Population ---------------------------------------------------------------


def test_group_combo_offers_the_vocabulary_groups(dialog, vocab):
    assert items(dialog(vocabulary=vocab)._group_combo) == ["", "Control", "Drug A"]


def test_strain_combo_offers_the_vocabulary_strains(dialog, vocab):
    assert items(dialog(vocabulary=vocab)._strain_combo) == ["", "C57BL/6J"]


def test_solution_combo_offers_the_vocabulary_solutions(dialog, vocab):
    assert items(dialog(vocabulary=vocab)._solution_combo) == ["", "Saline"]


def test_route_combo_offers_the_vocabulary_routes(dialog, vocab):
    """Route was already a combo; only its source of items changes."""
    assert items(dialog(vocabulary=vocab)._route_combo) == ["", "IP", "SC"]


def test_sex_combo_offers_the_vocabulary_sexes(dialog, vocab):
    assert items(dialog(vocabulary=vocab)._sex_combo) == ["", "Male", "Female"]


def test_nothing_is_preselected_for_a_new_subject(dialog, vocab):
    """A combo that defaults to its first item would silently label every new
    animal 'Control'. Every one of these must start blank."""
    d = dialog(vocabulary=vocab)

    assert d._group_combo.currentText() == ""
    assert d._strain_combo.currentText() == ""
    assert d._solution_combo.currentText() == ""
    assert d._route_combo.currentText() == ""
    assert d._sex_combo.currentText() == ""


def test_the_vocabulary_defaults_to_the_library_on_disk(dialog, library_dir):
    """No injected vocabulary: read the lab's own file."""
    stored = Vocabulary()
    stored.add("groups", "Cohort B")
    assert save(stored, library_dir) is True

    assert "Cohort B" in items(dialog()._group_combo)


# --- Free text still works ----------------------------------------------------


def test_a_group_not_in_the_vocabulary_can_still_be_typed(dialog, vocab):
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._group_combo.setCurrentText("Drug Z")

    assert d.get_subject().group == "Drug Z"


def test_an_empty_vocabulary_leaves_every_editable_field_usable(dialog, library_dir):
    """The experience of a user who skipped setup entirely."""
    empty = Vocabulary(groups=[], strains=[], solutions=[], routes=[], sexes=[])
    d = dialog(vocabulary=empty)
    d._subject_id_edit.setText("M001")
    d._group_combo.setCurrentText("Control")
    d._strain_combo.setCurrentText("C57BL/6J")
    d._solution_combo.setCurrentText("Saline")
    d._route_combo.setCurrentText("IP")
    d._sex_combo.setCurrentText("Male")

    subject = d.get_subject()

    assert subject.group == "Control"
    assert subject.strain == "C57BL/6J"
    assert subject.solution == "Saline"
    assert subject.route == "IP"
    assert subject.sex == "Male"


def test_a_fresh_install_still_offers_the_standard_routes_and_sexes(dialog, library_dir):
    """`load` on a missing file yields defaults, so skipping setup costs nothing."""
    d = dialog(vocabulary=load(library_dir))

    assert items(d._route_combo) == ROUTE_OPTIONS
    assert items(d._sex_combo) == SEX_OPTIONS


# --- get_subject: the create-new branch ---------------------------------------


def test_creating_a_new_subject_persists_every_combo_field(dialog, vocab):
    """`get_subject`'s create branch -- the one a naive test would cover alone."""
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._group_combo.setCurrentText("Drug A")
    d._strain_combo.setCurrentText("BALB/c")
    d._solution_combo.setCurrentText("Vehicle")
    d._route_combo.setCurrentText("SC")
    d._sex_combo.setCurrentText("Female")

    subject = d.get_subject()

    assert subject.subject_id == "M001"
    assert subject.group == "Drug A"
    assert subject.strain == "BALB/c"
    assert subject.solution == "Vehicle"
    assert subject.route == "SC"
    assert subject.sex == "Female"


# --- get_subject: the update-existing branch ----------------------------------


def test_editing_an_existing_subject_persists_every_combo_field(dialog, vocab):
    """`get_subject`'s update branch, which reads each field a second time.

    Miss it and editing an animal silently drops its group, strain and
    solution while creating one keeps working.
    """
    existing = Subject(
        subject_id="M001",
        group="Control",
        strain="C57BL/6J",
        solution="Saline",
        route="IP",
        sex="Male",
    )
    d = dialog(subject=existing, vocabulary=vocab)
    d._group_combo.setCurrentText("Drug A")
    d._strain_combo.setCurrentText("BALB/c")
    d._solution_combo.setCurrentText("Vehicle")
    d._route_combo.setCurrentText("SC")
    d._sex_combo.setCurrentText("Female")

    subject = d.get_subject()

    assert subject is existing
    assert subject.group == "Drug A"
    assert subject.strain == "BALB/c"
    assert subject.solution == "Vehicle"
    assert subject.route == "SC"
    assert subject.sex == "Female"


def test_editing_shows_the_subjects_existing_values(dialog, vocab):
    existing = Subject(
        subject_id="M001", group="Control", strain="C57BL/6J", solution="Saline", route="IP"
    )
    d = dialog(subject=existing, vocabulary=vocab)

    assert d._group_combo.currentText() == "Control"
    assert d._strain_combo.currentText() == "C57BL/6J"
    assert d._solution_combo.currentText() == "Saline"
    assert d._route_combo.currentText() == "IP"


# --- Recorded values survive a round trip, for every field alike ---------------
#
# These are parametrised over all five fields rather than written out for the
# ones that came to mind. Sex was the field that came to mind last, was built
# inline instead of through the shared helper, and was the only one that
# silently discarded a recorded value it did not recognise.


def test_every_vocabulary_field_is_editable(dialog, vocab):
    """One combo behaving unlike its four neighbours is how the sex data-loss
    bug hid: a reader had to notice the asymmetry to suspect it."""
    d = dialog(vocabulary=vocab)

    for _, widget, _ in VOCABULARY_FIELDS:
        assert getattr(d, widget).isEditable(), f"{widget} is not editable"


@pytest.mark.parametrize("attribute, widget, list_name", VOCABULARY_FIELDS, ids=FIELD_IDS)
def test_a_value_the_vocabulary_never_had_survives_open_and_save(
    dialog, vocab, attribute, widget, list_name
):
    """An old .glider file must load, display and re-save unchanged.

    Opening a subject and pressing OK must never alter what was recorded. This
    is a provenance tool; silently rewriting metadata is its worst failure.
    """
    recorded = UNRECOGNISED_VALUES[attribute]
    existing = Subject(subject_id="M001", **{attribute: recorded})

    d = dialog(subject=existing, vocabulary=vocab)

    assert getattr(d, widget).currentText() == recorded
    assert getattr(d.get_subject(), attribute) == recorded


@pytest.mark.parametrize("attribute, widget, list_name", VOCABULARY_FIELDS, ids=FIELD_IDS)
def test_a_value_removed_from_the_vocabulary_survives_open_and_save(
    dialog, vocab, attribute, widget, list_name
):
    """Tidying a list must not rewrite the subjects already recorded with it.

    Before the vocabulary existed these option sets were fixed constants, so
    no user could provoke this. Making them editable in Lab Setup is what
    turned it into a live data-loss path.
    """
    recorded = RECOGNISED_VALUES[attribute]
    existing = Subject(subject_id="M001", **{attribute: recorded})
    assert vocab.remove(list_name, recorded) is True  # the lab tidies its list

    d = dialog(subject=existing, vocabulary=vocab)

    assert getattr(d, widget).currentText() == recorded
    assert getattr(d.get_subject(), attribute) == recorded


# --- Learn on save ------------------------------------------------------------
#
# Learning is deliberately narrow: only a term the user *typed into this form*
# teaches the vocabulary. A term merely loaded from the subject does not, which
# is what makes removing one in Lab Setup stick. Re-learning whatever a subject
# still carried meant "remove" silently failed for exactly the terms anyone
# wants to remove -- the ones already in saved files -- with no user edit
# involved at all: opening a subject and pressing OK was enough.


@pytest.mark.parametrize("attribute, widget, list_name", VOCABULARY_FIELDS, ids=FIELD_IDS)
def test_a_novel_value_in_any_field_is_learned(
    dialog, vocab, library_dir, attribute, widget, list_name
):
    """Every vocabulary-backed combo must be reachable by learn-on-save.

    The sexes entry was dead code while the sex combo refused typing.
    """
    novel = UNRECOGNISED_VALUES[attribute]
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    getattr(d, widget).setCurrentText(novel)

    press_ok(d)

    assert novel in vocab.get(list_name)
    assert novel in load(library_dir).get(list_name)


def test_a_novel_value_is_learned_and_written(dialog, vocab, library_dir):
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._strain_combo.setCurrentText("BALB/c")

    press_ok(d)

    assert "BALB/c" in vocab.strains
    assert "BALB/c" in load(library_dir).strains


def test_a_novel_value_typed_while_editing_is_also_learned(dialog, vocab, library_dir):
    existing = Subject(subject_id="M001", group="Control")
    d = dialog(subject=existing, vocabulary=vocab)
    d._group_combo.setCurrentText("Drug Q")

    press_ok(d)

    assert "Drug Q" in load(library_dir).groups


def test_nothing_novel_means_the_file_is_not_rewritten(dialog, vocab, monkeypatch):
    """`add`'s bool return exists precisely so this save is skipped."""
    saves = []
    monkeypatch.setattr(
        subject_dialog_module, "save", lambda v, d: saves.append(d) is None and True
    )
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._group_combo.setCurrentText("Control")
    d._route_combo.setCurrentText("IP")

    press_ok(d)

    assert saves == []


def test_a_differently_cased_value_is_not_a_new_entry(dialog, vocab, monkeypatch):
    """Typing 'control' must not fork the group, nor trigger a write."""
    saves = []
    monkeypatch.setattr(
        subject_dialog_module, "save", lambda v, d: saves.append(d) is None and True
    )
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._group_combo.setCurrentText("control")

    press_ok(d)

    assert vocab.groups == ["Control", "Drug A"]
    assert saves == []


def test_a_failed_save_does_not_cost_the_user_the_subject(dialog, vocab, monkeypatch):
    """Losing a vocabulary entry must never lose the animal just typed."""
    monkeypatch.setattr(subject_dialog_module, "save", lambda v, d: False)
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._group_combo.setCurrentText("Drug Z")

    press_ok(d)
    subject = d.get_subject()

    assert subject.subject_id == "M001"
    assert subject.group == "Drug Z"


def test_blank_fields_are_not_learned(dialog, vocab, library_dir, monkeypatch):
    saves = []
    monkeypatch.setattr(
        subject_dialog_module, "save", lambda v, d: saves.append(d) is None and True
    )
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")

    press_ok(d)

    assert saves == []
    assert "" not in vocab.groups
    assert "" not in vocab.strains
    assert "" not in vocab.solutions


# --- Learning is a write, so it belongs on the accept path only ----------------


def test_reading_the_form_writes_nothing(dialog, vocab, library_dir):
    """``get_subject`` is a getter, and a getter must not touch the disk.

    It used to add to the vocabulary and rewrite ``vocabulary.json`` with no
    guard on the result code, so Cancel was safe only because the one caller
    happened to check ``Accepted`` first. That is a caller convention, not a
    property of the dialog, and the next caller does not inherit it.
    """
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._group_combo.setCurrentText("Drug Z")

    d.get_subject()

    assert "Drug Z" not in vocab.groups
    assert not (library_dir / "vocabulary.json").exists()


def test_a_cancelled_dialog_teaches_the_vocabulary_nothing(dialog, vocab, library_dir):
    """Cancel means the animal was not entered; nothing it named was either."""
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._group_combo.setCurrentText("Drug Z")

    d.reject()

    assert "Drug Z" not in vocab.groups
    assert not (library_dir / "vocabulary.json").exists()


def test_a_rejected_required_field_learns_nothing(dialog, vocab, library_dir, monkeypatch):
    """OK with no Subject ID is refused, so nothing on the form is real yet."""
    monkeypatch.setattr(subject_dialog_module.QMessageBox, "warning", lambda *a, **k: None)
    d = dialog(vocabulary=vocab)
    d._group_combo.setCurrentText("Drug Z")

    press_ok(d)

    assert "Drug Z" not in vocab.groups
    assert not (library_dir / "vocabulary.json").exists()


# --- Removal has to stick ------------------------------------------------------


@pytest.mark.parametrize("attribute, widget, list_name", VOCABULARY_FIELDS, ids=FIELD_IDS)
def test_a_term_the_lab_removed_is_not_relearned_from_an_old_subject(
    dialog, vocab, library_dir, attribute, widget, list_name
):
    """Removing a term in Lab Setup must not be undone by opening a subject.

    This needed no user edit at all: the term was still on the subject, so
    learn-on-save put it straight back. Which meant "remove" failed for exactly
    the terms a lab wants to remove -- the ones already written into saved
    files -- and the only way to make one stick was to never open an animal
    recorded with it again.
    """
    recorded = RECOGNISED_VALUES[attribute]
    existing = Subject(subject_id="M001", **{attribute: recorded})
    assert vocab.remove(list_name, recorded) is True  # the lab tidies its list
    before = list(vocab.get(list_name))

    d = dialog(subject=existing, vocabulary=vocab)
    press_ok(d)

    assert vocab.get(list_name) == before, f"{recorded} came back"
    assert not (library_dir / "vocabulary.json").exists()
    assert getattr(d.get_subject(), attribute) == recorded, "the subject's own value changed"


def test_a_removed_term_stays_removed_when_the_subject_is_otherwise_edited(
    dialog, vocab, library_dir
):
    """Editing another field must not smuggle the untouched one back in."""
    existing = Subject(subject_id="M001", group="Control", strain="C57BL/6J")
    assert vocab.remove("groups", "Control") is True

    d = dialog(subject=existing, vocabulary=vocab)
    d._strain_combo.setCurrentText("BALB/c")
    press_ok(d)

    assert vocab.groups == ["Drug A"]
    assert "BALB/c" in vocab.strains
    assert load(library_dir).groups == ["Drug A"]


def test_retyping_a_removed_term_teaches_it_again(dialog, vocab, library_dir):
    """Removal is durable, not permanent: the user is still in charge.

    A snapshot-and-compare would get this wrong -- the text ends where it
    started -- so what is tracked is whether the field was edited at all, not
    whether its value moved.
    """
    existing = Subject(subject_id="M001", group="Control")
    assert vocab.remove("groups", "Control") is True

    d = dialog(subject=existing, vocabulary=vocab)
    d._group_combo.setCurrentText("")
    d._group_combo.setCurrentText("Control")
    press_ok(d)

    assert vocab.groups == ["Drug A", "Control"]
    assert load(library_dir).groups == ["Drug A", "Control"]


# --- Completion must not rewrite what was typed --------------------------------


def test_typing_a_term_that_is_a_prefix_of_a_longer_one_records_what_was_typed(
    dialog, qtbot, library_dir
):
    """Qt's default inline completion silently upgrades a typed term.

    With ``Control Group`` on the list, typing ``Control`` left the longer
    term in the box and that is what got recorded -- so an animal the user
    labelled Control was filed under Control Group. Real pairs: ``C57`` vs
    ``C57BL/6J``, ``Saline`` vs ``Saline + vehicle``. A popup offers the longer
    term without ever putting it in the field.
    """
    d = dialog(vocabulary=Vocabulary(groups=["Control Group"]))
    d._subject_id_edit.setText("M001")

    # Shown, activated and focused: QLineEdit only runs its completer for a
    # focused widget, so an unshown dialog cannot see this at all.
    d.show()
    d.activateWindow()
    d._group_combo.setFocus()
    qtbot.wait(1)

    qtbot.keyClicks(d._group_combo.lineEdit(), "Control")

    assert d._group_combo.currentText() == "Control"
    assert d.get_subject().group == "Control"


def test_every_vocabulary_combo_completes_in_a_popup(dialog, vocab):
    """The mechanism, pinned for all five: inline completion is the default,
    so a combo built without changing it silently gets the old behaviour."""
    from PyQt6.QtWidgets import QCompleter

    d = dialog(vocabulary=vocab)

    for _, widget, _ in VOCABULARY_FIELDS:
        completer = getattr(d, widget).completer()
        assert completer is not None, f"{widget} has no completer"
        assert completer.completionMode() == QCompleter.CompletionMode.PopupCompletion, widget
