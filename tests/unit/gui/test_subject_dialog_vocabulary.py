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

    subject = d.get_subject()

    assert subject.group == "Control"
    assert subject.strain == "C57BL/6J"
    assert subject.solution == "Saline"
    assert subject.route == "IP"


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


def test_editing_a_subject_whose_values_predate_the_vocabulary(dialog, vocab):
    """An old .glider file must load unchanged even if nothing matches."""
    existing = Subject(subject_id="M001", group="Legacy Cohort", strain="Wistar")
    d = dialog(subject=existing, vocabulary=vocab)

    assert d._group_combo.currentText() == "Legacy Cohort"
    assert d._strain_combo.currentText() == "Wistar"

    subject = d.get_subject()
    assert subject.group == "Legacy Cohort"
    assert subject.strain == "Wistar"


# --- Learn on save ------------------------------------------------------------


def test_a_novel_value_is_learned_and_written(dialog, vocab, library_dir):
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._strain_combo.setCurrentText("BALB/c")

    d.get_subject()

    assert "BALB/c" in vocab.strains
    assert "BALB/c" in load(library_dir).strains


def test_a_novel_value_typed_while_editing_is_also_learned(dialog, vocab, library_dir):
    existing = Subject(subject_id="M001", group="Control")
    d = dialog(subject=existing, vocabulary=vocab)
    d._group_combo.setCurrentText("Drug Q")

    d.get_subject()

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

    d.get_subject()

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

    d.get_subject()

    assert vocab.groups == ["Control", "Drug A"]
    assert saves == []


def test_a_failed_save_does_not_cost_the_user_the_subject(dialog, vocab, monkeypatch):
    """Losing a vocabulary entry must never lose the animal just typed."""
    monkeypatch.setattr(subject_dialog_module, "save", lambda v, d: False)
    d = dialog(vocabulary=vocab)
    d._subject_id_edit.setText("M001")
    d._group_combo.setCurrentText("Drug Z")

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

    d.get_subject()

    assert saves == []
    assert "" not in vocab.groups
    assert "" not in vocab.strains
    assert "" not in vocab.solutions
