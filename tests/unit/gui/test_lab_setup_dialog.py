"""The lab setup form: where a lab's vocabulary is defined before it is used.

Three properties here are load-bearing, and each of them is a way the feature
fails quietly rather than loudly.

**Skip is a real exit.** The person doing first launch is often not the person
who knows the lab's strains. A wizard that punishes skipping gets junk typed
into it, so Skip closes the dialog and writes nothing at all.

**Done must not lose what was typed.** ``vocabulary.save`` returns ``False``
rather than raising when the library directory cannot be written, so a dialog
that ignores the return value closes on a failed write and silently discards
the whole vocabulary the user just entered.

**Every edit goes through ``Vocabulary.add``/``remove``.** ``get`` hands out
the live list, so appending to it directly bypasses the case- and
Unicode-folding that is the entire reason this feature exists -- ``Control``
and ``control`` would become two treatment groups again, entered through the
very form meant to prevent that.
"""

import pytest
from PyQt6.QtCore import Qt

from glider.core.vocabulary import LISTS, Vocabulary, load, save
from glider.gui.dialogs.lab_setup_dialog import LabSetupDialog


@pytest.fixture(autouse=True)
def library_dir(tmp_path, monkeypatch):
    """Point the vocabulary file at a tmp dir, never the developer's ~/.glider."""
    from glider.core.config import get_config

    path = tmp_path / "library"
    path.mkdir()
    monkeypatch.setattr(get_config().paths, "library_dir", path)
    return path


@pytest.fixture
def dialog(qtbot):
    """A dialog that has been ``show()``n, as every real one has.

    Showing is not cosmetic. ``QDialog.showEvent`` promotes the first
    auto-default button in a ``QDialogButtonBox`` to the window's default
    button, so Done is ``isDefault() is False`` before ``show()`` and True
    after. A suite that never shows the dialog therefore tests a widget in a
    state no user is ever in, and cannot see Return reaching the default
    button at all -- which is exactly how "type a term and press Enter" came
    to also save the file and close the form.
    """

    def _make(**kwargs):
        d = LabSetupDialog(**kwargs)
        qtbot.addWidget(d)
        d.show()
        return d

    return _make


def type_value(qtbot, editor, value: str) -> None:
    """Enter a value the way a user does: type it, press Enter."""
    editor.entry.setText(value)
    qtbot.keyClick(editor.entry, Qt.Key.Key_Return)


# --- Every list is offered -----------------------------------------------------


def test_every_vocabulary_list_has_an_editor(dialog):
    """Iterating LISTS rather than naming five: a sixth list must not be
    silently unreachable in the only form that defines them."""
    d = dialog()

    assert set(d.editors) == set(LISTS)


def test_the_editors_appear_in_the_order_the_vocabulary_declares(dialog):
    assert list(dialog().editors) == list(LISTS)


def test_every_editor_is_labelled(dialog):
    d = dialog()

    for name in LISTS:
        assert d.editors[name].title().strip()


# --- What the form starts from -------------------------------------------------


def test_the_form_starts_from_the_lab_vocabulary_on_disk(dialog, library_dir):
    stored = Vocabulary()
    stored.add("groups", "Cohort B")
    assert save(stored, library_dir) is True

    assert dialog().editors["groups"].values() == ["Cohort B"]


def test_a_fresh_install_still_shows_the_standard_routes(dialog):
    """`load` yields defaults for routes and sexes, so setup starts useful."""
    assert "IP" in dialog().editors["routes"].values()


def test_an_injected_vocabulary_is_shown(dialog):
    vocab = Vocabulary(groups=["Control", "Drug A"])

    assert dialog(vocabulary=vocab).editors["groups"].values() == ["Control", "Drug A"]


# --- Adding --------------------------------------------------------------------


def test_typing_a_value_and_pressing_enter_adds_it(dialog, qtbot):
    d = dialog(vocabulary=Vocabulary())

    type_value(qtbot, d.editors["groups"], "Control")

    assert d.editors["groups"].values() == ["Control"]


def test_the_entry_field_clears_so_the_next_value_can_be_typed(dialog, qtbot):
    d = dialog(vocabulary=Vocabulary())

    type_value(qtbot, d.editors["groups"], "Control")

    assert d.editors["groups"].entry.text() == ""


def test_each_added_value_gets_its_own_row(dialog, qtbot):
    d = dialog(vocabulary=Vocabulary())
    editor = d.editors["groups"]

    type_value(qtbot, editor, "Control")
    type_value(qtbot, editor, "Drug A")

    assert editor.row_count() == 2
    assert editor.values() == ["Control", "Drug A"]


def test_a_value_is_stored_trimmed(dialog, qtbot):
    d = dialog(vocabulary=Vocabulary())

    type_value(qtbot, d.editors["groups"], "  Control  ")

    assert d.editors["groups"].values() == ["Control"]


def test_pressing_enter_adds_the_term_and_leaves_the_form_open(dialog, qtbot, library_dir):
    """The interaction the form documents must not also be the way out of it.

    ``QDialogButtonBox`` promotes its AcceptRole button to the window default
    once the dialog is shown, and ``QLineEdit`` lets Return fall through to the
    default button as well as emitting ``returnPressed``. So Enter added the
    term *and* pressed Done: the file was written and the form closed on the
    first term it was given. On a first launch that is the whole vocabulary --
    one entry -- because the caller burns the seen-it flag before the dialog
    opens and never offers it again.
    """
    d = dialog(vocabulary=Vocabulary())

    type_value(qtbot, d.editors["groups"], "Control")

    assert d.editors["groups"].values() == ["Control"]
    assert d.isVisible(), "Enter closed the form"
    assert not (library_dir / "vocabulary.json").exists(), "Enter saved the vocabulary"


def test_neither_button_is_the_window_default(dialog):
    """The mechanism behind the test above, pinned directly.

    Leave either button auto-default and Return in any of the five entry
    fields reaches it again.
    """
    d = dialog(vocabulary=Vocabulary())

    assert d.done_button.isDefault() is False
    assert d.skip_button.isDefault() is False
    assert d.done_button.autoDefault() is False
    assert d.skip_button.autoDefault() is False


def test_enter_stays_harmless_in_every_list(dialog, qtbot, library_dir):
    """Five entry fields, one shared default button: all five have to be safe."""
    d = dialog(vocabulary=Vocabulary(routes=[], sexes=[]))

    for name in LISTS:
        type_value(qtbot, d.editors[name], f"value-for-{name}")
        assert d.isVisible(), f"Enter in {name} closed the form"

    assert not (library_dir / "vocabulary.json").exists()


def test_every_list_accepts_input(dialog, qtbot):
    """Not just groups -- a list wired up but never connected looks fine."""
    d = dialog(vocabulary=Vocabulary(routes=[], sexes=[]))

    for name in LISTS:
        type_value(qtbot, d.editors[name], f"value-for-{name}")

    for name in LISTS:
        assert f"value-for-{name}" in d.editors[name].values()


# --- Refusals fold silently ----------------------------------------------------


def test_a_duplicate_folds_silently(dialog, qtbot):
    """Typing 'Control' twice earns one row and no scolding."""
    d = dialog(vocabulary=Vocabulary())
    editor = d.editors["groups"]

    type_value(qtbot, editor, "Control")
    type_value(qtbot, editor, "Control")

    assert editor.values() == ["Control"]
    assert editor.row_count() == 1
    assert d.error_label.text() == ""


def test_a_differently_cased_duplicate_folds_too(dialog, qtbot):
    """The failure the whole feature exists to prevent, entered by hand."""
    d = dialog(vocabulary=Vocabulary())
    editor = d.editors["groups"]

    type_value(qtbot, editor, "Control")
    type_value(qtbot, editor, "control")

    assert editor.values() == ["Control"]
    assert editor.row_count() == 1


def test_a_whitespace_only_value_is_refused(dialog, qtbot):
    d = dialog(vocabulary=Vocabulary())
    editor = d.editors["groups"]

    type_value(qtbot, editor, "   ")

    assert editor.values() == []
    assert editor.row_count() == 0


def test_an_empty_entry_is_refused(dialog, qtbot):
    d = dialog(vocabulary=Vocabulary())
    editor = d.editors["groups"]

    type_value(qtbot, editor, "")

    assert editor.row_count() == 0


# --- Removing ------------------------------------------------------------------


def test_removing_a_row_drops_the_value(dialog):
    d = dialog(vocabulary=Vocabulary(groups=["Control", "Drug A"]))
    editor = d.editors["groups"]

    editor.remove_button("Control").click()

    assert editor.values() == ["Drug A"]
    assert editor.row_count() == 1


def test_a_removed_value_can_be_added_again(dialog, qtbot):
    d = dialog(vocabulary=Vocabulary(groups=["Control"]))
    editor = d.editors["groups"]

    editor.remove_button("Control").click()
    type_value(qtbot, editor, "Control")

    assert editor.values() == ["Control"]
    assert editor.row_count() == 1


def test_removing_one_row_leaves_the_others_removable(dialog):
    """A rebuilt row list whose buttons still point at old values removes the
    wrong entry, which is invisible until the second click."""
    d = dialog(vocabulary=Vocabulary(groups=["Control", "Drug A", "Vehicle"]))
    editor = d.editors["groups"]

    editor.remove_button("Drug A").click()
    editor.remove_button("Vehicle").click()

    assert editor.values() == ["Control"]


# --- Skip ----------------------------------------------------------------------


def test_skip_writes_nothing(dialog, qtbot, library_dir):
    """Skip is a first-class exit, not a booby prize: no file, no half-vocabulary."""
    d = dialog(vocabulary=Vocabulary())
    type_value(qtbot, d.editors["groups"], "Control")

    d.skip_button.click()

    assert not (library_dir / "vocabulary.json").exists()


def test_skip_closes_the_dialog(dialog):
    d = dialog(vocabulary=Vocabulary())

    d.skip_button.click()

    assert not d.isVisible()


def test_skip_leaves_an_existing_vocabulary_file_untouched(dialog, qtbot, library_dir):
    stored = Vocabulary(groups=["Cohort B"])
    assert save(stored, library_dir) is True

    d = dialog()
    type_value(qtbot, d.editors["groups"], "Typed But Abandoned")
    d.skip_button.click()

    assert load(library_dir).groups == ["Cohort B"]


def test_the_form_does_not_mutate_the_vocabulary_it_was_given(dialog, qtbot):
    """Skip must cost nothing, in memory as well as on disk."""
    vocab = Vocabulary(groups=["Control"])
    d = dialog(vocabulary=vocab)

    type_value(qtbot, d.editors["groups"], "Drug A")
    d.editors["groups"].remove_button("Control").click()
    d.skip_button.click()

    assert vocab.groups == ["Control"]


# --- Done ----------------------------------------------------------------------


def test_done_writes_and_the_values_survive_a_reload(dialog, qtbot, library_dir):
    d = dialog(vocabulary=Vocabulary())
    type_value(qtbot, d.editors["groups"], "Control")
    type_value(qtbot, d.editors["strains"], "C57BL/6J")

    d.done_button.click()

    reloaded = load(library_dir)
    assert reloaded.groups == ["Control"]
    assert reloaded.strains == ["C57BL/6J"]


def test_done_persists_a_removal(dialog, library_dir):
    d = dialog(vocabulary=Vocabulary(groups=["Control", "Drug A"]))

    d.editors["groups"].remove_button("Control").click()
    d.done_button.click()

    assert load(library_dir).groups == ["Drug A"]


def test_done_closes_the_dialog(dialog):
    d = dialog(vocabulary=Vocabulary())

    d.done_button.click()

    assert not d.isVisible()


def test_a_saved_vocabulary_still_folds_case_on_the_next_visit(dialog, qtbot, library_dir):
    """Round trip: the fold has to survive the file, not just the widget."""
    d = dialog(vocabulary=Vocabulary())
    type_value(qtbot, d.editors["groups"], "Control")
    d.done_button.click()

    again = dialog()
    type_value(qtbot, again.editors["groups"], "control")

    assert again.editors["groups"].values() == ["Control"]


# --- A save that cannot happen -------------------------------------------------


def test_a_failed_save_keeps_the_dialog_open_with_a_visible_message(dialog, qtbot, tmp_path):
    """A file where the library directory should be: `save` returns False.

    Closing here would discard the whole vocabulary the user just typed, with
    nothing on disk and no explanation.
    """
    from glider.core.config import get_config

    (tmp_path / "blocker").write_text("not a directory", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(get_config().paths, "library_dir", tmp_path / "blocker" / "lib")
        d = dialog(vocabulary=Vocabulary())
        type_value(qtbot, d.editors["groups"], "Control")

        d.done_button.click()

        assert d.isVisible()
        assert "save" in d.error_label.text().lower()
        assert d.error_label.isVisible()


def test_a_failed_save_keeps_what_was_typed(dialog, qtbot, tmp_path):
    from glider.core.config import get_config

    (tmp_path / "blocker").write_text("not a directory", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(get_config().paths, "library_dir", tmp_path / "blocker" / "lib")
        d = dialog(vocabulary=Vocabulary())
        type_value(qtbot, d.editors["groups"], "Control")

        d.done_button.click()

        assert d.editors["groups"].values() == ["Control"]


def test_a_retried_save_that_succeeds_clears_the_message_and_closes(dialog, qtbot, tmp_path):
    """The user fixes the directory and clicks Done again; nothing was lost."""
    from glider.core.config import get_config

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        library_dir = blocker / "lib"
        patch.setattr(get_config().paths, "library_dir", library_dir)
        d = dialog(vocabulary=Vocabulary())
        type_value(qtbot, d.editors["groups"], "Control")
        d.done_button.click()
        assert d.isVisible()

        blocker.unlink()
        d.done_button.click()

        assert not d.isVisible()
        assert d.error_label.text() == ""
        assert load(library_dir).groups == ["Control"]


def test_skip_still_works_after_a_failed_save(dialog, qtbot, tmp_path):
    """The escape hatch must not be the thing that breaks when writing fails."""
    from glider.core.config import get_config

    (tmp_path / "blocker").write_text("not a directory", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(get_config().paths, "library_dir", tmp_path / "blocker" / "lib")
        d = dialog(vocabulary=Vocabulary())
        type_value(qtbot, d.editors["groups"], "Control")
        d.done_button.click()

        d.skip_button.click()

        assert not d.isVisible()
