"""The experiment type is one value: every window that shows it follows the same one."""

import pytest

pytestmark = pytest.mark.usefixtures("qtbot")


def test_session_review_and_the_experiment_share_one_type(main_window_factory, qtbot):
    win = main_window_factory()
    win._open_session_review()
    review = win._session_review_window
    qtbot.addWidget(review)
    review._type_combo.setCurrentIndex(review._type_combo.findText("Pup Retrieval"))
    assert win._core.session.metadata.experiment_type == "Pup Retrieval"
    assert win._core.session.is_dirty
    win._set_experiment_type("Open Field")
    assert review._type_combo.currentText() == "Open Field"


def test_batch_pose_follows_the_shared_type(main_window_factory, qtbot):
    win = main_window_factory()
    win._set_experiment_type("Pup Retrieval")
    win._open_pose_batch()
    qtbot.addWidget(win._pose_batch_window)
    assert win._pose_batch_window._experiment_type == "Pup Retrieval"
    win._set_experiment_type("Open Field")
    assert win._pose_batch_window._experiment_type == "Open Field"


def test_the_experiment_tab_form_feeds_the_other_windows(main_window_factory, qtbot):
    """Picking in the Experiment tab's Protocol box reaches Session Review too."""
    win = main_window_factory()
    win._open_session_review()
    review = win._session_review_window
    qtbot.addWidget(review)
    form = win._experiment_details
    if form is None:  # the tab builds its sections on first look
        win._experiment_detail_pages()
        form = win._experiment_details
    form._protocol_combo.setCurrentIndex(form._protocol_combo.findText("Pup Retrieval"))
    assert win._core.session.metadata.experiment_type == "Pup Retrieval"
    assert review._type_combo.currentText() == "Pup Retrieval"


def test_loading_another_session_re_broadcasts_the_type(main_window_factory, qtbot):
    win = main_window_factory()
    win._open_session_review()
    review = win._session_review_window
    qtbot.addWidget(review)
    win._core.session.metadata.experiment_type = "Pup Retrieval"
    win.session_changed.emit()
    assert review._type_combo.currentText() == "Pup Retrieval"


def test_the_type_is_announced_only_when_it_changes(main_window_factory, qtbot):
    win = main_window_factory()
    win._broadcast_experiment_type()
    with qtbot.assertNotEmitted(win.experiment_type_changed):
        win._broadcast_experiment_type()
        win.session_changed.emit()
    with qtbot.waitSignal(win.experiment_type_changed) as sig:
        win._set_experiment_type("Pup Retrieval")
    assert sig.args == ["Pup Retrieval"]


def _preset_session(win, experiment_type="Pup Retrieval"):
    """Make the open experiment a Run-tab preset (one with run fields)."""
    from glider.core.experiment_session import RunField

    md = win._core.session.metadata
    md.experiment_type = experiment_type
    md.run_fields = [RunField("dam_id", "Dam ID", True)]
    win._run_header.refresh()
    return md


def _count_refreshes(view, monkeypatch):
    calls = []
    real = view.refresh
    monkeypatch.setattr(view, "refresh", lambda: (calls.append(1), real())[1])
    return calls


def test_the_run_header_picker_follows_an_experiment_tab_edit(main_window_factory, qtbot):
    win = main_window_factory()
    md = _preset_session(win)
    assert win._run_header._picker.text() == "Pup Retrieval ▾"
    win._experiment_detail_pages()
    win._experiment_details._protocol_combo.setCurrentText("Open Field")
    assert md.experiment_type == "Open Field"
    assert win._run_header._picker.text() == "Open Field ▾"


def test_an_open_settings_dialog_follows_a_session_review_pick_and_keeps_it(
    main_window_factory, qtbot
):
    win = main_window_factory()
    win._on_open_experiment_dialog()
    dialog = win._experiment_dialog
    qtbot.addWidget(dialog)
    win._open_session_review()
    review = win._session_review_window
    qtbot.addWidget(review)
    review._type_combo.setCurrentIndex(review._type_combo.findText("Pup Retrieval"))
    assert dialog._protocol_combo.currentText() == "Pup Retrieval"
    # The next edit in the dialog writes every widget back to the metadata; it
    # must carry the picked type, not the one the box showed before.
    dialog._name_edit.setText("Litter 7")
    md = win._core.session.metadata
    assert md.name == "Litter 7"
    assert md.experiment_type == "Pup Retrieval"


def test_the_form_that_was_typed_in_is_left_alone(main_window_factory, qtbot, monkeypatch):
    """Refreshing the originating form mid-keystroke would reset its cursor."""
    win = main_window_factory()
    _preset_session(win)
    win._experiment_detail_pages()
    win._on_open_experiment_dialog()
    qtbot.addWidget(win._experiment_dialog)
    tab_form, settings = win._experiment_details, win._experiment_dialog
    tab_refreshes = _count_refreshes(tab_form, monkeypatch)
    settings_refreshes = _count_refreshes(settings, monkeypatch)
    header_refreshes = _count_refreshes(win._run_header, monkeypatch)

    tab_form._protocol_combo.setCurrentText("Open Field")
    assert tab_refreshes == []
    assert len(settings_refreshes) == 1
    assert len(header_refreshes) == 1
    assert settings._protocol_combo.currentText() == "Open Field"

    settings._protocol_combo.setCurrentText("Rotarod")
    assert len(settings_refreshes) == 1
    assert len(tab_refreshes) == 1
    assert len(header_refreshes) == 2
    assert tab_form._protocol_combo.currentText() == "Rotarod"


def test_an_edit_that_leaves_the_type_alone_refreshes_nothing(
    main_window_factory, qtbot, monkeypatch
):
    win = main_window_factory()
    _preset_session(win)
    win._experiment_detail_pages()
    win._on_open_experiment_dialog()
    qtbot.addWidget(win._experiment_dialog)
    win._broadcast_experiment_type()
    refreshes = [
        _count_refreshes(view, monkeypatch)
        for view in (win._experiment_details, win._experiment_dialog, win._run_header)
    ]
    win._experiment_details._name_edit.setText("Litter 8")
    assert refreshes == [[], [], []]


# ---------------------------------------------- the form's one Protocol box


def _form(qtbot, **meta):
    from glider.core.experiment_session import ExperimentSession
    from glider.gui.dialogs.experiment_dialog import ExperimentDialog

    session = ExperimentSession()
    for key, value in meta.items():
        setattr(session.metadata, key, value)
    form = ExperimentDialog(session=session)
    qtbot.addWidget(form)
    return form, session.metadata


def test_the_protocol_box_is_the_type_dropdown(qtbot):
    from PyQt6.QtWidgets import QFormLayout

    form, _md = _form(qtbot)
    combo, layout = form._protocol_combo, form._info_layout
    assert combo.isEditable()
    assert [combo.itemText(i) for i in range(3)] == ["None", "Open Field", "Pup Retrieval"]
    assert layout.labelForField(combo).text() == "Protocol:"
    labels = [
        layout.itemAt(row, QFormLayout.ItemRole.LabelRole).widget().text()
        for row in range(layout.rowCount())
    ]
    assert "Type:" not in labels and labels.count("Protocol:") == 1


def test_picking_a_protocol_writes_the_shared_type(qtbot):
    form, md = _form(qtbot)
    combo = form._protocol_combo
    with qtbot.waitSignal(form.metadata_changed):
        combo.setCurrentIndex(combo.findText("Pup Retrieval"))
    assert md.experiment_type == "Pup Retrieval"
    combo.setCurrentIndex(combo.findText("None"))
    assert md.experiment_type == ""


def test_a_typed_protocol_is_stored_as_typed_and_none_means_none(qtbot):
    form, md = _form(qtbot)
    form._protocol_combo.setEditText("  Pup Retrieval (P4) ")
    assert md.experiment_type == "Pup Retrieval (P4)"
    form._protocol_combo.setEditText("None")
    assert md.experiment_type == ""


def test_the_old_protocol_text_survives_every_edit(qtbot):
    form, md = _form(qtbot, protocol="OFT-001")
    form._protocol_combo.setCurrentIndex(form._protocol_combo.findText("Open Field"))
    form._name_edit.setText("Litter 9")
    form._notes_edit.setPlainText("dim light")
    assert md.protocol == "OFT-001"
    assert md.to_dict()["protocol"] == "OFT-001"


@pytest.mark.parametrize(
    ("stored", "shown"),
    [("Open Field", "Open Field"), ("", "None"), ("Pup Retrieval (P4)", "Pup Retrieval (P4)")],
)
def test_a_loaded_protocol_is_shown(qtbot, stored, shown):
    form, md = _form(qtbot, experiment_type=stored)
    assert form._protocol_combo.currentText() == shown
    assert md.experiment_type == stored  # loading is not an edit
