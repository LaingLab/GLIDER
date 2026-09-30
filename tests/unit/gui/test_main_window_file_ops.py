"""MainWindow New/Open/Save guards (review F1, F2, F3, F5).

Bypasses the heavy ``__init__`` and drives the handlers against fakes.
"""

from __future__ import annotations

import types

import pytest
from PyQt6.QtWidgets import QMessageBox

from glider.core.experiment_session import SessionState
from glider.gui.commands import UndoStack

pytestmark = pytest.mark.usefixtures("qtbot")


class _Recorder:
    def __init__(self):
        self.zone_config = None

    def set_zone_configuration(self, cfg):
        self.zone_config = cfg

    def set_cv_processor(self, _cv):
        pass


def _window(state=SessionState.IDLE, busy=False):
    from glider.gui.main_window import MainWindow

    w = MainWindow.__new__(MainWindow)
    cleared = []
    w._core = types.SimpleNamespace(
        state=state,
        is_experiment_busy=busy,
        hardware_manager=types.SimpleNamespace(clear=lambda: cleared.append(True)),
        new_session=lambda: None,
        session=None,
        cv_processor=_Recorder(),
        tracking_logger=_Recorder(),
        data_recorder=_Recorder(),
    )
    w._cleared = cleared
    w._graph_view = types.SimpleNamespace(clear_graph=lambda: None)
    w._undo_stack = UndoStack()
    for attr in (
        "_camera_panel",
        "_node_library_panel",
        "_node_editor",
        "_hardware_panel",
        "_dash_hardware_panel",
        "_experiment_dialog",
    ):
        setattr(w, attr, None)
    w.session_changed = types.SimpleNamespace(emit=lambda: None)
    w._reset_experiment_page = lambda: None
    w._check_save = lambda: True
    w._update_undo_redo_actions = lambda: None
    w._zone_config = object()
    return w


def test_new_is_refused_while_running(monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = _window(state=SessionState.RUNNING)
    w._on_new()
    assert w._cleared == [], "hardware torn down under a live run"


def test_new_clears_undo_and_pushes_fresh_zones():
    w = _window()
    w._undo_stack.push(types.SimpleNamespace(description=lambda: "x"))
    old_zones = w._zone_config

    w._on_new()

    assert not w._undo_stack.can_undo()
    assert w._core.cv_processor.zone_config is w._zone_config
    assert w._core.tracking_logger.zone_config is w._zone_config
    assert w._core.data_recorder.zone_config is not old_zones


def test_check_save_keeps_work_when_save_fails(monkeypatch):
    from glider.gui.main_window import MainWindow

    w = MainWindow.__new__(MainWindow)
    w._core = types.SimpleNamespace(session=types.SimpleNamespace(is_dirty=True))
    w._view_manager = types.SimpleNamespace(is_runner_mode=False)
    w._on_save = lambda: False  # failed or cancelled Save-As
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save)
    assert w._check_save() is False
