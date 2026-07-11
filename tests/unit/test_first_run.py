"""Tests for the first-run flow, including the Take-the-Tour integration.

Dialog-level tests monkeypatch ``QMessageBox.exec`` to click a button
programmatically (button signals are wired at addButton time, so ``click()``
sets ``clickedButton`` without a running event loop). Orchestration tests stub
``show_welcome_dialog`` entirely and drive ``run_first_run_if_needed`` with a
fake parent, so no real window is needed.
"""

from __future__ import annotations

import types
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QMessageBox

from glider import first_run
from glider.first_run import (
    FIRST_RUN_COMPLETE_KEY,
    run_first_run_if_needed,
    show_welcome_dialog,
)


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture(autouse=True)
def _data_dir_in_tmp(tmp_path, monkeypatch):
    """Keep ensure_data_dir out of the real ~/Documents during tests."""
    monkeypatch.setattr(first_run, "default_data_dir", lambda: tmp_path / "GLIDER")


def _exec_clicking(monkeypatch, button_text: str, seen_buttons: list | None = None):
    """Patch QMessageBox.exec to record buttons and click ``button_text``."""

    def fake_exec(self):
        labels = [b.text() for b in self.buttons()]
        if seen_buttons is not None:
            seen_buttons.extend(labels)
        for b in self.buttons():
            if b.text() == button_text:
                b.click()
                return 0
        return 0  # no click — simulates closing the dialog

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)


# --- show_welcome_dialog -----------------------------------------------------


def test_tour_offer_shows_tour_and_skip_buttons(qtbot, monkeypatch, tmp_path):
    seen: list = []
    _exec_clicking(monkeypatch, "Take the Tour", seen)
    choice = show_welcome_dialog(None, tmp_path, offer_tour=True)
    assert choice == "tour"
    assert "Take the Tour" in seen
    assert "Skip" in seen
    assert "Start" not in seen


def test_skip_returns_start(qtbot, monkeypatch, tmp_path):
    _exec_clicking(monkeypatch, "Skip")
    assert show_welcome_dialog(None, tmp_path, offer_tour=True) == "start"


def test_no_offer_keeps_original_buttons(qtbot, monkeypatch, tmp_path):
    seen: list = []
    _exec_clicking(monkeypatch, "Start", seen)
    choice = show_welcome_dialog(None, tmp_path, offer_tour=False)
    assert choice == "start"
    assert "Take the Tour" not in seen
    assert "Skip" not in seen


def test_dialog_close_counts_as_start(qtbot, monkeypatch, tmp_path):
    _exec_clicking(monkeypatch, "NoSuchButton")  # nothing clicked
    assert show_welcome_dialog(None, tmp_path, offer_tour=True) == "start"


# --- run_first_run_if_needed --------------------------------------------------


def _builder_parent():
    return types.SimpleNamespace(is_runner_mode=False, _start_tour=Mock())


def test_tour_choice_starts_tour_and_marks_complete(monkeypatch, settings):
    monkeypatch.setattr(first_run, "show_welcome_dialog", Mock(return_value="tour"))
    parent = _builder_parent()
    run_first_run_if_needed(parent, settings=settings)
    parent._start_tour.assert_called_once()
    assert settings.value(FIRST_RUN_COMPLETE_KEY, False, type=bool) is True


def test_skip_choice_does_not_start_tour(monkeypatch, settings):
    monkeypatch.setattr(first_run, "show_welcome_dialog", Mock(return_value="start"))
    parent = _builder_parent()
    run_first_run_if_needed(parent, settings=settings)
    parent._start_tour.assert_not_called()
    assert settings.value(FIRST_RUN_COMPLETE_KEY, False, type=bool) is True


def test_builder_parent_gets_tour_offer(monkeypatch, settings):
    dialog = Mock(return_value="start")
    monkeypatch.setattr(first_run, "show_welcome_dialog", dialog)
    run_first_run_if_needed(_builder_parent(), settings=settings)
    assert dialog.call_args.kwargs["offer_tour"] is True


def test_runner_parent_gets_no_tour_offer(monkeypatch, settings):
    dialog = Mock(return_value="start")
    monkeypatch.setattr(first_run, "show_welcome_dialog", dialog)
    parent = types.SimpleNamespace(is_runner_mode=True, _start_tour=Mock())
    run_first_run_if_needed(parent, settings=settings)
    assert dialog.call_args.kwargs["offer_tour"] is False


def test_second_run_shows_nothing(monkeypatch, settings):
    dialog = Mock(return_value="start")
    monkeypatch.setattr(first_run, "show_welcome_dialog", dialog)
    settings.setValue(FIRST_RUN_COMPLETE_KEY, True)
    run_first_run_if_needed(_builder_parent(), settings=settings)
    dialog.assert_not_called()


def test_tour_failure_does_not_break_first_run(monkeypatch, settings):
    monkeypatch.setattr(first_run, "show_welcome_dialog", Mock(return_value="tour"))
    parent = types.SimpleNamespace(
        is_runner_mode=False, _start_tour=Mock(side_effect=RuntimeError("boom"))
    )
    # Must not raise, and must still mark completion.
    run_first_run_if_needed(parent, settings=settings)
    assert settings.value(FIRST_RUN_COMPLETE_KEY, False, type=bool) is True
