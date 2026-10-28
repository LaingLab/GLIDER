"""The one-time GPU offer at launch (#221), and the pose-model link on Tools (#222).

Uses the same bare ``MainWindow`` as ``test_lab_setup_entry.py``: the full
``__init__`` needs a live core, and these only need the methods under test.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QMainWindow, QMessageBox  # noqa: E402

from glider.first_run import FIRST_RUN_COMPLETE_KEY  # noqa: E402
from glider.gui import main_window as main_window_module  # noqa: E402
from glider.gui.main_window import GPU_SETUP_OFFERED_KEY, MainWindow  # noqa: E402
from glider.vision import gpu_setup  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> QSettings:
    s = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    s.setValue(FIRST_RUN_COMPLETE_KEY, True)
    return s


@pytest.fixture
def window(qtbot):
    win = MainWindow.__new__(MainWindow)  # skip the heavy __init__
    QMainWindow.__init__(win)
    win._view_manager = types.SimpleNamespace(is_runner_mode=False)
    win.installs = []
    win._install_cuda_torch = win.installs.append
    yield win
    win.deleteLater()


@pytest.fixture
def answer(monkeypatch):
    """Answer the offer without a modal; records how often it was asked."""
    record = types.SimpleNamespace(button=QMessageBox.StandardButton.No, asked=0)

    def _question(*_args, **_kwargs):
        record.asked += 1
        return record.button

    monkeypatch.setattr(QMessageBox, "question", _question)
    return record


def test_offered_once_and_installs_on_yes(window, settings, answer, monkeypatch):
    monkeypatch.setattr(gpu_setup, "needs_cuda_torch", lambda: (12, 8))
    answer.button = QMessageBox.StandardButton.Yes

    assert window.offer_gpu_setup_once(settings) is True
    assert window.installs == [(12, 8)]
    assert window.offer_gpu_setup_once(settings) is False
    assert answer.asked == 1


def test_declining_is_remembered(window, settings, answer, monkeypatch):
    monkeypatch.setattr(gpu_setup, "needs_cuda_torch", lambda: (12, 8))

    window.offer_gpu_setup_once(settings)

    assert window.installs == []
    assert settings.value(GPU_SETUP_OFFERED_KEY, False, type=bool) is True


def test_not_offered_without_a_gpu_to_use(window, settings, answer, monkeypatch):
    monkeypatch.setattr(gpu_setup, "needs_cuda_torch", lambda: None)

    assert window.offer_gpu_setup_once(settings) is False
    assert answer.asked == 0
    # Not spent: a GPU fitted later still gets asked.
    assert settings.value(GPU_SETUP_OFFERED_KEY, False, type=bool) is False


def test_waits_for_the_welcome_instead_of_stacking_on_it(window, tmp_path, answer, monkeypatch):
    fresh = QSettings(str(tmp_path / "fresh.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(gpu_setup, "needs_cuda_torch", lambda: (12, 8))
    retries = []
    monkeypatch.setattr(main_window_module.QTimer, "singleShot", lambda ms, fn: retries.append(fn))

    assert window.offer_gpu_setup_once(fresh) is False
    assert answer.asked == 0
    assert len(retries) == 1, "gave up instead of trying again later"


def test_no_tool_card_links_out_to_ultralytics_platform(window):
    """Pose models are labelled in the Animal Annotator now, not on a website."""
    window._create_tools_page()
    keys = {card.key for card in window._tools_page.cards()}

    assert "pose_model" not in keys
    assert "annotator" in keys


# --- Install GPU Support, any time after the launch offer --------------------


def test_install_gpu_support_installs_after_the_offer_was_declined(
    window, settings, answer, monkeypatch
):
    """The launch offer is asked once; this is the way back to it."""
    settings.setValue(GPU_SETUP_OFFERED_KEY, True)
    monkeypatch.setattr(gpu_setup, "gpu_support", lambda: ((12, 8), ""))
    answer.button = QMessageBox.StandardButton.Yes

    window._on_tool_chosen("gpu_install")

    assert window.installs == [(12, 8)]


def test_install_gpu_support_says_why_when_there_is_nothing_to_install(window, answer, monkeypatch):
    monkeypatch.setattr(gpu_setup, "gpu_support", lambda: (None, "No NVIDIA GPU driver answered"))
    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda _p, _t, text: told.append(text))

    window._on_install_gpu_support()

    assert told == ["No NVIDIA GPU driver answered"]
    assert window.installs == []
    assert answer.asked == 0


def test_install_gpu_support_is_on_the_tools_menu(qtbot):
    win = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(win)
    win._view_manager = types.SimpleNamespace(is_runner_mode=False)
    win._setup_menu()
    tools = next(m for m in win.menus() if m.title().replace("&", "") == "Tools")
    assert any("Install GPU Support" in a.text().replace("&", "") for a in tools.actions())
    win.deleteLater()
