"""The "Pup segmentation" card: shown only under pup retrieval, and what it sends."""

import pytest
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from glider.core.experiment_types import PUP_RETRIEVAL
from glider.gui.pose_batch.window import PoseBatchWindow

from .test_window import _ready


@pytest.fixture
def window(qtbot):
    w = PoseBatchWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def captured(monkeypatch):
    """The kwargs ``_start_worker`` hands the worker, without running a thread.

    Same stub as test_animal_count.py: real signals and a run()/cancel() so the
    connects in _start_worker and the teardown close both work, and
    QThread.start patched to a no-op.
    """
    seen = {}

    class StubWorker(QObject):
        progress = pyqtSignal(int, int)
        video_progress = pyqtSignal(int, int)
        log = pyqtSignal(str)
        finished = pyqtSignal(object)
        failed = pyqtSignal(str)

        def __init__(self, *args, **kwargs):
            super().__init__()
            seen.update(kwargs)

        def moveToThread(self, _thread):
            pass

        def run(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr("glider.gui.pose_batch.worker.PoseBatchWorker", StubWorker)
    monkeypatch.setattr(QThread, "start", lambda self: None)
    return seen


def test_the_card_is_hidden_until_the_type_unlocks_pup_retrieval(window):
    assert window._pup_card.isHidden()
    window.set_experiment_type(PUP_RETRIEVAL)
    assert not window._pup_card.isHidden()
    window.set_experiment_type("")
    assert window._pup_card.isHidden()


def test_the_pup_model_and_litter_reach_the_worker(window, tmp_path, captured):
    window.set_experiment_type(PUP_RETRIEVAL)
    _ready(window, tmp_path)
    window._set_pup_model(tmp_path / "pups.pt")
    window._litter_spin.setValue(6)
    window._start_worker()
    assert captured["pup_model"] == tmp_path / "pups.pt"
    assert captured["litter_size"] == 6


def test_no_pup_model_sends_none(window, tmp_path, captured):
    window.set_experiment_type(PUP_RETRIEVAL)
    _ready(window, tmp_path)
    window._start_worker()
    assert captured["pup_model"] is None


def test_a_chosen_pup_model_is_not_sent_outside_pup_retrieval(window, tmp_path, captured):
    # The card is hidden then; a model picked earlier must not run a pass the
    # user can no longer see.
    window.set_experiment_type(PUP_RETRIEVAL)
    window._set_pup_model(tmp_path / "pups.pt")
    window.set_experiment_type("")
    _ready(window, tmp_path)
    window._start_worker()
    assert captured["pup_model"] is None


def test_clearing_the_model_disables_the_litter_size(window, tmp_path):
    assert not window._litter_spin.isEnabled()
    window._set_pup_model(tmp_path / "pups.pt")
    assert window._litter_spin.isEnabled()
    assert window._pup_model_field.text() == "pups.pt"
    window._pup_clear.click()
    assert window._pup_model_path is None
    assert window._pup_model_field.text() == ""
    assert not window._litter_spin.isEnabled()
