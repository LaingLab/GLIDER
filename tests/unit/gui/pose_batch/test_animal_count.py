"""The animal count is the one knob that is never guessed."""

import pytest
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from glider.gui.pose_batch.window import PoseBatchWindow


@pytest.fixture
def window(qtbot):
    w = PoseBatchWindow()
    qtbot.addWidget(w)
    return w


def test_it_defaults_to_one_animal(window):
    assert window._animals_spin.value() == 1


def test_it_will_not_accept_zero_animals(window):
    window._animals_spin.setValue(0)
    assert window._animals_spin.value() >= 1


def test_the_tooltip_says_what_a_wrong_value_looks_like(window):
    tip = window._animals_spin.toolTip().lower()
    assert "too high" in tip or "too low" in tip


def test_the_count_reaches_the_worker(window, monkeypatch, tmp_path):
    # _start_worker connects PoseBatchWorker.{progress,video_progress,log,
    # finished,failed} and QThread.started -> worker.run before it ever starts
    # the thread, so the stub needs real signals and a run() -- a bare object
    # raises AttributeError on the first .connect() and never reaches start().
    # QThread.start is patched to a no-op so nothing runs on a background
    # thread: the kwargs are already captured by the time start() is called.
    # cancel() is needed too -- qtbot closes the window at teardown, and
    # closeEvent calls self._worker.cancel() while self._thread is still set.
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

    video = tmp_path / "session01.mp4"
    video.write_bytes(b"x")
    window._videos = [video]
    window._model_path = tmp_path / "m.pt"
    window._animals_spin.setValue(3)

    window._start_worker()
    assert seen["n_animals"] == 3
