"""retire_thread never lets a running QThread be destroyed (F6-F8)."""

import threading

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget

from glider.gui import qthreads


class _Blocked(QObject):
    done = pyqtSignal()

    def __init__(self, gate):
        super().__init__()
        self._gate = gate

    def run(self):
        self._gate.wait(5)
        self.done.emit()


def test_a_busy_thread_outlives_its_parent_and_is_released_after(qtbot):
    gate = threading.Event()
    parent = QWidget()
    thread = QThread(parent)
    worker = _Blocked(gate)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.start()

    assert qthreads.retire_thread(thread, worker, timeout_ms=0) is False
    assert any(k[0] is thread for k in qthreads._PARKED)
    parent.deleteLater()  # must not take the running thread with it
    qtbot.wait(20)
    assert thread.isRunning()

    gate.set()
    qtbot.waitUntil(lambda: all(k[0] is not thread for k in qthreads._PARKED), timeout=3000)
