"""Stopping a worker QThread without ever destroying it while it runs.

Dropping the last Python reference to a running ``QThread`` (or deleting it)
aborts the process with "QThread: Destroyed while thread is still running".
``quit()`` only ends the event loop, so a worker inside a model load or a
blocking inference call can outlast any reasonable wait.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QThread

logger = logging.getLogger(__name__)

# Threads (and their workers) that did not stop in time, held until they finish.
_PARKED: list[tuple] = []


def retire_thread(thread: QThread, *keep_alive, timeout_ms: int = 5000) -> bool:
    """``quit()`` ``thread``; free it only once it has really exited.

    Returns True if it stopped within ``timeout_ms``. Otherwise the thread and
    ``keep_alive`` (its worker) are parked here until ``finished``, so the
    caller can drop its own references safely either way.
    """
    thread.quit()
    if thread.wait(timeout_ms):
        thread.deleteLater()
        return True
    logger.warning("Worker thread still busy after %d ms; releasing it when it ends", timeout_ms)
    key = (thread, *keep_alive)
    _PARKED.append(key)
    # A parented thread would still die with its parent; hold it here instead.
    thread.setParent(None)

    def _release() -> None:
        # Identity, not ==: workers need not be hashable or comparable.
        for i, parked in enumerate(_PARKED):
            if parked is key:
                del _PARKED[i]
                thread.deleteLater()
                return

    thread.finished.connect(_release)
    if thread.isFinished():  # ended between wait() and connect()
        _release()
    return False
