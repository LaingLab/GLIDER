"""Notice when the open experiment changes on disk outside GLIDER.

An agent editing through glider-mcp writes the ``.glider`` file the operator
has open. The watcher tells the window, which reloads when there is nothing to
lose and otherwise shows :class:`ExternalChangeBar` so the operator chooses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PyQt6.QtCore import QFileSystemWatcher, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


def _fingerprint(path: str) -> str | None:
    """Hash of the file's bytes, or None if it can't be read (deleted, mid-move)."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


class ExperimentFileWatcher(QObject):
    """Emits ``changed_externally(path)`` when the watched file's contents change.

    Compares contents, not timestamps, so GLIDER's own save is silent as long
    as the window calls :meth:`watch` again right after saving -- which also
    re-reads the fingerprint. A burst of writes collapses into one signal.
    """

    changed_externally = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None, debounce_ms: int = 300) -> None:
        super().__init__(parent)
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_event)
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(debounce_ms)
        self._settle_timer.timeout.connect(self._settle)
        self._path: str | None = None
        self._known: str | None = None

    @property
    def path(self) -> str | None:
        return self._path

    def watch(self, path: str | None) -> None:
        """Watch ``path``, taking its current contents as known (it was just opened or saved)."""
        self.stop()
        if not path:
            return
        self._path = str(path)
        self._known = _fingerprint(self._path)
        self._arm()

    def acknowledge(self) -> None:
        """Take what is on disk now as known, so the same version is not reported again."""
        if self._path is not None:
            self._known = _fingerprint(self._path)

    def stop(self) -> None:
        self._settle_timer.stop()
        if self._watcher.files():
            self._watcher.removePaths(self._watcher.files())
        self._path = None
        self._known = None

    def _arm(self) -> None:
        # Saving by swapping a temp file in replaces the file the OS was
        # watching, so the watch can silently drop; re-add it every time.
        if (
            self._path is not None
            and Path(self._path).exists()
            and self._path not in self._watcher.files()
        ):
            self._watcher.addPath(self._path)

    def _on_event(self, _path: str) -> None:
        self._settle_timer.start()

    def _settle(self) -> None:
        if self._path is None:
            return
        self._arm()
        current = _fingerprint(self._path)
        if current is None or current == self._known:
            return  # deleted or moved away, or the same contents (our own save)
        self._known = current
        self.changed_externally.emit(self._path)


class ExternalChangeBar(QWidget):
    """Slim bar offering Reload / Keep mine when the open file changed and edits are unsaved."""

    reload_requested = pyqtSignal()
    keep_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("externalChangeBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)
        self._message = QLabel(
            "This experiment was changed outside GLIDER, and you have unsaved changes."
        )
        self._message.setWordWrap(True)
        layout.addWidget(self._message, 1)
        self._reload = QPushButton("Reload")
        self._reload.setToolTip("Open the version on disk; your unsaved changes are discarded")
        self._reload.clicked.connect(self.reload_requested)
        layout.addWidget(self._reload)
        self._keep = QPushButton("Keep mine")
        self._keep.setToolTip("Keep your version; saving will overwrite the one on disk")
        self._keep.clicked.connect(self.keep_requested)
        layout.addWidget(self._keep)
        self.hide()
