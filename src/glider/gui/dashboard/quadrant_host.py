"""A single framed dashboard slot: title bar + one hosted panel, with picker
and drag-to-swap. Layout changes are surfaced as signals; DashboardView owns
the actual reassignment so this widget stays free of layout policy.
"""

from __future__ import annotations

from PyQt6.QtCore import QMimeData, Qt, pyqtSignal
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from glider.gui.dashboard.panel_registry import PANEL_KEYS, PANEL_NAMES


class QuadrantHost(QWidget):
    """Hosts exactly one panel; emits pick/drag-swap intents to the parent."""

    panel_selected = pyqtSignal(str, str)  # (quadrant_id, panel_key)
    swap_requested = pyqtSignal(str, str)  # (source_quadrant_id, target_quadrant_id)

    _MIME = "application/x-glider-quadrant"

    def __init__(self, quadrant_id: str, parent=None):
        super().__init__(parent)
        self.quadrant_id = quadrant_id
        self.current_panel_key: str | None = None
        self._panel: QWidget | None = None
        self.setAcceptDrops(True)
        self.setProperty("dashboardQuadrant", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._title_bar = _QuadrantTitleBar(self)
        layout.addWidget(self._title_bar)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(0)
        layout.addWidget(self._body, 1)

    # --- panel hosting ---

    def set_panel(self, panel: QWidget, panel_key: str, title: str) -> QWidget | None:
        """Install `panel`; return the previously hosted panel (or None).

        The previous panel is reparented out (not deleted) only if it is still
        parented under this host's body — so a caller that has already moved it
        elsewhere (e.g. DashboardView parking all panels to a bench holder before
        reassigning) does not get it clobbered back to a null parent.
        """
        previous = self._panel
        if previous is not None and previous.parent() is self._body:
            self._body_layout.removeWidget(previous)
            previous.setParent(None)
        self._panel = panel
        self.current_panel_key = panel_key
        self._body_layout.addWidget(panel)  # reparents panel to self._body
        panel.show()
        self._title_bar.set_title(title)
        return previous

    def title_text(self) -> str:
        return self._title_bar.title_text()

    # --- picker ---

    def show_picker_menu(self) -> None:
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        for key in PANEL_KEYS:
            action = menu.addAction(PANEL_NAMES[key])
            action.setCheckable(True)
            action.setChecked(key == self.current_panel_key)
            action.triggered.connect(lambda _checked=False, k=key: self.trigger_pick(k))
        menu.exec(self._title_bar.picker_global_pos())

    def trigger_pick(self, panel_key: str) -> None:
        """Test seam + menu callback: request showing `panel_key` here."""
        self.panel_selected.emit(self.quadrant_id, panel_key)

    # --- drop target ---

    def dragEnterEvent(self, event):  # noqa: N802 (Qt override)
        if event.mimeData().hasFormat(self._MIME):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802 (Qt override)
        if not event.mimeData().hasFormat(self._MIME):
            return
        source_id = bytes(event.mimeData().data(self._MIME)).decode("utf-8")
        if source_id and source_id != self.quadrant_id:
            self.swap_requested.emit(source_id, self.quadrant_id)
        event.acceptProposedAction()


class _QuadrantTitleBar(QWidget):
    """Draggable title bar with a panel name and a picker button."""

    def __init__(self, host: QuadrantHost):
        super().__init__(host)
        self._host = host
        self._press_pos = None
        self.setProperty("dashboardQuadrantTitle", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(32)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 0, 6, 0)
        self._label = QLabel("")
        row.addWidget(self._label)
        row.addStretch()
        self._picker = QPushButton("▾")  # down triangle
        self._picker.setFixedSize(28, 24)
        self._picker.clicked.connect(self._host.show_picker_menu)
        row.addWidget(self._picker)

    def set_title(self, text: str) -> None:
        self._label.setText(text)

    def title_text(self) -> str:
        return self._label.text()

    def picker_global_pos(self):
        return self._picker.mapToGlobal(self._picker.rect().bottomLeft())

    # --- drag source ---

    def mousePressEvent(self, event):  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 (Qt override)
        if not (event.buttons() & Qt.MouseButton.LeftButton) or self._press_pos is None:
            return
        if (
            event.position().toPoint() - self._press_pos
        ).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(QuadrantHost._MIME, self._host.quadrant_id.encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)
