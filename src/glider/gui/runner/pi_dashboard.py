"""The Pi's Dashboard tab: the node graph over a tabbed panel.

A vertical splitter, graph on top and a ``QTabWidget`` under it, the two
starting at half the height each and resizable by the grip between them.
It is layout only: the caller adds the real panels with ``add_tab`` and the
colours come from ``touch.qss`` through the object names.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QSplitter, QTabWidget, QVBoxLayout, QWidget

from glider.gui.node_graph.graph_view import NodeGraphView


class PiDashboard(QWidget):
    """Graph on top, ``Nodes · Properties · Hardware · Camera · Control`` under it."""

    def __init__(self, graph: NodeGraphView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("piDashboard")
        self._keys: list[str] = []
        self._sized = False
        self._graph = graph
        self._fit_pending = True  # frame the flow on the first show

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setObjectName("piDashboardSplitter")
        self._splitter.setChildrenCollapsible(False)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("piDashboardTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.tabBar().setExpanding(True)

        self._splitter.addWidget(graph)
        self._splitter.addWidget(self._tabs)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)

        graph.node_selected.connect(self._on_node_selected)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # The graph's size hint would win the first layout, so split the height
        # evenly once, on the first show; after that the grip owns the sizes.
        if not self._sized:
            self._sized = True
            half = self._splitter.height() // 2
            self._splitter.setSizes([half, half])
        if self._fit_pending:
            self._fit_pending = False
            # After this layout pass, so the fit sees the graph's real size.
            QTimer.singleShot(0, self._graph.fit_to_screen)

    def fit_graph(self) -> None:
        """Frame the whole flow now, or on the next show if this is hidden.

        A kiosk has no wheel to go looking for a flow with, so a freshly opened
        one is put in view rather than wherever the last one left the camera.
        """
        if self.isVisible():
            self._graph.fit_to_screen()
        else:
            self._fit_pending = True

    def _on_node_selected(self, _node_id: str) -> None:
        self.set_current("properties")

    def add_tab(self, key: str, label: str, widget: QWidget) -> None:
        self._keys.append(key)
        self._tabs.addTab(widget, label)

    def set_current(self, key: str) -> None:
        if key in self._keys:
            self._tabs.setCurrentIndex(self._keys.index(key))

    def current(self) -> str:
        return self._keys[self._tabs.currentIndex()]

    def keys(self) -> list[str]:
        return list(self._keys)

    def splitter(self) -> QSplitter:
        return self._splitter

    def tabs(self) -> QTabWidget:
        return self._tabs
