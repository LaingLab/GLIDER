"""
Runner Dashboard - Main dashboard view for experiment execution.
"""

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.core.flow_engine import FlowEngine
    from glider.nodes.base_node import GliderNode

logger = logging.getLogger(__name__)


class RunnerDashboard(QWidget):
    """
    Touch-optimized dashboard for experiment execution.

    Dynamically creates widgets for nodes marked as visible_in_runner.
    Supports vertical scrolling with kinetic scrolling enabled.
    """

    # Signals
    widget_value_changed = pyqtSignal(str, object)  # node_id, value

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._flow_engine: FlowEngine | None = None
        self._widgets: dict[str, QWidget] = {}
        self._layout_mode = "vertical"  # "vertical", "horizontal", "grid"
        self._columns = 1

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the dashboard UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area for dashboard content
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Enable kinetic scrolling for touch
        from PyQt6.QtWidgets import QScroller

        QScroller.grabGesture(
            self._scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )

        # Content widget
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 10, 10, 10)
        self._content_layout.setSpacing(10)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll)

    def set_flow_engine(self, engine: "FlowEngine") -> None:
        """Set the flow engine to display widgets from."""
        self._flow_engine = engine

    def set_layout_mode(self, mode: str, columns: int = 1) -> None:
        """
        Set the layout mode for widgets.

        Args:
            mode: "vertical", "horizontal", or "grid"
            columns: Number of columns for grid layout
        """
        self._layout_mode = mode
        self._columns = columns
        self.rebuild_dashboard()

    def rebuild_dashboard(self) -> None:
        """Rebuild the dashboard from the current flow engine state."""
        # Clear existing widgets
        self.clear_widgets()

        if not self._flow_engine:
            return

        # Get visible nodes
        visible_nodes = [
            node
            for node in self._flow_engine.nodes.values()
            if hasattr(node, "visible_in_runner") and node.visible_in_runner
        ]

        if not visible_nodes:
            # Show placeholder
            placeholder = QLabel("No dashboard widgets configured")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 16px;")
            self._content_layout.insertWidget(0, placeholder)
            self._widgets["_placeholder"] = placeholder
            return

        # Create widgets based on layout mode
        if self._layout_mode == "grid":
            self._create_grid_layout(visible_nodes)
        else:
            self._create_linear_layout(visible_nodes)

        logger.info(f"Dashboard rebuilt with {len(visible_nodes)} widgets")

    def _create_linear_layout(self, nodes: list["GliderNode"]) -> None:
        """Create a linear (vertical/horizontal) layout."""
        from glider.gui.runner.widget_factory import WidgetFactory

        for node in nodes:
            widget = WidgetFactory.create_widget(node)
            if widget:
                self._content_layout.insertWidget(
                    self._content_layout.count() - 1, widget  # Before stretch
                )
                self._widgets[node.id] = widget

                # Connect widget value changes
                if hasattr(widget, "value_changed"):
                    widget.value_changed.connect(
                        lambda v, n=node: self.widget_value_changed.emit(n.id, v)
                    )

    def _create_grid_layout(self, nodes: list["GliderNode"]) -> None:
        """Create a grid layout."""
        from glider.gui.runner.widget_factory import WidgetFactory

        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(10)

        for i, node in enumerate(nodes):
            row = i // self._columns
            col = i % self._columns

            widget = WidgetFactory.create_widget(node)
            if widget:
                grid_layout.addWidget(widget, row, col)
                self._widgets[node.id] = widget

                # Connect widget value changes
                if hasattr(widget, "value_changed"):
                    widget.value_changed.connect(
                        lambda v, n=node: self.widget_value_changed.emit(n.id, v)
                    )

        self._content_layout.insertWidget(0, grid_container)

    def clear_widgets(self) -> None:
        """Clear all dashboard widgets."""
        for widget in self._widgets.values():
            widget.setParent(None)
            widget.deleteLater()
        self._widgets.clear()

    def update_widget(self, node_id: str, value: object) -> None:
        """Update a specific widget with new data."""
        widget = self._widgets.get(node_id)
        if widget and hasattr(widget, "set_value"):
            widget.set_value(value)

    def get_widget(self, node_id: str) -> QWidget | None:
        """Get a widget by node ID."""
        return self._widgets.get(node_id)
