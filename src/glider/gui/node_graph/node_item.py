"""
Node Item - Visual representation of a node in the graph.

Handles rendering, interaction, and embedded widgets using
QGraphicsProxyWidget.
"""

import sys

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QStyleOptionGraphicsItem,
    QWidget,
)

from glider.gui.node_graph.port_item import PortItem, PortType
from glider.gui.styles import colors


def _get_system_font() -> str:
    """Get the system font for the current platform."""
    if sys.platform == "darwin":
        return ".AppleSystemUIFont"
    elif sys.platform == "win32":
        return "Segoe UI"
    return "DejaVu Sans"


class NodeSignals(QObject):
    """Signals for NodeItem (QGraphicsItem can't have signals directly)."""

    position_changed = pyqtSignal(str, float, float)
    selected_changed = pyqtSignal(str, bool)


class NodeItem(QGraphicsRectItem):
    """
    Visual representation of a node in the graph.

    Features:
    - Header with node name
    - Input and output ports
    - Embedded widgets via QGraphicsProxyWidget
    - Selection and drag behavior
    - Category-based coloring
    """

    # Category colors
    CATEGORY_COLORS = {
        "hardware": colors.Q_CAT_HARDWARE,
        "logic": colors.Q_CAT_LOGIC,
        "interface": colors.Q_CAT_INTERFACE,
        "script": colors.Q_CAT_SCRIPT,
        "default": colors.Q_CAT_DEFAULT,
    }

    CATEGORY_GRADIENTS = {
        "hardware": colors.CAT_HARDWARE_GRADIENT,
        "logic": colors.CAT_LOGIC_GRADIENT,
        "interface": colors.CAT_INTERFACE_GRADIENT,
        "script": colors.CAT_SCRIPT_GRADIENT,
        "default": colors.CAT_DEFAULT_GRADIENT,
    }

    # Node dimensions
    MIN_WIDTH = 150
    HEADER_HEIGHT = 30
    PORT_HEIGHT = 24
    PORT_SPACING = 4
    CORNER_RADIUS = 8

    def __init__(self, node_id: str, node_type: str, category: str = "default"):
        super().__init__()

        self._node_id = node_id
        self._node_type = node_type
        self._category = category
        self._title = node_type

        # Signals
        self.signals = NodeSignals()
        self.position_changed = self.signals.position_changed
        self.selected_changed = self.signals.selected_changed

        # Ports
        self._input_ports: list[PortItem] = []
        self._output_ports: list[PortItem] = []

        # Embedded widget
        self._widget_proxy: QGraphicsProxyWidget | None = None
        self._widget_height = 0

        # Appearance
        self._header_color = self.CATEGORY_COLORS.get(category, self.CATEGORY_COLORS["default"])
        self._body_color = colors.Q_NODE_BODY
        self._selected_border_color = colors.Q_NODE_SELECTED
        self._border_width = 2

        # Setup (order matters: create header before updating size)
        self._create_header()
        self._setup_item()

    def _setup_item(self) -> None:
        """Configure item properties."""
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        # Initial size
        self._update_size()

    def _create_header(self) -> None:
        """Create the header text."""
        self._header_text = QGraphicsTextItem(self._title, self)
        self._header_text.setDefaultTextColor(colors.Q_NODE_HEADER_TEXT)
        font = QFont(_get_system_font(), 10, QFont.Weight.Bold)
        self._header_text.setFont(font)
        self._header_text.setPos(10, 5)

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_type(self) -> str:
        return self._node_type

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = value
        self._header_text.setPlainText(value)

    @property
    def input_ports(self) -> list[PortItem]:
        return self._input_ports

    @property
    def output_ports(self) -> list[PortItem]:
        return self._output_ports

    def add_input_port(self, name: str, port_type: PortType = PortType.DATA) -> PortItem:
        """Add an input port to the node."""
        port = PortItem(name, port_type, is_output=False, parent=self)
        port._port_index = len(self._input_ports)
        self._input_ports.append(port)
        self._update_port_positions()
        self._update_size()
        return port

    def add_output_port(self, name: str, port_type: PortType = PortType.DATA) -> PortItem:
        """Add an output port to the node."""
        port = PortItem(name, port_type, is_output=True, parent=self)
        port._port_index = len(self._output_ports)
        self._output_ports.append(port)
        self._update_port_positions()
        self._update_size()
        return port

    def set_widget(self, widget: QWidget) -> None:
        """Set an embedded widget in the node."""
        if self._widget_proxy:
            self._widget_proxy.setWidget(None)
            self.scene().removeItem(self._widget_proxy)

        self._widget_proxy = QGraphicsProxyWidget(self)
        self._widget_proxy.setWidget(widget)
        self._widget_height = widget.sizeHint().height() + 10
        self._update_size()
        self._update_widget_position()

    def _update_size(self) -> None:
        """Update the node size based on content."""
        num_ports = max(len(self._input_ports), len(self._output_ports))
        ports_height = num_ports * (self.PORT_HEIGHT + self.PORT_SPACING)

        height = self.HEADER_HEIGHT + ports_height + self._widget_height + 10
        width = max(self.MIN_WIDTH, self._get_text_width() + 40)

        self.setRect(0, 0, width, height)

    def _get_text_width(self) -> float:
        """Get the width needed for the header text."""
        return self._header_text.boundingRect().width()

    def _update_port_positions(self) -> None:
        """Update port positions."""
        rect = self.rect()

        # Input ports (left side)
        y = self.HEADER_HEIGHT + self.PORT_SPACING
        for port in self._input_ports:
            port.setPos(0, y + self.PORT_HEIGHT / 2)
            y += self.PORT_HEIGHT + self.PORT_SPACING

        # Output ports (right side)
        y = self.HEADER_HEIGHT + self.PORT_SPACING
        for port in self._output_ports:
            port.setPos(rect.width(), y + self.PORT_HEIGHT / 2)
            y += self.PORT_HEIGHT + self.PORT_SPACING

    def _update_widget_position(self) -> None:
        """Update embedded widget position."""
        if self._widget_proxy:
            rect = self.rect()
            num_ports = max(len(self._input_ports), len(self._output_ports))
            y = self.HEADER_HEIGHT + num_ports * (self.PORT_HEIGHT + self.PORT_SPACING) + 5
            self._widget_proxy.setPos(10, y)
            self._widget_proxy.widget().setFixedWidth(int(rect.width() - 20))

    def get_port_scene_pos(self, port_index: int, is_output: bool) -> QPointF:
        """Get the scene position of a port."""
        ports = self._output_ports if is_output else self._input_ports
        if 0 <= port_index < len(ports):
            port = ports[port_index]
            return self.mapToScene(port.pos())
        return self.scenePos()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        """Paint the node."""
        rect = self.rect()

        # Body background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._body_color))
        painter.drawRoundedRect(rect, self.CORNER_RADIUS, self.CORNER_RADIUS)

        # Header
        header_rect = QRectF(0, 0, rect.width(), self.HEADER_HEIGHT)
        gradient = QLinearGradient(0, 0, 0, self.HEADER_HEIGHT)
        grad_tuple = self.CATEGORY_GRADIENTS.get(
            self._category, self.CATEGORY_GRADIENTS["default"]
        )
        gradient.setColorAt(0, QColor(grad_tuple[0]))
        gradient.setColorAt(1, QColor(grad_tuple[1]))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(header_rect, self.CORNER_RADIUS, self.CORNER_RADIUS)
        # Fill bottom corners
        painter.drawRect(
            QRectF(0, self.HEADER_HEIGHT - self.CORNER_RADIUS, rect.width(), self.CORNER_RADIUS)
        )

        # Border
        if self.isSelected():
            pen = QPen(self._selected_border_color, self._border_width + 1)
        else:
            alpha_color = colors.qcolor_with_alpha(self._header_color, 0.3)
            pen = QPen(alpha_color, 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, self.CORNER_RADIUS, self.CORNER_RADIUS)

        # Port labels
        painter.setPen(QPen(colors.Q_NODE_PORT_LABEL))
        font = QFont(_get_system_font(), 9)
        painter.setFont(font)

        y = self.HEADER_HEIGHT + self.PORT_SPACING + 4
        for port in self._input_ports:
            painter.drawText(QPointF(15, y + self.PORT_HEIGHT / 2), port.name)
            y += self.PORT_HEIGHT + self.PORT_SPACING

        y = self.HEADER_HEIGHT + self.PORT_SPACING + 4
        for port in self._output_ports:
            text_width = painter.fontMetrics().horizontalAdvance(port.name)
            painter.drawText(
                QPointF(rect.width() - text_width - 15, y + self.PORT_HEIGHT / 2), port.name
            )
            y += self.PORT_HEIGHT + self.PORT_SPACING

    def itemChange(self, change, value):
        """Handle item changes."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            pos = value
            self.position_changed.emit(self._node_id, pos.x(), pos.y())
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.selected_changed.emit(self._node_id, bool(value))
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:
        """Handle hover enter."""
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        """Handle hover leave."""
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        """Handle mouse press."""
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release."""
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)
