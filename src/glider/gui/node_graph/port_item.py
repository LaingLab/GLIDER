"""
Port Item - Visual representation of node input/output ports.
"""

from enum import Enum, auto

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem


class PortSignals(QObject):
    """Signals for PortItem (QGraphicsItem can't have signals directly)."""

    connection_started = pyqtSignal(object)  # port item
    connection_finished = pyqtSignal(object)  # port item


class PortType(Enum):
    """Types of ports."""

    DATA = auto()  # Data flow port
    EXEC = auto()  # Execution flow port


class PortItem(QGraphicsEllipseItem):
    """
    Visual representation of a node port.

    Data ports are circular, exec ports are triangular.
    """

    # Port dimensions
    PORT_RADIUS = 6

    # Port colors by type
    PORT_COLORS = {
        PortType.DATA: QColor(100, 180, 255),  # Blue
        PortType.EXEC: QColor(255, 255, 255),  # White
    }

    def __init__(
        self,
        name: str,
        port_type: PortType = PortType.DATA,
        is_output: bool = False,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)

        self._name = name
        self._port_type = port_type
        self._is_output = is_output
        self._connected = False
        self._hovered = False
        self._port_index = 0  # Set by parent node

        # Signals
        self.signals = PortSignals()
        self.connection_started = self.signals.connection_started
        self.connection_finished = self.signals.connection_finished

        # Setup
        self._setup_item()

    def _setup_item(self) -> None:
        """Configure item properties."""
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        # This flag is critical - allows port to receive events before parent node
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)

        # Set rect for circular port
        r = self.PORT_RADIUS
        self.setRect(-r, -r, r * 2, r * 2)

    @property
    def name(self) -> str:
        return self._name

    @property
    def port_type(self) -> PortType:
        return self._port_type

    @property
    def is_output(self) -> bool:
        return self._is_output

    @property
    def is_connected(self) -> bool:
        return self._connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._connected = value
        self.update()

    def get_scene_center(self) -> QPointF:
        """Get the center point in scene coordinates."""
        return self.scenePos()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        """Paint the port."""
        color = self.PORT_COLORS.get(self._port_type, self.PORT_COLORS[PortType.DATA])

        if self._hovered:
            color = color.lighter(130)

        if self._port_type == PortType.EXEC:
            # Draw triangle for exec ports
            self._draw_exec_port(painter, color)
        else:
            # Draw circle for data ports
            self._draw_data_port(painter, color)

    def _draw_data_port(self, painter: QPainter, color: QColor) -> None:
        """Draw a circular data port."""
        r = self.PORT_RADIUS

        # Outer ring
        painter.setPen(QPen(color.darker(120), 2))
        if self._connected:
            painter.setBrush(QBrush(color))
        else:
            painter.setBrush(QBrush(color.darker(150)))

        painter.drawEllipse(QRectF(-r, -r, r * 2, r * 2))

    def _draw_exec_port(self, painter: QPainter, color: QColor) -> None:
        """Draw a triangular exec port."""
        r = self.PORT_RADIUS

        # Triangle points
        if self._is_output:
            points = [
                QPointF(-r, -r),
                QPointF(r, 0),
                QPointF(-r, r),
            ]
        else:
            points = [
                QPointF(r, -r),
                QPointF(-r, 0),
                QPointF(r, r),
            ]

        painter.setPen(QPen(color.darker(120), 2))
        if self._connected:
            painter.setBrush(QBrush(color))
        else:
            painter.setBrush(QBrush(color.darker(150)))

        from PyQt6.QtGui import QPolygonF

        painter.drawPolygon(QPolygonF(points))

    def hoverEnterEvent(self, event) -> None:
        """Handle hover enter."""
        self._hovered = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        """Handle hover leave."""
        self._hovered = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        """Handle mouse press to start connection."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.connection_started.emit(self)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release to finish connection."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.connection_finished.emit(self)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
