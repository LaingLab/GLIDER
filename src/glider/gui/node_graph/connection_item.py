"""
Connection Item - Visual representation of connections between nodes.
"""

from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPathItem

from glider.gui.styles import colors

if TYPE_CHECKING:
    from glider.gui.node_graph.node_item import NodeItem


class ConnectionItem(QGraphicsPathItem):
    """
    Visual representation of a connection between node ports.

    Draws a bezier curve between the output port of one node
    and the input port of another.
    """

    # Connection colors
    DATA_COLOR = colors.Q_PORT_DATA
    EXEC_COLOR = colors.Q_PORT_EXEC
    ACTIVE_COLOR = colors.Q_CONN_ACTIVE

    def __init__(
        self,
        connection_id: str,
        from_node: "NodeItem",
        from_port: int,
        to_node: "NodeItem",
        to_port: int,
        is_exec: bool = False,
    ):
        super().__init__()

        self._connection_id = connection_id
        self._from_node = from_node
        self._from_port = from_port
        self._to_node = to_node
        self._to_port = to_port
        self._is_exec = is_exec
        self._active = False

        # Appearance
        self._line_width = 3 if is_exec else 2
        self._color = self.EXEC_COLOR if is_exec else self.DATA_COLOR

        # Setup
        self._setup_item()
        self.update_path()

    def _setup_item(self) -> None:
        """Configure item properties."""
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)  # Draw behind nodes

    @property
    def connection_id(self) -> str:
        return self._connection_id

    @property
    def from_node_id(self) -> str:
        return self._from_node.node_id

    @property
    def to_node_id(self) -> str:
        return self._to_node.node_id

    @property
    def from_port(self) -> int:
        return self._from_port

    @property
    def to_port(self) -> int:
        return self._to_port

    @property
    def is_active(self) -> bool:
        return self._active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """Set active state (for visualizing data flow)."""
        self._active = value
        self.update()

    def update_path(self) -> None:
        """Update the bezier path based on node positions."""
        start = self._from_node.get_port_scene_pos(self._from_port, is_output=True)
        end = self._to_node.get_port_scene_pos(self._to_port, is_output=False)

        path = self._create_bezier_path(start, end)
        self.setPath(path)

    def _create_bezier_path(self, start: QPointF, end: QPointF) -> QPainterPath:
        """Create a bezier curve path between two points."""
        path = QPainterPath()
        path.moveTo(start)

        # Calculate control points
        dx = abs(end.x() - start.x())
        abs(end.y() - start.y())
        ctrl_offset = max(50, min(dx * 0.5, 150))

        # Horizontal offset for control points
        ctrl1 = QPointF(start.x() + ctrl_offset, start.y())
        ctrl2 = QPointF(end.x() - ctrl_offset, end.y())

        path.cubicTo(ctrl1, ctrl2, end)
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:
        """Paint the connection."""
        # Determine color
        if self._active:
            color = self.ACTIVE_COLOR
        elif self.isSelected():
            color = colors.Q_CONN_SELECTED
        else:
            color = self._color

        # Draw path
        pen = QPen(color, self._line_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        if self._is_exec:
            pen.setStyle(Qt.PenStyle.SolidLine)
        else:
            pen.setStyle(Qt.PenStyle.SolidLine)

        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())

        # Draw arrow at end for exec connections
        if self._is_exec:
            self._draw_arrow(painter, color)

    def _draw_arrow(self, painter: QPainter, color: QColor) -> None:
        """Draw an arrow at the end of exec connections."""
        path = self.path()
        if path.isEmpty():
            return

        # Get end point and direction
        end = path.pointAtPercent(1.0)
        near_end = path.pointAtPercent(0.95)

        dx = end.x() - near_end.x()
        dy = end.y() - near_end.y()

        import math

        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.01:
            return

        # Normalize
        dx /= length
        dy /= length

        # Arrow dimensions
        arrow_size = 8
        arrow_angle = 0.5  # radians

        # Arrow points
        p1 = QPointF(
            end.x() - arrow_size * (dx * math.cos(arrow_angle) - dy * math.sin(arrow_angle)),
            end.y() - arrow_size * (dy * math.cos(arrow_angle) + dx * math.sin(arrow_angle)),
        )
        p2 = QPointF(
            end.x() - arrow_size * (dx * math.cos(arrow_angle) + dy * math.sin(arrow_angle)),
            end.y() - arrow_size * (dy * math.cos(arrow_angle) - dx * math.sin(arrow_angle)),
        )

        from PyQt6.QtGui import QBrush, QPolygonF

        arrow = QPolygonF([end, p1, p2])
        painter.setBrush(QBrush(color))
        painter.drawPolygon(arrow)


class TempConnectionItem(QGraphicsPathItem):
    """Temporary connection while user is dragging to create a new connection."""

    def __init__(self, start: QPointF, is_exec: bool = False):
        super().__init__()

        self._start = start
        self._end = start
        self._is_exec = is_exec
        self._line_width = 3 if is_exec else 2
        self._color = ConnectionItem.EXEC_COLOR if is_exec else ConnectionItem.DATA_COLOR

        self.setZValue(-1)

    def set_end(self, end: QPointF) -> None:
        """Update the end point."""
        self._end = end
        self._update_path()

    def _update_path(self) -> None:
        """Update the bezier path."""
        path = QPainterPath()
        path.moveTo(self._start)

        dx = abs(self._end.x() - self._start.x())
        ctrl_offset = max(50, min(dx * 0.5, 150))

        ctrl1 = QPointF(self._start.x() + ctrl_offset, self._start.y())
        ctrl2 = QPointF(self._end.x() - ctrl_offset, self._end.y())

        path.cubicTo(ctrl1, ctrl2, self._end)
        self.setPath(path)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        """Paint the temporary connection."""
        pen = QPen(self._color, self._line_width)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawPath(self.path())
