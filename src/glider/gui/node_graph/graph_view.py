"""
Node Graph View - The main canvas for visual scripting.

Uses Qt's QGraphicsView/QGraphicsScene for efficient rendering
of nodes, connections, and interactive editing.
"""

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QWidget,
)

if TYPE_CHECKING:
    from glider.core.flow_engine import FlowEngine
    from glider.gui.node_graph.connection_item import ConnectionItem
    from glider.gui.node_graph.node_item import NodeItem

logger = logging.getLogger(__name__)


class NodeGraphScene(QGraphicsScene):
    """Custom scene for the node graph with grid background."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid_size = 20
        self._grid_color = QColor(50, 50, 50)
        self._grid_color_major = QColor(70, 70, 70)
        self._background_color = QColor(30, 30, 30)

        # Set scene rect
        self.setSceneRect(-10000, -10000, 20000, 20000)
        self.setBackgroundBrush(QBrush(self._background_color))

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw grid background."""
        super().drawBackground(painter, rect)

        # Draw grid
        left = int(rect.left()) - (int(rect.left()) % self._grid_size)
        top = int(rect.top()) - (int(rect.top()) % self._grid_size)

        # Minor grid lines
        pen = QPen(self._grid_color, 0.5)
        painter.setPen(pen)

        lines = []
        x = left
        while x < rect.right():
            lines.append((QPointF(x, rect.top()), QPointF(x, rect.bottom())))
            x += self._grid_size

        y = top
        while y < rect.bottom():
            lines.append((QPointF(rect.left(), y), QPointF(rect.right(), y)))
            y += self._grid_size

        for line in lines:
            painter.drawLine(line[0], line[1])

        # Major grid lines (every 5th line)
        pen = QPen(self._grid_color_major, 1)
        painter.setPen(pen)

        major_size = self._grid_size * 5
        left = int(rect.left()) - (int(rect.left()) % major_size)
        top = int(rect.top()) - (int(rect.top()) % major_size)

        x = left
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += major_size

        y = top
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += major_size


class NodeGraphView(QGraphicsView):
    """
    Main view for the node graph editor.

    Supports:
    - Pan and zoom navigation
    - Drag-and-drop node creation
    - Connection drawing
    - Selection and multi-selection
    - Context menus
    """

    # Signals
    node_created = pyqtSignal(str, float, float)  # node_type, x, y
    node_deleted = pyqtSignal(str)  # node_id
    node_selected = pyqtSignal(str)  # node_id
    node_moved = pyqtSignal(str, float, float)  # node_id, x, y
    connection_created = pyqtSignal(
        str, int, str, int, str
    )  # from_node, from_port, to_node, to_port, conn_type
    connection_deleted = pyqtSignal(str)  # connection_id

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        # Create scene
        self._scene = NodeGraphScene(self)
        self.setScene(self._scene)

        # Configure view
        self._setup_view()

        # State
        self._nodes: dict[str, NodeItem] = {}
        self._connections: dict[str, ConnectionItem] = {}
        self._flow_engine: FlowEngine | None = None

        # Interaction state
        self._panning = False
        self._pan_start = QPointF()
        self._connecting = False
        self._temp_connection = None
        self._connection_start_port = None
        self._connection_start_node = None
        self._selected_nodes: list[str] = []

        # Zoom settings
        self._zoom = 1.0
        self._zoom_min = 0.1
        self._zoom_max = 3.0

    def _setup_view(self) -> None:
        """Configure the graphics view."""
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Enable drag and drop
        self.setAcceptDrops(True)

    def set_flow_engine(self, engine: "FlowEngine") -> None:
        """Set the flow engine for node management."""
        self._flow_engine = engine

    @property
    def nodes(self) -> dict[str, "NodeItem"]:
        """Get all node items."""
        return self._nodes.copy()

    @property
    def connections(self) -> dict[str, "ConnectionItem"]:
        """Get all connection items."""
        return self._connections.copy()

    def add_node(self, node_id: str, node_type: str, x: float, y: float) -> "NodeItem":
        """Add a node to the graph."""
        from glider.gui.node_graph.node_item import NodeItem

        node_item = NodeItem(node_id, node_type)
        node_item.setPos(x, y)
        self._scene.addItem(node_item)
        self._nodes[node_id] = node_item

        # Connect signals
        node_item.position_changed.connect(self._on_node_moved)
        node_item.selected_changed.connect(self._on_node_selected)

        # Connect port signals for connection creation
        self._connect_port_signals(node_item)

        logger.debug(f"Added node: {node_id} at ({x}, {y})")
        return node_item

    def _connect_port_signals(self, node_item: "NodeItem") -> None:
        """Connect port signals for connection creation."""
        for port in node_item.input_ports:
            port.connection_started.connect(
                lambda p, n=node_item: self._on_port_connection_started(n, p)
            )
            port.connection_finished.connect(
                lambda p, n=node_item: self._on_port_connection_finished(n, p)
            )
        for port in node_item.output_ports:
            port.connection_started.connect(
                lambda p, n=node_item: self._on_port_connection_started(n, p)
            )
            port.connection_finished.connect(
                lambda p, n=node_item: self._on_port_connection_finished(n, p)
            )

    def _on_port_connection_started(self, node: "NodeItem", port) -> None:
        """Handle start of connection from a port."""
        from glider.gui.node_graph.connection_item import TempConnectionItem

        self._connecting = True
        self._connection_start_port = port
        self._connection_start_node = node

        # Create temporary connection
        start_pos = port.get_scene_center()
        is_exec = port.port_type.name == "EXEC"
        self._temp_connection = TempConnectionItem(start_pos, is_exec)
        self._scene.addItem(self._temp_connection)

        logger.debug(f"Connection started from {node.node_id}:{port.name}")

    def _on_port_connection_finished(self, node: "NodeItem", port) -> None:
        """Handle end of connection at a port."""
        if not self._connecting or self._temp_connection is None:
            return

        # Remove temp connection
        self._scene.removeItem(self._temp_connection)
        self._temp_connection = None

        start_port = self._connection_start_port
        start_node = self._connection_start_node

        # Validate connection
        if start_node is None or start_port is None:
            self._connecting = False
            return

        # Can't connect to same node
        if start_node.node_id == node.node_id:
            logger.debug("Cannot connect node to itself")
            self._connecting = False
            return

        # Must connect output to input (or vice versa)
        if start_port.is_output == port.is_output:
            logger.debug("Cannot connect same port types")
            self._connecting = False
            return

        # Determine from/to based on which is output
        if start_port.is_output:
            from_node_id = start_node.node_id
            from_port = start_port._port_index
            to_node_id = node.node_id
            to_port = port._port_index
            conn_type = "exec" if start_port.port_type.name == "EXEC" else "data"
        else:
            from_node_id = node.node_id
            from_port = port._port_index
            to_node_id = start_node.node_id
            to_port = start_port._port_index
            conn_type = "exec" if port.port_type.name == "EXEC" else "data"

        # Emit connection created signal
        self.connection_created.emit(from_node_id, from_port, to_node_id, to_port, conn_type)

        # Mark ports as connected
        start_port.is_connected = True
        port.is_connected = True

        self._connecting = False
        logger.debug(f"Connection created: {from_node_id}:{from_port} -> {to_node_id}:{to_port}")

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph."""
        if node_id in self._nodes:
            node_item = self._nodes.pop(node_id)
            self._scene.removeItem(node_item)

            # Remove associated connections
            connections_to_remove = [
                cid
                for cid, conn in self._connections.items()
                if conn.from_node_id == node_id or conn.to_node_id == node_id
            ]
            for cid in connections_to_remove:
                self.remove_connection(cid)

            logger.debug(f"Removed node: {node_id}")

    def add_connection(
        self,
        connection_id: str,
        from_node_id: str,
        from_port: int,
        to_node_id: str,
        to_port: int,
    ) -> "ConnectionItem":
        """Add a connection between nodes."""
        from glider.gui.node_graph.connection_item import ConnectionItem

        from_node = self._nodes.get(from_node_id)
        to_node = self._nodes.get(to_node_id)

        if not from_node or not to_node:
            raise ValueError("Invalid node IDs for connection")

        conn_item = ConnectionItem(
            connection_id,
            from_node,
            from_port,
            to_node,
            to_port,
        )
        self._scene.addItem(conn_item)
        self._connections[connection_id] = conn_item

        logger.debug(f"Added connection: {connection_id}")
        return conn_item

    def remove_connection(self, connection_id: str) -> None:
        """Remove a connection."""
        if connection_id in self._connections:
            conn_item = self._connections.pop(connection_id)
            self._scene.removeItem(conn_item)
            logger.debug(f"Removed connection: {connection_id}")

    def clear_graph(self) -> None:
        """Clear all nodes and connections."""
        for node_id in list(self._nodes.keys()):
            self.remove_node(node_id)
        self._nodes.clear()
        self._connections.clear()

    def center_on_nodes(self) -> None:
        """Center view on all nodes."""
        if self._nodes:
            items_rect = self._scene.itemsBoundingRect()
            self.fitInView(items_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()

    def _on_node_moved(self, node_id: str, x: float, y: float) -> None:
        """Handle node position changes."""
        # Update connected wires
        for conn in self._connections.values():
            if conn.from_node_id == node_id or conn.to_node_id == node_id:
                conn.update_path()

        # Emit signal for session update
        self.node_moved.emit(node_id, x, y)

    def _on_node_selected(self, node_id: str, selected: bool) -> None:
        """Handle node selection changes."""
        if selected:
            if node_id not in self._selected_nodes:
                self._selected_nodes.append(node_id)
            self.node_selected.emit(node_id)
        else:
            if node_id in self._selected_nodes:
                self._selected_nodes.remove(node_id)

    # Event handlers
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle zoom with mouse wheel."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = self._zoom * factor

        if self._zoom_min <= new_zoom <= self._zoom_max:
            self._zoom = new_zoom
            self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for panning and connection creation."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on a port
            scene_pos = self.mapToScene(event.position().toPoint())
            item = self._scene.itemAt(scene_pos, self.transform())

            from glider.gui.node_graph.port_item import PortItem

            if isinstance(item, PortItem):
                # Start connection from this port
                self._start_connection_from_port(item)
                event.accept()
                return

            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def _start_connection_from_port(self, port) -> None:
        """Start a connection from a port."""
        from glider.gui.node_graph.connection_item import TempConnectionItem

        # Find the parent node
        parent_node = port.parentItem()
        if parent_node is None:
            return

        self._connecting = True
        self._connection_start_port = port
        self._connection_start_node = parent_node

        # Create temporary connection
        start_pos = port.get_scene_center()
        is_exec = port.port_type.name == "EXEC"
        self._temp_connection = TempConnectionItem(start_pos, is_exec)
        self._scene.addItem(self._temp_connection)

        logger.debug(f"Connection started from port: {port.name}")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for panning and connection drawing."""
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
        elif self._connecting and self._temp_connection is not None:
            # Update temporary connection end point
            scene_pos = self.mapToScene(event.position().toPoint())
            self._temp_connection.set_end(scene_pos)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton and self._connecting:
            # Check if released on a port
            scene_pos = self.mapToScene(event.position().toPoint())
            item = self._scene.itemAt(scene_pos, self.transform())

            from glider.gui.node_graph.port_item import PortItem

            if isinstance(item, PortItem):
                self._finish_connection_at_port(item)
            else:
                # Cancel connection - not released on a port
                if self._temp_connection is not None:
                    self._scene.removeItem(self._temp_connection)
                    self._temp_connection = None
                self._connecting = False
                self._connection_start_port = None
                self._connection_start_node = None
        else:
            super().mouseReleaseEvent(event)

    def _finish_connection_at_port(self, target_port) -> None:
        """Finish a connection at a target port."""
        # Remove temp connection
        if self._temp_connection is not None:
            self._scene.removeItem(self._temp_connection)
            self._temp_connection = None

        start_port = self._connection_start_port
        start_node = self._connection_start_node
        target_node = target_port.parentItem()

        # Reset state
        self._connecting = False
        self._connection_start_port = None
        self._connection_start_node = None

        # Validate connection
        if start_node is None or start_port is None or target_node is None:
            return

        # Can't connect to same node
        if start_node.node_id == target_node.node_id:
            logger.debug("Cannot connect node to itself")
            return

        # Must connect output to input (or vice versa)
        if start_port.is_output == target_port.is_output:
            logger.debug("Cannot connect same port types (output to output or input to input)")
            return

        # Determine from/to based on which is output
        if start_port.is_output:
            from_node_id = start_node.node_id
            from_port = start_port._port_index
            to_node_id = target_node.node_id
            to_port = target_port._port_index
            conn_type = "exec" if start_port.port_type.name == "EXEC" else "data"
        else:
            from_node_id = target_node.node_id
            from_port = target_port._port_index
            to_node_id = start_node.node_id
            to_port = start_port._port_index
            conn_type = "exec" if target_port.port_type.name == "EXEC" else "data"

        # Emit connection created signal
        self.connection_created.emit(from_node_id, from_port, to_node_id, to_port, conn_type)

        # Mark ports as connected
        start_port.is_connected = True
        target_port.is_connected = True

        logger.debug(f"Connection created: {from_node_id}:{from_port} -> {to_node_id}:{to_port}")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Delete:
            # Delete selected connections first
            from glider.gui.node_graph.connection_item import ConnectionItem

            selected_items = self._scene.selectedItems()
            for item in selected_items:
                if isinstance(item, ConnectionItem):
                    conn_id = item.connection_id
                    self.connection_deleted.emit(conn_id)
                    self.remove_connection(conn_id)

            # Delete selected nodes
            for node_id in self._selected_nodes.copy():
                self.node_deleted.emit(node_id)
                self.remove_node(node_id)
        elif event.key() == Qt.Key.Key_Home:
            # Reset view
            self.resetTransform()
            self._zoom = 1.0
            self.centerOn(0, 0)
        elif event.key() == Qt.Key.Key_F:
            # Fit all nodes in view
            self.center_on_nodes()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:
        """Show context menu."""
        menu = QMenu(self)

        # Add node submenu
        add_menu = menu.addMenu("Add Node")

        # Flow nodes
        flow_menu = add_menu.addMenu("Flow")
        flow_menu.addAction(
            "Start Experiment", lambda: self._add_node_at_cursor("StartExperiment", event.pos())
        )
        flow_menu.addAction(
            "End Experiment", lambda: self._add_node_at_cursor("EndExperiment", event.pos())
        )
        flow_menu.addAction("Delay", lambda: self._add_node_at_cursor("Delay", event.pos()))

        # I/O nodes
        io_menu = add_menu.addMenu("I/O")
        io_menu.addAction("Output", lambda: self._add_node_at_cursor("Output", event.pos()))
        io_menu.addAction("Input", lambda: self._add_node_at_cursor("Input", event.pos()))

        menu.addSeparator()

        # Check for selected items (nodes or connections)
        from glider.gui.node_graph.connection_item import ConnectionItem

        selected_items = self._scene.selectedItems()
        has_selected_connections = any(isinstance(item, ConnectionItem) for item in selected_items)

        if self._selected_nodes or has_selected_connections:
            menu.addAction("Delete Selected", self._delete_selected)

        menu.exec(event.globalPos())

    def _add_node_at_cursor(self, node_type: str, pos) -> None:
        """Add a node at the cursor position."""
        scene_pos = self.mapToScene(pos)
        self.node_created.emit(node_type, scene_pos.x(), scene_pos.y())

    def _delete_selected(self) -> None:
        """Delete all selected nodes and connections."""
        # Delete selected connections first
        from glider.gui.node_graph.connection_item import ConnectionItem

        selected_items = self._scene.selectedItems()
        for item in selected_items:
            if isinstance(item, ConnectionItem):
                conn_id = item.connection_id
                self.connection_deleted.emit(conn_id)
                self.remove_connection(conn_id)

        # Delete selected nodes
        for node_id in self._selected_nodes.copy():
            self.node_deleted.emit(node_id)
            self.remove_node(node_id)

    # Drag and drop handlers
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasText():
            text = event.mimeData().text()
            if text.startswith("node:"):
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Handle drag move event."""
        if event.mimeData().hasText():
            text = event.mimeData().text()
            if text.startswith("node:"):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop event to create a new node."""
        if event.mimeData().hasText():
            text = event.mimeData().text()
            if text.startswith("node:"):
                node_type = text[5:]  # Remove "node:" prefix
                # Convert drop position to scene coordinates
                scene_pos = self.mapToScene(event.position().toPoint())
                # Emit signal to create node
                self.node_created.emit(node_type, scene_pos.x(), scene_pos.y())
                event.acceptProposedAction()
                return
        event.ignore()
