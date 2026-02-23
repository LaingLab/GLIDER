"""
Flow Function Editor Dialog.

Provides a GUI for creating and editing flow function definitions
with a visual graph editor for the internal node graph.
"""

import logging
import uuid
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from glider.core.flow_function import (
    FlowFunctionDefinition,
    FlowFunctionParameter,
    InternalConnectionConfig,
    InternalNodeConfig,
    ParameterType,
)

logger = logging.getLogger(__name__)


class ParameterEditorWidget(QWidget):
    """Widget for editing flow function parameters."""

    parameters_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Parameters table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Name", "Type", "Default", "Description"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._table)

        # Buttons
        btn_layout = QHBoxLayout()
        self._add_btn = QPushButton("Add Parameter")
        self._add_btn.clicked.connect(self._add_parameter)
        btn_layout.addWidget(self._add_btn)

        self._remove_btn = QPushButton("Remove Parameter")
        self._remove_btn.clicked.connect(self._remove_parameter)
        btn_layout.addWidget(self._remove_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _add_parameter(self):
        row = self._table.rowCount()
        self._table.insertRow(row)

        # Name
        name_item = QTableWidgetItem(f"param{row + 1}")
        self._table.setItem(row, 0, name_item)

        # Type combo
        type_combo = QComboBox()
        for pt in ParameterType:
            type_combo.addItem(pt.value, pt)
        self._table.setCellWidget(row, 1, type_combo)

        # Default value
        default_item = QTableWidgetItem("")
        self._table.setItem(row, 2, default_item)

        # Description
        desc_item = QTableWidgetItem("")
        self._table.setItem(row, 3, desc_item)

        self.parameters_changed.emit()

    def _remove_parameter(self):
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)
            self.parameters_changed.emit()

    def get_parameters(self) -> list[FlowFunctionParameter]:
        """Get the list of parameter definitions."""
        params = []
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            type_combo = self._table.cellWidget(row, 1)
            default_item = self._table.item(row, 2)
            desc_item = self._table.item(row, 3)

            if name_item and type_combo:
                name = name_item.text()
                param_type = type_combo.currentData()
                default_text = default_item.text() if default_item else ""
                description = desc_item.text() if desc_item else ""

                # Parse default value
                default_value = None
                if default_text:
                    if param_type == ParameterType.INT:
                        try:
                            default_value = int(default_text)
                        except ValueError:
                            pass
                    elif param_type == ParameterType.FLOAT:
                        try:
                            default_value = float(default_text)
                        except ValueError:
                            pass
                    elif param_type == ParameterType.BOOL:
                        default_value = default_text.lower() in ("true", "1", "yes")
                    else:
                        default_value = default_text

                params.append(
                    FlowFunctionParameter(
                        name=name,
                        param_type=param_type,
                        default_value=default_value,
                        description=description,
                    )
                )
        return params

    def set_parameters(self, params: list[FlowFunctionParameter]):
        """Set the parameter definitions."""
        self._table.setRowCount(0)
        for param in params:
            row = self._table.rowCount()
            self._table.insertRow(row)

            self._table.setItem(row, 0, QTableWidgetItem(param.name))

            type_combo = QComboBox()
            for pt in ParameterType:
                type_combo.addItem(pt.value, pt)
            type_combo.setCurrentText(param.param_type.value)
            self._table.setCellWidget(row, 1, type_combo)

            default_str = ""
            if param.default_value is not None:
                default_str = str(param.default_value)
            self._table.setItem(row, 2, QTableWidgetItem(default_str))

            self._table.setItem(row, 3, QTableWidgetItem(param.description))


class SimpleNodeItem:
    """Simple representation of a node in the graph preview."""

    def __init__(self, node_config: InternalNodeConfig, x: float, y: float):
        self.config = node_config
        self.x = x
        self.y = y
        self.width = 120
        self.height = 40


class FlowGraphPreview(QGraphicsView):
    """Simple preview of the flow function's internal graph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self._nodes: dict[str, Any] = {}
        self._connections: list[InternalConnectionConfig] = []

        self.setMinimumHeight(200)
        self.setRenderHint(self.renderHints().Antialiasing)

    def set_graph(
        self, nodes: list[InternalNodeConfig], connections: list[InternalConnectionConfig]
    ):
        """Set the graph to display."""
        self._scene.clear()
        self._nodes.clear()

        # Draw nodes
        x_offset = 50
        y_offset = 50
        for i, node in enumerate(nodes):
            x = node.position[0] if node.position[0] != 0 else x_offset + (i % 3) * 150
            y = node.position[1] if node.position[1] != 0 else y_offset + (i // 3) * 80

            # Draw node rectangle
            rect = self._scene.addRect(x, y, 120, 40)
            rect.setBrush(QBrush(QColor("#3a3a3a")))
            rect.setPen(QPen(QColor("#5a9bd4"), 2))

            # Draw node label
            text = self._scene.addText(node.node_type)
            text.setDefaultTextColor(QColor("#ffffff"))
            text.setPos(x + 5, y + 10)

            self._nodes[node.id] = {"rect": rect, "x": x, "y": y}

        # Draw connections
        for conn in connections:
            if conn.from_node in self._nodes and conn.to_node in self._nodes:
                from_info = self._nodes[conn.from_node]
                to_info = self._nodes[conn.to_node]

                # Draw line from right edge of from_node to left edge of to_node
                line = self._scene.addLine(
                    from_info["x"] + 120, from_info["y"] + 20, to_info["x"], to_info["y"] + 20
                )
                line.setPen(QPen(QColor("#5a9bd4"), 2))

        # Fit view
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class NodeListEditor(QWidget):
    """Widget for adding/removing internal nodes."""

    nodes_changed = pyqtSignal()

    def __init__(self, available_node_types: list[str], parent=None):
        super().__init__(parent)
        self._available_types = available_node_types
        self._nodes: list[InternalNodeConfig] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add node controls
        add_layout = QHBoxLayout()

        self._type_combo = QComboBox()
        self._type_combo.addItems(self._available_types)
        add_layout.addWidget(self._type_combo)

        self._add_btn = QPushButton("Add Node")
        self._add_btn.clicked.connect(self._add_node)
        add_layout.addWidget(self._add_btn)

        layout.addLayout(add_layout)

        # Node list
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self._list)

        # Remove button
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.clicked.connect(self._remove_node)
        layout.addWidget(self._remove_btn)

    def _add_node(self):
        node_type = self._type_combo.currentText()
        node_id = f"{node_type.lower()}_{uuid.uuid4().hex[:8]}"

        config = InternalNodeConfig(
            id=node_id,
            node_type=node_type,
            position=(len(self._nodes) * 150, 50),
        )
        self._nodes.append(config)

        item = QListWidgetItem(f"{node_type} ({node_id[:12]}...)")
        item.setData(Qt.ItemDataRole.UserRole, node_id)
        self._list.addItem(item)

        self.nodes_changed.emit()

    def _remove_node(self):
        row = self._list.currentRow()
        if row >= 0:
            item = self._list.takeItem(row)
            if item:
                node_id = item.data(Qt.ItemDataRole.UserRole)
                self._nodes = [n for n in self._nodes if n.id != node_id]
                self.nodes_changed.emit()

    def get_nodes(self) -> list[InternalNodeConfig]:
        """Get the list of internal node configs."""
        return self._nodes

    def set_nodes(self, nodes: list[InternalNodeConfig]):
        """Set the internal nodes."""
        self._nodes = nodes
        self._list.clear()
        for node in nodes:
            item = QListWidgetItem(f"{node.node_type} ({node.id[:12]}...)")
            item.setData(Qt.ItemDataRole.UserRole, node.id)
            self._list.addItem(item)


class ConnectionEditor(QWidget):
    """Widget for editing connections between internal nodes."""

    connections_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connections: list[InternalConnectionConfig] = []
        self._node_ids: list[str] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add connection controls
        add_layout = QHBoxLayout()

        add_layout.addWidget(QLabel("From:"))
        self._from_combo = QComboBox()
        add_layout.addWidget(self._from_combo)

        add_layout.addWidget(QLabel("To:"))
        self._to_combo = QComboBox()
        add_layout.addWidget(self._to_combo)

        self._add_btn = QPushButton("Connect")
        self._add_btn.clicked.connect(self._add_connection)
        add_layout.addWidget(self._add_btn)

        layout.addLayout(add_layout)

        # Connection list
        self._list = QListWidget()
        layout.addWidget(self._list)

        # Remove button
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.clicked.connect(self._remove_connection)
        layout.addWidget(self._remove_btn)

    def update_nodes(self, node_ids: list[str]):
        """Update available nodes for connections."""
        self._node_ids = node_ids
        current_from = self._from_combo.currentData()
        current_to = self._to_combo.currentData()

        self._from_combo.clear()
        self._to_combo.clear()

        for node_id in node_ids:
            short_id = node_id[:20] + "..." if len(node_id) > 20 else node_id
            self._from_combo.addItem(short_id, node_id)
            self._to_combo.addItem(short_id, node_id)

        # Restore selection if possible
        idx = self._from_combo.findData(current_from)
        if idx >= 0:
            self._from_combo.setCurrentIndex(idx)
        idx = self._to_combo.findData(current_to)
        if idx >= 0:
            self._to_combo.setCurrentIndex(idx)

    def _add_connection(self):
        from_node = self._from_combo.currentData()
        to_node = self._to_combo.currentData()

        if not from_node or not to_node:
            return
        if from_node == to_node:
            QMessageBox.warning(self, "Invalid Connection", "Cannot connect a node to itself.")
            return

        conn_id = f"conn_{uuid.uuid4().hex[:8]}"
        config = InternalConnectionConfig(
            id=conn_id,
            from_node=from_node,
            from_output=0,
            to_node=to_node,
            to_input=0,
            connection_type="exec",
        )
        self._connections.append(config)

        from_short = from_node[:12] + "..." if len(from_node) > 12 else from_node
        to_short = to_node[:12] + "..." if len(to_node) > 12 else to_node
        item = QListWidgetItem(f"{from_short} -> {to_short}")
        item.setData(Qt.ItemDataRole.UserRole, conn_id)
        self._list.addItem(item)

        self.connections_changed.emit()

    def _remove_connection(self):
        row = self._list.currentRow()
        if row >= 0:
            item = self._list.takeItem(row)
            if item:
                conn_id = item.data(Qt.ItemDataRole.UserRole)
                self._connections = [c for c in self._connections if c.id != conn_id]
                self.connections_changed.emit()

    def get_connections(self) -> list[InternalConnectionConfig]:
        """Get the connection configurations."""
        return self._connections

    def set_connections(self, connections: list[InternalConnectionConfig]):
        """Set the connections."""
        self._connections = connections
        self._list.clear()
        for conn in connections:
            from_short = conn.from_node[:12] + "..." if len(conn.from_node) > 12 else conn.from_node
            to_short = conn.to_node[:12] + "..." if len(conn.to_node) > 12 else conn.to_node
            item = QListWidgetItem(f"{from_short} -> {to_short}")
            item.setData(Qt.ItemDataRole.UserRole, conn.id)
            self._list.addItem(item)


class FlowFunctionDialog(QDialog):
    """Dialog for creating/editing flow function definitions."""

    def __init__(
        self,
        definition: FlowFunctionDefinition | None = None,
        available_node_types: list[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._definition = definition or FlowFunctionDefinition()
        self._available_node_types = available_node_types or [
            "FlowFunctionEntry",
            "FlowFunctionExit",
            "Delay",
            "Output",
            "Input",
            "MotorGovernor",
        ]
        self._setup_ui()
        self._load_definition()

    def _setup_ui(self):
        self.setWindowTitle("Flow Function Editor")
        self.setMinimumSize(800, 700)

        layout = QVBoxLayout(self)

        # Function info
        info_group = QGroupBox("Function Information")
        info_layout = QFormLayout(info_group)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g., Turn Up")
        info_layout.addRow("Name:", self._name_edit)

        self._desc_edit = QTextEdit()
        self._desc_edit.setMaximumHeight(60)
        self._desc_edit.setPlaceholderText("Description of what this function does")
        info_layout.addRow("Description:", self._desc_edit)

        layout.addWidget(info_group)

        # Tabs for different sections
        tabs = QTabWidget()

        # Parameters tab
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        self._param_editor = ParameterEditorWidget()
        params_layout.addWidget(self._param_editor)
        tabs.addTab(params_widget, "Parameters")

        # Flow Graph tab
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)

        # Splitter for nodes/connections and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: nodes and connections
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Nodes
        nodes_group = QGroupBox("Nodes")
        nodes_layout = QVBoxLayout(nodes_group)
        self._node_editor = NodeListEditor(self._available_node_types)
        self._node_editor.nodes_changed.connect(self._on_graph_changed)
        nodes_layout.addWidget(self._node_editor)
        left_layout.addWidget(nodes_group)

        # Connections
        conn_group = QGroupBox("Connections")
        conn_layout = QVBoxLayout(conn_group)
        self._conn_editor = ConnectionEditor()
        self._conn_editor.connections_changed.connect(self._on_graph_changed)
        conn_layout.addWidget(self._conn_editor)
        left_layout.addWidget(conn_group)

        splitter.addWidget(left_widget)

        # Right side: preview
        preview_group = QGroupBox("Graph Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._graph_preview = FlowGraphPreview()
        preview_layout.addWidget(self._graph_preview)
        splitter.addWidget(preview_group)

        splitter.setSizes([300, 500])
        graph_layout.addWidget(splitter)

        tabs.addTab(graph_widget, "Flow Graph")

        layout.addWidget(tabs)

        # Entry/Exit node selection
        entry_exit_layout = QHBoxLayout()

        entry_exit_layout.addWidget(QLabel("Entry Node:"))
        self._entry_combo = QComboBox()
        entry_exit_layout.addWidget(self._entry_combo)

        entry_exit_layout.addWidget(QLabel("Exit Nodes:"))
        self._exit_list = QListWidget()
        self._exit_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._exit_list.setMaximumHeight(60)
        entry_exit_layout.addWidget(self._exit_list)

        layout.addLayout(entry_exit_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_definition(self):
        """Load the definition into the UI."""
        self._name_edit.setText(self._definition.name)
        self._desc_edit.setPlainText(self._definition.description)
        self._param_editor.set_parameters(self._definition.parameters)
        self._node_editor.set_nodes(self._definition.nodes)
        self._conn_editor.set_connections(self._definition.connections)

        self._update_node_combos()
        self._update_preview()

        # Set entry node
        if self._definition.entry_node_id:
            idx = self._entry_combo.findData(self._definition.entry_node_id)
            if idx >= 0:
                self._entry_combo.setCurrentIndex(idx)

        # Set exit nodes
        for i in range(self._exit_list.count()):
            item = self._exit_list.item(i)
            if item:
                node_id = item.data(Qt.ItemDataRole.UserRole)
                if node_id in self._definition.exit_node_ids:
                    item.setSelected(True)

    def _on_graph_changed(self):
        """Handle changes to the graph."""
        self._update_node_combos()
        self._update_preview()

    def _update_node_combos(self):
        """Update entry/exit node selection combos."""
        nodes = self._node_editor.get_nodes()
        node_ids = [n.id for n in nodes]

        # Update connection editor
        self._conn_editor.update_nodes(node_ids)

        # Update entry combo
        current_entry = self._entry_combo.currentData()
        self._entry_combo.clear()
        for node in nodes:
            short_id = node.id[:20] + "..." if len(node.id) > 20 else node.id
            self._entry_combo.addItem(f"{node.node_type} ({short_id})", node.id)

        idx = self._entry_combo.findData(current_entry)
        if idx >= 0:
            self._entry_combo.setCurrentIndex(idx)

        # Update exit list
        selected_exits = [
            self._exit_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._exit_list.count())
            if self._exit_list.item(i).isSelected()
        ]

        self._exit_list.clear()
        for node in nodes:
            short_id = node.id[:20] + "..." if len(node.id) > 20 else node.id
            item = QListWidgetItem(f"{node.node_type} ({short_id})")
            item.setData(Qt.ItemDataRole.UserRole, node.id)
            self._exit_list.addItem(item)
            if node.id in selected_exits:
                item.setSelected(True)

    def _update_preview(self):
        """Update the graph preview."""
        nodes = self._node_editor.get_nodes()
        connections = self._conn_editor.get_connections()
        self._graph_preview.set_graph(nodes, connections)

    def _on_accept(self):
        """Handle OK button."""
        # Validate
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Function name is required.")
            return

        # Build definition
        self._definition.name = name
        self._definition.description = self._desc_edit.toPlainText()
        self._definition.parameters = self._param_editor.get_parameters()
        self._definition.nodes = self._node_editor.get_nodes()
        self._definition.connections = self._conn_editor.get_connections()
        self._definition.entry_node_id = self._entry_combo.currentData()

        # Collect exit nodes
        exit_nodes = []
        for i in range(self._exit_list.count()):
            item = self._exit_list.item(i)
            if item and item.isSelected():
                exit_nodes.append(item.data(Qt.ItemDataRole.UserRole))
        self._definition.exit_node_ids = exit_nodes

        self.accept()

    def get_definition(self) -> FlowFunctionDefinition:
        """Get the flow function definition."""
        return self._definition
