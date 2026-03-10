"""
Node Library Panel - Dock widget for the node library with draggable node buttons.

Provides the library of available nodes organized by category, plus
custom device, flow function, and zone node sections.
"""

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import QMimeData, Qt, pyqtSignal
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.gui.node_graph.graph_view import NodeGraphView

logger = logging.getLogger(__name__)


class DraggableNodeButton(QPushButton):
    """A button that can be dragged to create nodes in the graph."""

    def __init__(self, node_type: str, display_name: str, category: str, parent=None):
        super().__init__(display_name, parent)
        self._node_type = node_type
        self._category = category

        self.setProperty("nodeCategory", category)
        self.setProperty("nodeButton", True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        if not hasattr(self, "_drag_start_pos"):
            return

        if (event.pos() - self._drag_start_pos).manhattanLength() < 10:
            return

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(f"node:{self._node_type}")
        drag.setMimeData(mime_data)

        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        drag.exec(Qt.DropAction.CopyAction)
        self.setCursor(Qt.CursorShape.OpenHandCursor)


class EditableDraggableButton(DraggableNodeButton):
    """A draggable button with context menu for edit/delete."""

    def __init__(
        self,
        node_type: str,
        display_name: str,
        category: str,
        on_edit=None,
        on_delete=None,
        parent=None,
    ):
        super().__init__(node_type, display_name, category, parent)
        self._on_edit = on_edit
        self._on_delete = on_delete
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos):
        menu = QMenu(self)

        edit_action = menu.addAction("Edit")
        edit_action.triggered.connect(self._handle_edit)

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(self._handle_delete)

        menu.exec(self.mapToGlobal(pos))

    def _handle_edit(self):
        if self._on_edit:
            self._on_edit()

    def _handle_delete(self):
        if self._on_delete:
            self._on_delete()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._on_edit:
                self._on_edit()
        else:
            super().mouseDoubleClickEvent(event)


class NodeLibraryPanel(QWidget):
    """Panel for the node library with draggable node items."""

    status_message = pyqtSignal(str, int)  # message, timeout_ms

    def __init__(self, session_fn, graph_view: "NodeGraphView", parent=None):
        """
        Args:
            session_fn: Callable that returns the current ExperimentSession (or None)
            graph_view: NodeGraphView instance for adding nodes
        """
        super().__init__(parent)
        self._session_fn = session_fn
        self._graph_view = graph_view

        # Zone configuration reference (set externally)
        self._zone_config = None

        self._setup_ui()

    @property
    def _session(self):
        return self._session_fn()

    def _setup_ui(self):
        """Build the node library panel UI."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Define available nodes by category
        node_categories = {
            "Flow": [
                ("StartExperiment", "Start Experiment", "Entry point - begins the experiment flow"),
                ("EndExperiment", "End Experiment", "Exit point - ends the experiment"),
                ("Delay", "Delay", "Wait for a specified duration"),
            ],
            "Functions": [
                (
                    "StartFunction",
                    "Start Function",
                    "Define a reusable function - set name in properties",
                ),
                ("EndFunction", "End Function", "End of function definition"),
            ],
            "Control": [
                ("Loop", "Loop", "Repeat actions N times (0 = infinite)"),
                ("WaitForInput", "Wait For Input", "Wait for input trigger before continuing"),
            ],
            "I/O": [
                ("Output", "Output", "Write to a device (digital or PWM)"),
                ("Input", "Input", "Read from a device (digital or analog)"),
            ],
            "Audio": [
                ("AudioPlayback", "Audio Playback", "Play an audio file (WAV/MP3)"),
            ],
            "Video": [
                ("VideoPlayback", "Video Playback", "Play a video file (MP4/AVI)"),
            ],
        }

        category_colors = {
            "Flow": "#2d5a7a",
            "Functions": "#2d7a7a",
            "Control": "#7a5a2d",
            "I/O": "#2d7a2d",
            "Audio": "#6a3a6a",
            "Video": "#2d5a6a",
        }

        for category, nodes in node_categories.items():
            color = category_colors.get(category, "#444")

            category_widget = QWidget()
            category_layout = QVBoxLayout(category_widget)
            category_layout.setContentsMargins(0, 0, 0, 0)
            category_layout.setSpacing(2)

            header_btn = QPushButton(f"\u25bc  {category}")
            header_btn.setCheckable(True)
            header_btn.setChecked(True)
            header_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            header_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 12px;
                    font-size: 13px;
                    font-weight: bold;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: {color}cc;
                }}
                QPushButton:checked {{
                    border-bottom-left-radius: 0px;
                    border-bottom-right-radius: 0px;
                }}
            """)
            category_layout.addWidget(header_btn)

            nodes_container = QWidget()
            nodes_container.setStyleSheet(f"""
                QWidget {{
                    background-color: {color}40;
                    border-bottom-left-radius: 4px;
                    border-bottom-right-radius: 4px;
                }}
            """)
            nodes_layout = QVBoxLayout(nodes_container)
            nodes_layout.setContentsMargins(4, 4, 4, 4)
            nodes_layout.setSpacing(2)

            for node_type, node_name, tooltip in nodes:
                node_btn = DraggableNodeButton(node_type, node_name, category)
                node_btn.setToolTip(tooltip)
                node_btn.clicked.connect(lambda checked, nt=node_type: self._add_node_to_center(nt))
                nodes_layout.addWidget(node_btn)

            category_layout.addWidget(nodes_container)

            def make_toggle(btn, container):
                def toggle(checked):
                    container.setVisible(checked)
                    btn.setText(
                        f"\u25bc  {btn.text()[3:]}" if checked else f"\u25b6  {btn.text()[3:]}"
                    )

                return toggle

            header_btn.toggled.connect(make_toggle(header_btn, nodes_container))

            layout.addWidget(category_widget)

        # Custom Devices section
        self._custom_devices_container = QWidget()
        self._custom_devices_layout = QVBoxLayout(self._custom_devices_container)
        self._custom_devices_layout.setContentsMargins(0, 0, 0, 0)
        self._custom_devices_layout.setSpacing(2)
        self._setup_custom_category(
            self._custom_devices_container,
            self._custom_devices_layout,
            "Custom Devices",
            "#6a4a8a",
            layout,
            add_new_callback=self._on_new_custom_device,
        )

        # Flow Functions section
        self._flow_functions_container = QWidget()
        self._flow_functions_layout = QVBoxLayout(self._flow_functions_container)
        self._flow_functions_layout.setContentsMargins(0, 0, 0, 0)
        self._flow_functions_layout.setSpacing(2)
        self._setup_custom_category(
            self._flow_functions_container,
            self._flow_functions_layout,
            "Flow Functions",
            "#4a6a8a",
            layout,
            add_new_callback=self._on_new_flow_function,
        )

        # Zones section
        self._zones_container = QWidget()
        self._zones_layout = QVBoxLayout(self._zones_container)
        self._zones_layout.setContentsMargins(0, 0, 0, 0)
        self._zones_layout.setSpacing(2)
        self._setup_custom_category(
            self._zones_container,
            self._zones_layout,
            "Zones",
            "#5a4a2d",
            layout,
            add_new_callback=None,
        )

        layout.addStretch()
        scroll_area.setWidget(container)

        self._node_library_container = container
        self._node_library_layout = layout

        outer_layout.addWidget(scroll_area)

    # --- Public API ---

    def refresh_custom_devices(self) -> None:
        """Refresh the custom devices in the node library."""
        while self._custom_devices_layout.count():
            item = self._custom_devices_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        session = self._session
        if session:
            definitions = session.custom_device_definitions
            if definitions:
                for def_id, def_dict in definitions.items():
                    name = def_dict.get("name", "Unknown")
                    btn = EditableDraggableButton(
                        f"CustomDevice:{def_id}",
                        name,
                        "Custom Devices",
                        on_edit=lambda did=def_id: self._edit_custom_device(did),
                        on_delete=lambda did=def_id: self._delete_custom_device(did),
                    )
                    btn.setToolTip(
                        f"{def_dict.get('description', '')}\n(Right-click to edit/delete)"
                    )
                    btn.clicked.connect(
                        lambda checked, did=def_id: self._add_custom_device_node(did)
                    )
                    self._custom_devices_layout.addWidget(btn)
            else:
                self._add_placeholder(self._custom_devices_layout, "No devices defined")
        else:
            self._add_placeholder(self._custom_devices_layout, "No devices defined")

    def refresh_flow_functions(self) -> None:
        """Refresh the flow functions in the node library."""
        while self._flow_functions_layout.count():
            item = self._flow_functions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        has_functions = False

        detected_functions = self._detect_graph_functions()
        if detected_functions:
            for func_info in detected_functions:
                has_functions = True
                func_name = func_info["name"]
                start_node_id = func_info["start_node_id"]

                btn = DraggableNodeButton(f"FunctionCall:{start_node_id}", func_name, "Functions")
                btn.setToolTip(f"Call function '{func_name}'")
                btn.clicked.connect(
                    lambda checked, nid=start_node_id, name=func_name: self._add_function_call_node(
                        nid, name
                    )
                )
                self._flow_functions_layout.addWidget(btn)

        if not has_functions:
            placeholder = QLabel("Define functions with StartFunction \u2192 EndFunction")
            placeholder.setStyleSheet("color: #888; padding: 8px; font-size: 10px;")
            placeholder.setWordWrap(True)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._flow_functions_layout.addWidget(placeholder)

    def refresh_zones(self, zone_config=None) -> None:
        """Refresh the zones in the node library."""
        if zone_config is not None:
            self._zone_config = zone_config

        while self._zones_layout.count():
            item = self._zones_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._zone_config and self._zone_config.zones:
            for zone in self._zone_config.zones:
                btn = DraggableNodeButton(f"ZoneInput:{zone.id}", f"Zone: {zone.name}", "Zones")
                btn.setToolTip(
                    f"Monitor zone '{zone.name}' for object occupancy\n"
                    f"Outputs: Occupied (bool), Object Count (int), On Enter, On Exit"
                )
                btn.clicked.connect(lambda checked, zid=zone.id: self._add_zone_node(zid))
                self._zones_layout.addWidget(btn)
        else:
            placeholder = QLabel("Create zones in Camera \u2192 Zones...")
            placeholder.setStyleSheet("color: #888; padding: 8px; font-size: 10px;")
            placeholder.setWordWrap(True)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._zones_layout.addWidget(placeholder)

    # --- Internal methods ---

    def _add_placeholder(self, layout, text: str):
        placeholder = QLabel(text)
        placeholder.setStyleSheet("color: #888; padding: 8px;")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

    def _setup_custom_category(
        self,
        nodes_container: QWidget,
        nodes_layout: QVBoxLayout,
        category_name: str,
        color: str,
        parent_layout: QVBoxLayout,
        add_new_callback=None,
    ) -> None:
        """Setup a custom category with a header and add button."""
        category_widget = QWidget()
        category_layout = QVBoxLayout(category_widget)
        category_layout.setContentsMargins(0, 0, 0, 0)
        category_layout.setSpacing(2)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        header_btn = QPushButton(f"\u25bc  {category_name}")
        header_btn.setCheckable(True)
        header_btn.setChecked(True)
        header_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        header_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: bold;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {color}cc;
            }}
            QPushButton:checked {{
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }}
        """)
        header_layout.addWidget(header_btn, 1)

        if add_new_callback:
            add_btn = QPushButton("+")
            add_btn.setFixedWidth(30)
            add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            add_btn.setToolTip(f"Add new {category_name.lower()[:-1]}")
            add_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-top-right-radius: 4px;
                    border-bottom-right-radius: 4px;
                    padding: 8px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color}cc;
                }}
            """)
            add_btn.clicked.connect(add_new_callback)
            header_layout.addWidget(add_btn)

        category_layout.addWidget(header_widget)

        nodes_container.setStyleSheet(f"""
            QWidget {{
                background-color: {color}40;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
        """)
        nodes_layout.setContentsMargins(4, 4, 4, 4)

        placeholder = QLabel("No items defined")
        placeholder.setStyleSheet("color: #888; padding: 8px;")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nodes_layout.addWidget(placeholder)

        category_layout.addWidget(nodes_container)

        def make_toggle(btn, container):
            def toggle(checked):
                container.setVisible(checked)
                btn.setText(f"\u25bc  {btn.text()[3:]}" if checked else f"\u25b6  {btn.text()[3:]}")

            return toggle

        header_btn.toggled.connect(make_toggle(header_btn, nodes_container))

        parent_layout.addWidget(category_widget)

    def _on_new_custom_device(self) -> None:
        """Open dialog to create a new custom device."""
        try:
            from glider.gui.dialogs.custom_device_dialog import CustomDeviceDialog

            dialog = CustomDeviceDialog(parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                definition = dialog.get_definition()
                session = self._session
                if session:
                    session.add_custom_device_definition(definition.to_dict())
                    self.refresh_custom_devices()
                    logger.info(f"Created custom device: {definition.name}")
        except ImportError as e:
            logger.warning(f"Could not import CustomDeviceDialog: {e}")
            QMessageBox.warning(self, "Not Available", "Custom device editor not available.")

    def _on_new_flow_function(self) -> None:
        """Open dialog to create a new flow function."""
        try:
            from glider.core.flow_engine import FlowEngine
            from glider.gui.dialogs.flow_function_dialog import FlowFunctionDialog

            available_types = FlowEngine.get_available_nodes()
            available_types.extend(["FlowFunctionEntry", "FlowFunctionExit", "Parameter"])

            dialog = FlowFunctionDialog(available_node_types=available_types, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                definition = dialog.get_definition()
                session = self._session
                if session:
                    session.add_flow_function_definition(definition.to_dict())
                    self.refresh_flow_functions()
                    logger.info(f"Created flow function: {definition.name}")
        except ImportError as e:
            logger.warning(f"Could not import FlowFunctionDialog: {e}")
            QMessageBox.warning(self, "Not Available", "Flow function editor not available.")

    def _add_zone_node(self, zone_id: str) -> None:
        """Add a zone input node to the graph."""
        center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
        self._graph_view.node_created.emit(f"ZoneInput:{zone_id}", center.x(), center.y())

    def _detect_graph_functions(self) -> list:
        """Detect complete function definitions in the graph."""
        session = self._session
        if not session:
            return []

        functions = []
        flow = session.flow

        start_nodes = [n for n in flow.nodes if n.node_type == "StartFunction"]

        for start_node in start_nodes:
            func_name = "MyFunction"
            if start_node.state:
                func_name = start_node.state.get("function_name", "MyFunction")

            if self._trace_to_end_function(start_node.id, flow):
                functions.append(
                    {
                        "name": func_name,
                        "start_node_id": start_node.id,
                    }
                )

        return functions

    def _trace_to_end_function(self, start_id: str, flow) -> bool:
        """Trace from a node to see if it eventually reaches an EndFunction."""
        visited = set()
        to_visit = [start_id]

        while to_visit:
            current_id = to_visit.pop()
            if current_id in visited:
                continue
            visited.add(current_id)

            for node in flow.nodes:
                if node.id == current_id and node.node_type == "EndFunction":
                    return True

            for conn in flow.connections:
                if conn.from_node == current_id:
                    to_visit.append(conn.to_node)

        return False

    def _add_function_call_node(self, start_node_id: str, func_name: str) -> None:
        """Add a FunctionCall node to the graph."""
        center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
        self._graph_view.node_created.emit(f"FunctionCall:{start_node_id}", center.x(), center.y())

    def _add_custom_device_node(self, definition_id: str) -> None:
        """Add a custom device action node to the graph."""
        center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
        self._graph_view.node_created.emit(f"CustomDevice:{definition_id}", center.x(), center.y())

    def _add_flow_function_node(self, definition_id: str) -> None:
        """Add a flow function node to the graph."""
        center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
        self._graph_view.node_created.emit(f"FlowFunction:{definition_id}", center.x(), center.y())

    def _edit_custom_device(self, definition_id: str) -> None:
        """Edit an existing custom device definition."""
        session = self._session
        if not session:
            return

        def_dict = session.get_custom_device_definition(definition_id)
        if not def_dict:
            QMessageBox.warning(self, "Error", "Custom device not found.")
            return

        try:
            from glider.core.custom_device import CustomDeviceDefinition
            from glider.gui.dialogs.custom_device_dialog import CustomDeviceDialog

            definition = CustomDeviceDefinition.from_dict(def_dict)
            dialog = CustomDeviceDialog(definition=definition, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                updated_def = dialog.get_definition()
                session.remove_custom_device_definition(definition_id)
                session.add_custom_device_definition(updated_def.to_dict())
                self.refresh_custom_devices()
                logger.info(f"Updated custom device: {updated_def.name}")
        except ImportError as e:
            logger.warning(f"Could not import CustomDeviceDialog: {e}")
            QMessageBox.warning(self, "Not Available", "Custom device editor not available.")

    def _delete_custom_device(self, definition_id: str) -> None:
        """Delete a custom device definition."""
        session = self._session
        if not session:
            return

        def_dict = session.get_custom_device_definition(definition_id)
        name = def_dict.get("name", "Unknown") if def_dict else "Unknown"

        result = QMessageBox.question(
            self,
            "Delete Custom Device",
            f"Are you sure you want to delete '{name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if result == QMessageBox.StandardButton.Yes:
            session.remove_custom_device_definition(definition_id)
            self.refresh_custom_devices()
            logger.info(f"Deleted custom device: {name}")

    def _edit_flow_function(self, definition_id: str) -> None:
        """Edit an existing flow function definition."""
        session = self._session
        if not session:
            return

        def_dict = session.get_flow_function_definition(definition_id)
        if not def_dict:
            QMessageBox.warning(self, "Error", "Flow function not found.")
            return

        try:
            from glider.core.flow_engine import FlowEngine
            from glider.core.flow_function import FlowFunctionDefinition
            from glider.gui.dialogs.flow_function_dialog import FlowFunctionDialog

            definition = FlowFunctionDefinition.from_dict(def_dict)
            available_types = FlowEngine.get_available_nodes()
            available_types.extend(["FlowFunctionEntry", "FlowFunctionExit", "Parameter"])

            dialog = FlowFunctionDialog(
                definition=definition, available_node_types=available_types, parent=self
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                updated_def = dialog.get_definition()
                session.remove_flow_function_definition(definition_id)
                session.add_flow_function_definition(updated_def.to_dict())
                self.refresh_flow_functions()
                logger.info(f"Updated flow function: {updated_def.name}")
        except ImportError as e:
            logger.warning(f"Could not import FlowFunctionDialog: {e}")
            QMessageBox.warning(self, "Not Available", "Flow function editor not available.")

    def _delete_flow_function(self, definition_id: str) -> None:
        """Delete a flow function definition."""
        session = self._session
        if not session:
            return

        def_dict = session.get_flow_function_definition(definition_id)
        name = def_dict.get("name", "Unknown") if def_dict else "Unknown"

        result = QMessageBox.question(
            self,
            "Delete Flow Function",
            f"Are you sure you want to delete '{name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if result == QMessageBox.StandardButton.Yes:
            session.remove_flow_function_definition(definition_id)
            self.refresh_flow_functions()
            logger.info(f"Deleted flow function: {name}")

    def _add_node_to_center(self, node_type: str) -> None:
        """Add a node to the center of the graph view."""
        center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
        self._graph_view.node_created.emit(node_type, center.x(), center.y())
