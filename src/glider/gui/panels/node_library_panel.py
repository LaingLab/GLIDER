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
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import colors

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
                ("Timer", "Timer", "Emit ticks at a regular interval"),
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
            "Behavior": [
                (
                    "BehaviorInput",
                    "Behavior Input",
                    "Trigger on live behavior classification - fires On Enter / On "
                    "Exit as the animal starts and stops a behavior",
                ),
            ],
            "Audio": [
                ("AudioPlayback", "Audio Playback", "Play an audio file (WAV/MP3)"),
            ],
            "Video": [
                ("VideoPlayback", "Video Playback", "Play a video file (MP4/AVI)"),
            ],
        }

        category_colors = {
            "Flow": colors.LIB_FLOW,
            "Functions": colors.LIB_FUNCTIONS,
            "Control": colors.LIB_CONTROL,
            "I/O": colors.LIB_IO,
            "Behavior": colors.LIB_BEHAVIOR,
            "Audio": colors.LIB_AUDIO,
            "Video": colors.LIB_VIDEO,
            "default": colors.BORDER,
        }

        for category, nodes in node_categories.items():
            color = category_colors.get(category, category_colors["default"])

            category_widget = QWidget()
            category_layout = QVBoxLayout(category_widget)
            category_layout.setContentsMargins(0, 0, 0, 8)
            category_layout.setSpacing(2)

            header_label = QLabel(category.upper())
            header_label.setStyleSheet(f"""
                padding: 6px 8px;
                font-size: 11px;
                font-weight: 600;
                color: {colors.TEXT_TERTIARY};
                letter-spacing: 0.5px;
            """)
            category_layout.addWidget(header_label)

            for node_type, node_name, tooltip in nodes:
                node_btn = DraggableNodeButton(node_type, node_name, category)
                node_btn.setToolTip(tooltip)
                node_btn.clicked.connect(lambda checked, nt=node_type: self._add_node_to_center(nt))
                self._apply_node_btn_style(node_btn, color)
                category_layout.addWidget(node_btn)

            layout.addWidget(category_widget)

        # Graph Functions section: call-buttons auto-detected from
        # StartFunction -> EndFunction chains in the graph (no editor dialog).
        self._flow_functions_container = QWidget()
        self._flow_functions_layout = QVBoxLayout(self._flow_functions_container)
        self._flow_functions_layout.setContentsMargins(0, 0, 0, 0)
        self._flow_functions_layout.setSpacing(2)
        self._setup_custom_category(
            self._flow_functions_container,
            self._flow_functions_layout,
            "Graph Functions",
            colors.LIB_FUNCTIONS,
            layout,
            add_new_callback=None,
        )

        # Plugin nodes: whatever plugins registered, discovered rather than
        # listed. A node type core has never heard of has no other way onto the
        # canvas -- the static categories above and the canvas context menu are
        # both hardcoded.
        self._plugin_nodes_container = QWidget()
        self._plugin_nodes_layout = QVBoxLayout(self._plugin_nodes_container)
        self._plugin_nodes_layout.setContentsMargins(0, 0, 0, 0)
        self._plugin_nodes_layout.setSpacing(2)
        self._setup_custom_category(
            self._plugin_nodes_container,
            self._plugin_nodes_layout,
            "Plugins",
            colors.LIB_PLUGINS,
            layout,
            add_new_callback=None,
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
            colors.LIB_ZONES,
            layout,
            add_new_callback=None,
        )

        self.refresh_plugin_nodes()

        layout.addStretch()
        scroll_area.setWidget(container)

        self._node_library_container = container
        self._node_library_layout = layout

        outer_layout.addWidget(scroll_area)

    # --- Public API ---

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
                self._apply_node_btn_style(btn, colors.LIB_FUNCTIONS)
                self._flow_functions_layout.addWidget(btn)

        if not has_functions:
            placeholder = QLabel("Define functions with StartFunction \u2192 EndFunction")
            placeholder.setProperty("textRole", "muted")
            placeholder.setWordWrap(True)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._flow_functions_layout.addWidget(placeholder)

    def refresh_plugin_nodes(self) -> None:
        """Rebuild the Plugins section from whatever nodes plugins registered.

        Called again after plugins load or reload, since a plugin installed
        from the Plugins window takes effect without a restart.
        """
        while self._plugin_nodes_layout.count():
            item = self._plugin_nodes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        from glider.core.flow_engine import FlowEngine
        from glider.plugins.plugin_manager import plugin_components

        entries = plugin_components("node")
        for node_type in sorted(entries):
            node_class = FlowEngine.get_node_class(node_type)
            definition = getattr(node_class, "definition", None)
            label = getattr(definition, "name", None) or node_type
            description = getattr(definition, "description", "") or ""
            btn = DraggableNodeButton(node_type, label, "Plugins")
            btn.setToolTip(f"{description}\n\nFrom plugin: {entries[node_type]}".strip())
            btn.clicked.connect(lambda checked, nt=node_type: self._add_node_to_center(nt))
            self._apply_node_btn_style(btn, colors.LIB_PLUGINS)
            self._plugin_nodes_layout.addWidget(btn)

        if not entries:
            placeholder = QLabel("No plugin nodes installed")
            placeholder.setProperty("textRole", "muted")
            placeholder.setWordWrap(True)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plugin_nodes_layout.addWidget(placeholder)

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
                self._apply_node_btn_style(btn, colors.LIB_ZONES)
                self._zones_layout.addWidget(btn)
        else:
            placeholder = QLabel("Create zones in Camera \u2192 Zones...")
            placeholder.setProperty("textRole", "muted")
            placeholder.setWordWrap(True)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._zones_layout.addWidget(placeholder)

    # --- Internal methods ---

    def _add_placeholder(self, layout, text: str):
        placeholder = QLabel(text)
        placeholder.setProperty("textRole", "muted")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

    def _apply_node_btn_style(self, btn: QPushButton, color: str) -> None:
        """Apply the standard left-border node button style."""
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors.with_alpha(color, 0.1)};
                border: none;
                border-left: 3px solid {color};
                border-radius: 0;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                padding: 7px 10px;
                font-size: 13px;
                color: {colors.TEXT_SECONDARY};
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {colors.with_alpha(color, 0.2)};
                color: {colors.TEXT_PRIMARY};
            }}
        """)

    def _setup_custom_category(
        self,
        nodes_container: QWidget,
        nodes_layout: QVBoxLayout,
        category_name: str,
        color: str,
        parent_layout: QVBoxLayout,
        add_new_callback=None,
    ) -> None:
        """Setup a custom category with a section label and optional add button."""
        category_widget = QWidget()
        category_layout = QVBoxLayout(category_widget)
        category_layout.setContentsMargins(0, 0, 0, 8)
        category_layout.setSpacing(2)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        header_label = QLabel(category_name.upper())
        header_label.setStyleSheet(f"""
            padding: 6px 8px;
            font-size: 11px;
            font-weight: 600;
            color: {colors.TEXT_TERTIARY};
            letter-spacing: 0.5px;
        """)
        header_layout.addWidget(header_label, 1)

        if add_new_callback:
            add_btn = QPushButton("+")
            add_btn.setFixedWidth(24)
            add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            add_btn.setToolTip(f"Add new {category_name.lower()[:-1]}")
            add_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {colors.with_alpha(color, 0.2)};
                    color: {colors.TEXT_TERTIARY};
                    border: none;
                    border-radius: 4px;
                    padding: 2px;
                    font-size: 14px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {colors.with_alpha(color, 0.4)};
                    color: {colors.TEXT_PRIMARY};
                }}
            """)
            add_btn.clicked.connect(add_new_callback)
            header_layout.addWidget(add_btn)

        category_layout.addWidget(header_widget)

        nodes_container.setStyleSheet("")
        nodes_layout.setContentsMargins(0, 0, 0, 0)

        placeholder = QLabel("No items defined")
        placeholder.setProperty("textRole", "muted")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nodes_layout.addWidget(placeholder)

        category_layout.addWidget(nodes_container)

        parent_layout.addWidget(category_widget)

    def _add_zone_node(self, zone_id: str) -> None:
        """Add a zone input node to the graph."""
        center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
        self._graph_view.node_created.emit(f"ZoneInput:{zone_id}", center.x(), center.y())

    def _detect_graph_functions(self) -> list:
        """Detect complete function definitions in the graph."""
        from glider.core.graph_functions import list_graph_functions

        session = self._session
        if not session:
            return []
        return [
            {"name": info.name, "start_node_id": info.start_node_id}
            for info in list_graph_functions(session)
            if info.has_end
        ]

    def _add_function_call_node(self, start_node_id: str, func_name: str) -> None:
        """Add a FunctionCall node to the graph."""
        center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
        self._graph_view.node_created.emit(f"FunctionCall:{start_node_id}", center.x(), center.y())

    def _add_node_to_center(self, node_type: str) -> None:
        """Add a node to the center of the graph view."""
        center = self._graph_view.mapToScene(self._graph_view.viewport().rect().center())
        self._graph_view.node_created.emit(node_type, center.x(), center.y())
