"""
Node Editor Controller - Controller for node graph callbacks and properties panel.

Manages node creation/deletion/selection/movement, connection creation/deletion,
properties panel updates, and undo/redo command integration.
"""

import logging
import uuid
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from glider.gui.commands import (
    Command,
    CreateConnectionCommand,
    CreateNodeCommand,
    DeleteConnectionCommand,
    DeleteNodeCommand,
    MoveNodeCommand,
    PropertyChangeCommand,
    UndoStack,
)

if TYPE_CHECKING:
    from glider.core.hardware_manager import HardwareManager
    from glider.gui.node_graph.graph_view import NodeGraphView
    from glider.vision.zones import ZoneConfiguration

logger = logging.getLogger(__name__)


class NodeEditorController(QObject):
    """Controller for node graph callbacks and properties panel."""

    flow_functions_changed = pyqtSignal()
    undo_redo_changed = pyqtSignal()
    status_message = pyqtSignal(str, int)  # message, timeout_ms

    def __init__(
        self,
        graph_view: "NodeGraphView",
        session_fn,
        hardware_manager: "HardwareManager",
        undo_stack: UndoStack,
        core,
        parent=None,
    ):
        """
        Args:
            graph_view: NodeGraphView instance
            session_fn: Callable that returns the current ExperimentSession (or None)
            hardware_manager: HardwareManager instance
            undo_stack: UndoStack instance
            core: GliderCore instance (for flow_engine binding)
        """
        super().__init__(parent)
        self._graph_view = graph_view
        self._session_fn = session_fn
        self._hardware_manager = hardware_manager
        self._undo_stack = undo_stack
        self._core = core

        # Zone configuration
        self._zone_config = None

        # Properties dock reference (set by MainWindow)
        self._properties_dock = None

    @property
    def _session(self):
        return self._session_fn()

    # --- Public API ---

    def connect_graph_signals(self) -> None:
        """Connect graph view signals to this controller."""
        self._graph_view.node_created.connect(self._on_node_created)
        self._graph_view.node_deleted.connect(self._on_node_deleted)
        self._graph_view.node_selected.connect(self._on_node_selected)
        self._graph_view.node_moved.connect(self._on_node_moved)
        self._graph_view.connection_created.connect(self._on_connection_created)
        self._graph_view.connection_deleted.connect(self._on_connection_deleted)

    def set_properties_dock(self, properties_dock) -> None:
        """Set the properties dock widget reference."""
        self._properties_dock = properties_dock

    def set_zone_configuration(self, config: "ZoneConfiguration") -> None:
        """Set the zone configuration."""
        self._zone_config = config

    def setup_node_ports(self, node_item, node_type: str) -> None:
        """Set up input/output ports for a node based on its type."""
        from glider.gui.node_graph.port_item import PortType

        nt = node_type.replace(" ", "")

        port_configs = {
            "StartExperiment": ([], [">next"]),
            "EndExperiment": ([">exec"], []),
            "Delay": ([">exec"], [">next"]),
            "Loop": ([">exec"], [">body", ">done"]),
            "WaitForInput": ([">exec"], [">triggered"]),
            "Output": ([">exec"], [">next"]),
            "Input": ([">exec"], ["value", ">next"]),
            "MotorGovernor": ([">exec"], [">next"]),
            "StartFunction": ([], [">next"]),
            "EndFunction": ([">exec"], []),
            "FunctionCall": ([">exec"], [">next"]),
            "ZoneInput": ([], ["Occupied", "Object Count", ">On Enter", ">On Exit"]),
        }

        inputs, outputs = port_configs.get(nt, ([">in"], [">out"]))

        for port_name in inputs:
            if port_name.startswith(">"):
                node_item.add_input_port(port_name[1:], PortType.EXEC)
            else:
                node_item.add_input_port(port_name, PortType.DATA)

        for port_name in outputs:
            if port_name.startswith(">"):
                node_item.add_output_port(port_name[1:], PortType.EXEC)
            else:
                node_item.add_output_port(port_name, PortType.DATA)

    def redo_command(self, command: Command) -> None:
        """Re-apply a command for redo."""
        if isinstance(command, CreateNodeCommand):
            self._on_node_created(command._node_type, command._x, command._y)
            if self._undo_stack._undo_stack:
                self._undo_stack._undo_stack.pop()
        elif isinstance(command, DeleteNodeCommand):
            node_id = command._node_id
            self._graph_view.remove_node(node_id)
            session = self._session
            if session:
                session.remove_node(node_id)
        elif isinstance(command, MoveNodeCommand):
            node_item = self._graph_view.nodes.get(command._node_id)
            if node_item:
                node_item.setPos(command._new_x, command._new_y)
            session = self._session
            if session:
                session.update_node_position(command._node_id, command._new_x, command._new_y)
        elif isinstance(command, CreateConnectionCommand):
            self._graph_view.add_connection(
                command._conn_id,
                command._from_node,
                command._from_port,
                command._to_node,
                command._to_port,
            )
            session = self._session
            if session:
                from glider.core.experiment_session import ConnectionConfig

                conn_config = ConnectionConfig(
                    id=command._conn_id,
                    from_node=command._from_node,
                    from_output=command._from_port,
                    to_node=command._to_node,
                    to_input=command._to_port,
                    connection_type=command._conn_type,
                )
                session.add_connection(conn_config)
        elif isinstance(command, DeleteConnectionCommand):
            self._graph_view.remove_connection(command._conn_id)
            session = self._session
            if session:
                session.remove_connection(command._conn_id)
        elif isinstance(command, PropertyChangeCommand):
            session = self._session
            if session:
                session.update_node_state(
                    command._node_id, {command._prop_name: command._new_value}
                )

    # --- Node graph event handlers ---

    def _on_node_created(self, node_type: str, x: float, y: float) -> None:
        """Handle node creation from graph view."""
        from glider.core.experiment_session import NodeConfig

        display_name = node_type
        actual_node_type = node_type
        definition_id = None
        initial_state = {}

        session = self._session

        if node_type.startswith("FunctionCall:"):
            start_node_id = node_type.split(":", 1)[1]
            actual_node_type = "FunctionCall"
            if session:
                start_node = session.get_node(start_node_id)
                if start_node and start_node.state:
                    display_name = start_node.state.get("function_name", "Function")
                else:
                    display_name = "Function"
                initial_state["function_start_id"] = start_node_id
                initial_state["function_name"] = display_name
        elif node_type.startswith("ZoneInput:"):
            zone_id = node_type.split(":", 1)[1]
            actual_node_type = "ZoneInput"
            if self._zone_config:
                for zone in self._zone_config.zones:
                    if zone.id == zone_id:
                        display_name = f"Zone: {zone.name}"
                        initial_state["zone_id"] = zone_id
                        initial_state["zone_name"] = zone.name
                        break
                else:
                    display_name = "Zone Input"
                    initial_state["zone_id"] = zone_id
            else:
                display_name = "Zone Input"
                initial_state["zone_id"] = zone_id

        node_type_normalized = actual_node_type.replace(" ", "")

        node_id = f"{actual_node_type.lower()}_{uuid.uuid4().hex[:8]}"

        category = "default"
        flow_nodes = ["StartExperiment", "EndExperiment", "Delay"]
        control_nodes = ["Loop", "WaitForInput"]
        io_nodes = ["Output", "Input", "MotorGovernor"]
        function_nodes = ["FunctionCall", "StartFunction", "EndFunction"]
        interface_nodes = ["ZoneInput"]

        if node_type_normalized in flow_nodes:
            category = "logic"
        elif node_type_normalized in control_nodes:
            category = "interface"
        elif node_type_normalized in io_nodes:
            category = "hardware"
        elif node_type_normalized in function_nodes:
            category = "logic"
        elif node_type_normalized in interface_nodes:
            category = "interface"

        node_item = self._graph_view.add_node(node_id, display_name, x, y)
        node_item._category = category
        node_item._header_color = node_item.CATEGORY_COLORS.get(
            category, node_item.CATEGORY_COLORS["default"]
        )
        node_item._actual_node_type = actual_node_type
        node_item._definition_id = definition_id

        self.setup_node_ports(node_item, actual_node_type)

        self._graph_view._connect_port_signals(node_item)

        if session:
            node_config = NodeConfig(
                id=node_id,
                node_type=actual_node_type,
                position=(x, y),
                state=initial_state,
                device_id=None,
                visible_in_runner=category == "interface",
            )
            session.add_node(node_config)

        command = CreateNodeCommand(self, node_id, actual_node_type, x, y)
        self._undo_stack.push(command)
        self.undo_redo_changed.emit()

        self.status_message.emit(f"Created node: {display_name}", 2000)

    def _on_node_deleted(self, node_id: str) -> None:
        """Handle node deletion from graph view."""
        node_data = {}
        node_item = self._graph_view.nodes.get(node_id)
        if node_item:
            node_data = {
                "id": node_id,
                "node_type": node_item.node_type,
                "x": node_item.pos().x(),
                "y": node_item.pos().y(),
            }

        session = self._session
        if session:
            node_config = session.get_node(node_id)
            if node_config:
                node_data["state"] = node_config.state
                node_data["device_id"] = node_config.device_id
                node_data["visible_in_runner"] = node_config.visible_in_runner

            session.remove_node(node_id)

        command = DeleteNodeCommand(self, node_id, node_data)
        self._undo_stack.push(command)
        self.undo_redo_changed.emit()

        self.status_message.emit(f"Deleted node: {node_id}", 2000)

    def _on_node_selected(self, node_id: str) -> None:
        """Handle node selection from graph view."""
        self._update_properties_panel(node_id)
        self.status_message.emit(f"Selected: {node_id}", 1000)

    def _on_node_moved(self, node_id: str, x: float, y: float) -> None:
        """Handle node movement from graph view."""
        session = self._session
        if session:
            session.update_node_position(node_id, x, y)

    def _on_connection_created(
        self, from_node: str, from_port: int, to_node: str, to_port: int, conn_type: str = "data"
    ) -> None:
        """Handle connection creation from graph view."""
        from glider.core.experiment_session import ConnectionConfig

        connection_id = f"conn_{uuid.uuid4().hex[:8]}"

        self._graph_view.add_connection(connection_id, from_node, from_port, to_node, to_port)

        session = self._session
        if session:
            conn_config = ConnectionConfig(
                id=connection_id,
                from_node=from_node,
                from_output=from_port,
                to_node=to_node,
                to_input=to_port,
                connection_type=conn_type,
            )
            session.add_connection(conn_config)
            logger.info(
                f"Saved connection: {from_node}:{from_port} -> {to_node}:{to_port} (type: {conn_type})"
            )

        command = CreateConnectionCommand(
            self, connection_id, from_node, from_port, to_node, to_port, conn_type
        )
        self._undo_stack.push(command)
        self.undo_redo_changed.emit()

        self.status_message.emit(f"Connected: {from_node} -> {to_node}", 2000)

        self.flow_functions_changed.emit()

    def _on_connection_deleted(self, connection_id: str) -> None:
        """Handle connection deletion from graph view."""
        conn_data = {"id": connection_id}

        session = self._session
        if session:
            conn_config = session.get_connection(connection_id)
            if conn_config:
                conn_data["from_node"] = conn_config.from_node
                conn_data["from_port"] = conn_config.from_output
                conn_data["to_node"] = conn_config.to_node
                conn_data["to_port"] = conn_config.to_input
                conn_data["conn_type"] = conn_config.connection_type

            session.remove_connection(connection_id)

        command = DeleteConnectionCommand(self, connection_id, conn_data)
        self._undo_stack.push(command)
        self.undo_redo_changed.emit()

        self.status_message.emit(f"Deleted connection: {connection_id}", 2000)

        self.flow_functions_changed.emit()

    @staticmethod
    def _add_section_header(layout: QFormLayout, text: str) -> None:
        """Add an uppercase section header label spanning the full row."""
        header = QLabel(text)
        header.setProperty("class", "props-section-header")
        layout.addRow(header)

    @staticmethod
    def _add_divider(layout: QFormLayout) -> None:
        """Add a horizontal divider line spanning the full row."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setProperty("class", "props-divider")
        layout.addRow(line)

    def _update_properties_panel(self, node_id: str) -> None:
        """Update the properties panel for the selected node."""
        if self._properties_dock is None:
            return

        node_item = self._graph_view.nodes.get(node_id)
        if node_item is None:
            return

        session = self._session
        node_config = None
        if session:
            node_config = session.get_node(node_id)

        props_widget = QWidget()
        props_layout = QFormLayout(props_widget)
        props_layout.setContentsMargins(12, 12, 12, 12)
        props_layout.setVerticalSpacing(6)

        # -- Node Info section --
        self._add_section_header(props_layout, "NODE INFO")
        props_layout.addRow("ID:", QLabel(node_id))
        props_layout.addRow("Type:", QLabel(node_item.node_type))

        if hasattr(node_item, "_actual_node_type") and node_item._actual_node_type:
            node_type = node_item._actual_node_type.replace(" ", "")
        else:
            node_type = node_item.node_type.replace(" ", "")

        self._add_divider(props_layout)

        # Add device selector for I/O nodes
        if node_type in ["Output", "Input", "WaitForInput", "MotorGovernor"]:
            self._add_section_header(props_layout, "DEVICE")
            device_combo = QComboBox()
            device_combo.addItem("-- Select Device --", None)
            current_device_id = node_config.device_id if node_config else None
            current_index = 0

            for i, (dev_id, device) in enumerate(self._hardware_manager.devices.items()):
                device_name = getattr(device, "name", dev_id)
                device_type = getattr(device, "device_type", "")
                device_combo.addItem(f"{device_name} ({device_type})", dev_id)
                if dev_id == current_device_id:
                    current_index = i + 1

            device_combo.setCurrentIndex(current_index)
            device_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=device_combo: self._on_node_device_changed(
                    nid, combo.currentData()
                )
            )
            props_layout.addRow("Device:", device_combo)

        elif node_type == "Delay":
            self._add_section_header(props_layout, "CONFIGURATION")

            saved_duration = 1.0
            saved_unit = "seconds"
            if node_config and node_config.state:
                saved_duration = float(node_config.state.get("duration", 1.0))
                saved_unit = node_config.state.get("unit", "seconds")

            duration_spin = QDoubleSpinBox()
            duration_spin.setDecimals(3)
            duration_spin.setRange(0.0, 3_600_000.0)
            duration_spin.setValue(saved_duration)
            duration_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "duration", val)
            )

            unit_combo = QComboBox()
            unit_combo.addItem("sec", "seconds")
            unit_combo.addItem("ms", "milliseconds")
            unit_combo.setCurrentIndex(0 if saved_unit == "seconds" else 1)
            unit_combo.currentIndexChanged.connect(
                lambda _idx, nid=node_id, c=unit_combo: self._on_node_property_changed(
                    nid, "unit", c.currentData()
                )
            )

            props_layout.addRow("Duration:", duration_spin)
            props_layout.addRow("Unit:", unit_combo)

        elif node_type == "Timer":
            self._add_section_header(props_layout, "CONFIGURATION")

            saved_interval = 1.0
            saved_unit = "seconds"
            if node_config and node_config.state:
                saved_interval = float(node_config.state.get("interval", 1.0))
                saved_unit = node_config.state.get("unit", "seconds")

            interval_spin = QDoubleSpinBox()
            interval_spin.setDecimals(3)
            interval_spin.setRange(0.0, 3_600_000.0)
            interval_spin.setValue(saved_interval)
            interval_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "interval", val)
            )

            unit_combo = QComboBox()
            unit_combo.addItem("sec", "seconds")
            unit_combo.addItem("ms", "milliseconds")
            unit_combo.setCurrentIndex(0 if saved_unit == "seconds" else 1)
            unit_combo.currentIndexChanged.connect(
                lambda _idx, nid=node_id, c=unit_combo: self._on_node_property_changed(
                    nid, "unit", c.currentData()
                )
            )

            props_layout.addRow("Interval:", interval_spin)
            props_layout.addRow("Unit:", unit_combo)

        elif node_type == "StartFunction":
            self._add_section_header(props_layout, "FUNCTION")
            name_edit = QLineEdit()
            name_edit.setPlaceholderText("Enter function name")
            saved_name = "MyFunction"
            if node_config and node_config.state:
                saved_name = node_config.state.get("function_name", "MyFunction")
            name_edit.setText(saved_name)
            name_edit.textChanged.connect(
                lambda text, nid=node_id: self._on_node_property_changed(nid, "function_name", text)
            )
            props_layout.addRow("Function Name:", name_edit)

            info_label = QLabel("Connect to EndFunction to define a reusable function.")
            info_label.setWordWrap(True)
            info_label.setProperty("textRole", "muted")
            props_layout.addRow(info_label)

        # Value control for Output node
        if node_type == "Output":
            self._add_divider(props_layout)
            self._add_section_header(props_layout, "VALUE")
            bound_device_type = None
            if node_config and node_config.device_id:
                bound_device = self._hardware_manager.get_device(node_config.device_id)
                if bound_device:
                    bound_device_type = getattr(bound_device, "device_type", None)

            if bound_device_type == "PWMOutput":
                pwm_spin = QSpinBox()
                pwm_spin.setRange(0, 255)
                saved_value = 0
                if node_config and node_config.state:
                    saved_value = node_config.state.get("value", 0)
                pwm_spin.setValue(int(saved_value))
                pwm_spin.valueChanged.connect(
                    lambda val, nid=node_id: self._on_node_property_changed(nid, "value", val)
                )
                props_layout.addRow("PWM Value (0-255):", pwm_spin)
            else:
                from PyQt6.QtWidgets import QRadioButton

                value_layout = QHBoxLayout()
                high_radio = QRadioButton("HIGH")
                low_radio = QRadioButton("LOW")

                saved_value = 1
                if node_config and node_config.state:
                    saved_value = node_config.state.get("value", 1)

                if saved_value:
                    high_radio.setChecked(True)
                else:
                    low_radio.setChecked(True)

                value_layout.addWidget(high_radio)
                value_layout.addWidget(low_radio)
                high_radio.toggled.connect(
                    lambda checked, nid=node_id: self._on_node_property_changed(
                        nid, "value", 1 if checked else 0
                    )
                )
                value_widget = QWidget()
                value_widget.setLayout(value_layout)
                props_layout.addRow("Value:", value_widget)

        elif node_type == "MotorGovernor":
            self._add_section_header(props_layout, "ACTION")
            action_combo = QComboBox()
            action_combo.addItem("Move Up", "up")
            action_combo.addItem("Move Down", "down")
            action_combo.addItem("Stop", "stop")

            saved_action = "stop"
            if node_config and node_config.state:
                saved_action = node_config.state.get("action", "stop")

            action_map = {"up": 0, "down": 1, "stop": 2}
            action_combo.setCurrentIndex(action_map.get(saved_action, 2))

            action_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=action_combo: self._on_node_property_changed(
                    nid, "action", combo.currentData()
                )
            )
            props_layout.addRow("Action:", action_combo)

        elif node_type == "Loop":
            self._add_section_header(props_layout, "LOOP SETTINGS")
            count_spin = QSpinBox()
            count_spin.setRange(0, 10000)
            count_spin.setSpecialValueText("Infinite")
            saved_count = 0
            if node_config and node_config.state:
                saved_count = node_config.state.get("count", 0)
            count_spin.setValue(saved_count)
            count_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "count", val)
            )
            props_layout.addRow("Iterations:", count_spin)

            delay_spin = QDoubleSpinBox()
            delay_spin.setRange(0.0, 3600.0)
            delay_spin.setDecimals(2)
            delay_spin.setSingleStep(0.1)
            saved_delay = 1.0
            if node_config and node_config.state:
                saved_delay = node_config.state.get("delay", 1.0)
            delay_spin.setValue(saved_delay)
            delay_spin.setSuffix(" sec")
            delay_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "delay", val)
            )
            props_layout.addRow("Delay:", delay_spin)

        if node_type == "WaitForInput":
            self._add_divider(props_layout)
            self._add_section_header(props_layout, "INPUT SETTINGS")
            mode_combo = QComboBox()
            mode_combo.addItem("Digital (Rising Edge)", "digital")
            mode_combo.addItem("Analog (Threshold)", "analog")
            mode_combo.addItem("Revolution (Turns)", "revolution")

            saved_mode = "digital"
            if node_config and node_config.state:
                saved_mode = node_config.state.get("threshold_mode", "digital")

            mode_idx = mode_combo.findData(saved_mode)
            mode_combo.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
            mode_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=mode_combo: self._on_node_property_changed(
                    nid, "threshold_mode", combo.currentData()
                )
            )
            props_layout.addRow("Mode:", mode_combo)

            threshold_spin = QSpinBox()
            threshold_spin.setRange(0, 65535)
            saved_threshold = 512
            if node_config and node_config.state:
                saved_threshold = node_config.state.get("threshold", 512)
            threshold_spin.setValue(saved_threshold)
            threshold_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "threshold", val)
            )
            props_layout.addRow("Threshold:", threshold_spin)

            direction_combo = QComboBox()
            direction_combo.addItem("Above Threshold", "above")
            direction_combo.addItem("Below Threshold", "below")

            saved_direction = "above"
            if node_config and node_config.state:
                saved_direction = node_config.state.get("threshold_direction", "above")

            direction_combo.setCurrentIndex(0 if saved_direction == "above" else 1)
            direction_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=direction_combo: self._on_node_property_changed(
                    nid, "threshold_direction", combo.currentData()
                )
            )
            props_layout.addRow("Direction:", direction_combo)

            # Revolution mode: how many full turns to wait for. Applies only
            # when Mode is "Revolution (Turns)"; ignored otherwise.
            turns_spin = QSpinBox()
            turns_spin.setRange(1, 100000)
            saved_turns = 1
            if node_config and node_config.state:
                saved_turns = node_config.state.get("turns_target", 1)
            turns_spin.setValue(saved_turns)
            turns_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "turns_target", val)
            )
            props_layout.addRow("Turns:", turns_spin)

            # Revolution mode: sensor full-scale range (e.g. 4096 for the
            # 12-bit AS5600 raw angle). A reading jump larger than half this
            # is treated as one wrap-around / completed turn.
            counts_spin = QSpinBox()
            counts_spin.setRange(2, 65535)
            saved_counts = 4096
            if node_config and node_config.state:
                saved_counts = node_config.state.get("counts_per_turn", 4096)
            counts_spin.setValue(saved_counts)
            counts_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "counts_per_turn", val)
            )
            props_layout.addRow("Counts/Turn:", counts_spin)

            # Revolution mode "ramp down to landing": as the angle nears the
            # wrap, ease a motor's PWM from drive speed down to a creep so it
            # coasts almost nothing and lands on ~0. Applies only in revolution
            # mode with a ramp device selected.
            ramp_check = QCheckBox("Ramp down to landing (revolution)")
            saved_ramp = False
            if node_config and node_config.state:
                saved_ramp = node_config.state.get("ramp_down", False)
            ramp_check.setChecked(saved_ramp)
            ramp_check.toggled.connect(
                lambda checked, nid=node_id: self._on_node_property_changed(
                    nid, "ramp_down", checked
                )
            )
            props_layout.addRow(ramp_check)

            ramp_device_combo = QComboBox()
            ramp_device_combo.addItem("-- Ramp Device (PWM) --", None)
            saved_ramp_device = None
            if node_config and node_config.state:
                saved_ramp_device = node_config.state.get("ramp_device_id")
            ramp_current_index = 0
            ramp_idx = 1
            for dev_id, device in self._hardware_manager.devices.items():
                if getattr(device, "device_type", "") != "PWMOutput":
                    continue
                device_name = getattr(device, "name", dev_id)
                ramp_device_combo.addItem(f"{device_name} (PWMOutput)", dev_id)
                if dev_id == saved_ramp_device:
                    ramp_current_index = ramp_idx
                ramp_idx += 1
            ramp_device_combo.setCurrentIndex(ramp_current_index)
            ramp_device_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=ramp_device_combo: self._on_node_property_changed(
                    nid, "ramp_device_id", combo.currentData()
                )
            )
            props_layout.addRow("Ramp Device:", ramp_device_combo)

            drive_pwm_spin = QSpinBox()
            drive_pwm_spin.setRange(0, 255)
            saved_drive_pwm = 100
            if node_config and node_config.state:
                saved_drive_pwm = node_config.state.get("drive_pwm", 100)
            drive_pwm_spin.setValue(saved_drive_pwm)
            drive_pwm_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "drive_pwm", val)
            )
            props_layout.addRow("Drive PWM:", drive_pwm_spin)

            creep_pwm_spin = QSpinBox()
            creep_pwm_spin.setRange(0, 255)
            saved_creep_pwm = 30
            if node_config and node_config.state:
                saved_creep_pwm = node_config.state.get("creep_pwm", 30)
            creep_pwm_spin.setValue(saved_creep_pwm)
            creep_pwm_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "creep_pwm", val)
            )
            props_layout.addRow("Creep PWM:", creep_pwm_spin)

            ramp_zone_spin = QSpinBox()
            ramp_zone_spin.setRange(1, 65535)
            saved_ramp_zone = 512
            if node_config and node_config.state:
                saved_ramp_zone = node_config.state.get("ramp_zone", 512)
            ramp_zone_spin.setValue(saved_ramp_zone)
            ramp_zone_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "ramp_zone", val)
            )
            props_layout.addRow("Ramp Zone (counts):", ramp_zone_spin)

            # Landing tolerance: stop the moment the angle is within this many
            # counts of 0 (either side of the wrap), instead of needing a clean
            # 4095->0 jump. 0 disables it (pure wrap detection). Keeps the motor
            # moving at landing, avoiding a low-PWM stall short of the target.
            land_tol_spin = QSpinBox()
            land_tol_spin.setRange(0, 65535)
            saved_land_tol = 0
            if node_config and node_config.state:
                saved_land_tol = node_config.state.get("land_tolerance", 0)
            land_tol_spin.setValue(saved_land_tol)
            land_tol_spin.setSpecialValueText("Off (exact wrap)")
            land_tol_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "land_tolerance", val)
            )
            props_layout.addRow("Landing tolerance:", land_tol_spin)

            timeout_spin = QDoubleSpinBox()
            timeout_spin.setRange(0.0, 3600.0)
            timeout_spin.setDecimals(1)
            timeout_spin.setSpecialValueText("No timeout")
            saved_timeout = 0.0
            if node_config and node_config.state:
                saved_timeout = node_config.state.get("timeout", 0.0)
            timeout_spin.setValue(saved_timeout)
            timeout_spin.setSuffix(" sec")
            timeout_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "timeout", val)
            )
            props_layout.addRow("Timeout:", timeout_spin)

            info_label = QLabel(
                "Digital mode: waits for rising edge (LOW \u2192 HIGH)\n"
                "Analog mode: waits for value to cross threshold\n"
                "Revolution mode: counts full turns of a wrap-around "
                "sensor (e.g. AS5600)\n"
                "Ramp down: eases the ramp device's PWM to a creep near the "
                "wrap so the motor lands on ~0"
            )
            info_label.setWordWrap(True)
            info_label.setProperty("textRole", "muted")
            props_layout.addRow(info_label)

        elif node_type == "DigitalRead":
            self._add_section_header(props_layout, "DIGITAL SETTINGS")
            # DigitalReadNode has full continuous-polling support in its start()/
            # _poll_loop() methods, but before this UI was added there was no way
            # to toggle it from the node editor — _continuous stayed False forever.
            continuous_check = QCheckBox("Enable continuous reading")
            saved_continuous = False
            if node_config and node_config.state:
                saved_continuous = node_config.state.get("continuous", False)
            continuous_check.setChecked(saved_continuous)
            continuous_check.toggled.connect(
                lambda checked, nid=node_id: self._on_node_property_changed(
                    nid, "continuous", checked
                )
            )
            props_layout.addRow(continuous_check)

            poll_spin = QDoubleSpinBox()
            poll_spin.setRange(0.01, 10.0)
            poll_spin.setDecimals(2)
            poll_spin.setSingleStep(0.05)
            saved_poll = 0.1
            if node_config and node_config.state:
                saved_poll = node_config.state.get("poll_interval", 0.1)
            poll_spin.setValue(saved_poll)
            poll_spin.setSuffix(" sec")
            poll_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "poll_interval", val)
            )
            props_layout.addRow("Poll Interval:", poll_spin)

            info_label = QLabel(
                "Continuous mode: automatically polls the digital input at the poll\n"
                "interval. The 'value' output updates on every poll and the 'exec'\n"
                "output fires downstream nodes after each read."
            )
            info_label.setWordWrap(True)
            info_label.setProperty("textRole", "muted")
            props_layout.addRow(info_label)

        elif node_type == "AnalogRead":
            self._add_section_header(props_layout, "ANALOG SETTINGS")
            continuous_check = QCheckBox("Enable continuous reading")
            saved_continuous = False
            if node_config and node_config.state:
                saved_continuous = node_config.state.get("continuous", False)
            continuous_check.setChecked(saved_continuous)
            continuous_check.toggled.connect(
                lambda checked, nid=node_id: self._on_node_property_changed(
                    nid, "continuous", checked
                )
            )
            props_layout.addRow(continuous_check)

            poll_spin = QDoubleSpinBox()
            poll_spin.setRange(0.01, 10.0)
            poll_spin.setDecimals(2)
            poll_spin.setSingleStep(0.05)
            saved_poll = 0.05
            if node_config and node_config.state:
                saved_poll = node_config.state.get("poll_interval", 0.05)
            poll_spin.setValue(saved_poll)
            poll_spin.setSuffix(" sec")
            poll_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "poll_interval", val)
            )
            props_layout.addRow("Poll Interval:", poll_spin)

            threshold_check = QCheckBox("Enable threshold checking")
            saved_threshold_enabled = False
            if node_config and node_config.state:
                saved_threshold_enabled = node_config.state.get("threshold_enabled", False)
            threshold_check.setChecked(saved_threshold_enabled)
            threshold_check.toggled.connect(
                lambda checked, nid=node_id: self._on_node_property_changed(
                    nid, "threshold_enabled", checked
                )
            )
            props_layout.addRow(threshold_check)

            threshold_spin = QSpinBox()
            threshold_spin.setRange(0, 1023)
            saved_threshold = 512
            if node_config and node_config.state:
                saved_threshold = node_config.state.get("threshold", 512)
            threshold_spin.setValue(saved_threshold)
            threshold_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "threshold", val)
            )
            props_layout.addRow("Threshold:", threshold_spin)

            visible_check = QCheckBox("Show live value in dashboard")
            saved_visible = False
            if node_config and node_config.state:
                saved_visible = node_config.state.get("visible_in_runner", False)
            visible_check.setChecked(saved_visible)
            visible_check.toggled.connect(
                lambda checked, nid=node_id: self._on_node_property_changed(
                    nid, "visible_in_runner", checked
                )
            )
            props_layout.addRow(visible_check)

            info_label = QLabel(
                "Continuous mode: automatically polls sensor at poll interval.\n"
                "Threshold: output 'threshold_exceeded' will be True when value > threshold.\n"
                "Dashboard: enable to show live analog value in runner view."
            )
            info_label.setWordWrap(True)
            info_label.setProperty("textRole", "muted")
            props_layout.addRow(info_label)

        elif node_type == "AudioPlayback":
            self._add_section_header(props_layout, "AUDIO")
            file_edit = QLineEdit()
            file_edit.setReadOnly(True)
            file_edit.setPlaceholderText("No file selected")
            saved_file = ""
            if node_config and node_config.state:
                saved_file = node_config.state.get("file_path", "")
            file_edit.setText(saved_file)

            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(
                lambda checked, nid=node_id, le=file_edit: self._browse_audio_file(nid, le)
            )

            file_layout = QHBoxLayout()
            file_layout.addWidget(file_edit, 1)
            file_layout.addWidget(browse_btn)
            file_widget = QWidget()
            file_widget.setLayout(file_layout)
            props_layout.addRow("File:", file_widget)

            device_combo = QComboBox()
            device_combo.addItem("System Default", None)
            saved_device_index = None
            if node_config and node_config.state:
                saved_device_index = node_config.state.get("device_index")

            try:
                import sounddevice as sd

                devices = sd.query_devices()
                hostapis = sd.query_hostapis()
                api_names = {i: h["name"] for i, h in enumerate(hostapis)}
                api_priority = {
                    "Windows DirectSound": 0,
                    "Windows WASAPI": 1,
                }
                best = {}
                for i, dev in enumerate(devices):
                    if dev["max_output_channels"] > 0:
                        norm = dev["name"][:28].rstrip()
                        api = api_names.get(dev["hostapi"], "")
                        prio = api_priority.get(api, 2)
                        prev = best.get(norm)
                        if prev is None or prio < prev[0]:
                            best[norm] = (prio, i, dev["name"])

                current_idx = 0
                for _norm, (_prio, i, name) in sorted(best.items(), key=lambda kv: kv[1][1]):
                    device_combo.addItem(name, i)
                    if i == saved_device_index:
                        current_idx = device_combo.count() - 1
                device_combo.setCurrentIndex(current_idx)
            except ImportError:
                device_combo.addItem("(sounddevice not installed)", None)
            except Exception as e:
                logger.warning(f"Could not enumerate audio devices: {e}")

            def on_audio_device_changed(idx, nid=node_id, combo=device_combo):
                dev_idx = combo.currentData()
                dev_name = combo.currentText()
                self._on_node_property_changed(nid, "device_index", dev_idx)
                self._on_node_property_changed(nid, "device_name", dev_name)

            device_combo.currentIndexChanged.connect(on_audio_device_changed)
            props_layout.addRow("Output Device:", device_combo)

            volume_spin = QDoubleSpinBox()
            volume_spin.setRange(0.0, 1.0)
            volume_spin.setDecimals(2)
            volume_spin.setSingleStep(0.05)
            saved_volume = 1.0
            if node_config and node_config.state:
                saved_volume = node_config.state.get("volume", 1.0)
            volume_spin.setValue(saved_volume)
            volume_spin.valueChanged.connect(
                lambda val, nid=node_id: self._on_node_property_changed(nid, "volume", val)
            )
            props_layout.addRow("Volume:", volume_spin)

        elif node_type == "VideoPlayback":
            self._add_section_header(props_layout, "VIDEO")
            file_edit = QLineEdit()
            file_edit.setReadOnly(True)
            file_edit.setPlaceholderText("No file selected")
            saved_file = ""
            if node_config and node_config.state:
                saved_file = node_config.state.get("file_path", "")
            file_edit.setText(saved_file)

            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(
                lambda checked, nid=node_id, le=file_edit: self._browse_video_file(nid, le)
            )

            file_layout = QHBoxLayout()
            file_layout.addWidget(file_edit, 1)
            file_layout.addWidget(browse_btn)
            file_widget = QWidget()
            file_widget.setLayout(file_layout)
            props_layout.addRow("File:", file_widget)

            monitor_combo = QComboBox()
            screens = QApplication.screens()
            saved_monitor = -1
            if node_config and node_config.state:
                saved_monitor = node_config.state.get("monitor_index", -1)
            current_idx = 0
            for i, screen in enumerate(screens):
                geo = screen.geometry()
                label = f"{screen.name()} ({geo.width()}x{geo.height()})"
                monitor_combo.addItem(label, i)
                if i == saved_monitor:
                    current_idx = i
            monitor_combo.setCurrentIndex(current_idx)
            monitor_combo.currentIndexChanged.connect(
                lambda idx, nid=node_id, combo=monitor_combo: self._on_node_property_changed(
                    nid, "monitor_index", combo.currentData()
                )
            )
            props_layout.addRow("Monitor:", monitor_combo)

        self._properties_dock.setWidget(props_widget)

    def _on_node_device_changed(self, node_id: str, device_id: str) -> None:
        """Handle device selection change for a node."""
        session = self._session
        if session:
            node_config = session.get_node(node_id)
            if node_config:
                node_config.device_id = device_id
                session._mark_dirty()
                logger.info(f"Node {node_id} device changed to: {device_id}")

                if device_id and hasattr(self._core, "flow_engine") and self._core.flow_engine:
                    runtime_node = self._core.flow_engine.get_node(node_id)
                    if runtime_node and hasattr(runtime_node, "bind_device"):
                        device = self._hardware_manager.get_device(device_id)
                        if device:
                            runtime_node.bind_device(device)
                            logger.info(f"Bound device '{device_id}' to runtime node {node_id}")

                self._update_properties_panel(node_id)

    def _on_node_property_changed(self, node_id: str, prop_name: str, value) -> None:
        """Handle property change for a node."""
        session = self._session
        if session:
            session.update_node_state(node_id, {prop_name: value})
            logger.info(f"Node {node_id} property '{prop_name}' changed to: {value}")

            if prop_name == "function_name":
                self.flow_functions_changed.emit()

    def _browse_audio_file(self, node_id: str, line_edit: QLineEdit) -> None:
        """Open a file dialog to select an audio file."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)",
        )
        if file_path:
            line_edit.setText(file_path)
            self._on_node_property_changed(node_id, "file_path", file_path)

    def _browse_video_file(self, node_id: str, line_edit: QLineEdit) -> None:
        """Open a file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;MP4 Files (*.mp4);;All Files (*)",
        )
        if file_path:
            line_edit.setText(file_path)
            self._on_node_property_changed(node_id, "file_path", file_path)
