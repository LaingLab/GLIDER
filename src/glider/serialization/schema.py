"""
JSON Schema - Data structures for experiment serialization.

Defines the schema for .glider experiment files following the
structure specified in the design document.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

# Current schema version
SCHEMA_VERSION = "1.0.0"


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    def __init__(self, message: str, path: str = "", details: dict[str, Any] | None = None):
        self.path = path
        self.details = details or {}
        full_message = f"{path}: {message}" if path else message
        super().__init__(full_message)


def _validate_type(value: Any, expected_type: type, field_name: str, path: str) -> None:
    """Validate that a value is of the expected type."""
    if not isinstance(value, expected_type):
        raise SchemaValidationError(
            f"Expected {expected_type.__name__}, got {type(value).__name__}",
            path=f"{path}.{field_name}" if path else field_name,
        )


def _validate_required(data: dict[str, Any], fields: list[str], path: str) -> None:
    """Validate that required fields are present."""
    missing = [f for f in fields if f not in data]
    if missing:
        raise SchemaValidationError(
            f"Missing required fields: {', '.join(missing)}",
            path=path,
        )


@dataclass
class PortSchema:
    """Schema for a node port definition."""

    name: str
    type: str  # "data" or "exec"
    data_type: str | None = None  # e.g., "int", "float", "bool", "any"


@dataclass
class NodeSchema:
    """Schema for a node in the flow graph."""

    id: str
    type: str  # Full node type path, e.g., "glider.nodes.hardware.DigitalWriteNode"
    title: str
    position: dict[str, float]  # {"x": float, "y": float}
    properties: dict[str, Any] = field(default_factory=dict)
    inputs: list[PortSchema] = field(default_factory=list)
    outputs: list[PortSchema] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "position": self.position,
            "properties": self.properties,
            "inputs": [
                {"name": p.name, "type": p.type, "data_type": p.data_type} for p in self.inputs
            ],
            "outputs": [
                {"name": p.name, "type": p.type, "data_type": p.data_type} for p in self.outputs
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "node") -> "NodeSchema":
        """Create from dictionary with validation."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}", path)

        _validate_required(data, ["id", "type", "title", "position"], path)
        _validate_type(data["id"], str, "id", path)
        _validate_type(data["type"], str, "type", path)
        _validate_type(data["title"], str, "title", path)
        _validate_type(data["position"], dict, "position", path)

        # Validate position has x and y
        position = data["position"]
        if "x" not in position or "y" not in position:
            raise SchemaValidationError("Position must have 'x' and 'y' keys", f"{path}.position")

        try:
            inputs = [PortSchema(**p) for p in data.get("inputs", [])]
            outputs = [PortSchema(**p) for p in data.get("outputs", [])]
        except TypeError as e:
            raise SchemaValidationError(f"Invalid port definition: {e}", f"{path}.ports") from e

        return cls(
            id=data["id"],
            type=data["type"],
            title=data["title"],
            position=position,
            properties=data.get("properties", {}),
            inputs=inputs,
            outputs=outputs,
        )


@dataclass
class ConnectionSchema:
    """Schema for a connection between nodes."""

    id: str
    from_node: str
    from_port: int
    to_node: str
    to_port: int
    connection_type: str = "data"  # "data" or "exec"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "connection") -> "ConnectionSchema":
        """Create from dictionary with validation."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}", path)

        _validate_required(data, ["id", "from_node", "from_port", "to_node", "to_port"], path)
        _validate_type(data["id"], str, "id", path)
        _validate_type(data["from_node"], str, "from_node", path)
        _validate_type(data["to_node"], str, "to_node", path)

        # from_port and to_port should be integers
        if not isinstance(data["from_port"], int):
            raise SchemaValidationError(
                f"Expected int, got {type(data['from_port']).__name__}", f"{path}.from_port"
            )
        if not isinstance(data["to_port"], int):
            raise SchemaValidationError(
                f"Expected int, got {type(data['to_port']).__name__}", f"{path}.to_port"
            )

        return cls(
            id=data["id"],
            from_node=data["from_node"],
            from_port=data["from_port"],
            to_node=data["to_node"],
            to_port=data["to_port"],
            connection_type=data.get("connection_type", "data"),
        )


@dataclass
class BoardConfigSchema:
    """Schema for a hardware board configuration."""

    id: str
    type: str  # e.g., "telemetrix", "pigpio"
    port: str | None = None  # Serial port for Arduino
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "board") -> "BoardConfigSchema":
        """Create from dictionary with validation."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}", path)

        _validate_required(data, ["id", "type"], path)
        _validate_type(data["id"], str, "id", path)
        _validate_type(data["type"], str, "type", path)

        port = data.get("port")
        if port is not None and not isinstance(port, str):
            raise SchemaValidationError(
                f"Expected str or null, got {type(port).__name__}", f"{path}.port"
            )

        settings = data.get("settings", {})
        if not isinstance(settings, dict):
            raise SchemaValidationError(
                f"Expected dict, got {type(settings).__name__}", f"{path}.settings"
            )

        return cls(
            id=data["id"],
            type=data["type"],
            port=port,
            settings=settings,
        )


@dataclass
class DeviceConfigSchema:
    """Schema for a hardware device configuration."""

    id: str
    type: str  # e.g., "digital_output", "analog_input", "servo"
    board_id: str
    pin: int
    name: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "device") -> "DeviceConfigSchema":
        """Create from dictionary with validation."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}", path)

        _validate_required(data, ["id", "type", "board_id", "pin"], path)
        _validate_type(data["id"], str, "id", path)
        _validate_type(data["type"], str, "type", path)
        _validate_type(data["board_id"], str, "board_id", path)

        if not isinstance(data["pin"], int):
            raise SchemaValidationError(
                f"Expected int, got {type(data['pin']).__name__}", f"{path}.pin"
            )

        name = data.get("name")
        if name is not None and not isinstance(name, str):
            raise SchemaValidationError(
                f"Expected str or null, got {type(name).__name__}", f"{path}.name"
            )

        settings = data.get("settings", {})
        if not isinstance(settings, dict):
            raise SchemaValidationError(
                f"Expected dict, got {type(settings).__name__}", f"{path}.settings"
            )

        return cls(
            id=data["id"],
            type=data["type"],
            board_id=data["board_id"],
            pin=data["pin"],
            name=name,
            settings=settings,
        )


@dataclass
class HardwareConfigSchema:
    """Schema for hardware configuration."""

    boards: list[BoardConfigSchema] = field(default_factory=list)
    devices: list[DeviceConfigSchema] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "boards": [b.to_dict() for b in self.boards],
            "devices": [d.to_dict() for d in self.devices],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "hardware") -> "HardwareConfigSchema":
        """Create from dictionary with validation."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}", path)

        boards_data = data.get("boards", [])
        devices_data = data.get("devices", [])

        if not isinstance(boards_data, list):
            raise SchemaValidationError(
                f"Expected list, got {type(boards_data).__name__}", f"{path}.boards"
            )
        if not isinstance(devices_data, list):
            raise SchemaValidationError(
                f"Expected list, got {type(devices_data).__name__}", f"{path}.devices"
            )

        boards = []
        for i, b in enumerate(boards_data):
            boards.append(BoardConfigSchema.from_dict(b, f"{path}.boards[{i}]"))

        devices = []
        for i, d in enumerate(devices_data):
            devices.append(DeviceConfigSchema.from_dict(d, f"{path}.devices[{i}]"))

        return cls(boards=boards, devices=devices)


@dataclass
class DashboardWidgetSchema:
    """Schema for a dashboard widget configuration."""

    node_id: str
    position: int  # Order in the dashboard
    size: str = "normal"  # "small", "normal", "large"
    visible: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DashboardWidgetSchema":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DashboardConfigSchema:
    """Schema for dashboard configuration."""

    layout_mode: str = "vertical"  # "vertical", "horizontal", "grid"
    columns: int = 1
    widgets: list[DashboardWidgetSchema] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "layout_mode": self.layout_mode,
            "columns": self.columns,
            "widgets": [w.to_dict() for w in self.widgets],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "dashboard") -> "DashboardConfigSchema":
        """Create from dictionary."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}", path)

        widgets = [DashboardWidgetSchema.from_dict(w) for w in data.get("widgets", [])]
        return cls(
            layout_mode=data.get("layout_mode", "vertical"),
            columns=data.get("columns", 1),
            widgets=widgets,
        )


@dataclass
class FlowConfigSchema:
    """Schema for flow graph configuration."""

    nodes: list[NodeSchema] = field(default_factory=list)
    connections: list[ConnectionSchema] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "flow") -> "FlowConfigSchema":
        """Create from dictionary with validation."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}", path)

        nodes_data = data.get("nodes", [])
        connections_data = data.get("connections", [])

        if not isinstance(nodes_data, list):
            raise SchemaValidationError(
                f"Expected list, got {type(nodes_data).__name__}", f"{path}.nodes"
            )
        if not isinstance(connections_data, list):
            raise SchemaValidationError(
                f"Expected list, got {type(connections_data).__name__}", f"{path}.connections"
            )

        nodes = []
        for i, n in enumerate(nodes_data):
            nodes.append(NodeSchema.from_dict(n, f"{path}.nodes[{i}]"))

        connections = []
        for i, c in enumerate(connections_data):
            connections.append(ConnectionSchema.from_dict(c, f"{path}.connections[{i}]"))

        return cls(nodes=nodes, connections=connections)


@dataclass
class MetadataSchema:
    """Schema for experiment metadata."""

    name: str
    description: str = ""
    author: str = ""
    created: str = ""
    modified: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        if not self.modified:
            self.modified = self.created

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "metadata") -> "MetadataSchema":
        """Create from dictionary with validation."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}", path)

        _validate_required(data, ["name"], path)
        _validate_type(data["name"], str, "name", path)

        # Validate optional string fields
        for field_name in ["description", "author", "created", "modified"]:
            if field_name in data and not isinstance(data[field_name], str):
                raise SchemaValidationError(
                    f"Expected str, got {type(data[field_name]).__name__}", f"{path}.{field_name}"
                )

        # Validate tags is a list of strings
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            raise SchemaValidationError(f"Expected list, got {type(tags).__name__}", f"{path}.tags")
        for i, tag in enumerate(tags):
            if not isinstance(tag, str):
                raise SchemaValidationError(
                    f"Expected str, got {type(tag).__name__}", f"{path}.tags[{i}]"
                )

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            created=data.get("created", ""),
            modified=data.get("modified", ""),
            tags=tags,
        )


@dataclass
class ExperimentSchema:
    """
    Root schema for a GLIDER experiment file.

    This is the top-level structure saved as a .glider file.
    """

    schema_version: str = SCHEMA_VERSION
    metadata: MetadataSchema = field(default_factory=lambda: MetadataSchema(name="Untitled"))
    hardware: HardwareConfigSchema = field(default_factory=HardwareConfigSchema)
    flow: FlowConfigSchema = field(default_factory=FlowConfigSchema)
    dashboard: DashboardConfigSchema = field(default_factory=DashboardConfigSchema)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "metadata": self.metadata.to_dict(),
            "hardware": self.hardware.to_dict(),
            "flow": self.flow.to_dict(),
            "dashboard": self.dashboard.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "") -> "ExperimentSchema":
        """Create from dictionary with validation."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}", path or "root")

        # Validate schema_version
        schema_version = data.get("schema_version", SCHEMA_VERSION)
        if not isinstance(schema_version, str):
            raise SchemaValidationError(
                f"Expected str, got {type(schema_version).__name__}", "schema_version"
            )

        # Validate metadata (required)
        metadata_data = data.get("metadata")
        if metadata_data is None:
            raise SchemaValidationError("Missing required field: metadata", path or "root")

        return cls(
            schema_version=schema_version,
            metadata=MetadataSchema.from_dict(metadata_data, "metadata"),
            hardware=HardwareConfigSchema.from_dict(data.get("hardware", {}), "hardware"),
            flow=FlowConfigSchema.from_dict(data.get("flow", {}), "flow"),
            dashboard=DashboardConfigSchema.from_dict(data.get("dashboard", {}), "dashboard"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentSchema":
        """Create from JSON string with validation."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(
                f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}",
                path="",
            ) from e
        return cls.from_dict(data)

    def update_modified(self) -> None:
        """Update the modified timestamp."""
        self.metadata.modified = datetime.now().isoformat()
