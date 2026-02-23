"""
Experiment Serializer - Save and load experiment files.

Handles conversion between GLIDER runtime objects and
JSON-serializable schema objects.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from glider.serialization.schema import (
    SCHEMA_VERSION,
    BoardConfigSchema,
    ConnectionSchema,
    DashboardConfigSchema,
    DeviceConfigSchema,
    ExperimentSchema,
    FlowConfigSchema,
    HardwareConfigSchema,
    MetadataSchema,
    NodeSchema,
    PortSchema,
    SchemaValidationError,
)

if TYPE_CHECKING:
    from glider.core.experiment_session import ExperimentSession
    from glider.core.flow_engine import FlowEngine
    from glider.core.hardware_manager import HardwareManager
    from glider.nodes.base_node import GliderNode

logger = logging.getLogger(__name__)


class ExperimentSerializer:
    """
    Serializer for GLIDER experiment files.

    Provides save/load functionality with schema validation
    and version migration support.
    """

    # File extension for GLIDER experiments
    FILE_EXTENSION = ".glider"

    def __init__(self):
        self._node_registry: dict[str, type[GliderNode]] = {}

    def register_node_type(self, node_type: str, node_class: type["GliderNode"]) -> None:
        """
        Register a node type for deserialization.

        Args:
            node_type: Full type path (e.g., "glider.nodes.hardware.DigitalWriteNode")
            node_class: The node class
        """
        self._node_registry[node_type] = node_class

    def save(
        self,
        path: Path,
        session: "ExperimentSession",
        flow_engine: Optional["FlowEngine"] = None,
        hardware_manager: Optional["HardwareManager"] = None,
    ) -> None:
        """
        Save an experiment session to a file.

        Args:
            path: File path to save to
            session: The experiment session to save
            flow_engine: Optional flow engine for node/connection data
            hardware_manager: Optional hardware manager for device config
        """
        # Build schema from session
        schema = self._session_to_schema(session, flow_engine, hardware_manager)

        # Update modified timestamp
        schema.update_modified()

        # Ensure .glider extension
        if not path.suffix == self.FILE_EXTENSION:
            path = path.with_suffix(self.FILE_EXTENSION)

        # Write atomically via temp file + rename to prevent corruption
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=".glider_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(schema.to_json(indent=2))
            os.replace(tmp_path, path)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(f"Saved experiment to {path}")

    def load(self, path: Path) -> ExperimentSchema:
        """
        Load an experiment schema from a file.

        Args:
            path: File path to load from

        Returns:
            The loaded experiment schema

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file cannot be read
            SchemaValidationError: If the file is malformed or invalid
            ValueError: If schema validation fails
        """
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            logger.error(f"Experiment file not found: {path}")
            raise
        except PermissionError:
            logger.error(f"Permission denied reading experiment file: {path}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {path}: {e}")
            raise SchemaValidationError(
                f"File encoding error: {e}. Ensure the file is UTF-8 encoded.",
                path=str(path),
            ) from e
        except OSError as e:
            logger.error(f"Error reading experiment file {path}: {e}")
            raise SchemaValidationError(
                f"Error reading file: {e}",
                path=str(path),
            ) from e

        # Validate file is not empty
        if not content.strip():
            raise SchemaValidationError(
                "File is empty",
                path=str(path),
            )

        try:
            schema = ExperimentSchema.from_json(content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: line {e.lineno}, column {e.colno}")
            raise SchemaValidationError(
                f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}",
                path=str(path),
            ) from e
        except SchemaValidationError:
            # Re-raise with file path context
            raise

        # Validate and migrate if needed
        try:
            schema = self._validate_and_migrate(schema)
        except ValueError as e:
            raise SchemaValidationError(str(e), path=str(path)) from e

        logger.info(f"Loaded experiment from {path}")
        return schema

    def apply_to_session(
        self,
        schema: ExperimentSchema,
        session: "ExperimentSession",
        flow_engine: Optional["FlowEngine"] = None,
        hardware_manager: Optional["HardwareManager"] = None,
    ) -> None:
        """
        Apply a loaded schema to a session.

        Args:
            schema: The experiment schema to apply
            session: The session to update
            flow_engine: Optional flow engine to populate
            hardware_manager: Optional hardware manager to configure
        """
        # Apply metadata
        session.name = schema.metadata.name
        session.metadata.description = schema.metadata.description
        session.metadata.author = schema.metadata.author

        # Apply hardware config
        if hardware_manager:
            self._apply_hardware_config(schema.hardware, hardware_manager)

        # Apply flow config
        if flow_engine:
            self._apply_flow_config(schema.flow, flow_engine)

        # Apply dashboard config
        dashboard_dict = schema.dashboard.to_dict()
        dashboard = session.dashboard
        for key, value in dashboard_dict.items():
            if hasattr(dashboard, key):
                setattr(dashboard, key, value)

        logger.info(f"Applied schema to session: {session.name}")

    def _session_to_schema(
        self,
        session: "ExperimentSession",
        flow_engine: Optional["FlowEngine"],
        hardware_manager: Optional["HardwareManager"],
    ) -> ExperimentSchema:
        """Convert a session to a schema."""
        # Build metadata
        metadata = MetadataSchema(
            name=session.name,
            description=session.metadata.description,
            author=session.metadata.author,
        )

        # Build hardware config
        hardware = HardwareConfigSchema()
        if hardware_manager:
            hardware = self._extract_hardware_config(hardware_manager)

        # Build flow config
        flow = FlowConfigSchema()
        if flow_engine:
            flow = self._extract_flow_config(flow_engine)

        # Build dashboard config
        dashboard = DashboardConfigSchema.from_dict(session.dashboard.to_dict())

        return ExperimentSchema(
            metadata=metadata,
            hardware=hardware,
            flow=flow,
            dashboard=dashboard,
        )

    def _extract_hardware_config(self, hardware_manager: "HardwareManager") -> HardwareConfigSchema:
        """Extract hardware configuration from manager."""
        boards = []
        devices = []

        # Extract board configs
        for board_id, board in hardware_manager.boards.items():
            board_config = BoardConfigSchema(
                id=board_id,
                type=type(board).__name__.lower().replace("board", ""),
                port=getattr(board, "port", None),
                settings=getattr(board, "settings", {}),
            )
            boards.append(board_config)

        # Extract device configs
        for device_id, device in hardware_manager.devices.items():
            device_config = DeviceConfigSchema(
                id=device_id,
                type=getattr(device, "device_type", "unknown"),
                board_id=getattr(device, "board_id", ""),
                pin=getattr(device, "pin", 0),
                name=getattr(device, "name", None),
                settings=getattr(device, "settings", {}),
            )
            devices.append(device_config)

        return HardwareConfigSchema(boards=boards, devices=devices)

    def _extract_flow_config(self, flow_engine: "FlowEngine") -> FlowConfigSchema:
        """Extract flow configuration from engine."""
        nodes = []
        connections = []

        # Extract nodes
        for node_id, node in flow_engine.nodes.items():
            # Get position from GUI metadata if available
            position = getattr(node, "gui_position", {"x": 0.0, "y": 0.0})

            # Build input ports
            inputs = []
            for i, inp in enumerate(getattr(node, "inputs", [])):
                port = PortSchema(
                    name=getattr(inp, "name", f"in_{i}"),
                    type="exec" if getattr(inp, "is_exec", False) else "data",
                    data_type=getattr(inp, "data_type", "any"),
                )
                inputs.append(port)

            # Build output ports
            outputs = []
            for i, out in enumerate(getattr(node, "outputs", [])):
                port = PortSchema(
                    name=getattr(out, "name", f"out_{i}"),
                    type="exec" if getattr(out, "is_exec", False) else "data",
                    data_type=getattr(out, "data_type", "any"),
                )
                outputs.append(port)

            node_schema = NodeSchema(
                id=node_id,
                type=f"{type(node).__module__}.{type(node).__name__}",
                title=getattr(node, "title", type(node).__name__),
                position=position,
                properties=self._extract_node_properties(node),
                inputs=inputs,
                outputs=outputs,
            )
            nodes.append(node_schema)

        # Extract connections
        for conn_id, conn in flow_engine.connections.items():
            conn_schema = ConnectionSchema(
                id=conn_id,
                from_node=conn.from_node_id,
                from_port=conn.from_port,
                to_node=conn.to_node_id,
                to_port=conn.to_port,
                connection_type="exec" if getattr(conn, "is_exec", False) else "data",
            )
            connections.append(conn_schema)

        return FlowConfigSchema(nodes=nodes, connections=connections)

    def _extract_node_properties(self, node: "GliderNode") -> dict[str, Any]:
        """Extract serializable properties from a node."""
        properties = {}

        # Get properties from node's property definitions
        for prop_name in getattr(node, "property_names", []):
            if hasattr(node, prop_name):
                value = getattr(node, prop_name)
                # Only include serializable values
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    properties[prop_name] = value

        # Common properties
        if hasattr(node, "visible_in_runner"):
            properties["visible_in_runner"] = node.visible_in_runner

        return properties

    def _apply_hardware_config(
        self, config: HardwareConfigSchema, hardware_manager: "HardwareManager"
    ) -> None:
        """Apply hardware configuration to manager."""
        # Clear existing config
        hardware_manager.clear()

        # Add boards
        for board_config in config.boards:
            hardware_manager.add_board(
                board_id=board_config.id,
                board_type=board_config.type,
                port=board_config.port,
                **board_config.settings,
            )

        # Add devices
        for device_config in config.devices:
            hardware_manager.add_device(
                device_id=device_config.id,
                device_type=device_config.type,
                board_id=device_config.board_id,
                pin=device_config.pin,
                name=device_config.name,
                **device_config.settings,
            )

    def _apply_flow_config(self, config: FlowConfigSchema, flow_engine: "FlowEngine") -> None:
        """Apply flow configuration to engine."""
        # Clear existing flow
        flow_engine.clear()

        # Create nodes, tracking which ones succeeded
        created_node_ids: set[str] = set()
        for node_schema in config.nodes:
            node_class = self._node_registry.get(node_schema.type)
            if node_class:
                node = flow_engine.create_node(
                    node_class,
                    node_id=node_schema.id,
                    **node_schema.properties,
                )
                # Set position for GUI
                if node:
                    node.gui_position = node_schema.position
                    created_node_ids.add(node_schema.id)
            else:
                logger.warning(f"Unknown node type: {node_schema.type}")

        # Create connections, skipping any that reference missing nodes
        for conn_schema in config.connections:
            if conn_schema.from_node not in created_node_ids:
                logger.warning(
                    f"Skipping connection: source node '{conn_schema.from_node}' was not created"
                )
                continue
            if conn_schema.to_node not in created_node_ids:
                logger.warning(
                    f"Skipping connection: target node '{conn_schema.to_node}' was not created"
                )
                continue
            flow_engine.connect(
                from_node_id=conn_schema.from_node,
                from_port=conn_schema.from_port,
                to_node_id=conn_schema.to_node,
                to_port=conn_schema.to_port,
            )

    def _validate_and_migrate(self, schema: ExperimentSchema) -> ExperimentSchema:
        """
        Validate schema and migrate from older versions if needed.

        Args:
            schema: The schema to validate

        Returns:
            Validated (and possibly migrated) schema

        Raises:
            ValueError: If schema is invalid and cannot be migrated
        """
        version = schema.schema_version

        # Check version compatibility
        major, minor, patch = map(int, version.split("."))
        current_major, current_minor, current_patch = map(int, SCHEMA_VERSION.split("."))

        if major > current_major:
            raise ValueError(f"Schema version {version} is newer than supported {SCHEMA_VERSION}")

        # Apply migrations for older versions
        if major < current_major or (major == current_major and minor < current_minor):
            schema = self._migrate_schema(schema, version, SCHEMA_VERSION)

        return schema

    def _migrate_schema(
        self, schema: ExperimentSchema, from_version: str, to_version: str
    ) -> ExperimentSchema:
        """
        Migrate schema from one version to another.

        Args:
            schema: The schema to migrate
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated schema
        """
        logger.info(f"Migrating schema from {from_version} to {to_version}")

        # Migration logic would go here for specific version upgrades
        # For now, just update the version
        schema.schema_version = to_version

        return schema


# Global serializer instance
_serializer: ExperimentSerializer | None = None


def get_serializer() -> ExperimentSerializer:
    """Get the global serializer instance."""
    global _serializer
    if _serializer is None:
        _serializer = ExperimentSerializer()
    return _serializer
