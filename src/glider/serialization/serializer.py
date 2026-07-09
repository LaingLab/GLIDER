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


# Conventional pin name per device type, used to convert legacy single-pin
# device entries into a pins dict (mirrors HardwareManager.add_device).
_LEGACY_PIN_NAMES = {
    "DigitalOutput": "output",
    "DigitalInput": "input",
    "AnalogInput": "input",
    "PWMOutput": "output",
    "Servo": "signal",
}


def _schema_device_pins(device: DeviceConfigSchema) -> dict[str, int]:
    """Pins dict for a device schema, converting the legacy single-pin form."""
    if device.pins:
        return dict(device.pins)
    if device.pin is not None:
        return {_LEGACY_PIN_NAMES.get(device.type, "pin"): device.pin}
    return {}


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

        The schema is applied incrementally into the live managers (boards,
        devices, nodes, connections); there is no rollback. If this raises
        partway through, the hardware manager may hold partially-applied
        state and the session model is NOT updated — callers should treat a
        raised exception as requiring a session reset (clear the managers
        and start over) rather than continuing with mixed state.

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

        # Sync the loaded hardware/flow into the *session model* too. The
        # session's own dataclasses are what ExperimentSession.save() /
        # to_dict() serialize; without this, a File > Save As after loading
        # a .glider file writes empty hardware/flow sections (data loss),
        # and UI reading session.hardware / session.flow sees nothing.
        from glider.core.experiment_session import (
            BoardConfig as SessionBoardConfig,
        )
        from glider.core.experiment_session import (
            ConnectionConfig as SessionConnectionConfig,
        )
        from glider.core.experiment_session import (
            DeviceConfig as SessionDeviceConfig,
        )
        from glider.core.experiment_session import (
            FlowConfig as SessionFlowConfig,
        )
        from glider.core.experiment_session import (
            HardwareConfig as SessionHardwareConfig,
        )
        from glider.core.experiment_session import (
            NodeConfig as SessionNodeConfig,
        )

        session._hardware = SessionHardwareConfig(
            boards=[
                SessionBoardConfig(
                    id=b.id,
                    driver_type=b.type,
                    port=b.port,
                    settings=dict(b.settings),
                )
                for b in schema.hardware.boards
            ],
            devices=[
                SessionDeviceConfig(
                    id=d.id,
                    device_type=d.type,
                    name=d.name or d.id,
                    board_id=d.board_id,
                    pins=_schema_device_pins(d),
                    settings=dict(d.settings),
                )
                for d in schema.hardware.devices
            ],
        )

        flow_nodes = []
        for n in schema.flow.nodes:
            # _apply_flow_config skips nodes it cannot create (unknown type,
            # create_node failure). The session model must mirror the live
            # engine, not the raw file — otherwise a Save As persists
            # phantom nodes the engine never had. When no engine was given,
            # nothing was filtered, so sync the file contents unfiltered.
            if flow_engine is not None and n.id not in flow_engine.nodes:
                continue
            props = n.properties or {}
            state = props.get("state") or {}
            if not isinstance(state, dict):
                state = {}
            node_type = n.type
            if flow_engine is not None:
                node_type = self._resolve_node_type(n.type, flow_engine) or n.type
            flow_nodes.append(
                SessionNodeConfig(
                    id=n.id,
                    node_type=node_type,
                    position=(
                        float(n.position.get("x", 0.0)),
                        float(n.position.get("y", 0.0)),
                    ),
                    state=state,
                    # KNOWN GAP: the .glider schema does not carry device
                    # bindings (NodeSchema has no device_id and no node writes
                    # one into its state), so this is None for every file that
                    # exists today. Hardware-node device bindings are still
                    # lost across the schema save/load path — follow-up work,
                    # out of scope for this task.
                    device_id=state.get("device_id"),
                    visible_in_runner=bool(props.get("visible_in_runner", False)),
                )
            )
        # Connections referencing a filtered-out node would dangle; keep
        # only those whose both endpoints survived the node sync.
        synced_node_ids = {node.id for node in flow_nodes}
        session._flow = SessionFlowConfig(
            nodes=flow_nodes,
            connections=[
                SessionConnectionConfig(
                    id=c.id,
                    from_node=c.from_node,
                    from_output=c.from_port,
                    to_node=c.to_node,
                    to_input=c.to_port,
                    connection_type=c.connection_type,
                )
                for c in schema.flow.connections
                if flow_engine is None
                or (c.from_node in synced_node_ids and c.to_node in synced_node_ids)
            ],
        )

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

        # Extract device configs. Read the real BaseDevice attributes: pin
        # assignments live in device._config.pins (dict of name -> pin), the
        # owning board is device.board (its .id is the board id), and settings
        # live in device._config.settings. (Earlier code read device.pin /
        # device.board_id / device.settings — none exist on BaseDevice — so
        # every device saved as pin=0 / board_id="" and never reloaded.)
        for device_id, device in hardware_manager.devices.items():
            dev_config = getattr(device, "_config", None)
            board = getattr(device, "board", None)
            device_config = DeviceConfigSchema(
                id=device_id,
                type=getattr(device, "device_type", "unknown"),
                board_id=getattr(board, "id", "") if board is not None else "",
                pins=dict(dev_config.pins) if dev_config is not None else {},
                name=getattr(device, "name", None),
                settings=dict(dev_config.settings) if dev_config is not None else {},
            )
            devices.append(device_config)

        return HardwareConfigSchema(boards=boards, devices=devices)

    def _extract_flow_config(self, flow_engine: "FlowEngine") -> FlowConfigSchema:
        """Extract flow configuration from engine.

        Node ``type`` is the short registered name (e.g. ``"DigitalWrite"``) —
        the same string ``flow_engine.register_node(name, cls)`` was called
        with. Loaders look this up directly in the engine's node registry.
        Connections are read via ``get_connections()`` which returns a list
        of dicts with int port indices (``from_output`` / ``to_input``);
        these are mapped onto ``ConnectionSchema``'s ``from_port`` /
        ``to_port`` fields.
        """
        nodes = []
        connections = []

        # Build a reverse map class -> registered name once, so each node looks
        # up its stable persisted type string.
        cls_to_name: dict[type, str] = {
            cls: name for name, cls in flow_engine._node_registry.items()  # noqa: SLF001
        }

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

            # Prefer the short registered name; fall back to definition name,
            # then to the class name. Avoids writing fully-qualified dotted
            # paths that the apply path would have to round-trip-strip.
            short_type = cls_to_name.get(type(node))
            if short_type is None:
                short_type = getattr(getattr(node, "definition", None), "name", None)
            if short_type is None:
                short_type = type(node).__name__

            node_schema = NodeSchema(
                id=node_id,
                type=short_type,
                title=getattr(node, "title", type(node).__name__),
                position=position,
                properties=self._extract_node_properties(node),
                inputs=inputs,
                outputs=outputs,
            )
            nodes.append(node_schema)

        # Extract connections. `get_connections()` returns a list of dicts
        # with keys: id, from_node, from_output, to_node, to_input, type.
        # (Earlier code attempted `flow_engine.connections.items()` which
        # tried to attribute-access a non-existent property AND treat the
        # dict-of-dicts as if it had .from_node_id — every save crashed.)
        for conn in flow_engine.get_connections():
            conn_schema = ConnectionSchema(
                id=conn["id"],
                from_node=conn["from_node"],
                from_port=conn["from_output"],
                to_node=conn["to_node"],
                to_port=conn["to_input"],
                connection_type=conn.get("type", "data"),
            )
            connections.append(conn_schema)

        return FlowConfigSchema(nodes=nodes, connections=connections)

    def _extract_node_properties(self, node: "GliderNode") -> dict[str, Any]:
        """Extract serializable properties from a node.

        Uses the node's ``get_state()`` API as the authoritative source of
        truth. Earlier code iterated ``getattr(node, "property_names", [])``
        — but no node class ever defined ``property_names``, so the loop
        body never ran and every per-node parameter (pin, threshold, ITI,
        camera index, …) was silently dropped on every save. Result: load
        appeared to succeed and every node reverted to its dataclass
        defaults.

        Common attributes outside the state dict (``visible_in_runner``,
        ``enabled``) are preserved as siblings to the ``state`` payload so
        old loaders that don't know about ``state`` still see them.
        """
        properties: dict[str, Any] = {}

        if hasattr(node, "visible_in_runner"):
            properties["visible_in_runner"] = bool(node.visible_in_runner)
        if hasattr(node, "_enabled"):
            properties["enabled"] = bool(node._enabled)

        if hasattr(node, "get_state") and callable(node.get_state):
            try:
                state = node.get_state()
            except Exception as e:
                logger.warning(
                    "get_state() raised on %s (%s); node state will be empty",
                    getattr(node, "_glider_id", "?"),
                    type(node).__name__,
                    exc_info=e,
                )
                state = {}
            if isinstance(state, dict) and state:
                properties["state"] = state

        return properties

    def _apply_hardware_config(
        self, config: HardwareConfigSchema, hardware_manager: "HardwareManager"
    ) -> None:
        """Apply hardware configuration to manager.

        Note: ``HardwareManager.add_board`` parameter is ``driver_type``,
        NOT ``board_type``. Earlier code passed the wrong kwarg, so every
        load crashed immediately with ``TypeError``.

        ``settings`` is passed through as ``**kwargs``; if a user-edited
        ``.glider`` file ever contains a settings key that collides with an
        explicit kwarg (``driver_type``, ``port``, ``board_id``), the kwarg
        collision will raise ``TypeError`` early — that's preferable to
        silently honoring the malformed override.
        """
        # Clear existing config
        hardware_manager.clear()

        # Add boards
        for board_config in config.boards:
            hardware_manager.add_board(
                board_id=board_config.id,
                driver_type=board_config.type,
                port=board_config.port,
                **board_config.settings,
            )

        # Add devices. Current-format files carry a pins dict (multi-pin
        # aware, loaded via add_device_multi_pin — an empty dict is valid:
        # zero-pin devices like BLEWrite have no GPIO pins to allocate).
        # Legacy files carry a single int `pin` and no pins dict; ONLY that
        # combination takes the legacy path, where add_device maps the pin
        # to the conventional pin name ("output"/"input"/"signal"). Branching
        # on `pins` alone would send current-format zero-pin devices through
        # add_device(pin=None), synthesizing a phantom {"pin": None} that
        # collides across devices ("Pin None is already claimed").
        for device_config in config.devices:
            if device_config.pin is not None and not device_config.pins:
                hardware_manager.add_device(
                    device_id=device_config.id,
                    device_type=device_config.type,
                    board_id=device_config.board_id,
                    pin=device_config.pin,
                    name=device_config.name,
                    **device_config.settings,
                )
            else:
                hardware_manager.add_device_multi_pin(
                    device_id=device_config.id,
                    device_type=device_config.type,
                    board_id=device_config.board_id,
                    pins=device_config.pins,
                    name=device_config.name,
                    **device_config.settings,
                )

    def _apply_flow_config(self, config: FlowConfigSchema, flow_engine: "FlowEngine") -> None:
        """Apply flow configuration to engine.

        Three changes vs. the previous version:

        1. ``flow_engine.create_node`` takes a *type string* as its second
           argument, not a class. Earlier code passed the resolved class
           object as the first positional, which landed it in ``node_id``
           — every call raised ``TypeError``.
        2. The persisted ``properties`` dict carries ``state`` (the
           node's full ``_state`` payload), ``visible_in_runner``, and
           ``enabled`` as separate keys. ``state`` is handed to
           ``create_node(state=...)`` which calls ``node.set_state(state)``
           internally; the others are applied to the constructed node.
        3. Connections use port *indices* (ints from the schema), so we
           call ``create_connection`` directly. ``flow_engine.connect``
           never existed and ``connect_nodes`` requires port names. The
           schema stores indices, so this is the right call to begin with.

        Backwards-compat for files written by earlier versions:
        ``node_schema.type`` may carry a fully-qualified dotted path like
        ``"glider.nodes.experiment_nodes.StartExperimentNode"``. We accept
        either a short name (matches registry directly) or a long name
        whose final ``Node`` suffix is stripped (e.g. ``StartExperimentNode``
        → registry name ``"StartExperiment"``).
        """
        # Clear existing flow
        flow_engine.clear()

        # Create nodes, tracking which ones succeeded
        created_node_ids: set[str] = set()
        for node_schema in config.nodes:
            node_type = self._resolve_node_type(node_schema.type, flow_engine)
            if node_type is None:
                logger.warning(f"Unknown node type: {node_schema.type}")
                continue

            props = dict(node_schema.properties or {})
            state = props.pop("state", None)
            visible_in_runner = props.pop("visible_in_runner", None)
            enabled = props.pop("enabled", None)

            # Position dict from schema: prefer (x, y) tuple for create_node.
            position_tuple: tuple[float, float] = (
                float(node_schema.position.get("x", 0.0)),
                float(node_schema.position.get("y", 0.0)),
            )

            try:
                node = flow_engine.create_node(
                    node_id=node_schema.id,
                    node_type=node_type,
                    position=position_tuple,
                    state=state,
                )
            except Exception as e:
                logger.warning(f"Failed to create node {node_schema.id} ({node_type}): {e}")
                continue

            if node is None:
                continue

            # Preserve original position dict for the GUI (some panels use
            # the dict shape directly).
            node.gui_position = node_schema.position

            if visible_in_runner is not None and hasattr(node, "_visible_in_runner"):
                node._visible_in_runner = bool(visible_in_runner)
            if enabled is not None and hasattr(node, "_enabled"):
                node._enabled = bool(enabled)

            created_node_ids.add(node_schema.id)

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
            try:
                flow_engine.create_connection(
                    connection_id=conn_schema.id,
                    from_node_id=conn_schema.from_node,
                    from_output=conn_schema.from_port,
                    to_node_id=conn_schema.to_node,
                    to_input=conn_schema.to_port,
                    connection_type=conn_schema.connection_type,
                )
            except Exception as e:
                logger.warning(
                    f"Skipping connection " f"{conn_schema.from_node}->{conn_schema.to_node}: {e}"
                )

    @staticmethod
    def _resolve_node_type(type_str: str, flow_engine: "FlowEngine") -> str | None:
        """Resolve a persisted node-type string against the engine registry.

        Accepts: the registered short name (preferred), or a fully-qualified
        ``module.ClassName`` string from older save files. For the long form,
        strip a trailing ``Node`` suffix to match the convention used by
        ``register_node`` (e.g. ``StartExperimentNode`` → ``"StartExperiment"``).
        """
        registry = flow_engine._node_registry  # noqa: SLF001
        if type_str in registry:
            return type_str
        # Try the bare class name from a dotted path
        short = type_str.rsplit(".", 1)[-1] if "." in type_str else type_str
        if short in registry:
            return short
        if short.endswith("Node") and short[:-4] in registry:
            return short[:-4]
        return None

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
        try:
            parts = version.split(".")
            major, minor = int(parts[0]), int(parts[1])
        except (ValueError, IndexError) as e:
            raise SchemaValidationError(f"Invalid schema version format: {version}") from e
        try:
            current_parts = SCHEMA_VERSION.split(".")
            current_major, current_minor = int(current_parts[0]), int(current_parts[1])
        except (ValueError, IndexError) as e:
            raise SchemaValidationError(
                f"Invalid current schema version format: {SCHEMA_VERSION}"
            ) from e

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
