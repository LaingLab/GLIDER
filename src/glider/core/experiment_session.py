"""
Experiment Session - The Model in GLIDER's MVC architecture.

Represents the current state of an experiment: the Hardware Map
(Boards/Devices) and the Logic Graph. Serializable to JSON.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Current state of the experiment session."""

    IDLE = auto()  # Not running
    INITIALIZING = auto()  # Connecting to hardware
    READY = auto()  # Hardware connected, ready to run
    RUNNING = auto()  # Experiment in progress
    PAUSED = auto()  # Experiment paused
    STOPPING = auto()  # Shutting down
    ERROR = auto()  # Error state


@dataclass
class Subject:
    """Subject/animal information for the experiment."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""  # User-defined ID (e.g., "M001")
    name: str = ""
    group: str = ""  # Treatment group (e.g., "Control", "Drug A")

    # Biological fields
    age: str = ""
    sex: str = ""  # "Male", "Female", "Unknown"
    weight: str = ""  # With units (e.g., "25.5 g")
    strain: str = ""  # Strain/genotype

    # Solution/Drug fields
    solution: str = ""  # Solution name
    concentration: str = ""  # e.g., "10 mg/mL"
    dose: str = ""  # e.g., "5 mg/kg"
    route: str = ""  # e.g., "IP", "IV", "PO", "SC"

    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "name": self.name,
            "group": self.group,
            "age": self.age,
            "sex": self.sex,
            "weight": self.weight,
            "strain": self.strain,
            "solution": self.solution,
            "concentration": self.concentration,
            "dose": self.dose,
            "route": self.route,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Subject":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            subject_id=data.get("subject_id", ""),
            name=data.get("name", ""),
            group=data.get("group", ""),
            age=data.get("age", ""),
            sex=data.get("sex", ""),
            weight=data.get("weight", ""),
            strain=data.get("strain", ""),
            solution=data.get("solution", ""),
            concentration=data.get("concentration", ""),
            dose=data.get("dose", ""),
            route=data.get("route", ""),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


@dataclass
class SessionMetadata:
    """Metadata about the experiment session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Experiment"
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    glider_version: str = "1.0.0"

    # Experiment fields
    protocol: str = ""
    experiment_type: str = ""  # e.g., "Open Field", "Morris Water Maze"
    experimenter: str = ""
    lab: str = ""
    project: str = ""
    notes: str = ""

    # Subject management
    subjects: list[Subject] = field(default_factory=list)
    active_subject_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "version": self.version,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "glider_version": self.glider_version,
            "protocol": self.protocol,
            "experiment_type": self.experiment_type,
            "experimenter": self.experimenter,
            "lab": self.lab,
            "project": self.project,
            "notes": self.notes,
            "subjects": [s.to_dict() for s in self.subjects],
            "active_subject_id": self.active_subject_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        subjects_data = data.get("subjects", [])
        subjects = [Subject.from_dict(s) for s in subjects_data]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Untitled Experiment"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            glider_version=data.get("glider_version", "1.0.0"),
            protocol=data.get("protocol", ""),
            experiment_type=data.get("experiment_type", ""),
            experimenter=data.get("experimenter", ""),
            lab=data.get("lab", ""),
            project=data.get("project", ""),
            notes=data.get("notes", ""),
            subjects=subjects,
            active_subject_id=data.get("active_subject_id"),
        )

    def get_active_subject(self) -> Optional[Subject]:
        """Get the currently active subject."""
        if not self.active_subject_id:
            return None
        for subject in self.subjects:
            if subject.id == self.active_subject_id:
                return subject
        return None

    def add_subject(self, subject: Subject) -> None:
        """Add a subject to the experiment."""
        self.subjects.append(subject)
        # If this is the first subject, make it active
        if len(self.subjects) == 1:
            self.active_subject_id = subject.id

    def remove_subject(self, subject_id: str) -> bool:
        """Remove a subject by ID. Returns True if removed."""
        for i, subject in enumerate(self.subjects):
            if subject.id == subject_id:
                self.subjects.pop(i)
                # Clear active if we removed the active subject
                if self.active_subject_id == subject_id:
                    self.active_subject_id = self.subjects[0].id if self.subjects else None
                return True
        return False

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        """Get a subject by ID."""
        for subject in self.subjects:
            if subject.id == subject_id:
                return subject
        return None

    def set_active_subject(self, subject_id: str) -> bool:
        """Set the active subject. Returns True if successful."""
        for subject in self.subjects:
            if subject.id == subject_id:
                self.active_subject_id = subject_id
                return True
        return False


@dataclass
class BoardConfig:
    """Configuration for a hardware board."""

    id: str
    driver_type: str  # e.g., "arduino", "raspberry_pi"
    port: Optional[str] = None
    board_type: Optional[str] = None  # e.g., "uno", "mega"
    auto_reconnect: bool = False
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "driver_type": self.driver_type,
            "port": self.port,
            "board_type": self.board_type,
            "auto_reconnect": self.auto_reconnect,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoardConfig":
        return cls(
            id=data["id"],
            driver_type=data["driver_type"],
            port=data.get("port"),
            board_type=data.get("board_type"),
            auto_reconnect=data.get("auto_reconnect", False),
            settings=data.get("settings", {}),
        )


@dataclass
class DeviceConfig:
    """Configuration for a hardware device."""

    id: str
    device_type: str  # e.g., "DigitalOutput", "DHT22"
    name: str
    board_id: str
    pins: dict[str, int]
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "device_type": self.device_type,
            "name": self.name,
            "board_id": self.board_id,
            "pins": self.pins,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceConfig":
        return cls(
            id=data["id"],
            device_type=data["device_type"],
            name=data["name"],
            board_id=data["board_id"],
            pins=data["pins"],
            settings=data.get("settings", {}),
        )


@dataclass
class CameraConfig:
    """Configuration for camera and computer vision settings."""

    camera_index: int = 0
    resolution: tuple = (640, 480)
    fps: int = 30
    exposure: int = -1  # -1 = auto
    brightness: int = 128
    contrast: int = 128
    cv_enabled: bool = True
    cv_backend: str = "background_subtraction"
    confidence_threshold: float = 0.5
    tracking_enabled: bool = True
    max_disappeared: int = 50
    video_recording_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_index": self.camera_index,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "exposure": self.exposure,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "cv_enabled": self.cv_enabled,
            "cv_backend": self.cv_backend,
            "confidence_threshold": self.confidence_threshold,
            "tracking_enabled": self.tracking_enabled,
            "max_disappeared": self.max_disappeared,
            "video_recording_enabled": self.video_recording_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        return cls(
            camera_index=data.get("camera_index", 0),
            resolution=tuple(data.get("resolution", [640, 480])),
            fps=data.get("fps", 30),
            exposure=data.get("exposure", -1),
            brightness=data.get("brightness", 128),
            contrast=data.get("contrast", 128),
            cv_enabled=data.get("cv_enabled", True),
            cv_backend=data.get("cv_backend", "background_subtraction"),
            confidence_threshold=data.get("confidence_threshold", 0.5),
            tracking_enabled=data.get("tracking_enabled", True),
            max_disappeared=data.get("max_disappeared", 50),
            video_recording_enabled=data.get("video_recording_enabled", True),
        )


@dataclass
class HardwareConfig:
    """Complete hardware configuration."""

    boards: list[BoardConfig] = field(default_factory=list)
    devices: list[DeviceConfig] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boards": [b.to_dict() for b in self.boards],
            "devices": [d.to_dict() for d in self.devices],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareConfig":
        return cls(
            boards=[BoardConfig.from_dict(b) for b in data.get("boards", [])],
            devices=[DeviceConfig.from_dict(d) for d in data.get("devices", [])],
        )


@dataclass
class NodeConfig:
    """Configuration for a flow node."""

    id: str
    node_type: str
    position: tuple = (0, 0)
    state: dict[str, Any] = field(default_factory=dict)
    device_id: Optional[str] = None  # For hardware nodes
    visible_in_runner: bool = False  # Show in dashboard

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "position": list(self.position),
            "state": self.state,
            "device_id": self.device_id,
            "visible_in_runner": self.visible_in_runner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeConfig":
        return cls(
            id=data["id"],
            node_type=data["node_type"],
            position=tuple(data.get("position", [0, 0])),
            state=data.get("state", {}),
            device_id=data.get("device_id"),
            visible_in_runner=data.get("visible_in_runner", False),
        )


@dataclass
class ConnectionConfig:
    """Configuration for a connection between nodes."""

    id: str
    from_node: str
    from_output: int
    to_node: str
    to_input: int
    connection_type: str = "data"  # "data" or "exec"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_node": self.from_node,
            "from_output": self.from_output,
            "to_node": self.to_node,
            "to_input": self.to_input,
            "connection_type": self.connection_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConnectionConfig":
        return cls(
            id=data["id"],
            from_node=data["from_node"],
            from_output=data["from_output"],
            to_node=data["to_node"],
            to_input=data["to_input"],
            connection_type=data.get("connection_type", "data"),
        )


@dataclass
class FlowConfig:
    """Configuration for the experiment flow graph."""

    nodes: list[NodeConfig] = field(default_factory=list)
    connections: list[ConnectionConfig] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowConfig":
        return cls(
            nodes=[NodeConfig.from_dict(n) for n in data.get("nodes", [])],
            connections=[ConnectionConfig.from_dict(c) for c in data.get("connections", [])],
        )


@dataclass
class DashboardConfig:
    """Configuration for the runner dashboard layout."""

    widgets: list[dict[str, Any]] = field(default_factory=list)
    layout: str = "vertical"  # "vertical", "horizontal", "grid"
    columns: int = 1  # For grid layout

    def to_dict(self) -> dict[str, Any]:
        return {
            "widgets": self.widgets,
            "layout": self.layout,
            "columns": self.columns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DashboardConfig":
        return cls(
            widgets=data.get("widgets", []),
            layout=data.get("layout", "vertical"),
            columns=data.get("columns", 1),
        )


@dataclass
class ZoneConfig:
    """Configuration for zones within the camera view."""

    zones: list[dict[str, Any]] = field(default_factory=list)  # Serialized Zone objects
    config_width: int = 0
    config_height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "zones": self.zones,
            "config_width": self.config_width,
            "config_height": self.config_height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneConfig":
        return cls(
            zones=data.get("zones", []),
            config_width=data.get("config_width", 0),
            config_height=data.get("config_height", 0),
        )


@dataclass
class AudioConfig:
    """Configuration for audio recording settings."""

    enabled: bool = False
    device_index: Optional[int] = None
    device_name: str = ""
    sample_rate: int = 44100
    channels: int = 1
    gain: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "gain": self.gain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioConfig":
        return cls(
            enabled=data.get("enabled", False),
            device_index=data.get("device_index"),
            device_name=data.get("device_name", ""),
            sample_rate=data.get("sample_rate", 44100),
            channels=data.get("channels", 1),
            gain=data.get("gain", 1.0),
        )


class ExperimentSession:
    """
    Represents the complete state of an experiment.

    The session is the single source of truth for the experiment
    configuration and runtime state. It is serializable to JSON
    for saving and loading experiments.
    """

    def __init__(self):
        """Initialize a new experiment session."""
        self._metadata = SessionMetadata()
        self._hardware = HardwareConfig()
        self._flow = FlowConfig()
        self._dashboard = DashboardConfig()
        self._camera = CameraConfig()
        self._zones = ZoneConfig()
        self._audio = AudioConfig()
        self._state = SessionState.IDLE
        self._dirty = False  # Has unsaved changes
        self._file_path: Optional[str] = None

        # Custom devices and flow functions
        self._custom_device_definitions: dict[str, Any] = {}  # id -> definition dict
        self._flow_function_definitions: dict[str, Any] = {}  # id -> definition dict

        # Callbacks for state changes
        self._state_callbacks: list[Callable[[SessionState], None]] = []
        self._change_callbacks: list[Callable[[], None]] = []

    @property
    def metadata(self) -> SessionMetadata:
        """Session metadata."""
        return self._metadata

    @property
    def hardware(self) -> HardwareConfig:
        """Hardware configuration."""
        return self._hardware

    @property
    def flow(self) -> FlowConfig:
        """Flow graph configuration."""
        return self._flow

    @property
    def dashboard(self) -> DashboardConfig:
        """Dashboard configuration."""
        return self._dashboard

    @property
    def camera(self) -> CameraConfig:
        """Camera configuration."""
        return self._camera

    @property
    def zones(self) -> ZoneConfig:
        """Zone configuration."""
        return self._zones

    @property
    def audio(self) -> AudioConfig:
        """Audio recording configuration."""
        return self._audio

    @property
    def custom_device_definitions(self) -> dict[str, Any]:
        """Custom device definitions (id -> definition dict)."""
        return self._custom_device_definitions

    @property
    def flow_function_definitions(self) -> dict[str, Any]:
        """Flow function definitions (id -> definition dict)."""
        return self._flow_function_definitions

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @state.setter
    def state(self, value: SessionState) -> None:
        if value != self._state:
            old_state = self._state
            self._state = value
            logger.debug(f"Session state changed: {old_state} -> {value}")
            for callback in self._state_callbacks:
                try:
                    callback(value)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    @property
    def is_dirty(self) -> bool:
        """Whether session has unsaved changes."""
        return self._dirty

    @property
    def file_path(self) -> Optional[str]:
        """Path to the session file, if saved."""
        return self._file_path

    @property
    def name(self) -> str:
        """Experiment name."""
        return self._metadata.name

    @name.setter
    def name(self, value: str) -> None:
        self._metadata.name = value
        self._mark_dirty()

    def on_state_change(self, callback: Callable[[SessionState], None]) -> None:
        """Register a callback for state changes."""
        self._state_callbacks.append(callback)

    def on_change(self, callback: Callable[[], None]) -> None:
        """Register a callback for any changes."""
        self._change_callbacks.append(callback)

    def _mark_dirty(self) -> None:
        """Mark the session as having unsaved changes."""
        self._dirty = True
        self._metadata.modified_at = datetime.now().isoformat()
        for callback in self._change_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Change callback error: {e}")

    def _mark_clean(self) -> None:
        """Mark the session as saved."""
        self._dirty = False

    # Board management
    def add_board(self, config: BoardConfig) -> None:
        """Add a board to the hardware configuration."""
        self._hardware.boards.append(config)
        self._mark_dirty()

    def remove_board(self, board_id: str) -> None:
        """Remove a board and its associated devices."""
        self._hardware.boards = [b for b in self._hardware.boards if b.id != board_id]
        self._hardware.devices = [d for d in self._hardware.devices if d.board_id != board_id]
        self._mark_dirty()

    def get_board(self, board_id: str) -> Optional[BoardConfig]:
        """Get a board by ID."""
        for board in self._hardware.boards:
            if board.id == board_id:
                return board
        return None

    def update_board(
        self,
        board_id: str,
        port: Optional[str] = None,
        board_type: Optional[str] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update a board's configuration."""
        board = self.get_board(board_id)
        if board is None:
            return False
        if port is not None:
            board.port = port
        if board_type is not None:
            board.board_type = board_type
        if settings is not None:
            board.settings = settings
        self._mark_dirty()
        return True

    # Device management
    def add_device(self, config: DeviceConfig) -> None:
        """Add a device to the hardware configuration."""
        self._hardware.devices.append(config)
        self._mark_dirty()

    def remove_device(self, device_id: str) -> None:
        """Remove a device."""
        self._hardware.devices = [d for d in self._hardware.devices if d.id != device_id]
        # Also remove nodes that reference this device
        self._flow.nodes = [n for n in self._flow.nodes if n.device_id != device_id]
        self._mark_dirty()

    def get_device(self, device_id: str) -> Optional[DeviceConfig]:
        """Get a device by ID."""
        for device in self._hardware.devices:
            if device.id == device_id:
                return device
        return None

    def update_device(
        self,
        device_id: str,
        name: Optional[str] = None,
        pins: Optional[dict[str, int]] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update a device's configuration."""
        device = self.get_device(device_id)
        if device is None:
            return False
        if name is not None:
            device.name = name
        if pins is not None:
            device.pins = pins
        if settings is not None:
            device.settings = settings
        self._mark_dirty()
        return True

    # Node management
    def add_node(self, config: NodeConfig) -> None:
        """Add a node to the flow graph."""
        self._flow.nodes.append(config)
        self._mark_dirty()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its connections."""
        self._flow.nodes = [n for n in self._flow.nodes if n.id != node_id]
        self._flow.connections = [
            c for c in self._flow.connections if c.from_node != node_id and c.to_node != node_id
        ]
        self._mark_dirty()

    def get_node(self, node_id: str) -> Optional[NodeConfig]:
        """Get a node by ID."""
        for node in self._flow.nodes:
            if node.id == node_id:
                return node
        return None

    def update_node_position(self, node_id: str, x: float, y: float) -> None:
        """Update a node's position."""
        node = self.get_node(node_id)
        if node:
            node.position = (x, y)
            self._mark_dirty()

    def update_node_state(self, node_id: str, state: dict[str, Any]) -> None:
        """Update a node's state."""
        node = self.get_node(node_id)
        if node:
            node.state.update(state)
            self._mark_dirty()

    # Connection management
    def add_connection(self, config: ConnectionConfig) -> None:
        """Add a connection between nodes."""
        self._flow.connections.append(config)
        self._mark_dirty()

    def remove_connection(self, connection_id: str) -> None:
        """Remove a connection."""
        self._flow.connections = [c for c in self._flow.connections if c.id != connection_id]
        self._mark_dirty()

    def get_connection(self, connection_id: str) -> Optional[ConnectionConfig]:
        """Get a connection by ID."""
        for conn in self._flow.connections:
            if conn.id == connection_id:
                return conn
        return None

    # Custom device definition management
    def add_custom_device_definition(self, definition_dict: dict[str, Any]) -> None:
        """Add a custom device definition."""
        def_id = definition_dict.get("id")
        if def_id:
            self._custom_device_definitions[def_id] = definition_dict
            self._mark_dirty()

    def remove_custom_device_definition(self, definition_id: str) -> None:
        """Remove a custom device definition."""
        if definition_id in self._custom_device_definitions:
            del self._custom_device_definitions[definition_id]
            self._mark_dirty()

    def get_custom_device_definition(self, definition_id: str) -> Optional[dict[str, Any]]:
        """Get a custom device definition by ID."""
        return self._custom_device_definitions.get(definition_id)

    # Flow function definition management
    def add_flow_function_definition(self, definition_dict: dict[str, Any]) -> None:
        """Add a flow function definition."""
        def_id = definition_dict.get("id")
        if def_id:
            self._flow_function_definitions[def_id] = definition_dict
            self._mark_dirty()

    def remove_flow_function_definition(self, definition_id: str) -> None:
        """Remove a flow function definition."""
        if definition_id in self._flow_function_definitions:
            del self._flow_function_definitions[definition_id]
            self._mark_dirty()

    def get_flow_function_definition(self, definition_id: str) -> Optional[dict[str, Any]]:
        """Get a flow function definition by ID."""
        return self._flow_function_definitions.get(definition_id)

    # Serialization
    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dictionary."""
        result = {
            "metadata": self._metadata.to_dict(),
            "hardware": self._hardware.to_dict(),
            "flow": self._flow.to_dict(),
            "dashboard": self._dashboard.to_dict(),
            "camera": self._camera.to_dict(),
            "zones": self._zones.to_dict(),
            "audio": self._audio.to_dict(),
        }
        # Only include custom definitions if there are any
        if self._custom_device_definitions:
            result["custom_devices"] = self._custom_device_definitions
        if self._flow_function_definitions:
            result["flow_functions"] = self._flow_function_definitions
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSession":
        """Create session from dictionary."""
        session = cls()
        session._metadata = SessionMetadata.from_dict(data.get("metadata", {}))
        session._hardware = HardwareConfig.from_dict(data.get("hardware", {}))
        session._flow = FlowConfig.from_dict(data.get("flow", {}))
        session._dashboard = DashboardConfig.from_dict(data.get("dashboard", {}))
        session._camera = CameraConfig.from_dict(data.get("camera", {}))
        session._zones = ZoneConfig.from_dict(data.get("zones", {}))
        session._audio = AudioConfig.from_dict(data.get("audio", {}))
        # Load custom definitions
        session._custom_device_definitions = data.get("custom_devices", {})
        session._flow_function_definitions = data.get("flow_functions", {})
        return session

    def to_json(self, indent: int = 2) -> str:
        """Serialize session to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentSession":
        """Create session from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, file_path: Optional[str] = None) -> str:
        """
        Save session to file.

        Args:
            file_path: Path to save to (uses existing path if None)

        Returns:
            Path to saved file
        """
        if file_path is None:
            if self._file_path is None:
                raise ValueError("No file path specified")
            file_path = self._file_path

        with open(file_path, "w") as f:
            f.write(self.to_json())

        self._file_path = file_path
        self._mark_clean()
        logger.info(f"Session saved to {file_path}")
        return file_path

    @classmethod
    def load(cls, file_path: str) -> "ExperimentSession":
        """
        Load session from file.

        Args:
            file_path: Path to load from

        Returns:
            Loaded session
        """
        with open(file_path) as f:
            session = cls.from_json(f.read())

        session._file_path = file_path
        session._mark_clean()
        logger.info(f"Session loaded from {file_path}")
        return session

    def clear(self) -> None:
        """Clear the session to a new state."""
        self._metadata = SessionMetadata()
        self._hardware = HardwareConfig()
        self._flow = FlowConfig()
        self._dashboard = DashboardConfig()
        self._camera = CameraConfig()
        self._zones = ZoneConfig()
        self._audio = AudioConfig()
        self._custom_device_definitions = {}
        self._flow_function_definitions = {}
        self._state = SessionState.IDLE
        self._file_path = None
        self._mark_dirty()
