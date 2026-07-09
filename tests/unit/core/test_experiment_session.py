"""
Tests for glider.core.experiment_session module.

Tests the ExperimentSession state machine and session management.
"""

from glider.core.experiment_session import (
    BoardConfig,
    DeviceConfig,
    ExperimentSession,
    NodeConfig,
    SessionMetadata,
    SessionState,
)


class TestSessionMetadata:
    """Tests for SessionMetadata dataclass."""

    def test_default_values(self):
        """Test SessionMetadata default values."""
        metadata = SessionMetadata()

        assert metadata.name == "Untitled Experiment"
        assert metadata.description == ""
        assert metadata.author == ""
        assert metadata.version == "1.0.0"

    def test_custom_name(self):
        """Test SessionMetadata with custom name."""
        metadata = SessionMetadata(name="My Experiment")

        assert metadata.name == "My Experiment"

    def test_to_dict(self):
        """Test SessionMetadata serialization."""
        metadata = SessionMetadata(
            name="Test Session", description="A test description", author="Test Author"
        )
        data = metadata.to_dict()

        assert data["name"] == "Test Session"
        assert data["description"] == "A test description"
        assert data["author"] == "Test Author"
        assert "id" in data
        assert "created_at" in data

    def test_from_dict(self):
        """Test SessionMetadata deserialization."""
        data = {
            "name": "Loaded Session",
            "description": "Loaded description",
            "author": "Loaded Author",
            "version": "1.0.0",
        }
        metadata = SessionMetadata.from_dict(data)

        assert metadata.name == "Loaded Session"
        assert metadata.description == "Loaded description"
        assert metadata.author == "Loaded Author"


class TestBoardConfig:
    """Tests for BoardConfig dataclass."""

    def test_creation(self):
        """Test BoardConfig creation."""
        config = BoardConfig(id="board_1", driver_type="arduino")

        assert config.id == "board_1"
        assert config.driver_type == "arduino"

    def test_optional_fields(self):
        """Test BoardConfig optional fields."""
        config = BoardConfig(
            id="board_1",
            driver_type="arduino",
            port="/dev/ttyUSB0",
            board_type="uno",
            auto_reconnect=True,
        )

        assert config.port == "/dev/ttyUSB0"
        assert config.board_type == "uno"
        assert config.auto_reconnect is True

    def test_to_dict(self):
        """Test BoardConfig serialization."""
        config = BoardConfig(
            id="board_1", driver_type="arduino", port="/dev/ttyUSB0", settings={"baud_rate": 115200}
        )
        data = config.to_dict()

        assert data["id"] == "board_1"
        assert data["driver_type"] == "arduino"
        assert data["port"] == "/dev/ttyUSB0"

    def test_from_dict(self):
        """Test BoardConfig deserialization."""
        data = {"id": "board_2", "driver_type": "raspberry_pi", "port": None, "settings": {}}
        config = BoardConfig.from_dict(data)

        assert config.id == "board_2"
        assert config.driver_type == "raspberry_pi"


class TestDeviceConfig:
    """Tests for DeviceConfig dataclass."""

    def test_creation(self):
        """Test DeviceConfig creation."""
        config = DeviceConfig(
            id="device_1",
            device_type="DigitalOutput",
            name="LED",
            board_id="board_1",
            pins={"signal": 13},
        )

        assert config.id == "device_1"
        assert config.board_id == "board_1"
        assert config.device_type == "DigitalOutput"
        assert config.name == "LED"
        assert config.pins == {"signal": 13}

    def test_to_dict(self):
        """Test DeviceConfig serialization."""
        config = DeviceConfig(
            id="device_1",
            device_type="DigitalOutput",
            name="LED",
            board_id="board_1",
            pins={"signal": 13},
            settings={"inverted": False},
        )
        data = config.to_dict()

        assert data["id"] == "device_1"
        assert data["pins"] == {"signal": 13}

    def test_from_dict(self):
        """Test DeviceConfig deserialization."""
        data = {
            "id": "sensor_1",
            "board_id": "board_1",
            "device_type": "AnalogInput",
            "name": "Temperature Sensor",
            "pins": {"analog": 0},
            "settings": {"smoothing": True},
        }
        config = DeviceConfig.from_dict(data)

        assert config.id == "sensor_1"
        assert config.device_type == "AnalogInput"
        assert config.pins == {"analog": 0}


class TestExperimentSession:
    """Tests for ExperimentSession class."""

    def test_session_init(self):
        """Test ExperimentSession initialization."""
        session = ExperimentSession()

        assert session.state == SessionState.IDLE
        assert session.name == "Untitled Experiment"
        assert session.metadata.version == "1.0.0"

    def test_session_set_name(self):
        """Test setting session name."""
        session = ExperimentSession()
        session.name = "My Experiment"

        assert session.name == "My Experiment"
        assert session.is_dirty is True

    def test_session_state_transitions(self):
        """Test session state transitions."""
        session = ExperimentSession()

        session.state = SessionState.RUNNING
        assert session.state == SessionState.RUNNING

        session.state = SessionState.PAUSED
        assert session.state == SessionState.PAUSED

        session.state = SessionState.IDLE
        assert session.state == SessionState.IDLE

    def test_all_state_values(self):
        """Test all SessionState values can be set."""
        session = ExperimentSession()

        for state in SessionState:
            session.state = state
            assert session.state == state

    def test_is_dirty_on_changes(self):
        """Test is_dirty flag on changes."""
        session = ExperimentSession()

        # Session may start dirty or clean depending on implementation
        session._mark_clean()
        assert session.is_dirty is False

        # Adding board should mark dirty
        session.add_board(BoardConfig(id="board_1", driver_type="arduino"))
        assert session.is_dirty is True

    def test_add_board(self):
        """Test adding a board to the session."""
        session = ExperimentSession()
        config = BoardConfig(id="board_1", driver_type="arduino")

        session.add_board(config)

        assert len(session.hardware.boards) == 1
        assert session.hardware.boards[0] == config

    def test_remove_board(self):
        """Test removing a board from the session."""
        session = ExperimentSession()
        config = BoardConfig(id="board_1", driver_type="arduino")
        session.add_board(config)

        session.remove_board("board_1")

        assert len(session.hardware.boards) == 0

    def test_add_device(self):
        """Test adding a device to the session."""
        session = ExperimentSession()
        config = DeviceConfig(
            id="device_1",
            device_type="DigitalOutput",
            name="LED",
            board_id="board_1",
            pins={"signal": 13},
        )

        session.add_device(config)

        assert len(session.hardware.devices) == 1
        assert session.hardware.devices[0] == config

    def test_remove_device(self):
        """Test removing a device from the session."""
        session = ExperimentSession()
        config = DeviceConfig(
            id="device_1",
            device_type="DigitalOutput",
            name="LED",
            board_id="board_1",
            pins={"signal": 13},
        )
        session.add_device(config)

        session.remove_device("device_1")

        assert len(session.hardware.devices) == 0

    def test_get_board(self):
        """Test getting a board by ID."""
        session = ExperimentSession()
        config = BoardConfig(id="board_1", driver_type="arduino")
        session.add_board(config)

        result = session.get_board("board_1")
        assert result == config

        result = session.get_board("nonexistent")
        assert result is None

    def test_get_device(self):
        """Test getting a device by ID."""
        session = ExperimentSession()
        config = DeviceConfig(
            id="device_1",
            device_type="DigitalOutput",
            name="LED",
            board_id="board_1",
            pins={"signal": 13},
        )
        session.add_device(config)

        result = session.get_device("device_1")
        assert result == config

        result = session.get_device("nonexistent")
        assert result is None

    def test_to_dict(self):
        """Test session serialization."""
        session = ExperimentSession()
        session.name = "Test Session"
        session.add_board(BoardConfig(id="board_1", driver_type="arduino"))
        session.add_device(
            DeviceConfig(
                id="led_1",
                device_type="DigitalOutput",
                name="LED",
                board_id="board_1",
                pins={"signal": 13},
            )
        )

        data = session.to_dict()

        assert "metadata" in data
        assert "hardware" in data
        assert "flow" in data
        assert "dashboard" in data

    def test_from_dict(self):
        """Test session deserialization."""
        data = {
            "metadata": {
                "name": "Loaded Session",
                "description": "Test",
                "author": "Author",
                "version": "1.0.0",
            },
            "hardware": {
                "boards": [{"id": "b1", "driver_type": "arduino"}],
                "devices": [
                    {
                        "id": "d1",
                        "board_id": "b1",
                        "device_type": "DigitalOutput",
                        "name": "Device",
                        "pins": {"signal": 13},
                    }
                ],
            },
            "flow": {"nodes": [], "connections": []},
            "dashboard": {"widgets": [], "layout": "vertical"},
        }

        session = ExperimentSession.from_dict(data)

        assert session.name == "Loaded Session"
        assert len(session.hardware.boards) == 1
        assert len(session.hardware.devices) == 1

    def test_clear(self):
        """Test clearing a session."""
        session = ExperimentSession()
        session.name = "Test"
        session.add_board(BoardConfig(id="b1", driver_type="arduino"))
        session.add_device(
            DeviceConfig(
                id="d1",
                device_type="DigitalOutput",
                name="Device",
                board_id="b1",
                pins={"signal": 13},
            )
        )

        session.clear()

        assert len(session.hardware.boards) == 0
        assert len(session.hardware.devices) == 0

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = ExperimentSession()
        original.name = "Roundtrip Test"
        original.metadata.description = "Testing roundtrip"
        original.add_board(BoardConfig(id="board_1", driver_type="arduino", port="/dev/ttyUSB0"))
        original.add_device(
            DeviceConfig(
                id="led_1",
                device_type="DigitalOutput",
                name="Status LED",
                board_id="board_1",
                pins={"signal": 13},
            )
        )

        data = original.to_dict()
        restored = ExperimentSession.from_dict(data)

        assert restored.name == original.name
        assert len(restored.hardware.boards) == len(original.hardware.boards)
        assert len(restored.hardware.devices) == len(original.hardware.devices)


class TestSessionStateEnum:
    """Tests for SessionState enum."""

    def test_state_values(self):
        """Test SessionState enum values."""
        assert SessionState.IDLE is not None
        assert SessionState.INITIALIZING is not None
        assert SessionState.READY is not None
        assert SessionState.RUNNING is not None
        assert SessionState.PAUSED is not None
        assert SessionState.STOPPING is not None
        assert SessionState.ERROR is not None

    def test_all_states_distinct(self):
        """Test that all states have distinct values."""
        states = list(SessionState)
        values = [s.value for s in states]
        assert len(values) == len(set(values))


class TestRemoveBoardFlowCleanup:
    """remove_board must mirror remove_device's flow-node cleanup."""

    def test_remove_board_removes_nodes_bound_to_its_devices(self):
        session = ExperimentSession()
        session.add_board(BoardConfig(id="b1", driver_type="mock"))
        session.add_board(BoardConfig(id="b2", driver_type="mock"))
        session.add_device(
            DeviceConfig(
                id="d1",
                device_type="DigitalOutput",
                name="LED1",
                board_id="b1",
                pins={"output": 13},
            )
        )
        session.add_device(
            DeviceConfig(
                id="d2",
                device_type="DigitalOutput",
                name="LED2",
                board_id="b2",
                pins={"output": 12},
            )
        )
        session.add_node(NodeConfig(id="n1", node_type="DigitalWrite", device_id="d1"))
        session.add_node(NodeConfig(id="n2", node_type="DigitalWrite", device_id="d2"))
        session.add_node(NodeConfig(id="n3", node_type="Delay"))

        session.remove_board("b1")

        # Board and its devices are gone
        assert session.get_board("b1") is None
        assert all(d.board_id != "b1" for d in session.hardware.devices)

        # Nodes bound to the removed board's devices are gone; others remain
        node_ids = {n.id for n in session.flow.nodes}
        assert "n1" not in node_ids, "Node bound to removed board's device left dangling"
        assert "n2" in node_ids
        assert "n3" in node_ids
