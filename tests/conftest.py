"""
GLIDER Test Configuration and Fixtures.

Provides shared fixtures for unit and integration tests.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Provide a factory for creating temporary files."""
    created_files = []

    def _create_file(name: str, content: str = "") -> Path:
        path = temp_dir / name
        path.write_text(content)
        created_files.append(path)
        return path

    yield _create_file


# =============================================================================
# Mock Hardware Fixtures
# =============================================================================


@pytest.fixture
def mock_board():
    """Provide a mock board with common capabilities."""
    board = MagicMock()
    board.id = "mock_board_1"
    board.name = "Mock Board"
    board.is_connected = True
    board.state = MagicMock()
    board.state.name = "CONNECTED"

    # Async methods
    board.connect = AsyncMock(return_value=True)
    board.disconnect = AsyncMock()
    board.digital_write = AsyncMock()
    board.digital_read = AsyncMock(return_value=False)
    board.analog_read = AsyncMock(return_value=512)
    board.pwm_write = AsyncMock()
    board.servo_write = AsyncMock()
    board.initialize = AsyncMock()
    board.shutdown = AsyncMock()

    # Capabilities
    board.capabilities = MagicMock()
    board.capabilities.digital_pins = list(range(14))
    board.capabilities.analog_pins = list(range(6))
    board.capabilities.pwm_pins = [3, 5, 6, 9, 10, 11]
    board.capabilities.servo_pins = [9, 10]
    board.capabilities.has_i2c = True
    board.capabilities.has_spi = True

    return board


@pytest.fixture
def mock_device(mock_board):
    """Provide a mock device."""
    device = MagicMock()
    device.id = "mock_device_1"
    device.name = "Mock Device"
    device.device_type = "DigitalOutput"
    device.board = mock_board
    device.is_initialized = True
    device.pin = 13

    # Async methods
    device.initialize = AsyncMock()
    device.shutdown = AsyncMock()
    device.write = AsyncMock()
    device.read = AsyncMock(return_value=0)

    return device


@pytest.fixture
def mock_hardware_manager(mock_board, mock_device):
    """Provide a mock hardware manager."""
    manager = MagicMock()
    manager.boards = {"mock_board_1": mock_board}
    manager.devices = {"mock_device_1": mock_device}

    # Async methods
    manager.create_board = AsyncMock(return_value=mock_board)
    manager.create_device = AsyncMock(return_value=mock_device)
    manager.connect_all = AsyncMock()
    manager.initialize_all_devices = AsyncMock()
    manager.emergency_stop = AsyncMock()
    manager.shutdown = AsyncMock()

    # Sync methods
    manager.get_board = MagicMock(return_value=mock_board)
    manager.get_device = MagicMock(return_value=mock_device)

    return manager


# =============================================================================
# Mock Vision Fixtures
# =============================================================================


@pytest.fixture
def mock_frame():
    """Provide a mock camera frame (640x480 BGR)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_frame_with_content():
    """Provide a mock frame with some content for CV testing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a white rectangle
    frame[100:200, 100:200] = [255, 255, 255]
    # Add a red circle area
    frame[300:350, 300:350] = [0, 0, 255]
    return frame


@pytest.fixture
def mock_camera_manager():
    """Provide a mock camera manager."""
    manager = MagicMock()
    manager.is_connected = False
    manager.is_streaming = False
    manager.current_fps = 30.0
    manager.settings = MagicMock()
    manager.settings.camera_index = 0
    manager.settings.resolution = (640, 480)
    manager.settings.fps = 30

    manager.connect = MagicMock(return_value=True)
    manager.disconnect = MagicMock()
    manager.start_streaming = MagicMock(return_value=True)
    manager.stop_streaming = MagicMock()
    manager.on_frame = MagicMock()
    manager.enumerate_cameras = MagicMock(return_value=[])

    return manager


@pytest.fixture
def mock_cv_processor():
    """Provide a mock CV processor."""
    processor = MagicMock()
    processor.is_initialized = True
    processor.settings = MagicMock()
    processor.settings.enabled = True

    processor.initialize = MagicMock()
    processor.process_frame = MagicMock(return_value=([], [], None))
    processor.draw_overlays = MagicMock(side_effect=lambda f, *args: f)

    return processor


# =============================================================================
# Mock Flow Engine Fixtures
# =============================================================================


@pytest.fixture
def mock_flow_engine():
    """Provide a mock flow engine."""
    engine = MagicMock()
    engine.is_running = False
    engine.is_paused = False
    engine.nodes = {}
    engine.connections = []

    # Async methods
    engine.initialize = AsyncMock()
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    engine.pause = AsyncMock()
    engine.resume = AsyncMock()

    # Sync methods
    engine.create_node = MagicMock()
    engine.delete_node = MagicMock()
    engine.connect_nodes = MagicMock()
    engine.disconnect_nodes = MagicMock()
    engine.load_from_session = MagicMock()
    engine.clear = MagicMock()

    return engine


# =============================================================================
# Mock Data Recorder Fixtures
# =============================================================================


@pytest.fixture
def mock_data_recorder(temp_dir):
    """Provide a mock data recorder."""
    recorder = MagicMock()
    recorder.is_recording = False
    recorder.file_path = temp_dir / "test_data.csv"
    recorder.sample_count = 0

    # Async methods
    recorder.start = AsyncMock()
    recorder.stop = AsyncMock()
    recorder.record_sample = AsyncMock()

    return recorder


# =============================================================================
# Mock Core Fixtures
# =============================================================================


@pytest.fixture
def mock_session():
    """Provide a mock experiment session."""
    from glider.core.types import SessionState

    session = MagicMock()
    session.name = "Test Session"
    session.state = SessionState.IDLE
    session.is_dirty = False
    session.metadata = MagicMock()
    session.metadata.name = "Test Session"
    session.metadata.description = "Test description"
    session.boards = {}
    session.devices = {}
    session.flow_config = {}
    session.dashboard_config = {}

    session.mark_dirty = MagicMock()
    session.mark_clean = MagicMock()
    session.to_dict = MagicMock(return_value={})

    return session


@pytest.fixture
def mock_core(mock_hardware_manager, mock_flow_engine, mock_session):
    """Provide a mock GliderCore."""
    core = MagicMock()
    core.hardware_manager = mock_hardware_manager
    core.flow_engine = mock_flow_engine
    core.session = mock_session
    core.is_running = False

    # Async methods
    core.initialize = AsyncMock()
    core.start_experiment = AsyncMock()
    core.stop_experiment = AsyncMock()
    core.pause_experiment = AsyncMock()
    core.resume_experiment = AsyncMock()
    core.emergency_stop = AsyncMock()
    core.shutdown = AsyncMock()

    return core


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_board_config() -> dict[str, Any]:
    """Provide sample board configuration."""
    return {
        "id": "arduino_1",
        "driver": "arduino",
        "name": "Arduino Uno",
        "port": "/dev/ttyUSB0",
        "settings": {},
    }


@pytest.fixture
def sample_device_config() -> dict[str, Any]:
    """Provide sample device configuration."""
    return {
        "id": "led_1",
        "board_id": "arduino_1",
        "device_type": "DigitalOutput",
        "name": "Status LED",
        "pin": 13,
        "settings": {},
    }


@pytest.fixture
def sample_node_config() -> dict[str, Any]:
    """Provide sample node configuration."""
    return {"id": "node_1", "type": "Delay", "position": [100, 100], "state": {"duration": 1.0}}


@pytest.fixture
def sample_connection_config() -> dict[str, Any]:
    """Provide sample connection configuration."""
    return {
        "source_node": "node_1",
        "source_port": "exec_out",
        "target_node": "node_2",
        "target_port": "exec_in",
    }


@pytest.fixture
def sample_experiment_schema() -> dict[str, Any]:
    """Provide a complete sample experiment schema."""
    return {
        "schema_version": "1.0",
        "metadata": {
            "name": "Test Experiment",
            "description": "A test experiment",
            "author": "Test Author",
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
        },
        "hardware": {"boards": [], "devices": []},
        "flow": {"nodes": [], "connections": []},
        "dashboard": {"widgets": [], "layout": {}},
        "custom_devices": [],
        "flow_functions": [],
    }


# =============================================================================
# Calibration Fixtures
# =============================================================================


@pytest.fixture
def sample_calibration_line():
    """Provide a sample calibration line."""
    return {
        "start_x": 0.1,
        "start_y": 0.5,
        "end_x": 0.9,
        "end_y": 0.5,
        "length": 100.0,
        "unit": "mm",
        "color": [0, 255, 0],
    }


@pytest.fixture
def sample_zone_config():
    """Provide a sample zone configuration."""
    return {
        "id": "zone_1",
        "name": "Test Zone",
        "shape": "rectangle",
        "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        "color": [255, 0, 0],
        "track_entry": True,
        "track_exit": True,
        "track_dwell": True,
    }


# =============================================================================
# PyQt Fixtures (for GUI tests)
# =============================================================================


@pytest.fixture
def qtbot_or_skip(request):
    """
    Provide qtbot if pytest-qt is available, otherwise skip test.

    Usage:
        def test_gui_thing(qtbot_or_skip):
            if qtbot_or_skip is None:
                pytest.skip("pytest-qt not available")
            # ... rest of test
    """
    import importlib.util

    if importlib.util.find_spec("pytestqt") is None:
        return None
    try:
        # Get qtbot from pytest-qt plugin
        return request.getfixturevalue("qtbot")
    except Exception:
        return None


# =============================================================================
# Utility Functions
# =============================================================================


def create_mock_node(node_type: str, node_id: str = "test_node"):
    """Create a mock node for testing."""
    node = MagicMock()
    node.id = node_id
    node.name = node_type
    node.definition = MagicMock()
    node.definition.name = node_type
    node.definition.inputs = []
    node.definition.outputs = []
    node._inputs = []
    node._outputs = []
    node._enabled = True
    node._error = None

    node.update_event = MagicMock()
    node.get_state = MagicMock(return_value={})
    node.set_state = MagicMock()

    return node


def create_async_context_manager(return_value=None):
    """Create a mock async context manager."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm
