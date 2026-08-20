"""
Round-trip regression test for the serializer's save → load path.

Before the 1.0 release-prep pass, the serializer's apply path called APIs
that did not exist: ``add_board(board_type=…)`` (real kwarg is
``driver_type``), ``create_node(node_class, …)`` (first arg is
``node_id: str``), ``flow_engine.connect(…)`` (no such method —
``create_connection`` and ``connect_nodes`` are the real ones), and the
save path iterated ``flow_engine.connections.items()`` (no such property
— the real attribute is ``_connections: list[dict]``, accessed via
``get_connections()``). The code was undetected because the schema tests
exercised dataclass validation but no end-to-end save/load test existed.

These tests do not require real hardware; they spin up a
``HardwareManager`` with a registered mock driver and put the system
through a real save → fresh-load → save cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glider.core.experiment_session import ExperimentSession
from glider.core.flow_engine import FlowEngine
from glider.core.hardware_manager import HardwareManager
from glider.hal.mock_board import MockBoard
from glider.nodes.control_nodes import register_control_nodes
from glider.nodes.experiment_nodes import register_experiment_nodes
from glider.nodes.flow_function_nodes import register_flow_function_nodes
from glider.nodes.hardware import register_hardware_nodes
from glider.nodes.interface import register_interface_nodes
from glider.nodes.interface.audio_nodes import register_audio_nodes
from glider.nodes.interface.video_nodes import register_video_nodes
from glider.nodes.logic import (
    register_comparison_nodes,
    register_logic_control_nodes,
    register_math_nodes,
)
from glider.nodes.logic.flow_nodes import register_logic_nodes
from glider.nodes.vision.zone_nodes import register_zone_nodes
from glider.serialization.serializer import ExperimentSerializer


def _make_engine_with_all_nodes(hardware_manager=None) -> FlowEngine:
    engine = FlowEngine(hardware_manager)
    register_experiment_nodes(engine)
    register_control_nodes(engine)
    register_logic_nodes(engine)
    register_flow_function_nodes(engine)
    register_zone_nodes(engine)
    register_audio_nodes(engine)
    register_video_nodes(engine)
    register_interface_nodes(engine)
    register_hardware_nodes(engine)
    register_math_nodes(engine)
    register_comparison_nodes(engine)
    register_logic_control_nodes(engine)
    return engine


@pytest.fixture
def hardware_manager_with_mock():
    """A HardwareManager wired with a registered mock driver."""
    hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    return hm


@pytest.fixture
def flow_engine_with_nodes():
    return _make_engine_with_all_nodes()


@pytest.fixture
def serializer():
    return ExperimentSerializer()


def test_save_with_no_state_does_not_raise(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """Empty-session save should not crash."""
    session = ExperimentSession()
    out = tmp_path / "empty.glider"
    serializer.save(
        out,
        session=session,
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_with_board_and_node_does_not_raise(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """Save with one board + one node + one connection — the path that
    historically crashed at every step.
    """
    hardware_manager_with_mock.add_board(board_id="board1", driver_type="mock", port=None)

    # Two nodes so we have a connection to serialize.
    flow_engine_with_nodes.create_node(node_id="start1", node_type="StartExperiment")
    flow_engine_with_nodes.create_node(node_id="end1", node_type="EndExperiment")
    flow_engine_with_nodes.create_connection(
        connection_id="c1",
        from_node_id="start1",
        from_output=0,
        to_node_id="end1",
        to_input=0,
        connection_type="exec",
    )

    session = ExperimentSession()
    out = tmp_path / "small.glider"
    serializer.save(
        out,
        session=session,
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )
    assert out.exists()


def test_round_trip_preserves_structure(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """Build a session, save, reload into a fresh engine, assert the
    structure (boards / nodes / connections) round-trips.
    """
    # Build initial state
    hardware_manager_with_mock.add_board(board_id="board1", driver_type="mock", port=None)
    flow_engine_with_nodes.create_node(node_id="start1", node_type="StartExperiment")
    flow_engine_with_nodes.create_node(node_id="end1", node_type="EndExperiment")
    flow_engine_with_nodes.create_connection(
        connection_id="c1",
        from_node_id="start1",
        from_output=0,
        to_node_id="end1",
        to_input=0,
        connection_type="exec",
    )

    session = ExperimentSession()
    session.name = "round-trip-test"
    out = tmp_path / "rt.glider"
    serializer.save(
        out,
        session=session,
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )

    # Now load into fresh containers
    fresh_session = ExperimentSession()
    fresh_engine = _make_engine_with_all_nodes()
    fresh_hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)

    schema = serializer.load(out)
    serializer.apply_to_session(
        schema,
        session=fresh_session,
        flow_engine=fresh_engine,
        hardware_manager=fresh_hm,
    )

    # Boards round-tripped
    assert (
        "board1" in fresh_hm.boards
    ), f"Board 'board1' did not round-trip. Loaded boards: {list(fresh_hm.boards)}"

    # Nodes round-tripped
    loaded_node_ids = set(fresh_engine.nodes.keys())
    assert {"start1", "end1"}.issubset(
        loaded_node_ids
    ), f"Expected nodes 'start1' and 'end1', got {loaded_node_ids}"

    # Connection round-tripped
    loaded_conns = fresh_engine.get_connections()
    assert any(
        c["from_node"] == "start1" and c["to_node"] == "end1" for c in loaded_conns
    ), f"start1 -> end1 connection lost on round-trip. Got: {loaded_conns}"


def test_node_state_round_trip(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """Per-node state (set via the node's own _state dict) must survive
    save → load. The pre-fix code dropped every node-local parameter
    silently because ``_extract_node_properties`` iterated a
    ``property_names`` attribute that no node class defined.
    """
    # DelayNode stores its duration in self._state — a canonical test
    flow_engine_with_nodes.create_node(node_id="start1", node_type="StartExperiment")
    flow_engine_with_nodes.create_node(
        node_id="delay1",
        node_type="Delay",
        state={"duration_seconds": 4.2, "use_input": False},
    )

    session = ExperimentSession()
    out = tmp_path / "state.glider"
    serializer.save(
        out,
        session=session,
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )

    fresh_engine = _make_engine_with_all_nodes()
    fresh_hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    fresh_session = ExperimentSession()

    schema = serializer.load(out)
    serializer.apply_to_session(
        schema,
        session=fresh_session,
        flow_engine=fresh_engine,
        hardware_manager=fresh_hm,
    )

    loaded_delay = fresh_engine.nodes.get("delay1")
    assert loaded_delay is not None, "delay1 node was not loaded back"
    state = loaded_delay.get_state()
    assert (
        state.get("duration_seconds") == 4.2
    ), f"DelayNode duration_seconds lost on round-trip. State: {state}"
    assert (
        state.get("use_input") is False
    ), f"DelayNode use_input lost on round-trip. State: {state}"


def test_device_round_trip_preserves_pins_board_and_settings(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """A device's pin map, owning board, name, and settings must survive
    save -> fresh-load. The broken code read attributes that don't exist
    on BaseDevice (.pin/.board_id/.settings) so every device saved as
    pin=0 / board_id="" and reload raised BoardNotFoundError."""
    hardware_manager_with_mock.add_board(board_id="board1", driver_type="mock", port=None)
    hardware_manager_with_mock.add_device(
        device_id="led1",
        device_type="DigitalOutput",
        board_id="board1",
        pin=13,
        name="Status LED",
        initial_state=False,
    )

    out = tmp_path / "dev.glider"
    serializer.save(
        out,
        session=ExperimentSession(),
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )

    fresh_hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    fresh_engine = _make_engine_with_all_nodes()
    schema = serializer.load(out)
    serializer.apply_to_session(
        schema,
        session=ExperimentSession(),
        flow_engine=fresh_engine,
        hardware_manager=fresh_hm,
    )

    device = fresh_hm.devices.get("led1")
    assert device is not None, f"Device lost on round-trip. Loaded: {list(fresh_hm.devices)}"
    assert device.pins == {"output": 13}
    assert device.board.id == "board1"
    assert device.name == "Status LED"
    assert device._config.settings.get("initial_state") is False


def test_legacy_single_pin_device_file_still_loads(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    flow_engine_with_nodes,
):
    """Old .glider files store a single int `pin` per device (no `pins`
    dict). They must still load, with the pin mapped to the conventional
    pin name for the device type (DigitalOutput -> "output")."""
    legacy = {
        "schema_version": "1.0.0",
        "metadata": {"name": "Legacy"},
        "hardware": {
            "boards": [{"id": "board1", "type": "mock", "port": None, "settings": {}}],
            "devices": [
                {
                    "id": "led1",
                    "type": "DigitalOutput",
                    "board_id": "board1",
                    "pin": 13,
                    "name": "LED",
                    "settings": {},
                }
            ],
        },
        "flow": {"nodes": [], "connections": []},
        "dashboard": {"widgets": []},
    }
    path = tmp_path / "legacy.glider"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    fresh_hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    schema = serializer.load(path)
    serializer.apply_to_session(
        schema,
        session=ExperimentSession(),
        flow_engine=flow_engine_with_nodes,
        hardware_manager=fresh_hm,
    )

    device = fresh_hm.devices.get("led1")
    assert device is not None
    assert device.pins == {"output": 13}


def test_apply_to_session_populates_session_model_so_save_as_keeps_data(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """apply_to_session must sync schema hardware/flow into the session's
    own dataclasses (what ExperimentSession.save() serializes). Otherwise
    a Save As after loading a .glider file writes empty sections."""
    hardware_manager_with_mock.add_board(board_id="board1", driver_type="mock", port=None)
    hardware_manager_with_mock.add_device(
        device_id="led1", device_type="DigitalOutput", board_id="board1", pin=13
    )
    flow_engine_with_nodes.create_node(node_id="start1", node_type="StartExperiment")
    flow_engine_with_nodes.create_node(node_id="end1", node_type="EndExperiment")
    flow_engine_with_nodes.create_connection(
        connection_id="c1",
        from_node_id="start1",
        from_output=0,
        to_node_id="end1",
        to_input=0,
        connection_type="exec",
    )

    out = tmp_path / "sync.glider"
    serializer.save(
        out,
        session=ExperimentSession(),
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )

    fresh_session = ExperimentSession()
    fresh_hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    fresh_engine = _make_engine_with_all_nodes()
    schema = serializer.load(out)
    serializer.apply_to_session(
        schema,
        session=fresh_session,
        flow_engine=fresh_engine,
        hardware_manager=fresh_hm,
    )

    # Session model mirrors what went into the managers
    assert [b.id for b in fresh_session.hardware.boards] == ["board1"]
    assert len(fresh_session.hardware.devices) == 1
    device = fresh_session.hardware.devices[0]
    assert device.id == "led1"
    assert device.board_id == "board1"
    assert device.pins == {"output": 13}
    node_ids = {n.id for n in fresh_session.flow.nodes}
    assert {"start1", "end1"}.issubset(node_ids)
    assert any(
        c.from_node == "start1" and c.to_node == "end1" for c in fresh_session.flow.connections
    )

    # A subsequent Save As must not wipe the loaded data
    resave = tmp_path / "resave.glider"
    fresh_session.save(str(resave))
    data = json.loads(resave.read_text(encoding="utf-8"))
    assert data["hardware"]["boards"], "Save As wrote an empty boards list"
    assert data["hardware"]["devices"], "Save As wrote an empty devices list"
    assert data["flow"]["nodes"], "Save As wrote an empty nodes list"
    assert data["flow"]["connections"], "Save As wrote an empty connections list"


def test_zero_pin_devices_round_trip(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """Devices with no GPIO pins (e.g. BLEWrite, required_pins == []) save
    with pins == {} and must reload through the multi-pin path. Routing
    them down the legacy single-pin branch calls add_device(pin=None),
    which synthesizes {"pin": None} — and a SECOND zero-pin device then
    collides on the phantom pin ("Pin None is already claimed")."""
    hardware_manager_with_mock.add_board(board_id="board1", driver_type="mock", port=None)
    hardware_manager_with_mock.add_device_multi_pin(
        device_id="ble1",
        device_type="BLEWrite",
        board_id="board1",
        pins={},
        name="Stim A",
        address="AA:BB:CC:DD:EE:01",
        char_uuid="0000ffe1-0000-1000-8000-00805f9b34fb",
    )
    hardware_manager_with_mock.add_device_multi_pin(
        device_id="ble2",
        device_type="BLEWrite",
        board_id="board1",
        pins={},
        name="Stim B",
        address="AA:BB:CC:DD:EE:02",
        char_uuid="0000ffe1-0000-1000-8000-00805f9b34fb",
    )

    out = tmp_path / "ble.glider"
    serializer.save(
        out,
        session=ExperimentSession(),
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )

    fresh_hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    fresh_engine = _make_engine_with_all_nodes()
    schema = serializer.load(out)
    serializer.apply_to_session(
        schema,
        session=ExperimentSession(),
        flow_engine=fresh_engine,
        hardware_manager=fresh_hm,
    )

    for device_id in ("ble1", "ble2"):
        device = fresh_hm.devices.get(device_id)
        assert device is not None, f"{device_id} lost on round-trip: {list(fresh_hm.devices)}"
        assert device.pins == {}, f"{device_id} grew phantom pins: {device.pins}"
    assert fresh_hm.devices["ble1"]._config.settings["address"] == "AA:BB:CC:DD:EE:01"
    assert fresh_hm.devices["ble2"]._config.settings["address"] == "AA:BB:CC:DD:EE:02"


def _bound_setup(serializer_, tmp_path_):
    """A saved file with one hardware node bound to one device."""
    hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    hm.add_board(board_id="board1", driver_type="mock", port=None)
    hm.add_device_multi_pin("led1", "DigitalOutput", "board1", pins={"output": 5}, name="LED")

    engine = _make_engine_with_all_nodes(hm)
    node = engine.create_node(
        node_id="out1", node_type="Output", position=(10.0, 20.0), device_id="led1"
    )
    assert node.device is not None, "precondition: create_node should have bound the device"

    out = tmp_path_ / "bound.glider"
    serializer_.save(out, session=ExperimentSession(), flow_engine=engine, hardware_manager=hm)
    return out


def _reload(serializer_, path):
    hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    engine = _make_engine_with_all_nodes(hm)
    session = ExperimentSession()
    serializer_.apply_to_session(
        serializer_.load(path), session=session, flow_engine=engine, hardware_manager=hm
    )
    return session, engine


def test_node_device_binding_survives_the_runtime_round_trip(
    tmp_path: Path, serializer: ExperimentSerializer
):
    """A hardware node bound to a device must still be bound after reload --
    otherwise every Output/Input/DeviceAction node silently comes back unbound
    and the experiment does nothing until each is re-picked by hand."""
    path = _bound_setup(serializer, tmp_path)

    _session, engine = _reload(serializer, path)

    node = engine.get_node("out1")
    assert node is not None, "node lost on round-trip"
    assert node.device is not None, "node came back unbound"
    assert node.device.name == "LED"


def test_node_device_binding_survives_into_the_session_model(
    tmp_path: Path, serializer: ExperimentSerializer
):
    """The session model is what the properties panel reads and what File >
    Save As re-serializes, so the binding has to land there too."""
    path = _bound_setup(serializer, tmp_path)

    session, _engine = _reload(serializer, path)

    node_config = session.get_node("out1")
    assert node_config is not None
    assert node_config.device_id == "led1"


def test_binding_survives_a_second_save(tmp_path: Path, serializer: ExperimentSerializer):
    """Save -> load -> save must not quietly drop the binding on the way out."""
    session, engine = _reload(serializer, _bound_setup(serializer, tmp_path))
    hm = engine._hardware_manager

    again = tmp_path / "again.glider"
    serializer.save(again, session=session, flow_engine=engine, hardware_manager=hm)

    _session2, engine2 = _reload(serializer, again)
    assert engine2.get_node("out1").device is not None


def test_unbound_node_saves_no_device_id(tmp_path: Path, serializer: ExperimentSerializer):
    """Nodes with no device must not gain a phantom binding."""
    hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    hm.add_board(board_id="board1", driver_type="mock", port=None)
    engine = _make_engine_with_all_nodes(hm)
    engine.create_node(node_id="delay1", node_type="Delay", position=(0.0, 0.0))

    out = tmp_path / "unbound.glider"
    serializer.save(out, session=ExperimentSession(), flow_engine=engine, hardware_manager=hm)

    data = json.loads(out.read_text())
    node = next(n for n in data["flow"]["nodes"] if n["id"] == "delay1")
    assert node.get("device_id") is None


def test_binding_to_a_device_missing_from_the_file_is_survivable(
    tmp_path: Path, serializer: ExperimentSerializer
):
    """A file naming a device that no longer exists must load the node unbound,
    not fail the whole load."""
    path = _bound_setup(serializer, tmp_path)
    data = json.loads(path.read_text())
    data["hardware"]["devices"] = []
    path.write_text(json.dumps(data))

    _session, engine = _reload(serializer, path)

    node = engine.get_node("out1")
    assert node is not None, "the whole load failed over one dangling binding"
    assert node.device is None


def test_single_pin_device_dict_stays_old_version_readable(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """A single-pin device must stay openable by OLD GLIDER versions, whose
    DeviceConfigSchema.from_dict lists "pin" in its required fields. So the
    save emits BOTH keys: the legacy int "pin" (which old parsers read) AND
    the current "pins" dict (which old parsers ignore as an unknown key).
    The file must also still round-trip through the CURRENT loader, reloading
    the device with pins == {"output": 13}."""
    hardware_manager_with_mock.add_board(board_id="board1", driver_type="mock", port=None)
    hardware_manager_with_mock.add_device(
        device_id="led1", device_type="DigitalOutput", board_id="board1", pin=13
    )

    out = tmp_path / "singlepin.glider"
    serializer.save(
        out,
        session=ExperimentSession(),
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    saved_devices = data["hardware"]["devices"]
    assert len(saved_devices) == 1
    # Both keys present so old versions read `pin`, new versions read `pins`.
    assert (
        saved_devices[0]["pin"] == 13
    ), f"Single-pin device dropped legacy pin: {saved_devices[0]}"
    assert saved_devices[0]["pins"] == {"output": 13}

    # And the file still round-trips through the CURRENT loader.
    fresh_hm = HardwareManager()
    HardwareManager.register_driver("mock", MockBoard)
    schema = serializer.load(out)
    serializer.apply_to_session(
        schema,
        session=ExperimentSession(),
        flow_engine=_make_engine_with_all_nodes(),
        hardware_manager=fresh_hm,
    )
    device = fresh_hm.devices.get("led1")
    assert device is not None
    assert device.pins == {"output": 13}


def test_zero_pin_device_dict_omits_legacy_pin_key(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    hardware_manager_with_mock,
    flow_engine_with_nodes,
):
    """A zero-pin device (e.g. BLEWrite, pins == {}) cannot be represented in
    the old single-`pin` schema at all, so its saved dict carries NO `pin`
    key. This is an inherent, accepted break: old versions cannot open files
    with zero-pin devices — writing "pin": null would only make the old
    parser fail with a misleading "Expected int, got NoneType"."""
    hardware_manager_with_mock.add_board(board_id="board1", driver_type="mock", port=None)
    hardware_manager_with_mock.add_device_multi_pin(
        device_id="ble1",
        device_type="BLEWrite",
        board_id="board1",
        pins={},
        name="Stim A",
        address="AA:BB:CC:DD:EE:01",
        char_uuid="0000ffe1-0000-1000-8000-00805f9b34fb",
    )

    out = tmp_path / "zeropin.glider"
    serializer.save(
        out,
        session=ExperimentSession(),
        flow_engine=flow_engine_with_nodes,
        hardware_manager=hardware_manager_with_mock,
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    saved_devices = data["hardware"]["devices"]
    assert len(saved_devices) == 1
    assert "pin" not in saved_devices[0], f"Zero-pin device carries legacy pin: {saved_devices[0]}"
    assert saved_devices[0]["pins"] == {}


def test_session_sync_excludes_nodes_the_engine_could_not_create(
    tmp_path: Path,
    serializer: ExperimentSerializer,
    flow_engine_with_nodes,
):
    """_apply_flow_config skips nodes whose type can't be resolved; the
    session-model sync must skip the same nodes (and their connections),
    otherwise Save As persists phantom nodes the live engine never had."""
    doc = {
        "schema_version": "1.0.0",
        "metadata": {"name": "Phantom"},
        "hardware": {"boards": [], "devices": []},
        "flow": {
            "nodes": [
                {
                    "id": "delay1",
                    "type": "Delay",
                    "title": "Delay",
                    "position": {"x": 0.0, "y": 0.0},
                    "properties": {},
                    "inputs": [],
                    "outputs": [],
                },
                {
                    "id": "bogus1",
                    "type": "TotallyBogusNodeType",
                    "title": "Bogus",
                    "position": {"x": 10.0, "y": 10.0},
                    "properties": {},
                    "inputs": [],
                    "outputs": [],
                },
            ],
            "connections": [
                {
                    "id": "c1",
                    "from_node": "delay1",
                    "from_port": 0,
                    "to_node": "bogus1",
                    "to_port": 0,
                    "connection_type": "exec",
                }
            ],
        },
        "dashboard": {"widgets": []},
    }
    path = tmp_path / "phantom.glider"
    path.write_text(json.dumps(doc), encoding="utf-8")

    session = ExperimentSession()
    schema = serializer.load(path)
    serializer.apply_to_session(
        schema,
        session=session,
        flow_engine=flow_engine_with_nodes,
        hardware_manager=None,
    )

    # The engine only created the resolvable node
    assert "delay1" in flow_engine_with_nodes.nodes
    assert "bogus1" not in flow_engine_with_nodes.nodes

    # The session model must agree with the engine, not the raw file
    node_ids = {n.id for n in session.flow.nodes}
    assert node_ids == {"delay1"}, f"Phantom node leaked into session model: {node_ids}"
    assert session.flow.connections == [], (
        "Connection referencing a phantom node leaked into session model: "
        f"{session.flow.connections}"
    )
