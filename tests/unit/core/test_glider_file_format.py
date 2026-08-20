"""One `.glider` format across the whole app.

GLIDER had two readers/writers that could not read each other's files:
``ExperimentSession`` (what File > Save / File > Open use, and what every file
in ``examples/`` is written in) and ``ExperimentSerializer`` (what
``load_experiment`` / ``save_experiment`` -- and therefore ``glider --file`` --
used). Each dropped domains the other carried: the serializer had no ``camera``,
``zones`` or ``manual_controls``; the session had no ``vision``.

These lock in that the session format is the one format, that it now carries
vision settings, and that a file written by the old serializer still opens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glider.core.experiment_session import (
    BoardConfig,
    DeviceConfig,
    ExperimentSession,
    NodeConfig,
)
from glider.core.glider_core import GliderCore
from glider.core.hardware_manager import HardwareManager
from glider.hal.mock_board import MockBoard
from glider.vision.cv_processor import CVSettings, DetectionBackend


@pytest.fixture
def core() -> GliderCore:
    HardwareManager.register_driver("mock", MockBoard)
    c = GliderCore()
    c._session = ExperimentSession()
    c._session.name = "format"
    return c


def _populated(session: ExperimentSession) -> ExperimentSession:
    """A session using every domain the two formats disagreed about."""
    session.add_board(BoardConfig(id="b1", driver_type="mock", port=None))
    session.add_device(
        DeviceConfig(
            id="led1", device_type="DigitalOutput", name="LED", board_id="b1", pins={"output": 5}
        )
    )
    session.add_node(NodeConfig(id="out1", node_type="Output", position=(1.0, 2.0)))
    session.camera.camera_index = 3
    session.zones.zones = [{"id": "z1", "name": "Arena"}]
    session.set_manual_controls([{"device_id": "led1", "label": "LED"}])
    return session


# --- the reported bug: `glider --file <a file the GUI saved>` -----------------


async def test_load_experiment_opens_a_gui_saved_file(core, tmp_path: Path):
    """This is what `glider --file` does. It used to raise SchemaValidationError
    on any file the GUI had written -- which is every file that exists."""
    path = tmp_path / "gui.glider"
    _populated(core.session).save(str(path))

    reopened = GliderCore()
    await reopened.load_experiment(path)

    assert reopened.session.get_board("b1") is not None
    assert reopened.session.get_device("led1") is not None
    assert reopened.session.get_node("out1") is not None


async def test_load_experiment_opens_the_shipped_examples(tmp_path: Path):
    """The examples are the reference files; every one is session-format."""
    examples = sorted(Path(__file__).resolve().parents[3].glob("examples/*.glider"))
    assert examples, "no example .glider files found"
    for example in examples:
        core = GliderCore()
        await core.load_experiment(example)
        assert core.session.hardware.boards, f"{example.name}: no boards loaded"
        assert core.session.flow.nodes, f"{example.name}: no nodes loaded"


# --- no domain is dropped any more -------------------------------------------


async def test_round_trip_keeps_camera_zones_and_manual_controls(core, tmp_path: Path):
    """The serializer dropped all three; a save/load through it lost them."""
    _populated(core.session)
    path = tmp_path / "domains.glider"

    await core.save_experiment(path)
    reopened = GliderCore()
    await reopened.load_experiment(path)

    assert reopened.session.camera.camera_index == 3
    assert reopened.session.zones.zones == [{"id": "z1", "name": "Arena"}]
    assert reopened.session.manual_controls == [{"device_id": "led1", "label": "LED"}]


async def test_save_experiment_writes_the_session_format(core, tmp_path: Path):
    """Both writers must now produce a file the other reader accepts."""
    _populated(core.session)
    path = tmp_path / "written.glider"

    await core.save_experiment(path)

    reloaded = ExperimentSession.load(str(path))
    assert reloaded.get_node("out1") is not None
    node = json.loads(path.read_text())["flow"]["nodes"][0]
    assert "node_type" in node


# --- vision settings, the one thing the session format lacked ----------------


def test_gui_save_persists_cv_settings(core, tmp_path: Path):
    """File > Save goes through save_session, which never wrote CV settings --
    so the model choice silently did not survive reopening."""
    core.cv_processor.update_settings(
        CVSettings(
            backend=DetectionBackend.YOLO_BYTETRACK,
            model_path="/models/mouse_pose.pt",
            keypoint_names=["nose", "tail"],
        )
    )
    path = tmp_path / "cv.glider"

    core.save_session(str(path))

    assert json.loads(path.read_text())["vision"]["backend"] == "YOLO_BYTETRACK"


def test_gui_open_restores_cv_settings(core, tmp_path: Path):
    core.cv_processor.update_settings(
        CVSettings(backend=DetectionBackend.YOLO_BYTETRACK, model_path="/models/mouse_pose.pt")
    )
    path = tmp_path / "cv.glider"
    core.save_session(str(path))

    reopened = GliderCore()
    reopened.load_session(str(path))

    assert reopened.cv_processor.settings.model_path == "/models/mouse_pose.pt"


def test_a_file_without_vision_leaves_cv_settings_alone(core, tmp_path: Path):
    """Every existing file predates the vision block; opening one must not
    stomp the operator's live configuration with defaults."""
    path = tmp_path / "legacy.glider"
    _populated(core.session).save(str(path))
    data = json.loads(path.read_text())
    assert "vision" not in data, "a session with no CV settings should not write the key"

    reopened = GliderCore()
    reopened.cv_processor.update_settings(
        CVSettings(backend=DetectionBackend.MOTION_ONLY, model_path="/keep/me.pt")
    )
    reopened.load_session(str(path))

    assert reopened.cv_processor.settings.backend == DetectionBackend.MOTION_ONLY
    assert reopened.cv_processor.settings.model_path == "/keep/me.pt"


# --- old serializer-format files still open ----------------------------------


async def test_a_serializer_written_file_still_opens(core, tmp_path: Path):
    """Files written by the old save_experiment must not become unreadable."""
    from glider.serialization import ExperimentSerializer

    _populated(core.session)
    # The serializer reads hardware from the manager, not the session model.
    await core.setup_hardware()
    legacy = tmp_path / "legacy_schema.glider"
    ExperimentSerializer().save(
        legacy,
        session=core.session,
        flow_engine=core.flow_engine,
        hardware_manager=core.hardware_manager,
        vision_settings={"backend": "YOLO_BYTETRACK", "model_path": "/m.pt"},
    )
    assert "schema_version" in json.loads(legacy.read_text())

    reopened = GliderCore()
    await reopened.load_experiment(legacy)

    assert reopened.session.get_board("b1") is not None
    assert reopened.cv_processor.settings.model_path == "/m.pt"


# --- flow-engine adoption on save_experiment ---------------------------------


async def test_save_experiment_adopts_a_graph_built_on_the_engine(core, tmp_path: Path):
    """save_experiment used to serialize the flow engine; scripts that build a
    graph there rather than through the editor must not save an empty flow."""
    from glider.nodes.experiment_nodes import register_experiment_nodes

    register_experiment_nodes(core.flow_engine)
    core.flow_engine.create_node(node_id="start", node_type="StartExperiment", position=(0.0, 0.0))
    assert core.session.get_node("start") is None  # only the engine knows it

    path = tmp_path / "engine_built.glider"
    await core.save_experiment(path)

    reopened = GliderCore()
    await reopened.load_experiment(path)
    assert reopened.session.get_node("start") is not None


async def test_save_experiment_never_empties_a_loaded_session(core, tmp_path: Path):
    """The engine is empty between File > Open and setup_flow. Replacing the
    session's flow from it there would silently wipe the experiment."""
    path = tmp_path / "saved.glider"
    _populated(core.session).save(str(path))

    reopened = GliderCore()
    reopened.load_session(str(path))  # session model populated, engine empty
    assert not reopened.flow_engine.nodes

    resaved = tmp_path / "resaved.glider"
    await reopened.save_experiment(resaved)

    assert json.loads(resaved.read_text())["flow"]["nodes"], "the flow was wiped on save"


async def test_a_real_example_round_trips_without_loss(tmp_path: Path):
    """The strongest guard for user data: open a shipped experiment, save it,
    reopen it, and account for every board, device, node, connection and
    manual control. dispense.glider is the example that uses manual_controls,
    one of the three domains the serializer used to drop."""
    example = Path(__file__).resolve().parents[3] / "examples" / "dispense.glider"
    core = GliderCore()
    await core.initialize()  # registers the node types the file references
    await core.load_experiment(example)
    before = core.session

    counts = (
        len(before.hardware.boards),
        len(before.hardware.devices),
        len(before.flow.nodes),
        len(before.flow.connections),
        len(before.manual_controls),
    )
    assert all(counts), f"example is not exercising every domain: {counts}"

    resaved = tmp_path / "resaved.glider"
    await core.save_experiment(resaved)

    reopened = GliderCore()
    await reopened.initialize()
    await reopened.load_experiment(resaved)
    after = reopened.session

    assert (
        len(after.hardware.boards),
        len(after.hardware.devices),
        len(after.flow.nodes),
        len(after.flow.connections),
        len(after.manual_controls),
    ) == counts
    assert after.camera.to_dict() == before.camera.to_dict()
    assert after.manual_controls == before.manual_controls
