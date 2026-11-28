"""A real core and a mock board running with the flow loop thread on."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from glider.core import flow_loop
from glider.core.experiment_session import BoardConfig, DeviceConfig, SessionState
from glider.core.glider_core import GliderCore
from glider.core.hardware_manager import HardwareManager
from glider.hal.mock_board import MockBoard


class RecordingBoard(MockBoard):
    """Notes the thread every digital write and e-stop ran on."""

    writes: list[tuple[int, bool, str]] = []
    estops: list[str] = []

    async def write_digital(self, pin: int, value: bool) -> None:
        RecordingBoard.writes.append((pin, value, threading.current_thread().name))
        await super().write_digital(pin, value)

    async def emergency_stop(self) -> None:
        RecordingBoard.estops.append(threading.current_thread().name)
        await super().emergency_stop()


@pytest.fixture
async def core(tmp_path: Path):
    RecordingBoard.writes = []
    RecordingBoard.estops = []
    previous = HardwareManager.get_driver_class("recording_mock")
    HardwareManager.register_driver("recording_mock", RecordingBoard)
    flow_loop.start(home_loop=asyncio.get_running_loop())
    core = GliderCore()
    try:
        await core.initialize(load_plugins=False)
        core.set_recording_directory(tmp_path)
        core.video_recording_enabled = False
        core.cv_processing_enabled = False
        yield core
    finally:
        await core.shutdown()
        flow_loop.stop()
        if previous is not None:
            HardwareManager.register_driver("recording_mock", previous)
        else:
            HardwareManager._driver_registry.pop("recording_mock", None)


async def _build(core: GliderCore) -> None:
    board_cfg = BoardConfig(id="b1", driver_type="recording_mock", port="MOCK")
    device_cfg = DeviceConfig(
        id="led", device_type="DigitalOutput", name="LED", board_id="b1", pins={"output": 13}
    )
    core.session.add_board(board_cfg)
    core.session.add_device(device_cfg)
    hm = core.hardware_manager
    await hm.create_board(board_cfg)
    await hm.create_device(device_cfg)

    fe = core.flow_engine
    fe.create_node(node_id="start", node_type="StartExperiment")
    fe.create_node(node_id="write", node_type="DigitalWrite", device_id="led")
    fe.create_node(
        node_id="delay", node_type="Delay", state={"duration_seconds": 30.0, "use_input": False}
    )
    fe.create_connection("c1", "start", 0, "write", 0, connection_type="exec")
    fe.create_connection("c2", "write", 0, "delay", 0, connection_type="exec")


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting")
        await asyncio.sleep(0.01)


async def test_flow_and_hardware_run_on_the_flow_loop_thread(core: GliderCore):
    await _build(core)

    await core.start_experiment()
    assert core.state is SessionState.RUNNING
    await _wait_for(lambda: RecordingBoard.writes)

    threads = {name for _pin, _value, name in RecordingBoard.writes}
    assert threads == {"glider-flow-loop"}

    await core.stop_experiment()
    assert core.state is SessionState.READY


async def test_manual_device_control_hops_from_the_gui_loop(core: GliderCore):
    await _build(core)
    await core.connect_hardware()
    device = core.hardware_manager.get_device("led")
    if not device._initialized:
        await device.initialize()

    from glider.core.device_drive import set_digital

    RecordingBoard.writes.clear()
    await set_digital(device, True)  # what the Device Control panel awaits
    assert RecordingBoard.writes
    assert {name for *_, name in RecordingBoard.writes} == {"glider-flow-loop"}


async def test_emergency_stop_from_the_gui_loop_mid_run(core: GliderCore):
    await _build(core)
    await core.start_experiment()
    await _wait_for(lambda: RecordingBoard.writes)

    await asyncio.wait_for(core.emergency_stop(), timeout=5.0)

    assert RecordingBoard.estops
    assert set(RecordingBoard.estops) == {"glider-flow-loop"}
    assert core.state is SessionState.ERROR


async def test_emergency_stop_still_runs_when_the_flow_loop_is_wedged(core, monkeypatch):
    await _build(core)
    await core.connect_hardware()
    monkeypatch.setattr(flow_loop, "ESTOP_PING_TIMEOUT_S", 0.05)

    release = threading.Event()
    flow_loop.active().call_soon(release.wait, 10.0)
    try:
        await asyncio.wait_for(core.emergency_stop(), timeout=5.0)
    finally:
        release.set()

    me = threading.current_thread().name
    assert me in RecordingBoard.estops
