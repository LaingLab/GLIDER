"""Stop/start safety in GliderCore: A1, A3/A4, A5, A6, A7, A9 and C1 (review 2026-09-28)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from glider.core import hardware_manager as hm_module
from glider.core.experiment_session import ExperimentSession, NodeConfig, SessionState
from glider.core.glider_core import GliderCore


def _core(tmp_path, state=SessionState.READY) -> GliderCore:
    core = GliderCore()
    core._session = ExperimentSession()
    core._session.state = state
    core.set_recording_directory(tmp_path)
    return core


async def test_estop_drives_hardware_safe_even_if_the_flow_stop_hangs(tmp_path):
    core = _core(tmp_path, SessionState.RUNNING)
    core._hardware_manager.emergency_stop = AsyncMock()
    core._flow_engine.stop = lambda: asyncio.Event().wait()  # never returns

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(core.emergency_stop(), timeout=0.2)

    core._hardware_manager.emergency_stop.assert_awaited()


async def test_a_paused_video_is_finalized_on_stop(tmp_path):
    core = _core(tmp_path)
    assert not core._video_recorder.is_recording  # as when PAUSED
    core._video_recorder.stop = AsyncMock(return_value=None)
    core._multi_video_recorder.stop = AsyncMock(return_value={})

    await core._stop_recorders()

    core._video_recorder.stop.assert_awaited_once()
    core._multi_video_recorder.stop.assert_awaited_once()


async def test_a_recorder_that_fails_to_start_refuses_the_run_and_rolls_back(tmp_path):
    core = _core(tmp_path)
    core._data_recorder.start = AsyncMock(side_effect=OSError("read-only"))

    with pytest.raises(RuntimeError, match="data recorder: read-only"):
        await core.start_experiment()

    assert core.state == SessionState.READY
    assert not core._flow_engine.is_running
    assert not core._event_logger.is_recording  # started, then closed again


async def test_a_flow_that_did_not_load_completely_is_not_started(tmp_path):
    core = _core(tmp_path)
    core._session.add_node(NodeConfig(id="n1", node_type="NoSuchPluginNode"))

    with pytest.raises(RuntimeError, match="did not load completely"):
        await core.start_experiment()

    assert not core._flow_engine.is_running
    assert not core._event_logger.is_recording


async def test_a_node_failure_stops_the_running_experiment(tmp_path):
    core = _core(tmp_path, SessionState.RUNNING)
    core._stop_experiment_locked = AsyncMock()

    core._on_flow_error("node-7", RuntimeError("write failed"))
    await asyncio.gather(*core._background_tasks)

    core._stop_experiment_locked.assert_awaited_once()


async def test_a_node_failure_outside_a_run_stops_nothing(tmp_path):
    core = _core(tmp_path, SessionState.READY)
    core._stop_experiment_locked = AsyncMock()

    core._on_flow_error("node-7", RuntimeError("manual run failed"))

    assert not core._background_tasks
    core._stop_experiment_locked.assert_not_awaited()


async def test_one_wedged_device_does_not_hold_up_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(hm_module, "DEVICE_IO_TIMEOUT_S", 0.05)
    core = _core(tmp_path)
    ok = SimpleNamespace(shutdown=AsyncMock())
    wedged = SimpleNamespace(shutdown=lambda: asyncio.Event().wait())
    core._hardware_manager._devices = {"wedged": wedged, "ok": ok}

    await asyncio.wait_for(core._set_all_devices_low(), timeout=1.0)

    ok.shutdown.assert_awaited_once()


async def test_setup_hardware_refuses_while_running(tmp_path):
    core = _core(tmp_path, SessionState.RUNNING)
    with pytest.raises(RuntimeError, match="while an experiment is running"):
        await core.setup_hardware()
    assert core.state == SessionState.RUNNING


async def test_clear_releases_connected_hardware():
    manager = hm_module.HardwareManager()
    board = SimpleNamespace(id="b", is_connected=True, disconnect=AsyncMock())
    device = SimpleNamespace(id="d", board=board, shutdown=AsyncMock())
    manager._boards = {"b": board}
    manager._devices = {"d": device}

    manager.clear()
    await asyncio.gather(*manager._detached_teardowns)

    assert not manager.boards and not manager.devices
    device.shutdown.assert_awaited_once()
    board.disconnect.assert_awaited_once()


def test_a_stalled_camera_reaches_core_error_callbacks(tmp_path):
    core = _core(tmp_path)
    seen = []
    core.on_error(lambda source, error: seen.append((source, str(error))))

    core._camera_manager._notify_error("no frame for 5 s")
    core._multi_camera_manager._on_camera_error("cam_1", "capture loop crashed")

    assert seen == [("camera", "no frame for 5 s"), ("camera:cam_1", "capture loop crashed")]
