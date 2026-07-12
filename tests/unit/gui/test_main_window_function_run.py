"""MainWindow._run_function_async — the Runner function-button coordinator.

Bypasses the heavy ``__init__`` (``__new__`` + no Qt base needed, the handler
touches no widgets beyond a stubbed controls object) and drives the handler
against fakes to prove the glue: hardware gate, lazy graph load, the transient
``FlowState.RUNNING`` toggle that gates exec propagation, the shared-runner
call, the running affordance, and the single-in-flight busy guard.
"""

from __future__ import annotations

import types

import pytest

from glider.core.flow_engine import FlowState

pytestmark = pytest.mark.asyncio


def _bare_window():
    from glider.gui.main_window import MainWindow

    return MainWindow.__new__(MainWindow)


class _Controls:
    def __init__(self):
        self.running: list[tuple[str, bool]] = []
        self.status: list[tuple[str, str]] = []
        self.cleared = 0

    def set_function_running(self, sid, running):
        self.running.append((sid, running))

    def show_status(self, text, level="error"):
        self.status.append((text, level))

    def clear_status(self):
        self.cleared += 1


class _Runner:
    def __init__(self, engine):
        self._engine = engine
        self.state_during_run = None

    async def execute(self, timeout=60.0, on_timeout=None):
        self.state_during_run = self._engine.state
        return True


class _Engine:
    def __init__(self):
        self.nodes = {}
        self.state = "PREV"
        self._runner = None

    def get_node(self, nid):
        return self.nodes.get(nid)

    def get_function_runner(self, start_node_id):
        if self._runner is None:
            self._runner = _Runner(self)
        return self._runner


def _core(engine, *, connected=True, state=None, busy=False):
    from glider.core.experiment_session import SessionState

    session = types.SimpleNamespace(
        flow=types.SimpleNamespace(
            nodes=[types.SimpleNamespace(id="s1", node_type="StartFunction", state={})],
            connections=[],
        )
    )

    def setup_flow():
        engine.nodes = {"s1": object()}  # lazy load populates the engine

    return types.SimpleNamespace(
        hardware_manager=types.SimpleNamespace(is_any_board_connected=lambda: connected),
        flow_engine=engine,
        session=session,
        setup_flow=setup_flow,
        state=state if state is not None else SessionState.IDLE,
        is_experiment_busy=busy,
    )


async def test_happy_path_lazy_loads_toggles_running_and_runs():
    win = _bare_window()
    engine = _Engine()
    win._core = _core(engine)
    win._runner_device_controls = _Controls()
    win._manual_run_busy = False

    await win._run_function_async("s1")

    # Lazy-loaded, ran through the shared runner while RUNNING, restored state.
    assert engine.nodes  # setup_flow populated it
    assert engine._runner.state_during_run == FlowState.RUNNING
    assert engine.state == "PREV"
    # Running affordance raised then cleared; status cleared on success.
    assert win._runner_device_controls.running == [("s1", True), ("s1", False)]
    assert win._runner_device_controls.cleared == 1
    assert win._manual_run_busy is False


async def test_no_hardware_is_reported_and_does_not_run():
    win = _bare_window()
    engine = _Engine()
    win._core = _core(engine, connected=False)
    win._runner_device_controls = _Controls()
    win._manual_run_busy = False

    await win._run_function_async("s1")

    assert engine._runner is None  # never ran
    assert engine.nodes == {}  # never lazy-loaded
    assert any("Connect hardware" in text for text, _ in win._runner_device_controls.status)


async def test_refuses_manual_run_during_a_real_experiment():
    from glider.core.experiment_session import SessionState

    win = _bare_window()
    engine = _Engine()
    win._core = _core(engine, state=SessionState.RUNNING)
    win._runner_device_controls = _Controls()
    win._manual_run_busy = False

    await win._run_function_async("s1")

    assert engine._runner is None  # never ran
    assert engine.nodes == {}  # never lazy-loaded
    assert any("Stop the experiment" in t for t, _ in win._runner_device_controls.status)


async def test_refuses_manual_run_while_start_stop_in_progress():
    win = _bare_window()
    engine = _Engine()
    win._core = _core(engine, busy=True)  # experiment lock held mid start/stop
    win._runner_device_controls = _Controls()
    win._manual_run_busy = False

    await win._run_function_async("s1")

    assert engine._runner is None
    assert any("Stop the experiment" in t for t, _ in win._runner_device_controls.status)


async def test_start_is_refused_while_a_function_is_running():
    win = _bare_window()
    win._manual_run_busy = True  # a manual function is in flight
    started: list = []
    notified: list = []
    win._core = types.SimpleNamespace(
        start_experiment=lambda: started.append(True),  # must NOT be awaited
    )
    win._notify_user = lambda *a, **k: notified.append((a, k))

    await win._start_async()

    assert started == []  # real start refused
    assert notified  # operator told why


async def test_busy_guard_blocks_a_second_run():
    win = _bare_window()
    engine = _Engine()
    win._core = _core(engine)
    win._runner_device_controls = _Controls()
    win._manual_run_busy = True  # a run is already in flight

    await win._run_function_async("s1")

    assert engine._runner is None
    assert win._runner_device_controls.running == []
    assert any(level == "info" for _, level in win._runner_device_controls.status)


async def test_missing_node_after_load_is_reported():
    win = _bare_window()
    engine = _Engine()
    core = _core(engine)
    core.setup_flow = lambda: setattr(engine, "nodes", {"other": object()})  # s1 absent
    win._core = core
    win._runner_device_controls = _Controls()
    win._manual_run_busy = False

    await win._run_function_async("s1")

    assert engine._runner is None
    assert any("no longer in the graph" in text for text, _ in win._runner_device_controls.status)
