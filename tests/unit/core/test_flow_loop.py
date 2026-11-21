"""Tests for the flow loop thread (core/flow_loop.py)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from glider.core import flow_loop
from glider.core.flow_loop import (
    call_on_gui,
    estop_entry,
    on_flow_loop,
    post_to_loop,
    wrap_public_coroutines,
)
from glider.hal.base_board import BaseBoard
from glider.hal.base_device import BaseDevice


@pytest.fixture
async def fl():
    """A running flow loop, homed on the test's loop; stopped afterwards."""
    started = flow_loop.start(home_loop=asyncio.get_running_loop())
    yield started
    flow_loop.stop()


class Probe:
    """Records which thread its coroutines ran on."""

    def __init__(self) -> None:
        self.threads: list[threading.Thread] = []

    @on_flow_loop
    async def where(self) -> threading.Thread:
        self.threads.append(threading.current_thread())
        return threading.current_thread()

    @on_flow_loop
    async def outer(self) -> tuple[threading.Thread, threading.Thread]:
        here = threading.current_thread()
        inner = await self.where()
        return here, inner

    @on_flow_loop
    async def boom(self) -> None:
        raise ValueError("boom")

    @on_flow_loop
    async def sleep(self, started: threading.Event, cancelled: threading.Event) -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise


# -- pass-through when off ------------------------------------------------------
async def test_runs_on_caller_thread_when_no_flow_loop():
    assert flow_loop.active() is None
    assert await Probe().where() is threading.current_thread()


async def test_enabled_by_setting_env_override(monkeypatch):
    monkeypatch.setenv(flow_loop.ENV_VAR, "1")
    assert flow_loop.enabled_by_setting() is True
    monkeypatch.setenv(flow_loop.ENV_VAR, "0")
    assert flow_loop.enabled_by_setting() is False


async def test_setting_defaults_off(monkeypatch):
    monkeypatch.delenv(flow_loop.ENV_VAR, raising=False)
    from glider.core.config import GliderConfig

    assert GliderConfig().timing.flow_loop_thread is False
    assert GliderConfig.from_dict({"timing": {"flow_loop_thread": True}}).timing.flow_loop_thread
    assert GliderConfig().to_dict()["timing"]["flow_loop_thread"] is False


# -- hopping ---------------------------------------------------------------------
async def test_hops_to_flow_loop_thread(fl):
    ran_on = await Probe().where()
    assert ran_on is not threading.current_thread()
    assert ran_on.name == "glider-flow-loop"


async def test_nested_calls_do_not_hop_twice(fl):
    outer, inner = await Probe().outer()
    assert outer is inner
    assert outer.name == "glider-flow-loop"


async def test_exception_propagates_to_caller(fl):
    with pytest.raises(ValueError, match="boom"):
        await Probe().boom()


async def test_cancelling_caller_cancels_flow_loop_task(fl):
    started, cancelled = threading.Event(), threading.Event()
    task = asyncio.ensure_future(Probe().sleep(started, cancelled))
    assert await asyncio.to_thread(started.wait, 2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(cancelled.wait, 2.0)


async def test_stop_clears_active():
    loop = asyncio.get_running_loop()
    flow_loop.start(home_loop=loop)
    assert flow_loop.active() is not None
    flow_loop.stop()
    assert flow_loop.active() is None
    assert await Probe().where() is threading.current_thread()


# -- hal classes -----------------------------------------------------------------
def test_board_and_device_subclasses_get_wrapped():
    class Dev(BaseDevice):
        async def poke(self):  # public: wrapped
            return 1

        async def _private(self):  # private: left alone
            return 2

    assert getattr(Dev.poke, "__glider_flow_loop__", False)
    assert not getattr(Dev._private, "__glider_flow_loop__", False)
    assert getattr(BaseDevice.execute_action, "__glider_flow_loop__", False)
    assert getattr(BaseBoard.emergency_stop, "__glider_flow_loop__", False)


def test_wrapping_keeps_abstract_methods_abstract():
    with pytest.raises(TypeError):
        BaseBoard()  # type: ignore[abstract]


def test_wrap_public_coroutines_is_idempotent():
    class C:
        async def go(self):
            return 1

    wrap_public_coroutines(C)
    first = C.go
    wrap_public_coroutines(C)
    assert C.go is first


# -- e-stop ----------------------------------------------------------------------
class Stopper:
    def __init__(self) -> None:
        self.thread: threading.Thread | None = None

    @estop_entry
    async def emergency_stop(self) -> str:
        self.thread = threading.current_thread()
        # Anything it awaits must not hop when bypassed.
        return (await Probe().where()).name


async def test_estop_runs_on_flow_loop_when_responsive(fl):
    s = Stopper()
    assert await s.emergency_stop() == "glider-flow-loop"
    assert s.thread.name == "glider-flow-loop"


async def test_estop_runs_locally_when_flow_loop_is_wedged(fl, monkeypatch):
    monkeypatch.setattr(flow_loop, "ESTOP_PING_TIMEOUT_S", 0.05)
    release = threading.Event()
    # Block the flow loop's thread with synchronous work.
    fl.call_soon(release.wait, 5.0)
    try:
        t0 = time.perf_counter()
        s = Stopper()
        ran_on = await asyncio.wait_for(s.emergency_stop(), timeout=2.0)
        assert time.perf_counter() - t0 < 1.0
        assert ran_on == threading.current_thread().name
        assert s.thread is threading.current_thread()
    finally:
        release.set()


async def test_estop_runs_inline_without_flow_loop():
    s = Stopper()
    assert await s.emergency_stop() == threading.current_thread().name


# -- calls back to the GUI loop --------------------------------------------------
async def test_call_on_gui_from_flow_loop_runs_on_home_loop(fl):
    home = threading.current_thread()

    @on_flow_loop
    async def from_flow():
        assert threading.current_thread() is not home
        return await call_on_gui(threading.current_thread)

    assert await from_flow() is home


async def test_call_on_gui_propagates_exceptions(fl):
    def bad():
        raise RuntimeError("qt said no")

    @on_flow_loop
    async def from_flow():
        return await call_on_gui(bad)

    with pytest.raises(RuntimeError, match="qt said no"):
        await from_flow()


async def test_call_on_gui_inline_without_flow_loop():
    assert await call_on_gui(threading.current_thread) is threading.current_thread()


async def test_post_to_loop_inline_on_own_loop_and_queued_from_other_thread():
    loop = asyncio.get_running_loop()
    seen: list[threading.Thread] = []
    assert post_to_loop(loop, lambda: seen.append(threading.current_thread()))
    assert seen == [threading.current_thread()]

    await asyncio.to_thread(post_to_loop, loop, lambda: seen.append(threading.current_thread()))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert seen[-1] is threading.current_thread()
    assert len(seen) == 2


def test_post_to_loop_false_without_loop():
    assert post_to_loop(None, lambda: None) is False
