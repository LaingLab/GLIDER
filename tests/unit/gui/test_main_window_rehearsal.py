"""Rehearse starts the experiment, and stops only the one it started.

Zone Input and Behavior Input only listen while the experiment runs, so a
rehearsal of a stopped experiment played the video to nobody. Bypasses the
heavy ``__init__`` the way test_main_window_function_run does.
"""

from __future__ import annotations

import asyncio
import types


def _window(state: str, *, start_works: bool = True, plays: bool = True):
    from glider.gui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)
    win._rehearsal_starting = False
    win._rehearsal_started_run = False
    win.calls = []
    core = types.SimpleNamespace(state=types.SimpleNamespace(name=state))
    win._core = core

    async def _start():
        win.calls.append("start")
        await asyncio.sleep(0)  # a real start awaits hardware
        if start_works:
            core.state.name = "RUNNING"

    async def _stop():
        win.calls.append("stop")
        core.state.name = "READY"

    def _begin(path):
        win.calls.append(f"play {path}")
        return plays

    win._start_async = _start
    win._stop_async = _stop
    win._camera_panel = types.SimpleNamespace(begin_rehearsal=_begin)
    return win


async def test_it_starts_a_stopped_experiment_then_plays_and_stops_it_after():
    win = _window("READY")

    await win._rehearse_async("a.mp4")
    await win._end_rehearsal_async()

    assert win.calls == ["start", "play a.mp4", "stop"]


async def test_an_experiment_the_operator_started_is_left_running():
    win = _window("RUNNING")

    await win._rehearse_async("a.mp4")
    await win._end_rehearsal_async()

    assert win.calls == ["play a.mp4"]


async def test_a_paused_run_is_resumed_but_not_stopped_after():
    win = _window("PAUSED")

    await win._rehearse_async("a.mp4")
    await win._end_rehearsal_async()

    assert win.calls == ["start", "play a.mp4"]


async def test_nothing_plays_when_the_experiment_will_not_start():
    """_start_async has already told the operator why."""
    win = _window("READY", start_works=False)

    await win._rehearse_async("a.mp4")

    assert win.calls == ["start"]


async def test_a_video_that_will_not_play_stops_the_run_it_started():
    win = _window("READY", plays=False)

    await win._rehearse_async("a.mp4")

    assert win.calls == ["start", "play a.mp4", "stop"]


async def test_a_second_click_while_starting_is_ignored():
    """Starting can take seconds (BLE). Two pumps would feed one flow."""
    win = _window("READY")

    await asyncio.gather(win._rehearse_async("a.mp4"), win._rehearse_async("a.mp4"))

    assert win.calls == ["start", "play a.mp4"]
