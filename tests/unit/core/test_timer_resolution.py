"""The timer-resolution request must never be the thing that breaks a run.

It exists to make timed waits more precise on Windows. Every failure mode --
no winmm, a refused period, an exception from the call -- has to degrade to
"keep the coarse tick" rather than propagate, because a delay that is 15 ms
long is a nuisance and a delay that raises ends an experiment.
"""

from __future__ import annotations

import sys

import pytest

from glider.core import timer_resolution as tr


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Each test starts from a clean, un-held state."""
    monkeypatch.setattr(tr, "_depth", 0)
    monkeypatch.setattr(tr, "_winmm", None)
    monkeypatch.setattr(tr, "_unavailable", False)
    yield


class _FakeWinmm:
    def __init__(self, result=0):
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def timeBeginPeriod(self, ms):  # noqa: N802 - mirrors the Win32 name
        self.calls.append(("begin", ms))
        return self.result

    def timeEndPeriod(self, ms):  # noqa: N802 - mirrors the Win32 name
        self.calls.append(("end", ms))
        return 0


def _on_windows(monkeypatch, winmm):
    monkeypatch.setattr(tr.sys, "platform", "win32")
    monkeypatch.setattr(tr, "_load_winmm", lambda: winmm)


# --- the happy path -----------------------------------------------------------


def test_it_raises_and_releases_the_tick(monkeypatch):
    winmm = _FakeWinmm()
    _on_windows(monkeypatch, winmm)

    with tr.high_resolution_timers():
        assert winmm.calls == [("begin", 1)]

    assert winmm.calls == [("begin", 1), ("end", 1)]


def test_overlapping_holders_share_one_request(monkeypatch):
    """Two delays running at once must not let the first one to finish drop
    the tick out from under the second."""
    winmm = _FakeWinmm()
    _on_windows(monkeypatch, winmm)

    with tr.high_resolution_timers():
        with tr.high_resolution_timers():
            assert winmm.calls == [("begin", 1)]
        # Inner block left; the outer one still needs the fine tick.
        assert winmm.calls == [("begin", 1)]

    assert winmm.calls == [("begin", 1), ("end", 1)]


def test_the_tick_is_released_when_the_body_raises(monkeypatch):
    winmm = _FakeWinmm()
    _on_windows(monkeypatch, winmm)

    with pytest.raises(RuntimeError):
        with tr.high_resolution_timers():
            raise RuntimeError("boom")

    assert winmm.calls == [("begin", 1), ("end", 1)]


def test_it_does_not_leak_the_holder_count(monkeypatch):
    winmm = _FakeWinmm()
    _on_windows(monkeypatch, winmm)

    for _ in range(3):
        with tr.high_resolution_timers():
            pass

    assert tr._depth == 0
    assert winmm.calls.count(("begin", 1)) == 3
    assert winmm.calls.count(("end", 1)) == 3


# --- every way it can be unavailable ------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows path")
def test_it_is_a_no_op_off_windows():
    with tr.high_resolution_timers():
        pass
    assert tr._depth == 0


def test_a_refused_period_is_not_released_later(monkeypatch):
    """Releasing a period that was never granted would unbalance Windows'
    own refcount and could lower another process's tick."""
    winmm = _FakeWinmm(result=97)  # anything non-zero is a refusal
    _on_windows(monkeypatch, winmm)

    with tr.high_resolution_timers():
        pass

    assert winmm.calls == [("begin", 1)]  # no matching end
    assert tr._depth == 0


def test_a_missing_winmm_is_survivable(monkeypatch):
    monkeypatch.setattr(tr.sys, "platform", "win32")
    monkeypatch.setattr(tr, "_load_winmm", lambda: None)

    with tr.high_resolution_timers():
        pass

    assert tr._depth == 0


def test_an_exception_from_the_call_is_survivable(monkeypatch):
    class _Exploding:
        def timeBeginPeriod(self, ms):  # noqa: N802
            raise OSError("winmm is having a day")

    _on_windows(monkeypatch, _Exploding())

    with tr.high_resolution_timers():
        pass  # must reach here

    assert tr._depth == 0


def test_a_failed_load_is_not_retried(monkeypatch):
    """The warning belongs in the log once, not once per delay."""
    monkeypatch.setattr(tr.sys, "platform", "win32")
    attempts = {"n": 0}

    def _explode(name):
        attempts["n"] += 1
        raise OSError("no winmm here")

    import ctypes

    monkeypatch.setattr(ctypes, "WinDLL", _explode, raising=False)

    for _ in range(3):
        tr._load_winmm()

    assert attempts["n"] == 1
    assert tr._unavailable is True
