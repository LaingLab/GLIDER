"""The launch splash: the floor, the hand-off, and the child process's lifecycle.

``MIN_VISIBLE_MS`` is a *minimum*, not a duration, and getting that backwards is
the whole bug the floor tests guard: a splash that closed on a timer while
initialisation was still running would hand the user the empty desktop the
splash exists to prevent.
"""

from __future__ import annotations

import subprocess
import time

from PyQt6.QtWidgets import QWidget

from glider.gui import splash as splash_mod
from glider.gui.splash import MIN_VISIBLE_MS, GliderSplash, show_splash


class _FakeProcess:
    """Stands in for the child: the floor logic must not need a real one."""

    def __init__(self) -> None:
        self.stdin = None
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def terminate(self) -> None:
        self.alive = False


def _splash(elapsed_s: float = 0.0) -> GliderSplash:
    screen = GliderSplash(_FakeProcess())
    screen._shown_at -= elapsed_s
    return screen


def test_the_floor_has_not_elapsed_immediately():
    assert 0 < _splash().remaining_ms() <= MIN_VISIBLE_MS


def test_a_slow_launch_hands_off_at_once(qtbot):
    """Initialisation that outlasts the floor must not be made to wait again."""
    screen = _splash(MIN_VISIBLE_MS / 1000 + 1)
    window = QWidget()
    qtbot.addWidget(window)

    assert screen.remaining_ms() == 0
    screen.finish_when_due(window)
    assert not screen.isVisible()
    # The hand-off is what reveals the window -- nobody showed it here.
    assert window.isVisible()


def test_the_window_stays_hidden_until_the_splash_hands_off(qtbot):
    """The main window must not appear beside a splash that is still up."""
    screen = _splash()
    window = QWidget()
    qtbot.addWidget(window)

    screen.finish_when_due(window)
    assert screen.isVisible()
    assert not window.isVisible()


def test_on_shown_runs_after_the_window_is_up(qtbot):
    """The first-run welcome is modal; it must not open over the splash."""
    screen = _splash(MIN_VISIBLE_MS / 1000 + 1)
    window = QWidget()
    qtbot.addWidget(window)

    seen: list[bool] = []
    screen.finish_when_due(window, on_shown=lambda: seen.append(window.isVisible()))
    assert seen == [True]


def test_a_missing_animation_costs_the_splash_and_nothing_else(monkeypatch, tmp_path):
    """Best-effort by contract: a branded launch is a nicety, starting is not."""
    monkeypatch.setattr(splash_mod, "SPLASH_GIF", tmp_path / "nope.gif")
    assert show_splash() is None


def test_the_child_plays_and_leaves_when_told():
    """The real process: it must come up, stay up, and exit when the parent lets go."""
    screen = show_splash()
    assert screen is not None
    try:
        time.sleep(1.5)  # long enough to have crashed on start-up if it was going to
        assert screen.isVisible(), "splash child exited on its own"
    finally:
        screen.close()
    screen._process.wait(timeout=10)
    assert not screen.isVisible()


def test_the_child_leaves_when_its_parent_dies():
    """A parent that crashes never calls close(); stdin closing must be enough."""
    process = subprocess.Popen(splash_mod._child_command(), stdin=subprocess.PIPE)
    try:
        time.sleep(1.5)
        assert process.poll() is None
        process.stdin.close()  # what the OS does when the parent goes
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
