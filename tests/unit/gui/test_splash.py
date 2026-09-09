"""The launch splash. The floor is the only logic here worth a test.

``MIN_VISIBLE_MS`` is a *minimum*, not a duration, and getting that backwards is
the whole bug this guards: a splash that closed on a timer while initialisation
was still running would hand the user the empty desktop the splash exists to
prevent.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from glider.gui.splash import MIN_VISIBLE_MS, show_splash


def test_show_splash_returns_a_visible_splash(qtbot):
    screen = show_splash()
    assert screen is not None
    qtbot.addWidget(screen)
    try:
        assert not screen.pixmap().isNull()
    finally:
        screen.close()


def test_the_floor_has_not_elapsed_immediately(qtbot):
    screen = show_splash()
    assert screen is not None
    qtbot.addWidget(screen)
    try:
        remaining = screen.remaining_ms()
        assert 0 < remaining <= MIN_VISIBLE_MS
    finally:
        screen.close()


def test_a_slow_launch_hands_off_at_once(qtbot, monkeypatch):
    """Initialisation that outlasts the floor must not be made to wait again.
    Simulated by moving the splash's start time into the past."""
    screen = show_splash()
    assert screen is not None
    qtbot.addWidget(screen)
    window = QWidget()
    qtbot.addWidget(window)

    screen._shown_at -= (MIN_VISIBLE_MS / 1000) + 1
    assert screen.remaining_ms() == 0

    screen.finish_when_due(window)
    assert not screen.isVisible()
    # The hand-off is what reveals the window -- nobody showed it here.
    assert window.isVisible()


def test_the_window_stays_hidden_until_the_splash_hands_off(qtbot):
    """The reported bug: the main window appeared before the logo did.

    The caller used to ``show()`` the window and then start the splash's timer,
    so for the rest of the floor the window sat on screen beside a splash that
    was no longer hiding anything. Revealing the window IS the hand-off now.
    """
    screen = show_splash()
    assert screen is not None
    qtbot.addWidget(screen)
    window = QWidget()
    qtbot.addWidget(window)

    screen.finish_when_due(window)
    try:
        assert screen.isVisible()
        assert not window.isVisible()
    finally:
        screen.close()


def test_on_shown_runs_after_the_window_is_up(qtbot):
    """The first-run welcome is modal; it must not open over the splash,
    belonging to a window nobody has been shown yet."""
    screen = show_splash()
    assert screen is not None
    qtbot.addWidget(screen)
    window = QWidget()
    qtbot.addWidget(window)
    screen._shown_at -= (MIN_VISIBLE_MS / 1000) + 1

    seen: list[bool] = []
    screen.finish_when_due(window, on_shown=lambda: seen.append(window.isVisible()))

    assert seen == [True]


def test_a_missing_icon_costs_the_splash_and_nothing_else(monkeypatch):
    """Best-effort by contract: a branded launch is a nicety, starting is not."""

    def _no_icon(_size: int = 512):
        raise FileNotFoundError("icon_512.png")

    monkeypatch.setattr("glider.assets.get_icon_path", _no_icon)
    assert show_splash() is None
