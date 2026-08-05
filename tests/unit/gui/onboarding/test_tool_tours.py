"""Walkthroughs for the tool windows: Batch Pose Tracking and Session Review.

These windows never pass through the first-launch welcome — they are opened on
demand from the Tools menu, by someone who has already used the rest of the
app. So each carries its own walkthrough, its own "seen it" flag, and a
Tutorial button to replay it.

The flags are the subtle part. One shared flag means finishing any walkthrough
silences all of them, and the one most people meet first is the one they would
lose.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget  # noqa: E402

from glider.gui.onboarding.tour import (  # noqa: E402
    BEHAVIOR_TOUR_COMPLETE_KEY,
    POSE_BATCH_TOUR_COMPLETE_KEY,
    SESSION_REVIEW_TOUR_COMPLETE_KEY,
    TOUR_COMPLETE_KEY,
    Tour,
    TourStep,
    behavior_steps,
    offer_tour_once,
    pose_batch_steps,
    session_review_steps,
    tour_complete,
)


@pytest.fixture
def settings(tmp_path) -> QSettings:
    """A QSettings backed by a fresh INI file — no cross-test leakage."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


class _Host(QWidget):
    def __init__(self):
        super().__init__()
        self.thing = QLabel("a target")
        QVBoxLayout(self).addWidget(self.thing)

    def tour_targets(self):
        return {"thing": self.thing}


# ---------------------------------------------------------------------------
# Completion flags
# ---------------------------------------------------------------------------

ALL_KEYS = [
    TOUR_COMPLETE_KEY,
    BEHAVIOR_TOUR_COMPLETE_KEY,
    POSE_BATCH_TOUR_COMPLETE_KEY,
    SESSION_REVIEW_TOUR_COMPLETE_KEY,
]


def test_every_walkthrough_has_its_own_flag():
    assert len(set(ALL_KEYS)) == len(ALL_KEYS)


@pytest.mark.parametrize("key", ALL_KEYS[1:])
def test_finishing_a_tool_walkthrough_leaves_the_main_tour_unseen(qtbot, settings, key):
    """Otherwise a tool tour robs someone of the tour they have never seen."""
    host = _Host()
    qtbot.addWidget(host)

    tour = Tour(host, steps=[TourStep("thing", "T", "B")], settings=settings, complete_key=key)
    tour.start()
    tour._finish()

    assert tour_complete(settings, key=key)
    assert not tour_complete(settings, key=TOUR_COMPLETE_KEY)


def test_skipping_counts_as_seen(qtbot, settings):
    host = _Host()
    qtbot.addWidget(host)
    tour = Tour(
        host,
        steps=[TourStep("thing", "T", "B")],
        settings=settings,
        complete_key=POSE_BATCH_TOUR_COMPLETE_KEY,
    )
    tour.start()
    tour._finish()  # what Skip and Esc both call

    assert tour_complete(settings, key=POSE_BATCH_TOUR_COMPLETE_KEY)


# ---------------------------------------------------------------------------
# offer_tour_once
# ---------------------------------------------------------------------------


def test_it_offers_the_first_time(qtbot, settings):
    host = _Host()
    qtbot.addWidget(host)
    assert offer_tour_once(host, [TourStep("thing", "T", "B")], "k/one", settings=settings)


def test_it_never_offers_twice(qtbot, settings):
    """A walkthrough that returns every time you open the window is a nag."""
    host = _Host()
    qtbot.addWidget(host)
    steps = [TourStep("thing", "T", "B")]

    offer_tour_once(host, steps, "k/one", settings=settings)
    assert offer_tour_once(host, steps, "k/one", settings=settings) is None


def test_a_finished_walkthrough_is_not_offered_again(qtbot, settings):
    host = _Host()
    qtbot.addWidget(host)
    settings.setValue("k/one", True)

    assert offer_tour_once(host, [TourStep("thing", "T", "B")], "k/one", settings=settings) is None


def test_only_one_walkthrough_runs_at_a_time(qtbot, settings):
    """Offered on first open AND replayable by button — both can fire."""
    host = _Host()
    qtbot.addWidget(host)
    steps = [TourStep("thing", "T", "B")]

    first = Tour(host, steps=steps, settings=settings)
    first.start()
    second = Tour(host, steps=steps, settings=settings)
    second.start()

    assert host._active_tour is second
    assert first._overlay is None, "the first overlay is still dimming the window"


def test_finishing_clears_the_active_walkthrough(qtbot, settings):
    host = _Host()
    qtbot.addWidget(host)
    tour = Tour(host, steps=[TourStep("thing", "T", "B")], settings=settings)
    tour.start()
    tour._finish()

    assert host._active_tour is None


def test_offering_one_does_not_silence_another(qtbot, settings):
    host = _Host()
    qtbot.addWidget(host)
    steps = [TourStep("thing", "T", "B")]

    offer_tour_once(host, steps, "k/one", settings=settings)
    assert offer_tour_once(host, steps, "k/two", settings=settings) is not None


# ---------------------------------------------------------------------------
# The step lists
# ---------------------------------------------------------------------------

WALKTHROUGHS = {
    "pose_batch": pose_batch_steps,
    "session_review": session_review_steps,
    "behavior": behavior_steps,
}


@pytest.mark.parametrize("name", sorted(WALKTHROUGHS))
def test_it_opens_and_closes_without_a_spotlight(name):
    """Welcome and finish are centred, with nothing to point at."""
    steps = WALKTHROUGHS[name]()
    assert steps[0].target_key is None
    assert steps[-1].target_key is None


@pytest.mark.parametrize("name", sorted(WALKTHROUGHS))
def test_no_step_is_empty(name):
    for step in WALKTHROUGHS[name]():
        assert step.title.strip()
        assert step.body.strip()


def test_the_pose_batch_steps_follow_the_form(name=None):
    """Out of order, the tour teaches someone to fill the form wrong."""
    keys = [s.target_key for s in pose_batch_steps() if s.target_key]
    assert keys == ["model", "bodyparts", "videos", "calibration", "filter", "run"]


def test_bodypart_order_is_called_out():
    """The one mistake here that never announces itself."""
    step = next(s for s in pose_batch_steps() if s.target_key == "bodyparts")
    assert "order" in f"{step.title} {step.body}".lower()


def test_the_ethogram_step_explains_the_drag():
    """Selecting a window is the whole point and nothing on screen says so."""
    step = next(s for s in session_review_steps() if s.target_key == "ethogram")
    assert "drag" in step.body.lower()


# ---------------------------------------------------------------------------
# Against the real windows
# ---------------------------------------------------------------------------


def _pose_batch_window(qtbot):
    from glider.gui.pose_batch.window import PoseBatchWindow

    win = PoseBatchWindow()
    qtbot.addWidget(win)
    return win


def _session_review_window(qtbot):
    from glider.gui.behavior.analysis_window import AnalysisWindow

    win = AnalysisWindow()
    qtbot.addWidget(win)
    return win


WINDOWS = {
    "pose_batch": (_pose_batch_window, pose_batch_steps),
    "session_review": (_session_review_window, session_review_steps),
}


@pytest.mark.parametrize("name", sorted(WINDOWS))
def test_every_target_is_registered_by_the_window(qtbot, name):
    """A step naming a key the window does not provide is a silent blank."""
    build, steps = WINDOWS[name]
    registry = build(qtbot).tour_targets()

    missing = [s.target_key for s in steps() if s.target_key and s.target_key not in registry]
    assert not missing, f"{name} steps target unregistered keys: {missing}"


@pytest.mark.parametrize("name", sorted(WINDOWS))
def test_the_registry_resolves_to_live_widgets(qtbot, name):
    build, _ = WINDOWS[name]
    for key, widget in build(qtbot).tour_targets().items():
        assert widget is not None, f"{name}: {key} resolved to None"


@pytest.mark.parametrize("name", sorted(WINDOWS))
def test_the_walkthrough_is_reachable_from_the_window(qtbot, name):
    """A tutorial nobody can replay is a tutorial seen once and lost."""
    build, _ = WINDOWS[name]
    win = build(qtbot)
    assert win._tour_btn is not None
    assert win._tour_btn.isEnabled()


@pytest.mark.parametrize("name", sorted(WINDOWS))
def test_running_the_whole_walkthrough_does_not_raise(qtbot, name, settings):
    """End to end on the real window, not a stand-in."""
    build, steps = WINDOWS[name]
    win = build(qtbot)
    tour = Tour(win, steps=steps(), settings=settings)
    tour.start()
    for _ in range(len(steps())):
        tour._next()


def test_the_cohort_step_brings_the_cohort_tab_forward(qtbot, settings):
    """The table lives behind a tab, so pointing at it has to raise it."""
    win = _session_review_window(qtbot)
    win._tables.setCurrentIndex(0)

    step = next(s for s in session_review_steps() if s.target_key == "cohort_table")
    tour = Tour(win, steps=[step], settings=settings)
    tour.start()

    assert win._tables.currentWidget() is win._cohort_table


@pytest.mark.parametrize("name", sorted(WINDOWS))
def test_showing_the_window_offers_the_walkthrough(qtbot, monkeypatch, name):
    """Wired to showEvent, or the 'first time' tutorial never fires by itself."""
    calls = []
    monkeypatch.setattr(
        "glider.gui.onboarding.tour.offer_tour_once",
        lambda host, steps, key, **kw: calls.append(key),
    )

    build, _ = WINDOWS[name]
    win = build(qtbot)
    win.show()

    assert len(calls) == 1
    assert calls[0].startswith("first_run/")
