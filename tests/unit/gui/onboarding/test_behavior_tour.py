"""A walkthrough for the Behavior Analysis window.

The existing tour covers the main window only. The behaviour tools are a
four-stage pipeline — annotate, train, review, apply — spread across tabs of
their own window, and nothing explained the order or what each stage is for.

Tabs are the interesting part: a target on the Train tab is not visible while
Annotate is showing, so spotlighting it lands on nothing. The tour only knew
how to raise dock widgets.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget  # noqa: E402

from glider.gui.onboarding.tour import Tour, TourStep, behavior_steps  # noqa: E402


class _TabbedHost(QWidget):
    """A minimal tabbed host with a tour registry, like the real window."""

    def __init__(self):
        super().__init__()
        self.tabs = QTabWidget()
        self.first = QLabel("on tab one")
        self.second = QLabel("on tab two")
        page_a, page_b = QWidget(), QWidget()
        QVBoxLayout(page_a).addWidget(self.first)
        QVBoxLayout(page_b).addWidget(self.second)
        self.tabs.addTab(page_a, "A")
        self.tabs.addTab(page_b, "B")
        QVBoxLayout(self).addWidget(self.tabs)

    def tour_targets(self):
        return {"first": self.first, "second": self.second}


# ---------------------------------------------------------------------------
# Tab awareness
# ---------------------------------------------------------------------------


def test_a_step_on_another_tab_brings_that_tab_forward(qtbot):
    """Otherwise the spotlight lands on a widget nobody can see."""
    host = _TabbedHost()
    qtbot.addWidget(host)
    host.tabs.setCurrentIndex(0)

    tour = Tour(host, steps=[TourStep("second", "T", "B")])
    tour.start()

    assert host.tabs.currentIndex() == 1


def test_a_step_on_the_current_tab_leaves_it_alone(qtbot):
    host = _TabbedHost()
    qtbot.addWidget(host)
    host.tabs.setCurrentIndex(0)

    tour = Tour(host, steps=[TourStep("first", "T", "B")])
    tour.start()

    assert host.tabs.currentIndex() == 0


def test_stepping_through_follows_the_tabs(qtbot):
    host = _TabbedHost()
    qtbot.addWidget(host)
    tour = Tour(
        host,
        steps=[TourStep("first", "1", "a"), TourStep("second", "2", "b")],
    )
    tour.start()
    assert host.tabs.currentIndex() == 0
    tour._next()
    assert host.tabs.currentIndex() == 1


def test_an_unknown_target_does_not_raise(qtbot):
    """A registry gap must not take the window down mid-tour."""
    host = _TabbedHost()
    qtbot.addWidget(host)
    Tour(host, steps=[TourStep("nope", "T", "B")]).start()


# ---------------------------------------------------------------------------
# The behaviour walkthrough itself
# ---------------------------------------------------------------------------


def test_there_are_steps():
    assert behavior_steps()


def test_it_opens_and_closes_without_a_spotlight():
    """Welcome and finish are centred, with nothing to point at."""
    steps = behavior_steps()
    assert steps[0].target_key is None
    assert steps[-1].target_key is None


def test_every_stage_of_the_pipeline_is_covered():
    text = " ".join(f"{s.title} {s.body}" for s in behavior_steps()).lower()
    for stage in ("annotate", "train", "review", "apply"):
        assert stage in text, f"the {stage} stage is never mentioned"


def test_the_steps_follow_the_pipeline_order():
    """Out of order, the tour teaches the wrong workflow."""
    keys = [s.target_key for s in behavior_steps() if s.target_key]
    tabs = [k for k in keys if k.startswith("tab_")]
    assert tabs == ["tab_annotate", "tab_train", "tab_review", "tab_apply"]


def test_every_target_is_registered_by_the_window(qtbot, tmp_path):
    """A step naming a key the window does not provide is a silent blank."""
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path)
    qtbot.addWidget(win)
    registry = win.tour_targets()

    missing = [
        s.target_key for s in behavior_steps() if s.target_key and s.target_key not in registry
    ]
    assert not missing, f"steps target unregistered keys: {missing}"


def test_the_registry_resolves_to_live_widgets(qtbot, tmp_path):
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path)
    qtbot.addWidget(win)
    for key, widget in win.tour_targets().items():
        assert widget is not None, f"{key} resolved to None"


def test_running_the_whole_tour_visits_every_tab(qtbot, tmp_path):
    """End to end on the real window, not a stand-in."""
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path)
    qtbot.addWidget(win)
    tour = Tour(win, steps=behavior_steps())
    tour.start()

    seen = {win.tabs.currentIndex()}
    for _ in range(len(behavior_steps()) - 1):
        tour._next()
        seen.add(win.tabs.currentIndex())

    assert seen == set(range(win.tabs.count()))


def test_the_window_can_start_its_own_tour(qtbot, tmp_path):
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path)
    qtbot.addWidget(win)
    win.start_tour()  # must not raise


def test_the_tour_is_reachable_from_the_window(qtbot, tmp_path):
    """A tutorial nobody can replay is a tutorial seen once and lost."""
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path)
    qtbot.addWidget(win)
    assert win._tour_btn is not None
    assert win._tour_btn.isEnabled()
