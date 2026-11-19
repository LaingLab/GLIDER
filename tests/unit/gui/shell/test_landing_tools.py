"""The landing page's tool column, and a recent list that cannot stop start-up.

Two failures this file pins down:

* The tools (Session Review, Batch Pose Tracking, ...) were unreachable until
  an experiment existed: the Tools tab is hidden on the landing page and the
  Tools menu is off the bar. None of them needs an experiment.
* A recent entry on a share the user had lost access to made ``Path.exists``
  raise while the landing page was being built, so GLIDER would not launch.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QLabel

from glider.gui.main_window import PAGE_LANDING
from glider.gui.panels.tools_page import ToolCard
from glider.gui.shell import landing
from glider.gui.shell.landing import (
    RECENT_KEY,
    STACK_BELOW_PX,
    LandingPage,
    forget_experiment,
    recent_experiments,
    remember_experiment,
)


@pytest.fixture
def settings(tmp_path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def _cards() -> list[ToolCard]:
    return [
        ToolCard(key="review", title="Session Review", description="d", glyph="review"),
        ToolCard(
            key="pose",
            title="Batch Pose Tracking",
            description="d",
            glyph="pose",
            available=False,
            unavailable_reason="Needs the vision extra.",
            install_extra="vision",
        ),
    ]


# ------------------------------------------------------------- the widget


def test_without_tool_cards_the_page_is_the_start_card_alone(qtbot, settings):
    page = LandingPage(settings=settings)
    qtbot.addWidget(page)
    assert page.tools() is None


def test_tool_cards_sit_beside_the_start_card_and_emit_their_key(qtbot, settings):
    page = LandingPage(settings=settings, tool_cards=_cards())
    qtbot.addWidget(page)

    tools = page.tools()
    assert tools is not None
    assert [c.key for c in tools.cards()] == ["review", "pose"]

    with qtbot.waitSignal(page.tool_chosen) as caught:
        tools._on_clicked("review")
    assert caught.args == ["review"]


def test_an_unavailable_tool_offers_its_install_through_the_page(qtbot, settings):
    page = LandingPage(settings=settings, tool_cards=_cards())
    qtbot.addWidget(page)

    with qtbot.assertNotEmitted(page.tool_chosen):
        page.tools()._on_clicked("pose")
    with qtbot.waitSignal(page.install_requested) as caught:
        page.tools().install_button("pose").click()
    assert caught.args == ["vision"]


def test_landing_cards_are_name_and_icon_only(qtbot, settings):
    """No subheading and no description line; the description is the tooltip."""
    page = LandingPage(settings=settings, tool_cards=_cards())
    qtbot.addWidget(page)
    page.show()
    tools = page.tools()
    assert tools.findChild(QLabel, "toolsSubheading") is None
    for label in tools.findChildren(QLabel):
        if label.objectName() in ("toolCardBody", "toolCardUnavailable"):
            assert not label.isVisible()
    assert tools.card_frames()["review"].toolTip() == "d"


def test_install_progress_still_shows_on_a_plain_card(qtbot, settings):
    page = LandingPage(settings=settings, tool_cards=_cards())
    qtbot.addWidget(page)
    page.show()
    page.tools().set_install_status("vision", "Installing…", busy=True)
    shown = [
        label.text()
        for label in page.tools().findChildren(QLabel)
        if label.isVisible() and label.objectName() == "toolCardUnavailable"
    ]
    assert shown == ["Installing…"]


def test_a_narrow_window_stacks_the_columns(qtbot, settings):
    page = LandingPage(settings=settings, tool_cards=_cards())
    qtbot.addWidget(page)

    page.resize(STACK_BELOW_PX + 200, 800)
    page.show()
    qtbot.waitUntil(lambda: not page.is_stacked())

    page.resize(STACK_BELOW_PX - 200, 800)
    qtbot.waitUntil(page.is_stacked)


# ------------------------------------------------------------- MainWindow


def test_every_tool_opens_from_the_landing_page(qtbot, main_window_factory, monkeypatch):
    """With nothing created or opened, each card reaches its window's opener."""
    window = main_window_factory(desktop_mode=True)
    assert window._stack.currentIndex() == PAGE_LANDING
    assert not window._has_open_experiment()

    opened: list[str] = []
    for name in (
        "_open_behavior_analysis",
        "_open_pose_batch",
        "_open_session_review",
        "_open_animal_annotator",
        "_open_multi_camera",
        "_on_gpu_check",
        "_on_install_gpu_support",
        "_on_open_plugins",
    ):
        monkeypatch.setattr(window, name, lambda n=name: opened.append(n))

    tools = window._landing_page.tools()
    keys = [card.key for card in tools.cards()]
    for key in keys:
        window._landing_page.tool_chosen.emit(key)

    assert len(opened) == len(keys)
    assert "_open_session_review" in opened
    assert "_open_pose_batch" in opened
    # Still on the landing page: opening a tool does not invent an experiment.
    assert window._stack.currentIndex() == PAGE_LANDING


def test_landing_offers_every_tools_tab_card_plus_plugins(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    on_tab = [c.key for c in window._tools_page.cards()]
    on_landing = [c.key for c in window._landing_page.tools().cards()]
    assert on_landing == [*on_tab, "plugins"]


def test_install_progress_reaches_both_sets_of_cards(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    pages = window._tool_pages()
    assert window._tools_page in pages
    assert window._landing_page.tools() in pages


def test_runner_mode_never_shows_the_landing_page(qtbot, main_window_factory):
    """The Pi layout is untouched: it opens on the operator view."""
    window = main_window_factory(desktop_mode=False)
    assert window._stack.currentIndex() != PAGE_LANDING


# ------------------------------------------------------------- recent list


def test_an_entry_that_raises_on_stat_is_hidden_not_fatal(settings, tmp_path, monkeypatch):
    """The launch failure: ``Path.exists`` raises ``PermissionError`` for a
    share you have lost access to, rather than returning False."""
    good = tmp_path / "good.glider"
    good.write_text("{}")
    denied = tmp_path / "denied.glider"
    settings.setValue(RECENT_KEY, [str(denied), str(good)])

    real_exists = Path.exists

    def _exists(self, *args, **kwargs):
        if self.name == "denied.glider":
            raise PermissionError(13, "Access is denied")
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", _exists)

    assert recent_experiments(settings) == [good]


def test_landing_page_builds_when_an_entry_raises(qtbot, settings, tmp_path, monkeypatch):
    settings.setValue(RECENT_KEY, [str(tmp_path / "gone.glider")])
    monkeypatch.setattr(
        Path, "exists", lambda self, *a, **k: (_ for _ in ()).throw(OSError(53, "netpath"))
    )
    page = LandingPage(settings=settings)
    qtbot.addWidget(page)
    assert page.recent_buttons() == []


def test_a_hung_network_path_cannot_hold_up_the_list(settings, tmp_path, monkeypatch):
    """A ``stat`` on a dead server blocks; the list must not wait for it."""
    good = tmp_path / "good.glider"
    good.write_text("{}")
    settings.setValue(RECENT_KEY, [str(tmp_path / "hung.glider"), str(good)])

    release = threading.Event()
    real_exists = Path.exists

    def _exists(self, *args, **kwargs):
        if self.name == "hung.glider":
            release.wait(30)
            return True
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", _exists)
    try:
        start = time.perf_counter()
        found = recent_experiments(settings, timeout=0.3)
        elapsed = time.perf_counter() - start
    finally:
        release.set()

    assert found == [good]
    assert elapsed < 5


def test_an_unreachable_entry_is_kept_for_when_its_drive_returns(settings, tmp_path):
    """Hidden on read, never dropped on write: saving something else must not
    forget the protocol on a share that happens to be offline today."""
    offline = tmp_path / "offline" / "protocol.glider"  # does not exist now
    settings.setValue(RECENT_KEY, [str(offline)])

    new = tmp_path / "new.glider"
    new.write_text("{}")
    remember_experiment(new, settings)

    stored = settings.value(RECENT_KEY, [], type=list)
    assert len(stored) == 2
    assert Path(stored[1]) == offline
    assert recent_experiments(settings) == [new]

    offline.parent.mkdir()
    offline.write_text("{}")
    assert recent_experiments(settings) == [new, offline]


def test_remember_and_forget_de_duplicate_without_touching_disk(settings, tmp_path, monkeypatch):
    def _no_resolve(self, *a, **k):  # resolving a dead share is itself a hang
        raise AssertionError("resolve() must not be called")

    monkeypatch.setattr(Path, "resolve", _no_resolve)
    path = tmp_path / "a.glider"
    remember_experiment(path, settings)
    remember_experiment(str(path), settings)
    assert len(settings.value(RECENT_KEY, [], type=list)) == 1
    forget_experiment(path, settings)
    assert settings.value(RECENT_KEY, [], type=list) == []


def test_probe_threads_are_daemons(settings, tmp_path, monkeypatch):
    """A hung probe must not stop GLIDER from exiting."""
    seen: list[bool] = []
    real_thread = threading.Thread

    def _thread(*args, **kwargs):
        seen.append(kwargs.get("daemon", False))
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(landing.threading, "Thread", _thread)
    settings.setValue(RECENT_KEY, [str(tmp_path / "x.glider")])
    recent_experiments(settings)
    assert seen == [True]


def test_the_tools_tab_is_name_and_icon_only_too(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    page = window._tools_page
    assert page.findChild(QLabel, "toolsSubheading") is None
    for label in page.findChildren(QLabel):
        if label.objectName() in ("toolCardBody", "toolCardUnavailable"):
            assert label.isHidden()
