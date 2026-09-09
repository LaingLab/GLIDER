"""The tab bar, the landing page, and how MainWindow drives both.

The window-level tests here are the ones that matter: the tab bar is wired in
both directions, and a one-way wiring looks completely fine until something
switches pages without touching the bar.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings

from glider.gui.main_window import (
    PAGE_BUILDER,
    PAGE_EXPERIMENT,
    PAGE_LANDING,
    PAGE_OPERATOR,
)
from glider.gui.shell.landing import (
    MAX_RECENT,
    LandingPage,
    forget_experiment,
    recent_experiments,
    remember_experiment,
)
from glider.gui.shell.tab_bar import TAB_KEYS, ShellTabBar


@pytest.fixture
def settings(tmp_path) -> QSettings:
    """A throwaway settings file, so no test touches the real recent list."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def _experiment(tmp_path, name: str) -> Path:
    path = tmp_path / f"{name}.glider"
    path.write_text("{}")
    return path


# --------------------------------------------------------------- the tab bar


def test_pressing_a_tab_emits_its_key(qtbot):
    bar = ShellTabBar()
    qtbot.addWidget(bar)

    with qtbot.waitSignal(bar.tab_selected) as caught:
        bar.buttons()["run"].click()

    assert caught.args == ["run"]
    assert bar.current() == "run"


def test_pressing_the_current_tab_emits_nothing(qtbot):
    """Re-pressing the active tab would otherwise re-run the page switch --
    which, on the Dashboard, tears the camera panel out and puts it back."""
    bar = ShellTabBar()
    qtbot.addWidget(bar)
    bar.buttons()["experiment"].click()

    emitted: list[str] = []
    bar.tab_selected.connect(emitted.append)
    bar.buttons()["experiment"].click()

    assert emitted == []


def test_set_current_moves_the_highlight_without_emitting(qtbot):
    """The return path. If this emitted, following a page change would bounce
    straight back into the switch that caused it."""
    bar = ShellTabBar()
    qtbot.addWidget(bar)

    emitted: list[str] = []
    bar.tab_selected.connect(emitted.append)
    bar.set_current("run")

    assert emitted == []
    assert bar.current() == "run"
    assert bar.buttons()["run"].isChecked()


def test_set_current_ignores_an_unknown_key(qtbot):
    bar = ShellTabBar()
    qtbot.addWidget(bar)
    bar.set_current("nonsense")
    assert bar.current() == TAB_KEYS[0]


# ---------------------------------------------------------- the recent store


def test_recent_is_newest_first_and_deduplicated(tmp_path, settings):
    first = _experiment(tmp_path, "first")
    second = _experiment(tmp_path, "second")

    remember_experiment(first, settings)
    remember_experiment(second, settings)
    remember_experiment(first, settings)  # re-opened: moves up, does not repeat

    assert recent_experiments(settings) == [first, second]


def test_recent_drops_paths_that_no_longer_exist(tmp_path, settings):
    """Pruned on read, so a list looked at after a file moved is still true."""
    kept = _experiment(tmp_path, "kept")
    gone = _experiment(tmp_path, "gone")
    remember_experiment(kept, settings)
    remember_experiment(gone, settings)
    gone.unlink()

    assert recent_experiments(settings) == [kept]


def test_recent_is_capped(tmp_path, settings):
    made = [_experiment(tmp_path, f"e{i}") for i in range(MAX_RECENT + 4)]
    for path in made:
        remember_experiment(path, settings)

    listed = recent_experiments(settings)
    assert len(listed) == MAX_RECENT
    assert listed[0] == made[-1]


def test_forget_removes_one_entry(tmp_path, settings):
    kept = _experiment(tmp_path, "kept")
    dropped = _experiment(tmp_path, "dropped")
    remember_experiment(kept, settings)
    remember_experiment(dropped, settings)

    forget_experiment(dropped, settings)

    assert recent_experiments(settings) == [kept]


def test_recent_survives_a_junk_settings_value(settings):
    """A settings file must never be able to stop the page drawing."""
    settings.setValue("landing/recent", "not-a-list")
    assert recent_experiments(settings) == []


# ------------------------------------------------------------- landing page


def test_landing_lists_recent_and_emits_the_path(qtbot, tmp_path, settings):
    path = _experiment(tmp_path, "protocol")
    remember_experiment(path, settings)

    page = LandingPage(settings=settings)
    qtbot.addWidget(page)

    rows = page.recent_buttons()
    assert [b.text() for b in rows] == ["protocol"]
    # The label is the stem; the full path is the tooltip, because two
    # protocols with the same stem in different folders are ordinary.
    assert rows[0].toolTip() == str(path)

    with qtbot.waitSignal(page.recent_requested) as caught:
        rows[0].click()
    assert caught.args == [str(path)]


def test_landing_with_no_recent_says_so(qtbot, settings):
    page = LandingPage(settings=settings)
    qtbot.addWidget(page)
    assert page.recent_buttons() == []


# ------------------------------------------------- MainWindow's navigation


def test_desktop_startup_lands_on_the_landing_page(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    assert window._stack.currentIndex() == PAGE_LANDING
    # Nothing is open, so the three tabs have nothing to switch between.
    assert not window._tab_bar.isVisible()


def test_runner_startup_skips_the_landing_page(qtbot, main_window_factory):
    """The Pi is a kiosk beside a rig: it opens on the run, never on a chooser."""
    window = main_window_factory(desktop_mode=False)
    assert window._stack.currentIndex() == PAGE_OPERATOR


def test_new_from_landing_enters_the_experiment_tab(qtbot, main_window_factory):
    """Not the Dashboard: the next thing to do after New is say what this
    experiment is and who is in it."""
    window = main_window_factory(desktop_mode=True)
    window.show()

    window._landing_page.new_requested.emit()

    assert window._stack.currentIndex() == PAGE_EXPERIMENT
    assert window._tab_bar.isVisible()
    assert window._tab_bar.current() == "experiment"


def test_experiment_is_the_first_tab(qtbot):
    """Order is part of the design, not an accident of the dict."""
    assert TAB_KEYS == ("experiment", "dashboard", "run")


def test_tabs_switch_pages(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()

    window._tab_bar.buttons()["run"].click()
    assert window._stack.currentIndex() == PAGE_OPERATOR

    window._tab_bar.buttons()["experiment"].click()
    assert window._stack.currentIndex() == PAGE_EXPERIMENT

    window._tab_bar.buttons()["dashboard"].click()
    assert window._stack.currentIndex() == PAGE_BUILDER


def test_the_highlight_follows_a_switch_that_did_not_start_at_the_bar(qtbot, main_window_factory):
    """The wiring's return path. Without it the View menu changes the page and
    leaves the bar claiming you are still where you were."""
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()

    window._enter_dashboard()  # what View > Dashboard calls
    assert window._tab_bar.current() == "run"

    window.switch_to_builder()
    assert window._tab_bar.current() == "dashboard"


def test_toggle_view_from_the_landing_page_reaches_the_operator_view(qtbot, main_window_factory):
    """Regression: ``_toggle_view`` used to ask "am I on page 0?", which stopped
    meaning "am I on the Builder?" the moment the stack grew past two pages. On
    the landing page it fell through and went to the Builder, so F11 appeared to
    do nothing."""
    window = main_window_factory(desktop_mode=True)
    window.show()
    assert window._stack.currentIndex() == PAGE_LANDING

    window._toggle_view()

    assert window._stack.currentIndex() == PAGE_OPERATOR


def test_opening_a_recent_experiment_records_it_and_leaves_landing(
    qtbot, main_window_factory, tmp_path
):
    window = main_window_factory(desktop_mode=True)
    window.show()
    saved = tmp_path / "recorded.glider"
    window._core.save_session(str(saved))

    window._landing_page.recent_requested.emit(str(saved))

    assert window._stack.currentIndex() == PAGE_EXPERIMENT
    assert recent_experiments(window._settings) == [saved.resolve()]


def test_a_failed_open_from_landing_stays_on_landing(
    qtbot, main_window_factory, tmp_path, monkeypatch
):
    """A mis-click must not dump the user onto an empty Builder and call that
    'opened'."""
    from PyQt6.QtWidgets import QMessageBox

    window = main_window_factory(desktop_mode=True)
    window.show()
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)

    window._landing_page.recent_requested.emit(str(tmp_path / "not-there.glider"))

    assert window._stack.currentIndex() == PAGE_LANDING


def test_create_main_window_does_not_navigate_off_the_landing_page(qtbot, tmp_path):
    """Regression, found by running the app rather than by the suite.

    ``create_main_window`` applies the startup view mode by calling
    ``switch_to_builder``, which navigated straight off the landing page the
    window had just chosen -- so the landing page was unreachable in the real
    app while every test above still passed, because they construct MainWindow
    directly and never come through this path.
    """
    import asyncio

    from PyQt6.QtWidgets import QApplication

    from glider.__main__ import create_main_window
    from glider.core.glider_core import GliderCore

    core = GliderCore()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(core.initialize())
    try:
        window = create_main_window(QApplication.instance(), core, force_mode="builder")
        qtbot.addWidget(window)
        assert window._stack.currentIndex() == PAGE_LANDING
        assert window.is_on_landing()
    finally:
        core.session._mark_clean()
        loop.run_until_complete(core.shutdown())
        loop.close()
