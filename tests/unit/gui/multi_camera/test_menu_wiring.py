"""The entry point onto the Multi-Camera window.

The window is useless if it cannot be reached; before this it existed only as
a checkbox buried in the camera panel. It then lived on the Tools menu, and
now on the Analyze tab -- the Tools menu still holds it, but off the bar, so
the card is the reachable route and the one worth testing.
"""

from __future__ import annotations


def test_the_analyze_tab_offers_multi_camera(main_window_factory):
    window = main_window_factory()
    titles = [card.title for card in window._analyze_page.cards()]
    assert "Multi-Camera Recording" in titles, titles


def test_the_card_opens_the_window(main_window_factory, qtbot):
    """The card is wired to the same handler the menu item used, rather than to
    a second copy of the open-once bookkeeping."""
    window = main_window_factory()
    window._analyze_page.tool_chosen.emit("multicam")
    assert window._multi_camera_window is not None
    qtbot.addWidget(window._multi_camera_window)


def test_opening_it_twice_reuses_one_window(main_window_factory, qtbot):
    window = main_window_factory()
    window._open_multi_camera()
    first = window._multi_camera_window
    qtbot.addWidget(first)
    window._open_multi_camera()
    assert window._multi_camera_window is first


def test_it_shares_the_core_manager_and_recorder(main_window_factory, qtbot):
    window = main_window_factory()
    window._open_multi_camera()
    tool = window._multi_camera_window
    qtbot.addWidget(tool)
    assert tool._manager is window._core.multi_camera_manager
    assert tool._recorder is window._core.multi_video_recorder
