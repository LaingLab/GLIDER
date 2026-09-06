"""The Tools menu entry point.

The window is useless if it cannot be reached; before this it existed only as
a checkbox buried in the camera panel.
"""

from __future__ import annotations


def _menu(window, title: str):
    for action in window.menuBar().actions():
        if action.text().replace("&", "") == title:
            return action.menu()
    raise AssertionError(f"no {title} menu")


def test_tools_menu_offers_multi_camera(main_window_factory):
    window = main_window_factory()
    titles = [a.text().replace("&", "") for a in _menu(window, "Tools").actions()]
    assert any("Multi-Camera Recording" in t for t in titles), titles


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
