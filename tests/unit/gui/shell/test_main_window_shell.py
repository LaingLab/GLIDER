"""The Builder hosted in the shell, and the docks gone.

These tests guard the *architectural* claim of the change, not just the
rendering: the Builder is a **page of the existing stack**, so switching to the
operator view swaps the whole frame away. That is what lets
``_hide_builder_docks`` / ``_show_builder_docks`` / ``_BUILDER_DOCK_ATTRS`` be
deleted rather than ported. ``test_builder_dock_workaround_is_gone`` and
``test_nothing_lingers_over_the_operator_view`` are a pair: the first asserts
the workaround is absent, the second asserts the behaviour it used to buy is
still there. Either one alone would let the bug back in.

Every window here is ``show()``n before anything is asserted about it, and the
window object is held by the test for its whole body -- ``qtbot.addWidget``
keeps only a weak reference, and a dropped top level takes its children with it
("wrapped C/C++ object has been deleted").
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QDockWidget, QWidget

from glider.gui.shell import AppShell, SidePanel
from glider.hal.base_board import BoardConnectionState

pytestmark = pytest.mark.usefixtures("qtbot")


LEFT_KEYS = ["nodes", "hardware", "control", "files"]
RIGHT_KEYS = ["properties", "camera"]


def _builder(main_window_factory):
    """A shown desktop MainWindow sitting on the Builder page."""
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()
    return window


def _is_descendant(widget: QWidget, ancestor: QWidget) -> bool:
    parent = widget.parent()
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent()
    return False


class _FakeBoard:
    """Just enough board for the strip: a name and a connection state.

    ``port`` is here because a *connected* board is read by more than the strip:
    ``compute_readiness`` asks the manager for a connected board's description,
    from a deferred refresh that lands mid-test. Without it the double raises
    inside the Qt event loop, which pytest-qt reports as a failure of whatever
    test happened to be running.
    """

    def __init__(self, name: str, state: BoardConnectionState, port: str | None = None) -> None:
        self.name = name
        self.state = state
        self.port = port

    @property
    def is_connected(self) -> bool:
        return self.state is BoardConnectionState.CONNECTED

    async def disconnect(self) -> None:
        """Reached by the core's shutdown when a test leaves a board behind."""
        self.state = BoardConnectionState.DISCONNECTED


# ------------------------------------------------------------------ the frame


def test_builder_page_is_the_shell(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    assert isinstance(window._builder_view, AppShell)
    assert window._stack.widget(0) is window._builder_view
    assert window._stack.currentWidget() is window._builder_view


def test_centre_is_the_graph(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    assert window._builder_view.centre is window._graph_view
    assert window._graph_view.isVisible()


def test_both_panels_are_side_panels_with_the_expected_tabs(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    shell = window._builder_view
    assert isinstance(shell.left, SidePanel)
    assert isinstance(shell.right, SidePanel)
    assert shell.left.keys() == LEFT_KEYS
    assert shell.right.keys() == RIGHT_KEYS


def test_panels_are_re_hosted_not_rebuilt(qtbot, main_window_factory):
    """Every panel in a tab is the *same object* the docks used to hold."""
    window = _builder(main_window_factory)
    shell = window._builder_view

    assert shell.left.widget_for("nodes") is window._node_library_panel
    assert shell.left.widget_for("hardware") is window._hardware_panel
    assert shell.left.widget_for("control") is window._device_control_panel
    assert shell.left.widget_for("files") is window._files_panel

    # Properties is hosted behind a swappable container -- the node editor
    # replaces its contents on every selection -- so identity is asserted on
    # what the container currently holds.
    assert shell.right.widget_for("properties") is window._properties_host
    assert window._properties_host.widget() is window._properties_widget

    # The camera is the one panel that moves between views, so it lives in a
    # slot the tab owns, exactly as the dashboard quadrant does.
    assert _is_descendant(window._camera_panel, shell.right)


def test_node_editor_writes_into_the_properties_host(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    assert window._node_editor._properties_dock is window._properties_host

    replacement = QWidget()
    window._properties_host.setWidget(replacement)
    assert window._properties_host.widget() is replacement


# ------------------------------------------------- the deleted dock workaround


def test_builder_dock_workaround_is_gone(qtbot, main_window_factory):
    """The frame is a stack page now, so the issue-#39 helpers have no job."""
    window = _builder(main_window_factory)
    for gone in (
        "_hide_builder_docks",
        "_show_builder_docks",
        "_BUILDER_DOCK_ATTRS",
        "_builder_dock_visibility",
        "_node_library_dock",
        "_properties_dock",
        "_hardware_dock",
        "_control_dock",
        "_camera_dock",
        "_files_dock",
    ):
        assert not hasattr(window, gone), gone


def test_no_builder_docks_are_created(qtbot, main_window_factory):
    """Analysis is the only dock left, and it is built lazily."""
    window = _builder(main_window_factory)
    assert window.findChildren(QDockWidget) == []


def test_nothing_lingers_over_the_operator_view(qtbot, main_window_factory):
    """The regression issue #39 named, now guarded by structure not a helper."""
    window = _builder(main_window_factory)
    panels = (
        window._node_library_panel,
        window._hardware_panel,
        window._device_control_panel,
        window._files_panel,
        window._properties_host,
        window._camera_panel,
    )

    window.switch_to_runner()
    assert window._stack.currentWidget() is window._operator_view
    assert not window._builder_view.isVisible()
    for panel in panels:
        assert not panel.isVisible(), panel.objectName() or type(panel).__name__

    window.switch_to_builder()
    assert window._builder_view.isVisible()
    assert window._node_library_panel.isVisible()  # the left panel's first tab
    assert window._properties_host.isVisible()  # the right panel's first tab


def test_switch_to_desktop_mode_leaves_the_builder_showing(qtbot, main_window_factory):
    """The dashboard's own "switch to desktop" button bypasses switch_to_builder."""
    window = _builder(main_window_factory)
    window._toggle_view()  # F11 into the dashboard
    assert not window._builder_view.isVisible()

    window._switch_to_desktop_mode()
    assert window._stack.currentIndex() == 0
    assert window._builder_view.isVisible()
    assert window._node_library_panel.isVisible()


# ---------------------------------------------------------------- the camera


def test_camera_survives_builder_dashboard_builder(qtbot, main_window_factory):
    """The full round trip: a live feed is what a stale parent would destroy."""
    window = _builder(main_window_factory)
    shell = window._builder_view
    camera = window._camera_panel
    assert camera is not None
    assert _is_descendant(camera, shell.right)

    window.switch_to_runner()
    assert _is_descendant(camera, window._dashboard_view)
    assert window._camera_panel is camera  # never duplicated

    window.switch_to_builder()
    assert _is_descendant(camera, shell.right)
    assert window._camera_panel is camera
    # The widget is still a live C++ object, not a dangling wrapper.
    assert camera.objectName() is not None

    shell.right.set_current("camera")
    assert camera.isVisible()


def test_camera_returns_to_the_builder_from_runner_mode(qtbot, main_window_factory):
    """Runner -> desktop must land the single panel in the shell, twice over."""
    window = main_window_factory(desktop_mode=False)  # runner: no builder panels yet
    window.show()
    camera = window._camera_panel

    window._switch_to_desktop_mode()  # first: builds the panels
    assert _is_descendant(camera, window._builder_view.right)

    window.switch_to_runner()
    assert _is_descendant(camera, window._runner_shell)

    window._switch_to_desktop_mode()  # repeat: panels exist, camera must still move
    assert _is_descendant(camera, window._builder_view.right)
    assert window._camera_panel is camera


# ------------------------------------------------------------ the View menu


def test_view_menu_toggles_collapse_and_expand_the_panels(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    shell = window._builder_view

    for action, panel in (
        (window._left_panel_action, shell.left),
        (window._right_panel_action, shell.right),
    ):
        assert action.isCheckable()
        assert action.isChecked()
        action.trigger()
        assert not panel.expanded
        action.trigger()
        assert panel.expanded


def test_view_menu_toggles_keep_the_strip_in_step(qtbot, main_window_factory):
    """A button describing a state that is not on screen is worse than none."""
    window = _builder(main_window_factory)
    shell = window._builder_view
    strip = shell.strip

    window._left_panel_action.trigger()
    assert not shell.left.expanded
    assert not strip.left_toggle().isChecked()

    window._right_panel_action.trigger()
    assert not shell.right.expanded
    assert not strip.right_toggle().isChecked()

    window._left_panel_action.trigger()
    assert strip.left_toggle().isChecked()


def test_collapsing_from_the_strip_moves_the_menu_action(qtbot, main_window_factory):
    """The return path: the panel is the truth, the action reflects it."""
    window = _builder(main_window_factory)
    shell = window._builder_view

    shell.strip.left_toggle().click()
    assert not shell.left.expanded
    assert not window._left_panel_action.isChecked()

    shell.left.set_expanded(True)
    assert window._left_panel_action.isChecked()


def test_view_menu_has_no_dangling_dock_toggles(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    view_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text().replace("&", "") == "View"
    )
    texts = [action.text().replace("&", "") for action in view_menu.actions()]
    assert "Left Panel" in texts
    assert "Right Panel" in texts
    assert "Node Library" not in texts  # the dock toggles went with the docks


# ------------------------------------------------------------------ the strip


def test_strip_shows_the_experiment_name_and_dirty_state(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    strip = window._builder_view.strip

    window._core.session.name = "Open Field Day 3"
    window._refresh_strip_experiment()
    assert strip.name_label().text() == "Open Field Day 3"
    assert strip.dirty_label().isVisible()  # setting the name marks it dirty

    window._core.session._dirty = False
    window._refresh_strip_experiment()
    assert not strip.dirty_label().isVisible()


def test_strip_run_state_follows_the_session(qtbot, main_window_factory):
    from glider.core.experiment_session import SessionState

    window = _builder(main_window_factory)
    strip = window._builder_view.strip

    window._on_core_state_change(SessionState.IDLE)
    assert strip.run_state() == "idle"

    window._on_core_state_change(SessionState.RUNNING)
    assert strip.run_state() == "running"

    window._on_core_state_change(SessionState.ERROR)
    assert strip.run_state() == "error"


def test_strip_says_recording_while_the_recorder_is_running(qtbot, main_window_factory):
    from glider.core.experiment_session import SessionState

    window = _builder(main_window_factory)
    window._core.data_recorder._recording = True
    window._on_core_state_change(SessionState.RUNNING)
    assert window._builder_view.strip.run_state() == "recording"


def test_strip_reports_a_paused_run_as_still_live(qtbot, main_window_factory):
    from glider.core.experiment_session import SessionState

    window = _builder(main_window_factory)
    strip = window._builder_view.strip
    window._on_core_state_change(SessionState.PAUSED)
    assert strip.run_state() == "running"
    assert "Paused" in strip.pill().text()


# --------------------------------------------------------------- device dots


def test_device_dots_come_from_real_board_state(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    strip = window._builder_view.strip
    assert strip.device_names() == []  # an empty rig says nothing

    boards = window._core.hardware_manager._boards
    boards["uno_1"] = _FakeBoard("Arduino Uno", BoardConnectionState.CONNECTED)
    boards["pi_1"] = _FakeBoard("Raspberry Pi", BoardConnectionState.ERROR)
    window._refresh_strip_devices()

    assert strip.device_names() == ["uno_1", "pi_1"]
    assert [dot.property("state") for dot in strip.device_dots()] == ["ok", "error"]


def test_two_boards_of_the_same_type_are_told_apart(qtbot, main_window_factory):
    """``board.name`` is the board *type*, so two Unos share it. A label that
    cannot say *which* board to go and look at is the thing the dot already
    could not say."""
    window = _builder(main_window_factory)
    boards = window._core.hardware_manager._boards
    boards["uno_left"] = _FakeBoard("Arduino Uno", BoardConnectionState.CONNECTED)
    boards["uno_right"] = _FakeBoard("Arduino Uno", BoardConnectionState.ERROR)
    window._refresh_strip_devices()

    names = window._builder_view.strip.device_names()
    assert names == ["uno_left", "uno_right"]
    assert len(set(names)) == len(names)


def test_a_board_dot_still_names_its_type_on_hover(qtbot, main_window_factory):
    """The label says which board; the tooltip says what to go and look for."""
    window = _builder(main_window_factory)
    window._core.hardware_manager._boards["uno_1"] = _FakeBoard(
        "Arduino Uno", BoardConnectionState.CONNECTED
    )
    window._refresh_strip_devices()

    tip = window._builder_view.strip.device_chips()[0].toolTip()
    assert "uno_1" in tip
    assert "Arduino Uno" in tip


@pytest.mark.parametrize(
    ("board_state", "word", "not_word"),
    [
        (BoardConnectionState.CONNECTING, "connecting", "reconnecting"),
        (BoardConnectionState.RECONNECTING, "reconnecting", "warn"),
        (BoardConnectionState.DISCONNECTED, "disconnected", "warn"),
        (BoardConnectionState.CONNECTED, "connected", "warn"),
    ],
)
def test_the_raw_board_state_survives_the_mapping(
    qtbot, main_window_factory, board_state, word, not_word
):
    """Connecting and Reconnecting are both amber, and they are not the same
    thing: one is a handshake in flight, the other a board that has already
    dropped mid-recording. The tooltip is where that distinction lives -- and
    ``warn`` is our word for the colour, not the board's word for itself."""
    window = _builder(main_window_factory)
    window._core.hardware_manager._boards["uno_1"] = _FakeBoard("Arduino Uno", board_state)
    window._refresh_strip_devices()

    tip = window._builder_view.strip.device_chips()[0].toolTip().lower()
    assert word in tip
    assert not_word not in tip


@pytest.mark.parametrize(
    ("board_state", "expected"),
    [
        (BoardConnectionState.CONNECTED, "ok"),
        (BoardConnectionState.CONNECTING, "warn"),
        (BoardConnectionState.RECONNECTING, "warn"),
        (BoardConnectionState.DISCONNECTED, "error"),
        (BoardConnectionState.ERROR, "error"),
    ],
)
def test_only_a_connected_board_is_ever_green(qtbot, main_window_factory, board_state, expected):
    """Painting `reconnecting` green is the one behaviour that would make the
    strip actively harmful, so every state is pinned, not just the happy one."""
    window = _builder(main_window_factory)
    window._core.hardware_manager._boards["a"] = _FakeBoard("Board", board_state)
    window._refresh_strip_devices()

    dot = window._builder_view.strip.device_dots()[0]
    assert dot.property("state") == expected
    if board_state is not BoardConnectionState.CONNECTED:
        assert dot.property("state") != "ok"


def test_an_unrecognised_board_state_is_neutral_not_green(qtbot, main_window_factory):
    """Driver vocabularies widen; the strip must not guess in their favour."""
    window = _builder(main_window_factory)
    window._core.hardware_manager._boards["a"] = _FakeBoard("Board", "calibrating")
    window._refresh_strip_devices()

    strip = window._builder_view.strip
    assert strip.device_dots()[0].property("state") == "unknown"
    assert "calibrating" in strip.device_chips()[0].toolTip()


def test_a_board_dropping_reaches_the_strip(qtbot, main_window_factory):
    """The whole reason the strip exists: a drop mid-run must be visible."""
    window = _builder(main_window_factory)
    boards = window._core.hardware_manager._boards
    board = _FakeBoard("Arduino", BoardConnectionState.CONNECTED)
    boards["a"] = board
    window._refresh_strip_devices()
    assert window._builder_view.strip.device_dots()[0].property("state") == "ok"

    board.state = BoardConnectionState.RECONNECTING
    window._on_hardware_connection_change("a", BoardConnectionState.RECONNECTING)
    assert window._builder_view.strip.device_dots()[0].property("state") == "warn"


# ----------------------------------------------------------- layout on restart


def test_the_builder_layout_survives_a_restart(qtbot, main_window_factory, tmp_path):
    """The Builder reset itself every launch before this; nothing saved it.

    ``AppShell`` knows how to write and read a layout but is a page of a stack,
    not a window, so the window is what has to call it -- on close, and again
    once the tabs exist to restore into.
    """
    from PyQt6.QtCore import QSettings

    settings = QSettings(str(tmp_path / "restart.ini"), QSettings.Format.IniFormat)

    first = main_window_factory(desktop_mode=True, settings=settings)
    first.show()
    first._builder_view.left.set_expanded(False)
    first._builder_view.right.set_current("camera")
    first.close()

    second = main_window_factory(desktop_mode=True, settings=settings)
    second.show()
    assert not second._builder_view.left.expanded
    assert second._builder_view.right.current_key() == "camera"
