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
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QDockWidget, QWidget

from glider.gui.shell import AppShell, SidePanel, app_shell
from glider.gui.styles import colors
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


# ------------------------------------------------------------- opening size
#
# The first launch on a machine is the one path no saved geometry can rescue:
# ``AppShell.restore_layout`` clamps what it reads back, but there is nothing
# to read until a close has written one, so a fresh install got the configured
# 1400x900 applied raw. On a narrower display the OS centres that, the left
# edge lands at a negative x, and the Builder's first panel tab goes off the
# side of the screen.


def test_a_first_launch_window_fits_the_screen(qtbot, main_window_factory, monkeypatch):
    """End to end, with nothing saved: the window opens inside the screen."""
    available = QRect(0, 48, 1280, 752)
    monkeypatch.setattr(app_shell, "primary_available_geometry", lambda: available)

    window = main_window_factory(desktop_mode=True)  # a throwaway, empty QSettings

    assert window._settings.value("shell/geometry") is None
    assert available.contains(window.geometry())
    assert window.minimumWidth() <= available.width()
    assert window.minimumHeight() <= available.height()


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


def _open_analysis_dock(window, monkeypatch, tmp_path):
    """Open the Analysis dock the way the app does, with a stub panel.

    The real ``AnalysisPanel`` wants a finished recording on disk; what is
    under test is the dock, so the panel is the part that is stood in for.
    """
    import glider.gui.panels.analysis as analysis_module

    class _StubAnalysisPanel(QWidget):
        def load_recording(self, _directory) -> bool:
            return True

    monkeypatch.setattr(analysis_module, "AnalysisPanel", _StubAnalysisPanel)
    window._on_open_analysis_panel(str(tmp_path))
    assert window._analysis_dock is not None
    assert window._analysis_dock.isVisible()
    return window._analysis_dock


def test_the_analysis_dock_does_not_linger_over_the_operator_view(
    qtbot, main_window_factory, monkeypatch, tmp_path
):
    """Analysis is the one dock that stayed, and it is still a dock -- so it is
    still painted over the operator view, which is the whole of issue #39. The
    guard above only passes because nothing in it ever opens one."""
    window = _builder(main_window_factory)
    dock = _open_analysis_dock(window, monkeypatch, tmp_path)

    window.switch_to_runner()
    assert not dock.isVisible()

    window.switch_to_builder()
    assert dock.isVisible()


def test_switch_to_desktop_mode_brings_the_analysis_dock_back_too(
    qtbot, main_window_factory, monkeypatch, tmp_path
):
    """The dashboard's own "switch to desktop" button bypasses
    switch_to_builder, exactly as it does for the camera."""
    window = _builder(main_window_factory)
    dock = _open_analysis_dock(window, monkeypatch, tmp_path)

    window._toggle_view()
    assert not dock.isVisible()

    window._switch_to_desktop_mode()
    assert dock.isVisible()


def test_an_analysis_dock_the_user_closed_stays_closed(
    qtbot, main_window_factory, monkeypatch, tmp_path
):
    """Hiding it on the way out must not turn into showing it on the way back."""
    window = _builder(main_window_factory)
    dock = _open_analysis_dock(window, monkeypatch, tmp_path)
    dock.close()
    assert not dock.isVisible()

    window.switch_to_runner()
    window.switch_to_builder()

    assert not dock.isVisible()


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


# The window title is set once and never updated, so the strip's marker is the
# only unsaved-work indicator in the shell. These drive real edits -- a node
# dropped on the canvas, a node dragged -- rather than calling the refresh.


def _clean(window) -> None:
    """The state a just-saved experiment is in."""
    window._core.session._mark_clean()
    window._refresh_strip_experiment()
    assert not window._builder_view.strip.dirty_label().isVisible()


def test_dropping_a_node_raises_the_dirty_marker(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    _clean(window)

    # What the canvas emits when a node is dropped on it.
    window._graph_view.node_created.emit("DigitalWrite", 100.0, 100.0)

    assert window._core.session.is_dirty
    assert window._builder_view.strip.dirty_label().isVisible()


def test_moving_a_node_raises_the_dirty_marker(qtbot, main_window_factory):
    """A drag marks the session dirty and pushes nothing on the undo stack, so
    a marker wired to undo/redo alone would stay silent through it."""
    window = _builder(main_window_factory)
    window._graph_view.node_created.emit("DigitalWrite", 100.0, 100.0)
    node_id = next(iter(window._graph_view.nodes))
    _clean(window)

    window._graph_view.node_moved.emit(node_id, 220.0, 140.0)

    assert window._core.session.is_dirty
    assert window._builder_view.strip.dirty_label().isVisible()


def test_a_new_session_starts_the_marker_watching_again(qtbot, main_window_factory):
    """``new_session`` replaces the session object. A hook registered on the
    old one would go quiet for the rest of the run."""
    window = _builder(main_window_factory)
    window._core.session._dirty = False
    window._on_new()
    _clean(window)

    window._graph_view.node_created.emit("DigitalWrite", 10.0, 10.0)

    assert window._builder_view.strip.dirty_label().isVisible()


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


# ---------------------------------------------------------- peripheral link drops
#
# _on_device_link_change is the device-channel sibling of
# _on_hardware_connection_change above, and the three things that make it safe
# are only guaranteed by reading the code, not by anything CI would catch: it
# must fire only while an experiment is RUNNING, it must never pause the run
# (a ten-second BLE dropout should not end a two-hour session), and it must
# never reach for _show_hardware_disconnection_dialog, the modal reserved for
# a board going away. Each is pinned separately below.


def test_a_device_drop_during_a_run_notifies_without_pausing_or_the_dialog(
    qtbot, main_window_factory, monkeypatch
):
    """The one case where the operator must actually be told."""
    from glider.core.experiment_session import SessionState

    window = _builder(main_window_factory)
    window._core.session.state = SessionState.RUNNING

    notified = []
    dialog_calls = []
    run_async_calls = []
    monkeypatch.setattr(window, "_notify_user", lambda *a, **kw: notified.append((a, kw)))
    monkeypatch.setattr(
        window,
        "_show_hardware_disconnection_dialog",
        lambda *a, **kw: dialog_calls.append((a, kw)),
    )
    monkeypatch.setattr(window, "_run_async", lambda coro: run_async_calls.append(coro))

    window._on_device_link_change("stim_1", BoardConnectionState.DISCONNECTED)

    assert notified  # the operator is told
    assert dialog_calls == []  # never the modal reserved for a board going away
    assert run_async_calls == []  # never paused -- a BLE dropout is not a reason to stop


def test_error_and_disconnected_get_different_wording(qtbot, main_window_factory, monkeypatch):
    """ERROR is the reconnect loop's own terminal give-up state (12 attempts --
    see BLEDevice.MAX_RECONNECT_ATTEMPTS): retrying is over by the time it
    fires, so it must not tell the operator to sit tight the way a fresh
    DISCONNECTED does."""
    from glider.core.experiment_session import SessionState

    window = _builder(main_window_factory)
    window._core.session.state = SessionState.RUNNING

    bodies = []
    monkeypatch.setattr(window, "_notify_user", lambda title, message, **kw: bodies.append(message))

    window._on_device_link_change("stim_1", BoardConnectionState.DISCONNECTED)
    window._on_device_link_change("stim_1", BoardConnectionState.ERROR)

    disconnected_body, error_body = bodies
    assert "retrying" in disconnected_body.lower()
    assert "stopped retrying" in error_body.lower()
    assert "is retrying" not in error_body.lower()


def test_a_device_drop_when_not_running_notifies_nobody(qtbot, main_window_factory, monkeypatch):
    """A device that never held a link during an idle rig is not news."""
    window = _builder(main_window_factory)  # session defaults to IDLE

    notified = []
    monkeypatch.setattr(window, "_notify_user", lambda *a, **kw: notified.append((a, kw)))

    window._on_device_link_change("stim_1", BoardConnectionState.DISCONNECTED)

    assert notified == []


@pytest.mark.parametrize(
    "state", [BoardConnectionState.CONNECTED, BoardConnectionState.RECONNECTING]
)
def test_a_non_drop_transition_notifies_nobody_even_while_running(
    qtbot, main_window_factory, monkeypatch, state
):
    """Connecting or retrying is not a drop, RUNNING or not."""
    from glider.core.experiment_session import SessionState

    window = _builder(main_window_factory)
    window._core.session.state = SessionState.RUNNING

    notified = []
    monkeypatch.setattr(window, "_notify_user", lambda *a, **kw: notified.append((a, kw)))

    window._on_device_link_change("stim_1", state)

    assert notified == []


@pytest.mark.parametrize(
    "state",
    [
        BoardConnectionState.CONNECTED,
        BoardConnectionState.CONNECTING,
        BoardConnectionState.RECONNECTING,
        BoardConnectionState.DISCONNECTED,
        BoardConnectionState.ERROR,
    ],
)
def test_a_device_link_change_always_refreshes_the_readouts(
    qtbot, main_window_factory, monkeypatch, state
):
    """Both readouts follow every transition, not just the ones that notify."""
    window = _builder(main_window_factory)

    refreshed = []
    monkeypatch.setattr(window, "_refresh_hardware_readouts", lambda: refreshed.append(state))

    window._on_device_link_change("stim_1", state)

    assert refreshed == [state]


@pytest.mark.parametrize(
    "state",
    [
        BoardConnectionState.CONNECTED,
        BoardConnectionState.RECONNECTING,
        BoardConnectionState.DISCONNECTED,
        BoardConnectionState.ERROR,
    ],
)
def test_a_device_link_change_repaints_the_tree_and_regates_the_controls(
    qtbot, main_window_factory, monkeypatch, state
):
    """The readouts were the only two surfaces that followed a link change.

    HardwarePanel.refresh_tree runs only from a user action, so the documented
    "the row moves to Disconnected" was not delivered; and the Device Control
    panel's greyed-button guard was evaluated only at device-selection time, so
    a drop while the panel was open left every button live.
    """
    window = _builder(main_window_factory)

    trees, controls = [], []
    for panel in (window._hardware_panel, window._dash_hardware_panel):
        monkeypatch.setattr(panel, "refresh_link_states", lambda p=panel: trees.append(p))
    monkeypatch.setattr(
        window._device_control_panel, "refresh_link_state", lambda: controls.append(state)
    )

    window._on_device_link_change("stim_1", state)

    assert len(trees) == 2, "both hardware trees follow the link"
    assert controls == [state]


def test_a_device_link_change_does_not_rebuild_anything(qtbot, main_window_factory, monkeypatch):
    """A rebuild would throw away what the operator typed.

    refresh_tree emits hardware_changed, which fans out to
    DeviceControlPanel.refresh_devices and clears its combo; refresh_devices
    then re-runs _on_device_selected, which rebuilds the argument fields. A BLE
    blip the operator did not cause must not cost them a period and a duration.
    """
    window = _builder(main_window_factory)

    rebuilt = []
    for panel in (window._hardware_panel, window._dash_hardware_panel):
        monkeypatch.setattr(panel, "refresh_tree", lambda: rebuilt.append("tree"))
    monkeypatch.setattr(
        window._device_control_panel, "refresh_devices", lambda: rebuilt.append("combo")
    )

    window._on_device_link_change("stim_1", BoardConnectionState.DISCONNECTED)

    assert rebuilt == []


# ---------------------------------------------------- the two hardware readouts
#
# The strip and the status bar describe the same rig. They are allowed to say
# different amounts; they are not allowed to disagree. The status bar preferred
# any connected board and stopped looking, so an Uno up and a Pi in ERROR read
# "Arduino Uno - Connected" at the bottom while the strip showed a red dot.


def _conn_text(window) -> str:
    return window._conn_label.text()


def _conn_is_green(window) -> bool:
    return colors.SUCCESS in window._conn_dot.styleSheet()


def test_the_status_bar_will_not_call_the_rig_healthy_while_a_board_is_down(
    qtbot, main_window_factory
):
    window = _builder(main_window_factory)
    boards = window._core.hardware_manager._boards
    boards["uno_1"] = _FakeBoard("Arduino Uno", BoardConnectionState.CONNECTED)
    boards["pi_1"] = _FakeBoard("Raspberry Pi", BoardConnectionState.ERROR)

    window._update_connection_status()

    assert not _conn_is_green(window)
    assert colors.ERROR in window._conn_dot.styleSheet()
    assert "pi_1" in _conn_text(window)  # and it names the one to go and look at


def test_the_status_bar_agrees_with_the_strip_about_a_healthy_rig(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    window._core.hardware_manager._boards["uno_1"] = _FakeBoard(
        "Arduino Uno", BoardConnectionState.CONNECTED
    )

    window._update_connection_status()

    assert _conn_is_green(window)
    assert "uno_1" in _conn_text(window)
    assert "Connected" in _conn_text(window)


def test_the_status_bar_counts_a_rig_of_more_than_one(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    boards = window._core.hardware_manager._boards
    boards["uno_1"] = _FakeBoard("Arduino Uno", BoardConnectionState.CONNECTED)
    boards["uno_2"] = _FakeBoard("Arduino Uno", BoardConnectionState.CONNECTED)

    window._update_connection_status()

    assert _conn_is_green(window)
    assert "2" in _conn_text(window)


def test_a_board_still_shaking_hands_is_not_down_and_is_not_up(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    window._core.hardware_manager._boards["uno_1"] = _FakeBoard(
        "Arduino Uno", BoardConnectionState.RECONNECTING
    )

    window._update_connection_status()

    assert not _conn_is_green(window)
    assert colors.WARNING in window._conn_dot.styleSheet()
    assert "Reconnecting" in _conn_text(window)


def test_an_empty_rig_still_says_so(qtbot, main_window_factory):
    window = _builder(main_window_factory)
    window._update_connection_status()
    assert _conn_text(window) == "No board"


def test_both_readouts_follow_the_roster_together(qtbot, main_window_factory):
    """Wiring one and not the other is how they came to disagree."""
    window = _builder(main_window_factory)
    window._core.hardware_manager._boards["uno_1"] = _FakeBoard(
        "Arduino Uno", BoardConnectionState.CONNECTED
    )
    window._hardware_panel.refresh_tree()
    assert _conn_is_green(window)
    assert window._builder_view.strip.device_names() == ["uno_1"]

    window._core.session._dirty = False
    window._on_new()

    assert _conn_text(window) == "No board"
    assert window._builder_view.strip.device_names() == []


# ------------------------------------------------------------- the board roster
#
# The dots had drop *detection* wired -- a board changing state reaches the
# strip -- and nothing at all wired to the board **set**. A board state
# transition is not a registration event, so a rig that was emptied or replaced
# left the strip rendering the old roster, green, while the manager held
# nothing. These three drive the real paths rather than the refresh helper.


def test_clearing_the_rig_empties_the_strip(qtbot, main_window_factory):
    """File -> New clears the hardware manager. The strip has to notice."""
    window = _builder(main_window_factory)
    strip = window._builder_view.strip
    window._core.hardware_manager.add_board("uno_1", "telemetrix", port="COM9")
    window._refresh_strip_devices()
    assert strip.device_names() == ["uno_1"]

    window._core.session._dirty = False  # so File -> New does not ask to save
    window._on_new()

    assert window._core.hardware_manager.boards == {}
    assert strip.device_names() == []
    assert strip.device_dots() == []


def test_opening_an_experiment_over_another_replaces_the_roster(
    qtbot, main_window_factory, tmp_path, monkeypatch
):
    """Open B over A and the strip must describe B's rig, not accumulate."""
    from PyQt6.QtWidgets import QFileDialog

    from glider.core.experiment_session import BoardConfig

    window = _builder(main_window_factory)
    strip = window._builder_view.strip

    # Experiment B, on disk: one Raspberry Pi and nothing else.
    window._core.session.hardware.boards.append(BoardConfig(id="pi_b", driver_type="raspberry_pi"))
    path = tmp_path / "b.glider"
    window._core.save_session(str(path))

    # Experiment A, in the window: one Arduino, and nothing of B's.
    window._core.new_session()
    window._core.hardware_manager.clear()
    window._core.hardware_manager.add_board("uno_a", "telemetrix", port="COM9")
    window._refresh_strip_devices()
    assert strip.device_names() == ["uno_a"]

    window._core.session._dirty = False
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )
    window._on_open()

    assert list(window._core.hardware_manager.boards) == ["pi_b"]
    assert strip.device_names() == ["pi_b"]


def test_a_board_registered_through_the_hardware_panel_reaches_the_strip(
    qtbot, main_window_factory
):
    """Every add and remove in the Hardware panel ends in ``refresh_tree``."""
    window = _builder(main_window_factory)
    strip = window._builder_view.strip
    assert strip.device_names() == []

    window._core.hardware_manager.add_board("uno_1", "telemetrix", port="COM9")
    window._hardware_panel.refresh_tree()

    assert strip.device_names() == ["uno_1"]


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
