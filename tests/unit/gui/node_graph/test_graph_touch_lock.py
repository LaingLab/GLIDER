"""Touch pan and the edit lock on the node graph (Pi touchscreen Runner layout)."""

import pytest
from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt
from PyQt6.QtGui import QContextMenuEvent, QDropEvent, QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QGraphicsView, QMenu

from glider.gui.node_graph.graph_view import NodeGraphView


@pytest.fixture
def view(qtbot):
    v = NodeGraphView()
    qtbot.addWidget(v)
    return v


def test_desktop_default_is_unchanged(view):
    assert view.dragMode() == QGraphicsView.DragMode.RubberBandDrag
    assert view.isInteractive() and not view.is_locked()
    assert not view._overlay.lock_btn.isVisibleTo(view)
    assert view._overlay.zoom_in_btn.width() == 32


def test_touch_mode_pans_with_one_finger_and_shows_the_lock(view):
    view.set_touch_mode(True)
    assert view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
    assert view._overlay.lock_btn.isVisibleTo(view)
    assert view._overlay.zoom_in_btn.width() >= 44
    view.set_touch_mode(False)
    assert view.dragMode() == QGraphicsView.DragMode.RubberBandDrag
    assert not view._overlay.lock_btn.isVisibleTo(view)
    assert view._overlay.zoom_in_btn.width() == 32


def test_lock_makes_the_view_look_only(qtbot, view):
    with qtbot.waitSignal(view.locked_changed) as sig:
        view.set_locked(True)
    assert sig.args == [True]
    assert not view.isInteractive()
    view.set_locked(False)
    assert view.isInteractive()


def test_locked_changed_fires_only_on_a_change(qtbot, view):
    with qtbot.assertNotEmitted(view.locked_changed):
        view.set_locked(False)


def test_lock_button_says_what_a_tap_will_do(view):
    btn = view._overlay.lock_btn
    assert btn.objectName() == "graphLockButton"
    assert btn.property("locked") is False
    assert "lock" in btn.text()
    view.set_locked(True)
    assert btn.property("locked") is True
    assert "edit" in btn.text()


def test_locked_view_refuses_library_drops(qtbot, view):
    view.set_locked(True)
    mime = QMimeData()
    mime.setText("node:Delay")
    drop = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    with qtbot.assertNotEmitted(view.node_created):
        view.dropEvent(drop)
    assert not drop.isAccepted()


def test_unlocked_view_still_takes_library_drops(qtbot, view):
    mime = QMimeData()
    mime.setText("node:Delay")
    drop = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    with qtbot.waitSignal(view.node_created):
        view.dropEvent(drop)


def test_locked_view_ignores_delete(qtbot, view, monkeypatch):
    view.set_locked(True)
    view._selected_nodes = ["n1"]
    monkeypatch.setattr(view, "remove_node", lambda _id: None)
    key = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    with qtbot.assertNotEmitted(view.node_deleted):
        view.keyPressEvent(key)


def test_locked_view_has_no_context_menu(view, monkeypatch):
    shown = []
    monkeypatch.setattr(QMenu, "exec", lambda self, *a: shown.append(self))
    view.set_locked(True)
    view.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(5, 5)))
    assert shown == []


def test_lock_button_toggles_and_can_be_disabled(qtbot, view):
    view.set_touch_mode(True)
    view.set_locked(True)
    qtbot.mouseClick(view._overlay.lock_btn, Qt.MouseButton.LeftButton)
    assert not view.is_locked()
    view.set_lock_enabled(False)
    assert not view._overlay.lock_btn.isEnabled()


@pytest.mark.parametrize("locked, wiring", [(False, True), (True, False)])
def test_pressing_a_port_wires_only_while_unlocked(view, locked, wiring):
    node = view.add_node("n1", "Delay", 0, 0)
    port = node.add_output_port("out")
    view.set_locked(locked)
    pos = QPointF(view.mapFromScene(port.get_scene_center()))
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        pos,
        view.viewport().mapToGlobal(pos),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mousePressEvent(press)
    assert view._connecting is wiring
