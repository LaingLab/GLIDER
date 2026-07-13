import pytest
from PyQt6.QtCore import QMimeData, QPointF, Qt
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import QApplication, QLabel

from glider.gui.dashboard.quadrant_host import QuadrantHost


@pytest.fixture
def host(qtbot):
    h = QuadrantHost(quadrant_id="top_left")
    qtbot.addWidget(h)
    return h


def test_set_panel_installs_and_titles(host):
    panel = QLabel("hello")
    host.set_panel(panel, panel_key="run_control", title="Run Control")
    assert host.current_panel_key == "run_control"
    assert host.title_text() == "Run Control"
    assert panel.parent() is not None


def test_set_panel_returns_previous_panel_without_destroying_it(host):
    first = QLabel("first")
    second = QLabel("second")
    host.set_panel(first, "run_control", "Run Control")
    returned = host.set_panel(second, "camera", "Camera")
    assert returned is first
    assert first.text() == "first"


def test_picker_emits_panel_selected(host, qtbot):
    host.set_panel(QLabel(), "run_control", "Run Control")
    with qtbot.waitSignal(host.panel_selected, timeout=1000) as blocker:
        host.trigger_pick("camera")
    assert blocker.args == ["top_left", "camera"]


def _drop_event(source_id: str) -> QDropEvent:
    # QDropEvent does not take ownership of the QMimeData; parent it to the
    # QApplication so it outlives this helper and the event can read it.
    mime = QMimeData()
    mime.setParent(QApplication.instance())
    mime.setData(QuadrantHost._MIME, source_id.encode("utf-8"))
    return QDropEvent(
        QPointF(0, 0),
        Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def test_drop_from_other_quadrant_emits_swap(host, qtbot):
    with qtbot.waitSignal(host.swap_requested, timeout=1000) as blocker:
        host.dropEvent(_drop_event("bottom_right"))
    assert blocker.args == ["bottom_right", "top_left"]


def test_drop_from_same_quadrant_is_noop(host):
    received = []
    host.swap_requested.connect(lambda s, t: received.append((s, t)))
    host.dropEvent(_drop_event("top_left"))
    assert received == []


def test_drop_with_wrong_mime_is_noop(host):
    received = []
    host.swap_requested.connect(lambda s, t: received.append((s, t)))
    mime = QMimeData()
    mime.setParent(QApplication.instance())
    mime.setText("not-a-quadrant")
    ev = QDropEvent(
        QPointF(0, 0),
        Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    host.dropEvent(ev)
    assert received == []
