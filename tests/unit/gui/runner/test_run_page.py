import pytest
from PyQt6.QtWidgets import QVBoxLayout

from glider.core.experiment_session import ExperimentSession, FlowConfig, NodeConfig, RunField
from glider.gui.runner.run_page import PiRunPage

pytestmark = pytest.mark.usefixtures("qtbot")


def _real_session(core, *, board=True, start_node=True):
    """A real session (the default mock's flow is not iterable as nodes)."""
    session = ExperimentSession()
    nodes = [NodeConfig(id="s", node_type="StartExperiment")] if start_node else []
    session._flow = FlowConfig(nodes=nodes)
    core.session = session
    core.hardware_manager.is_any_board_connected = lambda: board
    core.last_flow_duration_s = None
    return session


def _page(qtbot, core):
    core.last_flow_duration_s = None
    page = PiRunPage(core)
    qtbot.addWidget(page)
    return page


def test_clock_follows_the_control(qtbot, mock_core):
    page = PiRunPage(mock_core)
    qtbot.addWidget(page)
    assert page.clock.objectName() == "piRunClock"
    assert page.clock.text() == "00:00.00"
    page.control.elapsed_updated.emit("01:02.03")
    assert page.clock.text() == "01:02.03"


def test_required_field_blocks_start(qtbot, mock_core):
    session = _real_session(mock_core)
    session.metadata.run_fields = [RunField("dam_id", "Dam ID", True)]
    page = _page(qtbot, mock_core)
    assert not page.control._start_btn.isEnabled()
    page.header._tags.value_edited.emit("dam_id", "D-12")
    page.control.refresh()
    assert page.control._start_btn.isEnabled()


def test_camera_slot_hidden_until_asked(qtbot, mock_core):
    page = PiRunPage(mock_core)
    qtbot.addWidget(page)
    assert isinstance(page.camera_slot.layout(), QVBoxLayout)
    assert page.camera_slot.layout().contentsMargins().top() == 0
    assert not page.camera_slot.isVisibleTo(page)
    page.set_camera_visible(True)
    assert page.camera_slot.isVisibleTo(page)
    page.set_camera_visible(False)
    assert not page.camera_slot.isVisibleTo(page)


def test_parts_are_in_the_documented_order(qtbot, mock_core):
    page = _page(qtbot, mock_core)
    assert page.objectName() == "piRunPage"
    layout = page.layout()
    parts = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert parts == [page.header, page.clock, page.devices, page.camera_slot, page.control]
    assert layout.stretch(2) == 1


def test_the_header_and_control_are_touch_variants(qtbot, mock_core):
    page = _page(qtbot, mock_core)
    assert not page.header._title.isVisibleTo(page.header)
    assert not page.header._picker.isVisibleTo(page.header)
    assert not page.control._timer.isVisibleTo(page.control)


def test_update_state_reaches_header_control_devices_and_clock(qtbot, mock_core):
    _real_session(mock_core)
    page = _page(qtbot, mock_core)
    page.update_state("RUNNING")
    assert page.header._pill.text() == "RUNNING"
    assert page.control.is_live
    assert page.devices._poll.isActive()
    assert page.clock.property("live") is True

    page.update_state("STOPPED")
    assert page.header._pill.text() == "READY"
    assert not page.control.is_live
    assert not page.devices._poll.isActive()
    assert page.clock.property("live") is False


def test_clock_is_restyled_when_live_flips(qtbot, mock_core):
    _real_session(mock_core)
    page = _page(qtbot, mock_core)
    page.setStyleSheet('QLabel#piRunClock[live="true"] { font-size: 41px; }')
    page.clock.ensurePolished()
    assert page.clock.font().pixelSize() != 41
    page.update_state("RUNNING")
    page.clock.ensurePolished()
    assert page.clock.font().pixelSize() == 41
    page.update_state("STOPPED")
    page.clock.ensurePolished()
    assert page.clock.font().pixelSize() != 41


def test_readiness_feeds_the_header_pill(qtbot, mock_core):
    session = _real_session(mock_core, board=False)
    session.hardware.boards.append(object())  # a board is configured, none connected
    page = _page(qtbot, mock_core)
    assert page.header._pill.text() == "NOT READY"  # synced at construction
    mock_core.hardware_manager.is_any_board_connected = lambda: True
    page.control.refresh()
    assert page.header._pill.text() == "READY"


@pytest.mark.parametrize("name", ["start", "stop", "pause", "resume"])
def test_control_signals_are_forwarded(qtbot, mock_core, name):
    page = _page(qtbot, mock_core)
    with qtbot.waitSignal(getattr(page, f"{name}_requested"), timeout=1000):
        getattr(page.control, f"{name}_requested").emit()


def test_refresh_refreshes_all_three(qtbot, mock_core, monkeypatch):
    page = _page(qtbot, mock_core)
    calls = []
    monkeypatch.setattr(page.header, "refresh", lambda: calls.append("header"))
    monkeypatch.setattr(page.control, "refresh", lambda: calls.append("control"))
    monkeypatch.setattr(page.devices, "refresh_devices", lambda: calls.append("devices"))
    page.refresh()
    assert calls == ["header", "control", "devices"]
