from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QPushButton

from glider.gui.panels.hardware_panel import HardwarePanel
from glider.hal.base_board import ConnectionState

pytestmark = pytest.mark.usefixtures("qtbot")


def _panel(qtbot, mock_hardware_manager, show_add_buttons=True):
    panel = HardwarePanel(
        mock_hardware_manager,
        session_fn=lambda: None,
        run_async_fn=lambda c: None,
        show_add_buttons=show_add_buttons,
    )
    qtbot.addWidget(panel)
    return panel


def test_add_buttons_shown_by_default(qtbot, mock_hardware_manager):
    panel = _panel(qtbot, mock_hardware_manager)
    texts = [b.text() for b in panel.findChildren(QPushButton)]
    assert "+ Board" in texts
    assert "+ Device" in texts


def test_add_buttons_hidden_when_disabled(qtbot, mock_hardware_manager):
    panel = _panel(qtbot, mock_hardware_manager, show_add_buttons=False)
    assert not any(b.text() in ("+ Board", "+ Device") for b in panel.findChildren(QPushButton))


# --- a link that moves without the hardware map changing ----------------------


class _LinkDevice:
    """A peripheral with a link of its own, on the mock board."""

    device_type = "Maimu"
    name = "Stimulator"
    pins: dict = {}

    def __init__(self, board, state):
        self.board = board
        self.link_state = state
        self._config = SimpleNamespace(settings={"address": "AA:BB"})


def _tree_with_device(qtbot, mock_hardware_manager, state):
    board = next(iter(mock_hardware_manager.boards.values()))
    device = _LinkDevice(board, state)
    mock_hardware_manager.devices = {"stim": device}
    mock_hardware_manager.get_device = lambda dev_id: device if dev_id == "stim" else None
    panel = _panel(qtbot, mock_hardware_manager)
    panel.refresh_tree()
    return panel, device


def _status_words(panel):
    tree = panel._hardware_tree
    return [
        tree.topLevelItem(b).child(c).text(2)
        for b in range(tree.topLevelItemCount())
        for c in range(tree.topLevelItem(b).childCount())
    ]


def test_refresh_link_states_repaints_the_status_column(qtbot, mock_hardware_manager):
    """The documented behaviour: a dropped peripheral's row moves to Disconnected."""
    panel, device = _tree_with_device(qtbot, mock_hardware_manager, ConnectionState.CONNECTED)
    assert _status_words(panel) == ["Ready"]

    device.link_state = ConnectionState.DISCONNECTED
    panel.refresh_link_states()

    assert _status_words(panel) == ["Disconnected"]


def test_refresh_link_states_does_not_announce_a_hardware_change(qtbot, mock_hardware_manager):
    """hardware_changed fans out to a full Device Control panel rebuild, which
    would clear the combo and any argument values typed into it. A link moving
    is not a change to the hardware map."""
    panel, device = _tree_with_device(qtbot, mock_hardware_manager, ConnectionState.CONNECTED)
    emitted = []
    panel.hardware_changed.connect(lambda: emitted.append(True))

    device.link_state = ConnectionState.RECONNECTING
    panel.refresh_link_states()

    assert emitted == []
    assert _status_words(panel) == ["Reconnecting…"]
