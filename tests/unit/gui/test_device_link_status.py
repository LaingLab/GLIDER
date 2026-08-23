"""Device link state as the GUI renders it.

One vocabulary module, because the tree, the Device Control panel and the
status strip were each inventing their own word for the same state -- which
is how the status bar came to read "Connected" beside a red dot.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QPushButton

from glider.gui.device_status import link_is_usable, link_status_text, link_strip_state
from glider.gui.panels.device_control_panel import DeviceControlPanel
from glider.hal.base_board import ConnectionState

pytestmark = pytest.mark.usefixtures("qtbot")


# --- the vocabulary -----------------------------------------------------------


@pytest.mark.parametrize(
    "state,expected",
    [
        (ConnectionState.CONNECTED, "Ready"),
        (ConnectionState.CONNECTING, "Connecting…"),
        (ConnectionState.RECONNECTING, "Reconnecting…"),
        (ConnectionState.DISCONNECTED, "Disconnected"),
        (ConnectionState.ERROR, "Error"),
    ],
)
def test_status_text(state, expected):
    assert link_status_text(state) == expected


@pytest.mark.parametrize(
    "state,expected",
    [
        (ConnectionState.CONNECTED, "ok"),
        (ConnectionState.CONNECTING, "warn"),
        (ConnectionState.RECONNECTING, "warn"),
        (ConnectionState.DISCONNECTED, "error"),
        (ConnectionState.ERROR, "error"),
    ],
)
def test_strip_state(state, expected):
    assert link_strip_state(state) == expected


def test_an_unknown_state_is_never_green():
    """A state nobody recognises does not get the benefit of the doubt."""
    assert link_strip_state(object()) == "unknown"
    assert link_status_text(object()) == "Unknown"


@pytest.mark.parametrize(
    "state,usable",
    [
        (ConnectionState.CONNECTED, True),
        (ConnectionState.RECONNECTING, False),
        (ConnectionState.DISCONNECTED, False),
        (ConnectionState.ERROR, False),
        (ConnectionState.CONNECTING, False),
    ],
)
def test_usable_only_when_connected(state, usable):
    assert link_is_usable(state) is usable


# --- the Device Control panel -------------------------------------------------


class _Device:
    device_type = "Maimu"
    name = "Stimulator"
    _initialized = True

    def __init__(self, state=ConnectionState.CONNECTED):
        self.link_state = state
        self.owns_link = True
        self.board = SimpleNamespace(is_connected=True)

    async def on(self):
        pass

    async def off(self):
        pass

    @property
    def actions(self):
        return {"on": self.on, "off": self.off}


def _panel(qtbot, device):
    manager = SimpleNamespace(
        devices={"dev1": device},
        get_device=lambda dev_id: device if dev_id == "dev1" else None,
    )
    panel = DeviceControlPanel(manager, lambda coro: coro.close())
    qtbot.addWidget(panel)
    panel.refresh_devices()
    panel._device_combo.setCurrentIndex(1)
    return panel


def test_panel_status_reads_the_link(qtbot):
    panel = _panel(qtbot, _Device(ConnectionState.RECONNECTING))
    assert "Reconnecting" in panel._device_status_label.text()


def test_panel_status_says_ready_when_up(qtbot):
    panel = _panel(qtbot, _Device(ConnectionState.CONNECTED))
    assert "Ready" in panel._device_status_label.text()


def test_action_buttons_are_dead_while_the_link_is_down(qtbot):
    """Offering a press that is certain to fail is worse than not offering it."""
    panel = _panel(qtbot, _Device(ConnectionState.DISCONNECTED))
    buttons = panel._actions_widget.findChildren(QPushButton)
    assert buttons, "expected on/off buttons to be built"
    assert all(not b.isEnabled() for b in buttons)


def test_action_buttons_are_live_when_the_link_is_up(qtbot):
    panel = _panel(qtbot, _Device(ConnectionState.CONNECTED))
    buttons = panel._actions_widget.findChildren(QPushButton)
    assert buttons and all(b.isEnabled() for b in buttons)
