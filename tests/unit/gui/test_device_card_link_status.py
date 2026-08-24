"""The two device-card panels read the link, not "has this been set up".

Task 6 converted the hardware tree and the Device Control panel; these two
were missed. RunnerPanel's cards repaint on a timer, so they went on asserting
a green "Ready" for a peripheral whose link was down -- while the status strip
showed a red dot for the same device two inches away. That is the exact
disagreement glider.gui.device_status exists to prevent, and the Runner card
is what an operator watches during a live run.

Both panels are driven through the same parametrised factory: they render the
same card and the requirement on them is identical, so a fix that lands on one
and not the other should fail here.
"""

from unittest.mock import MagicMock

import pytest

from glider.gui.styles import colors
from glider.hal.base_board import ConnectionState

pytestmark = pytest.mark.usefixtures("qtbot")


def _runner_panel(qtbot, core):
    from glider.gui.panels.runner_panel import RunnerPanel

    panel = RunnerPanel(core, MagicMock())
    qtbot.addWidget(panel)
    return panel, panel._update_runner_device_states


def _states_panel(qtbot, core):
    from glider.gui.dashboard.panels.device_states_panel import DeviceStatesPanel

    panel = DeviceStatesPanel(core)
    qtbot.addWidget(panel)
    return panel, panel._update_device_states


PANELS = [
    pytest.param(_runner_panel, id="runner"),
    pytest.param(_states_panel, id="dashboard"),
]


def _card(qtbot, core, device, build, state):
    device.link_state = state
    # Initialized *and* gone is the whole point: the old check could not tell
    # this apart from a device that is genuinely up.
    device._initialized = True
    device.device_type = "Maimu"
    core.hardware_manager.devices = {"stim": device}
    panel, repaint = build(qtbot, core)
    panel.refresh_devices()
    return panel, repaint, panel._runner_device_cards["stim"]._ready_label


@pytest.mark.parametrize("build", PANELS)
def test_a_live_link_still_reads_ready_in_green(qtbot, mock_core, mock_device, build):
    _panel, _repaint, label = _card(qtbot, mock_core, mock_device, build, ConnectionState.CONNECTED)
    assert label.text() == "Ready"
    assert colors.SUCCESS in label.styleSheet()


@pytest.mark.parametrize("build", PANELS)
def test_a_dropped_link_is_not_a_green_ready(qtbot, mock_core, mock_device, build):
    _panel, _repaint, label = _card(
        qtbot, mock_core, mock_device, build, ConnectionState.DISCONNECTED
    )
    assert label.text() == "Disconnected"
    assert colors.SUCCESS not in label.styleSheet()
    assert colors.ERROR in label.styleSheet()


@pytest.mark.parametrize("build", PANELS)
def test_a_retrying_link_reads_as_retrying(qtbot, mock_core, mock_device, build):
    _panel, _repaint, label = _card(
        qtbot, mock_core, mock_device, build, ConnectionState.RECONNECTING
    )
    assert label.text() == "Reconnecting…"
    assert colors.WARNING in label.styleSheet()


@pytest.mark.parametrize("build", PANELS)
def test_the_repaint_follows_a_drop(qtbot, mock_core, mock_device, build):
    """The timer-driven update path, not just the card that was built once."""
    _panel, repaint, label = _card(qtbot, mock_core, mock_device, build, ConnectionState.CONNECTED)
    assert label.text() == "Ready"

    mock_device.link_state = ConnectionState.DISCONNECTED
    repaint()

    assert label.text() == "Disconnected"
    assert colors.SUCCESS not in label.styleSheet()
