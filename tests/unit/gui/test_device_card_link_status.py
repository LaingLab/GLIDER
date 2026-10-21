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

import pytest

from glider.gui.styles import colors
from glider.hal.base_board import ConnectionState
from glider.hal.value_spec import KIND_SWITCH, ActionValueSpec

pytestmark = pytest.mark.usefixtures("qtbot")


def _device_grid(qtbot, core):
    from glider.gui.runner.device_grid import DeviceCardGrid

    panel = DeviceCardGrid(core)
    qtbot.addWidget(panel)
    return panel, panel.update_states


def _controls_panel(qtbot, core):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    panel = RunnerDeviceControls(core.hardware_manager)
    qtbot.addWidget(panel)
    return panel, panel.refresh_values


PANELS = [
    pytest.param(_device_grid, id="device-grid"),
    pytest.param(_controls_panel, id="device-controls"),
]


def _card(qtbot, core, device, build, state):
    device.link_state = state
    # Initialized *and* gone is the whole point: the old check could not tell
    # this apart from a device that is genuinely up.
    device._initialized = True
    device.device_type = "Maimu"
    device.actions = {"set": lambda *a: None}
    device.value_spec = lambda n: ActionValueSpec(KIND_SWITCH, 0, 1)
    core.hardware_manager.devices = {"stim": device}
    panel, repaint = build(qtbot, core)
    if hasattr(panel, "refresh_devices"):
        panel.refresh_devices()
        return panel, repaint, panel.cards()["stim"]._ready_label
    panel.refresh()
    return panel, repaint, _RowLabel(panel._rows["stim"])


class _RowLabel:
    """Adapts a device row (link text + dot) to the card label's text/styleSheet."""

    def __init__(self, refs):
        self._refs = refs

    def text(self):
        return self._refs.link.text()

    def styleSheet(self):
        return self._refs.dot.styleSheet()


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
