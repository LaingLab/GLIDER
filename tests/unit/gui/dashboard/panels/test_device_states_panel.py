import pytest

from glider.gui.dashboard.panels.device_states_panel import DeviceStatesPanel

pytestmark = pytest.mark.usefixtures("qtbot")


@pytest.fixture
def panel(qtbot, mock_core):
    p = DeviceStatesPanel(mock_core)
    qtbot.addWidget(p)
    return p


def test_placeholder_shown_when_no_devices(panel, mock_core):
    # The shared mock_hardware_manager.devices is non-empty by default; clear it
    # first so the "no devices" placeholder path is exercised. Use isVisibleTo
    # (not isVisible) — offscreen, a widget with no shown ancestor reports
    # isVisible()==False regardless.
    mock_core.hardware_manager.devices = {}
    panel.refresh_devices()
    assert panel._runner_no_devices.isVisibleTo(panel) is True
    assert not panel._runner_device_cards


def test_cards_created_for_devices(panel, mock_core, mock_device):
    mock_core.hardware_manager.devices = {"dev1": mock_device}
    panel.refresh_devices()
    assert "dev1" in panel._runner_device_cards


def test_running_starts_refresh_timer(panel):
    panel.update_state("RUNNING")
    assert panel._device_refresh_timer.isActive()


def test_idle_stops_refresh_timer(panel):
    panel.update_state("RUNNING")
    panel.update_state("IDLE")
    assert not panel._device_refresh_timer.isActive()


def test_analog_card_renders_value_and_voltage(qtbot, mock_core, mock_device):
    mock_device.device_type = "AnalogInput"
    mock_device._last_value = 512
    mock_core.hardware_manager.devices = {"a1": mock_device}
    panel = DeviceStatesPanel(mock_core)
    qtbot.addWidget(panel)
    panel.refresh_devices()
    text = panel._runner_device_cards["a1"]._state_label.text()
    assert "512" in text
    assert "2.50V" in text  # 512/1023*5 = 2.50


def test_update_device_states_reflects_state_change(qtbot, mock_core, mock_device):
    mock_device.device_type = "DigitalOutput"
    mock_device._state = True
    mock_core.hardware_manager.devices = {"d1": mock_device}
    panel = DeviceStatesPanel(mock_core)
    qtbot.addWidget(panel)
    panel.refresh_devices()
    assert panel._runner_device_cards["d1"]._state_label.text() == "HIGH"
    mock_device._state = False
    panel._update_device_states()
    assert panel._runner_device_cards["d1"]._state_label.text() == "LOW"
