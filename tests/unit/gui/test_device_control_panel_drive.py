"""DeviceControlPanel drives devices via the shared device_drive helper.

Regression guard for the device_drive refactor (Task 4): the ON button must
still end up calling the device's set_state through device_drive.set_digital
once the coroutine scheduled via run_async_fn is awaited, and the existing
status-label / dialog behavior must be unchanged.
"""

from glider.gui.panels.device_control_panel import DeviceControlPanel


class _Board:
    is_connected = True


class _Device:
    def __init__(self, device_type="DigitalOutput"):
        self.id = "dev_1"
        self.name = "led"
        self.device_type = device_type
        self.board = _Board()
        self._initialized = True
        self.pins = {}
        self.calls = []

    async def set_state(self, value):
        self.calls.append(("set_state", value))


class _HW:
    def __init__(self, device):
        self.devices = {device.id: device}
        self._device = device

    def get_device(self, device_id):
        return self._device if device_id == self._device.id else None


def _select_only_device(panel):
    panel.refresh_devices()
    panel._device_combo.setCurrentIndex(1)  # index 0 is the "-- Select Device --" placeholder


async def test_on_button_drives_device_via_set_digital(qtbot):
    device = _Device()
    scheduled = []
    panel = DeviceControlPanel(_HW(device), run_async_fn=scheduled.append)
    qtbot.addWidget(panel)
    _select_only_device(panel)

    panel._on_btn.click()

    # _set_digital_output schedules exactly one coroutine via run_async_fn.
    assert len(scheduled) == 1
    await scheduled[0]

    assert device.calls == [("set_state", True)]
    assert panel._device_status_label.text() == "Status: Output set to ON"


async def test_toggle_button_drives_device_via_toggle_digital(qtbot):
    class _ToggleDevice(_Device):
        def __init__(self):
            super().__init__()
            self.state = False

        async def toggle(self):
            self.calls.append(("toggle",))
            self.state = not self.state

    device = _ToggleDevice()
    scheduled = []
    panel = DeviceControlPanel(_HW(device), run_async_fn=scheduled.append)
    qtbot.addWidget(panel)
    _select_only_device(panel)

    panel._toggle_btn.click()

    assert len(scheduled) == 1
    await scheduled[0]

    assert device.calls == [("toggle",)]
    assert panel._device_status_label.text() == "Status: Output toggled"
