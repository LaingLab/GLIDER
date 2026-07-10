from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("qtbot")


def _hw(devices):
    return SimpleNamespace(devices=devices)


def test_digital_on_emits_signal(qtbot):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    dev = SimpleNamespace(device_type="DigitalOutput", id="d1")
    w = RunnerDeviceControls(_hw({"d1": dev}))
    qtbot.addWidget(w)
    w.refresh()
    with qtbot.waitSignal(w.set_digital_requested) as sig:
        w._buttons["d1"]["on"].click()
    assert sig.args == ["d1", True]


def test_digital_off_emits_signal(qtbot):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    dev = SimpleNamespace(device_type="DigitalOutput", id="d1")
    w = RunnerDeviceControls(_hw({"d1": dev}))
    qtbot.addWidget(w)
    w.refresh()
    with qtbot.waitSignal(w.set_digital_requested) as sig:
        w._buttons["d1"]["off"].click()
    assert sig.args == ["d1", False]


def test_digital_toggle_emits_signal(qtbot):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    dev = SimpleNamespace(device_type="DigitalOutput", id="d1")
    w = RunnerDeviceControls(_hw({"d1": dev}))
    qtbot.addWidget(w)
    w.refresh()
    with qtbot.waitSignal(w.toggle_digital_requested) as sig:
        w._buttons["d1"]["toggle"].click()
    assert sig.args == ["d1"]


def test_pwm_slider_emits_signal(qtbot):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    dev = SimpleNamespace(device_type="PWMOutput", id="p1")
    w = RunnerDeviceControls(_hw({"p1": dev}))
    qtbot.addWidget(w)
    w.refresh()
    with qtbot.waitSignal(w.set_pwm_requested) as sig:
        w._sliders["p1"].setValue(200)
    assert sig.args == ["p1", 200]


def test_only_output_devices_shown(qtbot):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    devs = {"a": SimpleNamespace(device_type="AnalogInput", id="a")}
    w = RunnerDeviceControls(_hw(devs))
    qtbot.addWidget(w)
    w.refresh()
    assert w._buttons == {} and w._sliders == {}


def test_refresh_rebuilds_cleanly(qtbot):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    dev = SimpleNamespace(device_type="DigitalOutput", id="d1")
    w = RunnerDeviceControls(_hw({"d1": dev}))
    qtbot.addWidget(w)
    w.refresh()
    w.refresh()  # second refresh must not duplicate/leak
    assert set(w._buttons.keys()) == {"d1"}


def test_input_device_has_read_button(qtbot):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    dev = SimpleNamespace(device_type="DigitalInput", id="in1")
    w = RunnerDeviceControls(_hw({"in1": dev}))
    qtbot.addWidget(w)
    w.refresh()
    assert "in1" in w._read_buttons
    with qtbot.waitSignal(w.read_requested) as sig:
        w._read_buttons["in1"].click()
    assert sig.args == ["in1"]


def test_set_read_value_updates_label(qtbot):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    dev = SimpleNamespace(device_type="AnalogInput", id="a1")
    w = RunnerDeviceControls(_hw({"a1": dev}))
    qtbot.addWidget(w)
    w.refresh()
    w.set_read_value("a1", "512\n2.50V")
    assert "512" in w._value_labels["a1"].text()
