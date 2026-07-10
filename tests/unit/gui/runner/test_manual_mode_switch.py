from types import SimpleNamespace

import pytest

from glider.core.experiment_session import ExperimentSession

pytestmark = pytest.mark.usefixtures("qtbot")


class _HW:
    def __init__(self, connected=True, devices=None):
        self._c = connected
        self.devices = devices if devices is not None else {}

    def is_any_board_connected(self):
        return self._c


class _Core:
    def __init__(self, session, connected=True, devices=None):
        self.session = session
        self.hardware_manager = _HW(connected, devices)


@pytest.fixture
def core():
    return _Core(ExperimentSession())


def test_running_shows_device_controls(qtbot, core):
    from glider.gui.runner.manual_control_panel import ManualControlPanel

    p = ManualControlPanel(core)
    qtbot.addWidget(p)
    p.update_state("RUNNING")
    assert p._stack.currentIndex() == 1


def test_idle_shows_function_grid(qtbot, core):
    from glider.gui.runner.manual_control_panel import ManualControlPanel

    p = ManualControlPanel(core)
    qtbot.addWidget(p)
    p.update_state("RUNNING")
    p.update_state("IDLE")
    assert p._stack.currentIndex() == 0


def test_manual_chain_running_does_not_switch(qtbot, core):
    from glider.gui.runner.manual_control_panel import ManualControlPanel

    p = ManualControlPanel(core)
    qtbot.addWidget(p)
    p.set_running("node-123")
    assert p._stack.currentIndex() == 0


def test_running_refreshes_device_controls(qtbot):
    from glider.gui.runner.manual_control_panel import ManualControlPanel

    dev = SimpleNamespace(device_type="DigitalOutput", id="d1")
    core = _Core(ExperimentSession(), devices={"d1": dev})
    p = ManualControlPanel(core)
    qtbot.addWidget(p)
    p.update_state("RUNNING")
    assert "d1" in p._device_controls._buttons


def test_panel_reemits_device_control_signals(qtbot, core):
    from glider.gui.runner.manual_control_panel import ManualControlPanel

    p = ManualControlPanel(core)
    qtbot.addWidget(p)
    p.update_state("RUNNING")

    with qtbot.waitSignal(p.set_digital_requested) as sig:
        p._device_controls.set_digital_requested.emit("d1", True)
    assert sig.args == ["d1", True]

    with qtbot.waitSignal(p.toggle_digital_requested) as sig:
        p._device_controls.toggle_digital_requested.emit("d1")
    assert sig.args == ["d1"]

    with qtbot.waitSignal(p.set_pwm_requested) as sig:
        p._device_controls.set_pwm_requested.emit("p1", 128)
    assert sig.args == ["p1", 128]
