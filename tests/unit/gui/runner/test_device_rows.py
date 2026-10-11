"""Each device section opens with one status row: link dot, name, type, live value."""

from types import SimpleNamespace

import pytest

from glider.hal.value_spec import KIND_SWITCH, ActionValueSpec

pytestmark = pytest.mark.usefixtures("qtbot")


def _dev(state=None, device_type="DigitalOutput", link=None):
    specs = {"set": ActionValueSpec(KIND_SWITCH, 0, 1)}
    return SimpleNamespace(
        device_type=device_type,
        actions={"set": lambda *a: None},
        value_spec=lambda name: specs.get(name),
        _state=state,
        link_state=link,
    )


def _controls(qtbot, devices):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    w = RunnerDeviceControls(SimpleNamespace(devices=devices))
    qtbot.addWidget(w)
    w.refresh()
    return w


def test_row_shows_live_value_and_type(qtbot):
    w = _controls(qtbot, {"cue_led": _dev(state=True)})
    row = w._rows["cue_led"]
    assert row.value.text() == "HIGH"


def test_refresh_values_follows_the_device(qtbot):
    dev = _dev(state=False)
    w = _controls(qtbot, {"cue_led": dev})
    assert w._rows["cue_led"].value.text() == "LOW"
    dev._state = True
    w.refresh_values()
    assert w._rows["cue_led"].value.text() == "HIGH"


def test_poll_runs_only_while_running(qtbot):
    w = _controls(qtbot, {"cue_led": _dev(state=False)})
    w.update_state("RUNNING")
    assert w._value_timer.isActive()
    w.update_state("STOPPED")
    assert not w._value_timer.isActive()


def test_manual_change_mid_run_is_labelled(qtbot):
    w = _controls(qtbot, {"cue_led": _dev(state=False)})
    w.update_state("RUNNING")
    w.on_action_succeeded("cue_led", "set", True)
    assert w._rows["cue_led"].manual.text().startswith("set by you ")
    assert w._rows["cue_led"].manual.isVisibleTo(w)


def test_manual_change_outside_a_run_is_not_labelled(qtbot):
    w = _controls(qtbot, {"cue_led": _dev(state=False)})
    w.on_action_succeeded("cue_led", "set", True)
    assert not w._rows["cue_led"].manual.isVisibleTo(w)


def test_manual_set_repaints_the_value_when_not_polling(qtbot):
    dev = _dev(state=False)
    w = _controls(qtbot, {"cue_led": dev})
    dev._state = True
    w.on_action_succeeded("cue_led", "set", True)
    assert w._rows["cue_led"].value.text() == "HIGH"
