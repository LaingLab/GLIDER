"""Generated runner controls: one widget per action, chosen from value_spec.

Confirms the action-keyed factory renders the right widget per kind, dispatches
by action name (so any device works), commits sliders on release (not per tick),
and skips redundant primitives (On/Off/Toggle -> one switch).
"""

from types import SimpleNamespace

import pytest

from glider.hal.value_spec import KIND_SWITCH, KIND_WHOLE, ActionValueSpec

pytestmark = pytest.mark.usefixtures("qtbot")


def _dev(specs: dict, device_type="X"):
    """Fake device: specs maps action name -> ActionValueSpec or None."""
    actions = {name: (lambda *a: None) for name in specs}
    return SimpleNamespace(
        device_type=device_type,
        actions=actions,
        value_spec=lambda name: specs.get(name),
    )


def _hw(devices):
    return SimpleNamespace(devices=devices)


def _widget(w, dev_id, action):
    from glider.gui.runner.device_controls import RunnerDeviceControls  # noqa: F401

    w.refresh()
    return w._widgets[(dev_id, action)]


def _controls(devices):
    from glider.gui.runner.device_controls import RunnerDeviceControls

    w = RunnerDeviceControls(_hw(devices))
    return w


def test_switch_action_renders_a_switch_and_emits_bool(qtbot):
    dev = _dev({"set": ActionValueSpec(KIND_SWITCH, 0, 1), "on": None, "off": None, "toggle": None})
    w = _controls({"d1": dev})
    qtbot.addWidget(w)
    switch = _widget(w, "d1", "set")["switch"]
    with qtbot.waitSignal(w.action_write_requested) as sig:
        switch.click()  # unchecked -> checked
    assert sig.args == ["d1", "set", True]
    # On/Off/Toggle are subsumed by the switch — not rendered.
    assert ("d1", "on") not in w._widgets
    assert ("d1", "toggle") not in w._widgets


def test_whole_action_renders_slider_and_commits_on_release(qtbot):
    dev = _dev({"set": ActionValueSpec(KIND_WHOLE, 0, 4095)})
    w = _controls({"p1": dev})
    qtbot.addWidget(w)
    widgets = _widget(w, "p1", "set")
    slider = widgets["slider"]
    slider.setValue(3000)  # dragging updates readout but must NOT emit
    assert widgets["spin"].value() == 3000
    with qtbot.waitSignal(w.action_write_requested) as sig:
        slider.sliderReleased.emit()  # release commits
    assert sig.args == ["p1", "set", 3000]


def test_spinbox_precise_entry_emits(qtbot):
    dev = _dev({"set": ActionValueSpec(KIND_WHOLE, 0, 4095)})
    w = _controls({"p1": dev})
    qtbot.addWidget(w)
    spin = _widget(w, "p1", "set")["spin"]
    spin.setValue(1234)
    with qtbot.waitSignal(w.action_write_requested) as sig:
        spin.editingFinished.emit()
    assert sig.args == ["p1", "set", 1234]


def test_huge_range_falls_back_to_entry_only(qtbot):
    dev = _dev({"set": ActionValueSpec(KIND_WHOLE, 0, 10_000_000)})
    w = _controls({"p1": dev})
    qtbot.addWidget(w)
    widgets = _widget(w, "p1", "set")
    assert widgets["slider"] is None  # slider suppressed; spin box remains
    assert widgets["spin"].maximum() == 10_000_000


def test_no_value_action_renders_a_command_button(qtbot):
    dev = _dev({"stop": None, "energize": None})
    w = _controls({"m1": dev})
    qtbot.addWidget(w)
    btn = _widget(w, "m1", "stop")["button"]
    with qtbot.waitSignal(w.action_fire_requested) as sig:
        btn.click()
    assert sig.args == ["m1", "stop"]


def test_read_action_renders_read_button_and_updates_label(qtbot):
    dev = _dev({"read": None}, device_type="AnalogInput")
    w = _controls({"a1": dev})
    qtbot.addWidget(w)
    read_btn = _widget(w, "a1", "read")["read"]
    with qtbot.waitSignal(w.read_requested) as sig:
        read_btn.click()
    assert sig.args == ["a1", "read"]
    w.set_read_value("a1", "read", "512")
    assert w._value_labels[("a1", "read")].text() == "512"


def test_custom_device_gets_controls_automatically(qtbot):
    # A device type the old three-family grid would have dropped entirely.
    dev = _dev({"set_rate": ActionValueSpec(KIND_WHOLE, 0, 100, unit="mL/min")}, device_type="Pump")
    w = _controls({"pump": dev})
    qtbot.addWidget(w)
    assert ("pump", "set_rate") in _all_widgets(w)


def test_refresh_rebuilds_cleanly(qtbot):
    dev = _dev({"set": ActionValueSpec(KIND_SWITCH, 0, 1)})
    w = _controls({"d1": dev})
    qtbot.addWidget(w)
    w.refresh()
    w.refresh()
    assert set(w._widgets.keys()) == {("d1", "set")}


def _all_widgets(w):
    w.refresh()
    return w._widgets


# --- status strip + optimistic feedback (polish increment) -------------------


def test_status_strip_shows_persistent_icon_text(qtbot):
    w = _controls({})
    qtbot.addWidget(w)
    assert w._status.isHidden()
    w.show_status("pump failed", level="error")
    assert not w._status.isHidden()
    assert "pump failed" in w._status.text()
    assert "✗" in w._status.text()  # icon, not color alone
    assert w._status.property("level") == "error"
    w.clear_status()
    assert w._status.isHidden()


def test_failed_write_reverts_optimistic_switch_and_shows_status(qtbot):
    dev = _dev({"set": ActionValueSpec(KIND_SWITCH, 0, 1)})
    w = _controls({"d1": dev})
    qtbot.addWidget(w)
    switch = _widget(w, "d1", "set")["switch"]
    switch.click()  # optimistic -> checked (ON)
    assert switch.isChecked() and switch.text() == "ON"
    w.on_action_failed("d1", "set", "set failed: boom")
    assert not switch.isChecked() and switch.text() == "OFF"
    assert "boom" in w._status.text() and not w._status.isHidden()
