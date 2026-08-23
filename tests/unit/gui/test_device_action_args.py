"""Pressing Pulse in the Builder's Device Control panel.

pulse(period_ms, duration_s) has two required arguments and the panel had
nowhere to put them, so _build_action_buttons rendered it disabled with a
tooltip pointing at a node. A device that declares ACTION_ARGS_SCHEMA now gets
real fields and a live button.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QPushButton, QSpinBox

from glider.gui.panels.device_control_panel import DeviceControlPanel
from glider.hal.base_board import ConnectionState

pytestmark = pytest.mark.usefixtures("qtbot")


class _Device:
    device_type = "Maimu"
    name = "Stimulator"
    _initialized = True
    owns_link = True
    link_state = ConnectionState.CONNECTED

    ACTION_ARGS_SCHEMA = {
        "pulse": [
            {
                "key": "period_ms",
                "label": "Period (ms)",
                "type": "int",
                "default": 500,
                "min": 1,
                "max": 3_600_000,
            },
            {
                "key": "duration_s",
                "label": "Duration (s)",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 86_400,
            },
        ],
    }

    def __init__(self):
        self.calls = []
        self.board = SimpleNamespace(is_connected=True)

    async def on(self):
        self.calls.append(("on", ()))

    async def pulse(self, period_ms, duration_s):
        self.calls.append(("pulse", (period_ms, duration_s)))

    async def fade(self, level):
        self.calls.append(("fade", (level,)))

    async def execute_action(self, name, *args):
        return await self.actions[name](*args)

    @property
    def actions(self):
        return {"on": self.on, "pulse": self.pulse, "fade": self.fade}

    def action_args_schema(self, action):
        return list(self.ACTION_ARGS_SCHEMA.get(action, ()))

    def action_needs_args(self, action):
        import inspect

        func = self.actions.get(action)
        if func is None:
            return False
        return any(
            p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in inspect.signature(func).parameters.values()
        )


def _panel(qtbot, device):
    def _runner(coro):
        # These are sync tests, so there is no loop to schedule onto. Run the
        # coroutine to completion inline; asyncio.run rather than
        # get_event_loop, which is deprecated outside a running loop.
        import asyncio

        asyncio.run(coro)

    manager = SimpleNamespace(
        devices={"dev1": device},
        get_device=lambda dev_id: device if dev_id == "dev1" else None,
    )
    panel = DeviceControlPanel(manager, _runner)
    qtbot.addWidget(panel)
    panel.refresh_devices()
    panel._device_combo.setCurrentIndex(1)
    return panel


def _button(panel, label):
    found = panel._actions_widget.findChildren(QPushButton)
    for btn in found:
        if btn.text() == label:
            return btn
    raise AssertionError(f"no {label!r} button; found {[b.text() for b in found]}")


def test_a_declared_action_is_pressable(qtbot):
    panel = _panel(qtbot, _Device())
    assert _button(panel, "pulse").isEnabled()


def test_its_fields_are_rendered_with_their_defaults(qtbot):
    panel = _panel(qtbot, _Device())
    widgets = panel._action_arg_widgets["pulse"]
    assert widgets["period_ms"][0].value() == 500
    assert widgets["duration_s"][0].value() == 10


def test_the_fields_are_spin_boxes_with_the_declared_bounds(qtbot):
    panel = _panel(qtbot, _Device())
    period = panel._action_arg_widgets["pulse"]["period_ms"][0]
    assert isinstance(period, QSpinBox)
    assert period.minimum() == 1
    assert period.maximum() == 3_600_000


def test_args_are_read_in_schema_order(qtbot):
    """Swapped, pulse would run a 10 ms train for 500 seconds."""
    panel = _panel(qtbot, _Device())
    panel._action_arg_widgets["pulse"]["period_ms"][0].setValue(250)
    panel._action_arg_widgets["pulse"]["duration_s"][0].setValue(30)
    assert panel._action_args("pulse") == [250, 30]


def test_pressing_it_calls_the_action_with_both_values(qtbot):
    device = _Device()
    panel = _panel(qtbot, device)
    panel._action_arg_widgets["pulse"]["period_ms"][0].setValue(250)
    panel._action_arg_widgets["pulse"]["duration_s"][0].setValue(30)
    _button(panel, "pulse").click()
    assert device.calls == [("pulse", (250, 30))]


def test_an_undeclared_argument_action_stays_disabled(qtbot):
    """fade(level) declares nothing; the old tooltip is still the right answer."""
    panel = _panel(qtbot, _Device())
    fade = _button(panel, "fade")
    assert not fade.isEnabled()
    assert "Device Action node" in fade.toolTip()


def test_a_no_argument_action_gets_no_fields(qtbot):
    panel = _panel(qtbot, _Device())
    assert "on" not in panel._action_arg_widgets


def test_switching_device_clears_the_fields(qtbot):
    """Stale widgets from the previous device would be read on the next press."""
    panel = _panel(qtbot, _Device())
    assert "pulse" in panel._action_arg_widgets
    panel._device_combo.setCurrentIndex(0)  # "-- Select Device --"
    assert panel._action_arg_widgets == {}


def test_a_down_link_disables_it_too(qtbot):
    device = _Device()
    device.link_state = ConnectionState.RECONNECTING
    panel = _panel(qtbot, device)
    assert not _button(panel, "pulse").isEnabled()
