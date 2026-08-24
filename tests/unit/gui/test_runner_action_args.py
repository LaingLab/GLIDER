"""Argument-taking actions in the Runner's touch controls.

The bug this closes is a crash, not a cosmetic one: pulse has no value_spec,
so _controls_for classified it as a plain fire button, and pressing it called
execute_action("pulse") with no arguments -- TypeError, every time.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QSpinBox

from glider.gui.runner.device_controls import RunnerDeviceControls

pytestmark = pytest.mark.usefixtures("qtbot")


class _Device:
    device_type = "Maimu"
    name = "Stimulator"

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

    async def on(self):
        pass

    async def off(self):
        pass

    async def pulse(self, period_ms, duration_s):
        pass

    async def fade(self, level):
        pass

    @property
    def actions(self):
        return {"on": self.on, "off": self.off, "pulse": self.pulse, "fade": self.fade}

    def value_spec(self, action):
        return None

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


def _controls(qtbot, device):
    manager = SimpleNamespace(devices={"dev1": device})
    widget = RunnerDeviceControls(manager, session_fn=lambda: None)
    qtbot.addWidget(widget)
    widget.refresh()
    return widget


def test_a_declared_action_gets_an_args_control(qtbot):
    device = _Device()
    kinds = {action: kind for kind, action, _spec in _controls(qtbot, device)._controls_for(device)}
    assert kinds["pulse"] == "action_args"


def test_a_no_argument_action_is_still_a_button(qtbot):
    device = _Device()
    kinds = {action: kind for kind, action, _spec in _controls(qtbot, device)._controls_for(device)}
    assert kinds["on"] == "button"


def test_an_undeclared_argument_action_is_not_a_bare_button(qtbot):
    """This is the crash: a bare fire button called fade() with no level."""
    device = _Device()
    kinds = {action: kind for kind, action, _spec in _controls(qtbot, device)._controls_for(device)}
    assert kinds.get("fade") != "button"


def test_the_fields_render_with_their_defaults(qtbot):
    widget = _controls(qtbot, _Device())
    values = sorted(s.value() for s in widget.findChildren(QSpinBox))
    assert values == [10, 500]


def test_pressing_it_emits_both_values(qtbot):
    device = _Device()
    widget = _controls(qtbot, device)
    fields = widget._widgets[("dev1", "pulse")]["args"]
    fields["period_ms"][0].setValue(250)
    fields["duration_s"][0].setValue(30)

    emitted = []
    widget.action_call_requested.connect(
        lambda dev_id, action, args: emitted.append((dev_id, action, list(args)))
    )
    button = widget._widgets[("dev1", "pulse")]["button"]
    button.click()
    assert emitted == [("dev1", "pulse", [250, 30])]


def test_the_args_are_in_schema_order(qtbot):
    """Swapped, a pulse runs a 10 ms train for 500 seconds."""
    device = _Device()
    widget = _controls(qtbot, device)
    emitted = []
    widget.action_call_requested.connect(lambda dev_id, action, args: emitted.append(list(args)))
    widget._widgets[("dev1", "pulse")]["button"].click()
    assert emitted == [[500, 10]]


def test_a_failure_does_not_try_to_revert_a_slider(qtbot):
    """The args control has no committed value to snap back to.

    Checked via ``isHidden()`` rather than ``isVisible()``: this widget is
    never shown (no top-level window is mapped in this test), so
    ``isVisible()`` is False regardless of ``setVisible(True)`` -- the same
    reason the pre-existing status-strip test in
    ``tests/unit/gui/runner/test_device_controls.py`` asserts
    ``not w._status.isHidden()`` rather than ``w._status.isVisible()``.
    """
    device = _Device()
    widget = _controls(qtbot, device)
    widget.on_action_failed("dev1", "pulse", "pulse failed: link down")
    assert not widget._status.isHidden()
