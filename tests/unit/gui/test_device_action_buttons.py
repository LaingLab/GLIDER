"""Manual control for a device type core has never heard of.

The Device Control panel showed controls for exactly two device types,
``DigitalOutput`` and ``PWMOutput``. Anything else -- every plugin device, and
several built-ins -- selected fine and then offered nothing to press.

That matters more than it sounds. With six identical stimulators on a bench,
being able to press **on** and watch which one lights up is how you find out
which is the one you just added.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QPushButton

from glider.gui.panels.device_control_panel import DeviceControlPanel

pytestmark = pytest.mark.usefixtures("qtbot")


class _Device:
    """A plugin-ish device: a few named actions, one of which takes arguments."""

    device_type = "Maimu"
    name = "Stimulator"
    _initialized = True

    def __init__(self):
        self.calls: list[str] = []
        self.board = SimpleNamespace(is_connected=True)
        self.fail = None

    async def on(self):
        self.calls.append("on")
        if self.fail:
            raise self.fail

    async def off(self):
        self.calls.append("off")

    async def pulse(self, period_ms, duration_s):
        self.calls.append("pulse")

    async def execute_action(self, name, *args):
        # What BaseDevice.execute_action does, minus the locking.
        return await self.actions[name](*args)

    @property
    def actions(self):
        return {"on": self.on, "off": self.off, "pulse": self.pulse}


def _panel(qtbot, device, run_async=None):
    ran: list = []

    def _default_runner(coro):
        ran.append(coro)
        coro.close()

    manager = SimpleNamespace(
        devices={"stim": device},
        get_device=lambda i: device if i == "stim" else None,
    )
    panel = DeviceControlPanel(manager, run_async or _default_runner)
    qtbot.addWidget(panel)
    panel._device_combo.addItem("Stimulator", "stim")
    panel._device_combo.setCurrentIndex(panel._device_combo.count() - 1)
    return panel, ran


def _action_buttons(panel):
    return {b.text(): b for b in panel._actions_widget.findChildren(QPushButton)}


def test_a_device_with_actions_gets_buttons(qtbot):
    panel, _ran = _panel(qtbot, _Device())

    assert set(_action_buttons(panel)) == {"on", "off", "pulse"}


def test_the_action_row_is_shown(qtbot):
    """Built but hidden would be the same as absent."""
    panel, _ran = _panel(qtbot, _Device())

    assert panel._actions_widget.isVisibleTo(panel)
    assert panel._control_group.isVisibleTo(panel)


def test_a_no_argument_action_is_pressable(qtbot):
    panel, _ran = _panel(qtbot, _Device())

    assert _action_buttons(panel)["on"].isEnabled()
    assert _action_buttons(panel)["off"].isEnabled()


def test_an_action_that_needs_arguments_is_disabled_not_hidden(qtbot):
    """Hiding it would read as the device not having a pulse; the tooltip says
    where to drive it from instead."""
    panel, _ran = _panel(qtbot, _Device())

    pulse = _action_buttons(panel)["pulse"]
    assert not pulse.isEnabled()
    assert "Device Action" in pulse.toolTip()


async def test_pressing_a_button_runs_the_action(qtbot):
    device = _Device()
    coros: list = []
    panel, _ran = _panel(qtbot, device, run_async=coros.append)

    _action_buttons(panel)["on"].click()

    assert len(coros) == 1
    await coros[0]
    assert device.calls == ["on"]


async def test_a_failing_action_is_reported_not_raised(qtbot):
    """Mid-session, a device that has wandered off must not take the panel down."""
    device = _Device()
    device.fail = RuntimeError("peripheral gone")
    coros: list = []
    panel, _ran = _panel(qtbot, device, run_async=coros.append)

    _action_buttons(panel)["on"].click()
    await coros[0]

    assert "failed" in panel._device_status_label.text()
    assert "peripheral gone" in panel._device_status_label.text()


def test_switching_devices_rebuilds_the_row(qtbot):
    """Stale buttons would run actions on the wrong device."""

    class _Other(_Device):
        device_type = "Other"

        @property
        def actions(self):
            return {"beep": self.on}

    device, other = _Device(), _Other()
    manager = SimpleNamespace(
        devices={"stim": device, "other": other},
        get_device=lambda i: {"stim": device, "other": other}.get(i),
    )
    panel = DeviceControlPanel(manager, lambda coro: coro.close())
    qtbot.addWidget(panel)
    panel._device_combo.addItem("Stimulator", "stim")
    panel._device_combo.setCurrentIndex(panel._device_combo.count() - 1)
    assert "on" in _action_buttons(panel)

    panel._device_combo.addItem("Other", "other")
    panel._device_combo.setCurrentIndex(panel._device_combo.count() - 1)

    assert set(_action_buttons(panel)) == {"beep"}


def test_a_device_with_no_actions_shows_no_row(qtbot):
    class _Bare:
        device_type = "Nothing"
        name = "bare"
        _initialized = True
        board = SimpleNamespace(is_connected=True)
        actions: dict = {}

    panel, _ran = _panel(qtbot, _Bare())

    assert not panel._actions_widget.isVisibleTo(panel)


def test_an_awkward_actions_property_does_not_break_the_panel(qtbot):
    """A device is third-party code; a raising property is its problem, not the
    panel's."""

    class _Awkward:
        device_type = "Awkward"
        name = "awkward"
        _initialized = True
        board = SimpleNamespace(is_connected=True)

        @property
        def actions(self):
            raise RuntimeError("nope")

    panel, _ran = _panel(qtbot, _Awkward())  # must not raise

    assert not panel._actions_widget.isVisibleTo(panel)


def test_digital_outputs_keep_their_own_controls(qtbot):
    """The bespoke ON/OFF/Toggle row is better than generic buttons; the action
    row must not displace it."""

    class _Digital:
        device_type = "DigitalOutput"
        name = "led"
        _initialized = True
        board = SimpleNamespace(is_connected=True)

        async def set(self, value):
            pass

        @property
        def actions(self):
            return {"set": self.set}

    panel, _ran = _panel(qtbot, _Digital())

    assert panel._digital_widget.isVisibleTo(panel)
    assert not panel._actions_widget.isVisibleTo(panel)
