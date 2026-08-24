"""Peripherals on the status strip.

A BLE board is the host adapter: it goes green when bleak imports and stays
there. Its peripherals are the things that actually come and go, so the strip
-- the one piece of chrome that cannot be dismissed -- has to show them, or a
drop is invisible unless you happen to have the Hardware panel open.
"""

from types import SimpleNamespace

import pytest

from glider.gui.main_window import _device_chips
from glider.hal.base_board import ConnectionState

pytestmark = pytest.mark.usefixtures("qtbot")


def _device(name, state, owns=True):
    return SimpleNamespace(
        name=name,
        owns_link=owns,
        link_state=state,
        device_type="Maimu",
    )


def test_a_pin_device_gets_no_chip():
    """It would only duplicate its board's dot."""
    devices = {"led": _device("LED", ConnectionState.CONNECTED, owns=False)}
    assert _device_chips(devices) == []


def test_a_peripheral_gets_a_chip():
    devices = {"d1": _device("Stimulator", ConnectionState.CONNECTED)}
    assert _device_chips(devices) == [("Stimulator", "ok", "Maimu · Ready")]


def test_a_dropped_peripheral_is_red():
    devices = {"d1": _device("Stimulator", ConnectionState.DISCONNECTED)}
    name, state, _detail = _device_chips(devices)[0]
    assert (name, state) == ("Stimulator", "error")


def test_reconnecting_is_amber_and_says_so_in_the_tooltip():
    """The four-colour mapping cannot tell 'retrying' from 'gone'. The tooltip can."""
    devices = {"d1": _device("Stimulator", ConnectionState.RECONNECTING)}
    _name, state, detail = _device_chips(devices)[0]
    assert state == "warn"
    assert "Reconnecting" in detail


def test_chips_keep_the_devices_order():
    devices = {
        "a": _device("Left", ConnectionState.CONNECTED),
        "b": _device("Right", ConnectionState.DISCONNECTED),
    }
    assert [name for name, _s, _d in _device_chips(devices)] == ["Left", "Right"]


def test_a_nameless_device_falls_back_to_its_id():
    devices = {"dev_7": SimpleNamespace(owns_link=True, link_state=ConnectionState.CONNECTED)}
    assert _device_chips(devices)[0][0] == "dev_7"


def test_an_awkward_device_does_not_take_the_strip_down():
    """A plugin device with a raising property must not blank the strip."""

    class _Awkward:
        owns_link = True

        @property
        def link_state(self):
            raise RuntimeError("plugin exploded")

    devices = {
        "bad": _Awkward(),
        "good": _device("Stimulator", ConnectionState.CONNECTED),
    }
    chips = _device_chips(devices)
    assert ("Stimulator", "ok", "Maimu · Ready") in chips


def test_two_default_named_peripherals_are_told_apart():
    """A bench of six identical, still-default-named stimulators is the reason
    peripherals are on the strip at all -- an indistinguishable chip defeats
    the whole feature. Mirrors test_two_boards_of_the_same_type_are_told_apart
    in tests/unit/gui/shell/test_main_window_shell.py."""
    devices = {
        "d1": _device("Maimu", ConnectionState.CONNECTED),
        "d2": _device("Maimu", ConnectionState.DISCONNECTED),
    }
    names = [name for name, _s, _d in _device_chips(devices)]
    assert len(set(names)) == len(names)


def test_a_uniquely_named_peripheral_keeps_its_plain_name():
    """Disambiguation is a collision response, not a default -- a rig where
    every device has already been given a distinct name should still read
    the operator-chosen name, not a name padded with its id."""
    devices = {
        "d1": _device("Left Stim", ConnectionState.CONNECTED),
        "d2": _device("Right Stim", ConnectionState.CONNECTED),
    }
    names = [name for name, _s, _d in _device_chips(devices)]
    assert names == ["Left Stim", "Right Stim"]
