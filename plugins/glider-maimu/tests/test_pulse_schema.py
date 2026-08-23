"""The Maimu's pulse arguments, as the control panels will render them.

period_ms is a period in milliseconds, not a frequency, and the firmware
atoi()s both fields -- so the declared bounds are the same whole-number,
at-least-1 contract MaimuDevice._whole_number enforces at call time.
"""

import inspect

from glider.hal.base_device import DeviceConfig
from glider_maimu.device import MaimuDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


def _device():
    return MaimuDevice(_FakeBoard(), DeviceConfig(), name="maimu")


def test_pulse_declares_both_arguments():
    assert [f["key"] for f in _device().action_args_schema("pulse")] == [
        "period_ms",
        "duration_s",
    ]


def test_pulse_is_the_only_declared_action():
    """on/off/write need nothing; declaring them would only add empty forms."""
    assert set(MaimuDevice.ACTION_ARGS_SCHEMA) == {"pulse"}


def test_the_declared_order_matches_the_signature():
    """The panels pass these positionally; a swap would invert period and duration."""
    params = list(inspect.signature(MaimuDevice.pulse).parameters)[1:]
    assert [f["key"] for f in _device().action_args_schema("pulse")] == params


def test_the_defaults_are_the_node_s_defaults():
    """A researcher moving between the node and the panel should see one number."""
    from glider_maimu.node import DEFAULT_DURATION_S, DEFAULT_PERIOD_MS

    fields = {f["key"]: f for f in _device().action_args_schema("pulse")}
    assert fields["period_ms"]["default"] == DEFAULT_PERIOD_MS
    assert fields["duration_s"]["default"] == DEFAULT_DURATION_S


def test_the_bounds_reject_zero():
    """_whole_number raises below 1; the spin box should not offer it."""
    fields = {f["key"]: f for f in _device().action_args_schema("pulse")}
    assert fields["period_ms"]["min"] == 1
    assert fields["duration_s"]["min"] == 1


def test_every_field_is_a_whole_number():
    """The firmware atoi()s both, so a float widget would silently truncate."""
    assert all(f["type"] == "int" for f in _device().action_args_schema("pulse"))
