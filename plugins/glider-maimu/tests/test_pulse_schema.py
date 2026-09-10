"""The Maimu's pulse arguments, as the control panels will render them.

period_ms is a period in milliseconds, not a frequency, and the firmware's
parser is strict -- a fractional or non-numeric field is rejected outright
rather than coerced (no more atoi() turning "fast" into 0). The declared
bounds are the same whole-number contract MaimuDevice._whole_number enforces
at call time (count and intensity_pct floor at 0, the other two at 1).
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


def test_pulse_declares_every_argument():
    assert [f["key"] for f in _device().action_args_schema("pulse")] == [
        "period_ms",
        "pulse_width_ms",
        "count",
        "intensity_pct",
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
    from glider_maimu.node import (
        DEFAULT_COUNT,
        DEFAULT_INTENSITY_PCT,
        DEFAULT_PERIOD_MS,
        DEFAULT_PULSE_WIDTH_MS,
    )

    fields = {f["key"]: f for f in _device().action_args_schema("pulse")}
    assert fields["period_ms"]["default"] == DEFAULT_PERIOD_MS
    assert fields["pulse_width_ms"]["default"] == DEFAULT_PULSE_WIDTH_MS
    assert fields["count"]["default"] == DEFAULT_COUNT
    assert fields["intensity_pct"]["default"] == DEFAULT_INTENSITY_PCT


def test_the_bounds_match_the_firmware():
    """Every value the panel offers must be one the firmware's parser accepts.

    count and intensity legitimately allow 0 -- 0 pulses means "run until
    stopped" and 0 intensity is an armed-but-dark control condition -- so a
    blanket floor of 1 would be wrong.
    """
    fields = {f["key"]: f for f in _device().action_args_schema("pulse")}
    assert (fields["period_ms"]["min"], fields["period_ms"]["max"]) == (1, 3_600_000)
    assert (fields["pulse_width_ms"]["min"], fields["pulse_width_ms"]["max"]) == (1, 3_600_000)
    assert (fields["count"]["min"], fields["count"]["max"]) == (0, 65_535)
    assert (fields["intensity_pct"]["min"], fields["intensity_pct"]["max"]) == (0, 100)


def test_the_node_s_pulse_fields_match_the_device_s():
    """The node's spin boxes and the device's guard must agree, field for field.

    node.py claims this in a comment. Without this test nothing enforced it, and
    the failure mode is a spin box that happily offers a value MaimuDevice.pulse
    raises on -- discovered mid-experiment, with an animal on the rig.
    """
    from glider_maimu.node import MaimuNode

    device_fields = {f["key"]: f for f in _device().action_args_schema("pulse")}
    node_fields = {f["key"]: f for f in MaimuNode.PROPERTIES_SCHEMA}

    # "mode" is the node's alone; every pulse argument must be present in both.
    assert set(device_fields) <= set(node_fields)

    for key, want in device_fields.items():
        got = node_fields[key]
        assert (got["min"], got["max"], got["default"], got["type"]) == (
            want["min"],
            want["max"],
            want["default"],
            want["type"],
        ), f"node and device disagree on {key}"


def test_every_field_is_a_whole_number():
    """The firmware's parser rejects a fractional field outright, so a float
    widget would let a researcher enter a value that silently does nothing."""
    assert all(f["type"] == "int" for f in _device().action_args_schema("pulse"))
