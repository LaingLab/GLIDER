"""A device declaring what its actions' arguments are.

Until now an action with required arguments was dead on both control
surfaces -- disabled in the Builder's Device Control panel, and a TypeError in
the Runner's manual controls, which called it with none. A declared schema is
what lets either one render fields and pass real values.

Same field vocabulary as SETTINGS_SCHEMA, so schema_form renders it with no
new widget code.
"""

from glider.hal.base_device import BaseDevice, DeviceConfig


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


class _Device(BaseDevice):
    ACTION_ARGS_SCHEMA = {
        "pulse": [
            {"key": "period_ms", "label": "Period (ms)", "type": "int", "default": 500},
            {"key": "duration_s", "label": "Duration (s)", "type": "int", "default": 10},
        ],
    }

    @property
    def device_type(self):
        return "Fake"

    @property
    def actions(self):
        return {
            "on": self.on,
            "pulse": self.pulse,
            "fade": self.fade,
            "write": self.write,
            # A genuinely signature-less callable, standing in for a
            # C-extension action a real device might expose. `int` has no
            # computed signature on this Python -- inspect.signature(int)
            # raises ValueError -- so it exercises the same
            # "unintrospectable" branch a foreign SDK call could hit.
            "mystery": int,
        }

    async def on(self):
        pass

    async def pulse(self, period_ms, duration_s):
        pass

    async def fade(self, level=5):
        pass

    async def write(self, *args):
        pass

    async def initialize(self):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False

    @classmethod
    def from_dict(cls, data, board):
        return cls(board, DeviceConfig())


def _device():
    return _Device(_FakeBoard(), DeviceConfig())


def test_the_default_schema_is_empty():
    assert BaseDevice.ACTION_ARGS_SCHEMA == {}


def test_a_declared_action_returns_its_fields():
    schema = _device().action_args_schema("pulse")
    assert [f["key"] for f in schema] == ["period_ms", "duration_s"]


def test_an_undeclared_action_returns_an_empty_list():
    assert _device().action_args_schema("on") == []


def test_an_unknown_action_returns_an_empty_list():
    assert _device().action_args_schema("nonsense") == []


def test_a_no_argument_action_needs_none():
    assert _device().action_needs_args("on") is False


def test_a_required_argument_action_needs_them():
    assert _device().action_needs_args("pulse") is True


def test_a_defaulted_argument_does_not_count_as_required():
    """fade(level=5) is pressable as-is; it must not be greyed out."""
    assert _device().action_needs_args("fade") is False


def test_varargs_do_not_count_as_required():
    """write(*args) validates its own emptiness and reports a real error."""
    assert _device().action_needs_args("write") is False


def test_an_unknown_action_needs_nothing():
    assert _device().action_needs_args("nonsense") is False


def test_an_unintrospectable_action_needs_nothing():
    """A callable inspect.signature() cannot read is reported as needing none.

    `int` genuinely has no computed signature on this Python --
    inspect.signature(int) raises ValueError, confirmed directly below --
    standing in for a C-extension action a real device might wrap. Reporting
    False here, rather than raising or guessing True, means the action stays
    offered instead of greyed out; if it actually needs arguments, calling it
    with none reports a real error, which beats hiding a working action
    behind an incorrect assumption.
    """
    import inspect

    import pytest

    with pytest.raises(ValueError):
        inspect.signature(int)

    assert _device().action_needs_args("mystery") is False


def test_declared_keys_match_the_real_parameters():
    """The schema is passed positionally, so a typo would swap the arguments."""
    import inspect

    device = _device()
    for action, schema in type(device).ACTION_ARGS_SCHEMA.items():
        params = [
            p.name
            for p in inspect.signature(device.actions[action]).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert [f["key"] for f in schema] == params[: len(schema)]
