import pytest

from glider.core.device_drive import set_digital, set_pwm, toggle_digital


class _Dev:
    def __init__(self):
        self.state = False
        self.value = 0
        self.calls = []

    async def set_state(self, v):
        self.calls.append(("set_state", v))
        self.state = v

    async def toggle(self):
        self.calls.append(("toggle",))
        self.state = not self.state

    async def set_value(self, v):
        self.calls.append(("set_value", v))
        self.value = v


async def test_set_digital_prefers_set_state():
    d = _Dev()
    await set_digital(d, True)
    assert d.calls == [("set_state", True)]


async def test_toggle_prefers_toggle_method():
    d = _Dev()
    await toggle_digital(d)
    assert d.calls == [("toggle",)]


async def test_set_pwm_prefers_set_value():
    d = _Dev()
    await set_pwm(d, 128)
    assert d.calls == [("set_value", 128)]


async def test_none_device_raises():
    with pytest.raises(ValueError):
        await set_digital(None, True)
