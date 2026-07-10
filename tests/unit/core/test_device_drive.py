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


class _In:
    def __init__(self, device_type, value, ref=5.0, initialized=True):
        self.device_type = device_type
        self._value = value
        self._reference_voltage = ref
        self._initialized = initialized

    async def read(self):
        return self._value


async def test_read_digital_input():
    from glider.core.device_drive import read_input

    assert await read_input(_In("DigitalInput", True)) == "HIGH (1)"
    assert await read_input(_In("DigitalInput", False)) == "LOW (0)"


async def test_read_analog_input_voltage_fallback():
    from glider.core.device_drive import read_input

    out = await read_input(_In("AnalogInput", 512))
    assert "512" in out and "2.50V" in out


async def test_read_input_none_raises():
    import pytest

    from glider.core.device_drive import read_input

    with pytest.raises(ValueError):
        await read_input(None)
