# tests/unit/plugins/test_rotary_encoder_behaviors.py
import importlib.util
from pathlib import Path

import pytest

from glider.hal.input_behavior import BehaviorContext

_ENC = Path(__file__).parents[3] / "examples" / "plugins" / "rotary_encoder" / "__init__.py"


def _mod():
    spec = importlib.util.spec_from_file_location("rotary_encoder_behaviors", _ENC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _AngleDevice:
    def __init__(self, values):
        self._values, self._i = list(values), 0
        self.id = "enc"

    async def execute_action(self, name, *a, **k):
        v = self._values[self._i]
        if self._i < len(self._values) - 1:
            self._i += 1
        return v

    async def read(self):
        return await self.execute_action("angle")


class _RecordingPWM:
    device_type = "PWMOutput"

    def __init__(self):
        self.writes = []

    async def set_value(self, value):
        self.writes.append(value)


class _HM:
    def __init__(self, mapping):
        self._m = mapping

    def get_device(self, dev_id):
        return self._m.get(dev_id)


_TWO_TURN = [0, 1000, 2000, 3000, 4000, 100, 1100, 2100, 3100, 4090, 200]


async def test_revolution_behavior_fires_after_one_turn():
    mod = _mod()
    beh = next(b for b in mod.RotaryEncoderDevice.INPUT_BEHAVIORS if b.key == "revolution")
    ctx = BehaviorContext(device=_AngleDevice(_TWO_TURN), poll_interval=0.0)
    value = await beh.wait_for_input(
        {"turns_target": 1, "counts_per_turn": 4096, "land_tolerance": 0, "ramp_down": False},
        ctx,
        timeout=1.0,
    )
    assert value == 100


async def test_revolution_ramp_stops_motor_at_landing():
    mod = _mod()
    beh = next(b for b in mod.RotaryEncoderDevice.INPUT_BEHAVIORS if b.key == "revolution")
    pwm = _RecordingPWM()
    ctx = BehaviorContext(
        device=_AngleDevice([3000, 3600, 3800, 4000, 4090, 50]),
        hardware_manager=_HM({"m": pwm}),
        poll_interval=0.0,
    )
    await beh.wait_for_input(
        {
            "turns_target": 1,
            "counts_per_turn": 4096,
            "land_tolerance": 0,
            "ramp_down": True,
            "ramp_device": "m",
            "drive_pwm": 100,
            "creep_pwm": 30,
            "ramp_zone": 512,
        },
        ctx,
        timeout=1.0,
    )
    assert pwm.writes[0] == 100
    assert pwm.writes[:-1] == sorted(pwm.writes[:-1], reverse=True)
    assert pwm.writes[-1] == 0  # cleanup stops the motor at the landing


async def test_ramp_motor_stopped_on_timeout():
    mod = _mod()
    beh = next(b for b in mod.RotaryEncoderDevice.INPUT_BEHAVIORS if b.key == "revolution")
    pwm = _RecordingPWM()
    ctx = BehaviorContext(
        device=_AngleDevice([0, 10, 20]),  # never completes a turn
        hardware_manager=_HM({"m": pwm}),
        poll_interval=0.0,
    )
    with pytest.raises(TimeoutError):
        await beh.wait_for_input(
            {
                "turns_target": 1,
                "counts_per_turn": 4096,
                "land_tolerance": 0,
                "ramp_down": True,
                "ramp_device": "m",
                "drive_pwm": 100,
                "creep_pwm": 30,
                "ramp_zone": 512,
            },
            ctx,
            timeout=0.15,
        )
    assert pwm.writes[-1] == 0


async def test_move_counts_behavior_reaches_target():
    mod = _mod()
    beh = next(b for b in mod.RotaryEncoderDevice.INPUT_BEHAVIORS if b.key == "counts")
    ctx = BehaviorContext(device=_AngleDevice([0, 100, 200, 300, 400, 500]), poll_interval=0.0)
    value = await beh.wait_for_input(
        {"counts_target": 400, "counts_per_turn": 4096, "land_tolerance": 0, "ramp_down": False},
        ctx,
        timeout=1.0,
    )
    assert value == 400


def test_behaviors_exposed_via_property():
    mod = _mod()
    from glider.hal.base_device import DeviceConfig

    dev = mod.RotaryEncoderDevice(None, DeviceConfig(pins={}, settings={}), name="e")
    keys = {b.key for b in dev.input_behaviors}
    assert {"revolution", "counts"} <= keys
