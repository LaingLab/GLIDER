"""Tests for the StepperA4988Device HAL device type.

The device lazy-imports ``gpiozero`` inside ``initialize()`` and claims its six
pins directly, bypassing the board abstraction (same precedent as
ADS1115Device/GenericI2CDevice). Tests inject a fake ``gpiozero`` module into
``sys.modules`` so pin writes are recorded regardless of host OS.
``step_delay`` is 0 in tests so pulse loops run instantly.
"""

import asyncio
import sys
import types

import pytest

from glider.hal.base_device import DEVICE_REGISTRY, DeviceConfig
from glider.hal.devices.stepper_a4988 import MICROSTEP_MODES, StepperA4988Device

PINS = {"step": 17, "dir": 27, "enable": 22, "ms1": 5, "ms2": 6, "ms3": 13}


class _FakeBoard:
    """Minimal board stand-in; the device never touches the board."""

    def __init__(self):
        self.id = "fake_board"


class FakePin:
    """Records on/off history for one claimed GPIO pin."""

    def __init__(self, pin, active_high=True, initial_value=False, **kwargs):
        self.pin = pin
        self.active_high = active_high
        self.is_on = bool(initial_value)
        self.on_count = 0
        self.closed = False

    def on(self):
        self.is_on = True
        self.on_count += 1

    def off(self):
        self.is_on = False

    def close(self):
        self.closed = True


@pytest.fixture
def fake_pins(monkeypatch):
    """Inject a fake ``gpiozero``; return dict of BCM pin number -> FakePin."""
    created: dict[int, FakePin] = {}
    module = types.ModuleType("gpiozero")

    def factory(pin, **kwargs):
        fp = FakePin(pin, **kwargs)
        created[pin] = fp
        return fp

    module.DigitalOutputDevice = factory
    monkeypatch.setitem(sys.modules, "gpiozero", module)
    return created


def _make_device(settings=None, name=None):
    merged = {"step_delay": 0.0, **(settings or {})}
    config = DeviceConfig(pins=dict(PINS), settings=merged)
    return StepperA4988Device(_FakeBoard(), config, name=name)


async def _initialized(settings=None):
    device = _make_device(settings)
    await device.initialize()
    return device


# --- Identity / configuration -------------------------------------------------


def test_device_type():
    assert _make_device().device_type == "StepperA4988"


def test_required_pins():
    assert _make_device().required_pins == ["step", "dir", "enable", "ms1", "ms2", "ms3"]


def test_actions():
    assert set(_make_device().actions) == {
        "move_steps",
        "move_turns",
        "stop",
        "energize",
        "de_energize",
    }


def test_settings_defaults():
    device = StepperA4988Device(_FakeBoard(), DeviceConfig(pins=dict(PINS), settings={}))
    assert device.steps_per_rev == 200
    assert device.steptype == "Full"
    assert device.step_delay == 0.005
    assert device.auto_disable is True


def test_settings_parsed_from_config():
    device = _make_device(settings={"steps_per_rev": 400, "steptype": "1/8", "auto_disable": False})
    assert device.steps_per_rev == 400
    assert device.steptype == "1/8"
    assert device.step_delay == 0.0
    assert device.auto_disable is False


def test_unknown_steptype_in_settings_raises():
    with pytest.raises(ValueError, match="steptype"):
        _make_device(settings={"steptype": "1/32"})


def test_nonpositive_steps_per_rev_raises():
    with pytest.raises(ValueError, match="steps_per_rev"):
        _make_device(settings={"steps_per_rev": 0})


def test_negative_step_delay_raises():
    with pytest.raises(ValueError, match="step_delay"):
        _make_device(settings={"step_delay": -0.001})


def test_microstep_truth_table_shape():
    # (factor, ms1, ms2, ms3) per the A4988 datasheet
    assert MICROSTEP_MODES == {
        "Full": (1, False, False, False),
        "Half": (2, True, False, False),
        "1/4": (4, False, True, False),
        "1/8": (8, True, True, False),
        "1/16": (16, True, True, True),
    }


# --- Serialization --------------------------------------------------------------


def test_to_dict_from_dict_round_trip():
    device = _make_device(settings={"steps_per_rev": 400}, name="knob")
    data = device.to_dict()
    clone = StepperA4988Device.from_dict(data, _FakeBoard())
    assert clone.device_type == "StepperA4988"
    assert clone.id == device.id
    assert clone.name == "knob"
    assert clone.pins == PINS
    assert clone.steps_per_rev == 400


# --- Lifecycle ------------------------------------------------------------------


async def test_initialize_claims_all_six_pins(fake_pins):
    device = await _initialized()
    assert device.is_initialized
    assert set(fake_pins) == set(PINS.values())


async def test_enable_pin_is_active_low(fake_pins):
    await _initialized()
    # Claimed with active_high=False so .on() == energized == pin LOW.
    assert fake_pins[PINS["enable"]].active_high is False


async def test_starts_de_energized(fake_pins):
    device = await _initialized()
    assert device.is_energized is False
    assert fake_pins[PINS["enable"]].is_on is False


async def test_initialize_applies_default_steptype_to_ms_pins(fake_pins):
    await _initialized(settings={"steptype": "1/8"})
    _, ms1, ms2, ms3 = MICROSTEP_MODES["1/8"]
    assert fake_pins[PINS["ms1"]].is_on == ms1
    assert fake_pins[PINS["ms2"]].is_on == ms2
    assert fake_pins[PINS["ms3"]].is_on == ms3


async def test_initialize_without_gpiozero_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpiozero", None)  # forces ImportError
    device = _make_device()
    with pytest.raises(RuntimeError, match="gpiozero"):
        await device.initialize()
    assert not device.is_initialized


async def test_shutdown_de_energizes_and_closes_pins(fake_pins):
    device = await _initialized()
    await device.shutdown()
    assert not device.is_initialized
    assert device.is_energized is False
    assert all(fp.closed for fp in fake_pins.values())


async def test_shutdown_without_initialize_is_safe():
    device = _make_device()
    await device.shutdown()  # must not raise
    assert not device.is_initialized


async def test_shutdown_clears_state_even_if_close_raises(fake_pins):
    device = await _initialized()

    def boom():
        raise RuntimeError("close failed")

    fake_pins[PINS["step"]].close = boom
    await device.shutdown()  # must not raise
    assert not device.is_initialized


# --- move_steps -----------------------------------------------------------------


async def test_move_steps_pulses_step_pin(fake_pins):
    device = await _initialized()
    result = await device.move_steps(4)
    assert result == 4
    assert fake_pins[PINS["step"]].on_count == 4


async def test_positive_steps_set_dir_high(fake_pins):
    device = await _initialized()
    await device.move_steps(2)
    assert fake_pins[PINS["dir"]].is_on is True


async def test_negative_steps_set_dir_low_and_return_signed(fake_pins):
    device = await _initialized()
    result = await device.move_steps(-3)
    assert result == -3
    assert fake_pins[PINS["dir"]].is_on is False
    assert fake_pins[PINS["step"]].on_count == 3


@pytest.mark.parametrize("mode", list(MICROSTEP_MODES))
async def test_per_move_steptype_drives_ms_pins(fake_pins, mode):
    device = await _initialized()
    await device.move_steps(1, steptype=mode)
    _, ms1, ms2, ms3 = MICROSTEP_MODES[mode]
    assert fake_pins[PINS["ms1"]].is_on == ms1
    assert fake_pins[PINS["ms2"]].is_on == ms2
    assert fake_pins[PINS["ms3"]].is_on == ms3


async def test_move_steps_unknown_steptype_raises(fake_pins):
    device = await _initialized()
    with pytest.raises(ValueError, match="steptype"):
        await device.move_steps(1, steptype="1/32")


async def test_move_steps_requires_steps(fake_pins):
    device = await _initialized()
    with pytest.raises(ValueError, match="steps"):
        await device.move_steps()


async def test_move_steps_uninitialized_raises():
    device = _make_device()
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.move_steps(1)


async def test_auto_disable_de_energizes_after_move(fake_pins):
    device = await _initialized()
    await device.move_steps(2)
    enable = fake_pins[PINS["enable"]]
    assert enable.on_count >= 1  # was energized during the move
    assert enable.is_on is False  # de-energized after
    assert device.is_energized is False


async def test_auto_disable_false_stays_energized(fake_pins):
    device = await _initialized(settings={"auto_disable": False})
    await device.move_steps(2)
    assert fake_pins[PINS["enable"]].is_on is True
    assert device.is_energized is True


async def test_move_zero_steps_pulses_nothing(fake_pins):
    device = await _initialized()
    assert await device.move_steps(0) == 0
    assert fake_pins[PINS["step"]].on_count == 0


async def test_fractional_steps_are_rounded(fake_pins):
    device = await _initialized()
    assert await device.move_steps(2.7) == 3
    assert fake_pins[PINS["step"]].on_count == 3


# --- move_turns -----------------------------------------------------------------


async def test_move_turns_full_step_conversion(fake_pins):
    device = await _initialized()  # steps_per_rev=200, Full
    result = await device.move_turns(0.5)
    assert result == 100
    assert fake_pins[PINS["step"]].on_count == 100


async def test_move_turns_scales_by_microstep_factor(fake_pins):
    device = await _initialized()
    result = await device.move_turns(0.25, steptype="1/4")
    assert result == 200  # 0.25 * 200 * 4


async def test_move_turns_negative_reverses(fake_pins):
    device = await _initialized()
    result = await device.move_turns(-0.1)
    assert result == -20
    assert fake_pins[PINS["dir"]].is_on is False


async def test_move_turns_requires_turns(fake_pins):
    device = await _initialized()
    with pytest.raises(ValueError, match="turns"):
        await device.move_turns()


# --- stop / energize ------------------------------------------------------------


async def test_stop_interrupts_move_mid_loop(fake_pins):
    device = await _initialized()
    step_pin = fake_pins[PINS["step"]]
    original_on = step_pin.on

    def on_and_stop():
        original_on()
        if step_pin.on_count >= 5:
            device._stop_event.set()  # deterministic mid-move stop

    step_pin.on = on_and_stop
    result = await device.move_steps(1000)
    assert result == 5
    assert step_pin.on_count == 5


async def test_move_after_stop_runs_normally(fake_pins):
    device = await _initialized()
    await device.stop()  # sets the event before any move
    result = await device.move_steps(3)
    assert result == 3  # event cleared on move entry


async def test_energize_and_de_energize(fake_pins):
    device = await _initialized()
    await device.energize()
    assert device.is_energized is True
    assert fake_pins[PINS["enable"]].is_on is True
    await device.de_energize()
    assert device.is_energized is False
    assert fake_pins[PINS["enable"]].is_on is False


async def test_energize_uninitialized_raises():
    device = _make_device()
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.energize()


# --- shutdown vs. in-flight moves -------------------------------------------------


async def test_shutdown_waits_for_inflight_move(fake_pins):
    # A real (tiny) step delay so the move is genuinely in flight when
    # shutdown runs; shutdown's stop event ends it within one step.
    device = await _initialized(settings={"step_delay": 0.001})
    move = asyncio.create_task(device.move_steps(100_000))
    await asyncio.sleep(0.02)  # let the pulse loop start
    await device.shutdown()
    result = await move  # must not raise
    assert 0 < result < 100_000
    assert all(fp.closed for fp in fake_pins.values())


async def test_move_after_shutdown_raises_runtime_error(fake_pins):
    device = await _initialized()
    await device.shutdown()
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.move_steps(5)


async def test_queued_move_behind_shutdown_raises_not_runs(fake_pins):
    # Move A in flight, move B queued on the lock, then e-stop. B must NOT
    # run after the e-stop (it would clear the stop event and physically
    # move the motor); it must raise instead, and shutdown must not block
    # for B's duration.
    device = await _initialized(settings={"step_delay": 0.001})
    move_a = asyncio.create_task(device.move_steps(100_000))
    await asyncio.sleep(0.02)  # A's pulse loop is running
    move_b = asyncio.create_task(device.move_steps(500))
    await asyncio.sleep(0.01)  # B is queued on the move lock
    await device.shutdown()
    assert 0 < await move_a < 100_000  # A interrupted, partial
    with pytest.raises(RuntimeError, match="not initialized"):
        await move_b


# --- Registry -------------------------------------------------------------------


def test_registered_in_device_registry():
    assert DEVICE_REGISTRY.get("StepperA4988") is StepperA4988Device


def test_create_device_from_dict_builds_stepper():
    # Loading a .glider file goes through create_device_from_dict, which only
    # knows types present in DEVICE_REGISTRY. (This test's own imports already
    # executed devices/__init__.py, so the runtime guarantee — that importing
    # bare glider.hal is enough — is proven by the fresh-process smoke check
    # in Task 9 Step 3, which is mandatory.)
    from glider.hal.base_device import create_device_from_dict

    data = _make_device(name="knob").to_dict()
    device = create_device_from_dict(data, _FakeBoard())
    assert isinstance(device, StepperA4988Device)
