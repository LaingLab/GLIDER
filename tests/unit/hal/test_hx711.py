"""Tests for the HX711Device HAL device type.

The device lazy-imports ``gpiozero`` inside ``initialize()`` and bit-bangs the
HX711's two-wire protocol directly (StepperA4988Device precedent). Tests inject
a fake ``gpiozero`` whose DOUT/SCK pair implements the chip side of the
protocol, so frames are decoded, validated, and cached with no hardware.
"""

import asyncio
import sys
import time
import types

import pytest

from glider.hal.base_device import DEVICE_REGISTRY, DeviceConfig
from glider.hal.devices.hx711 import GAIN_PULSES, HX711Device

PINS = {"dout": 5, "sck": 6}


class _FakeBoard:
    """Minimal board stand-in; the device never touches the board."""

    def __init__(self):
        self.id = "fake_board"


def _make_device(settings=None, name=None):
    config = DeviceConfig(pins=dict(PINS), settings=dict(settings or {}))
    return HX711Device(_FakeBoard(), config, name=name)


# --- Identity / configuration -------------------------------------------------


def test_device_type():
    assert _make_device().device_type == "HX711"


def test_required_pins():
    assert _make_device().required_pins == ["dout", "sck"]


def test_actions():
    assert set(_make_device().actions) == {"read", "read_raw", "tare"}


def test_settings_defaults():
    device = _make_device()
    assert device.gain == 128
    assert device.scale == 1.0
    assert device.offset == 0.0


def test_settings_parsed_from_config():
    device = _make_device(settings={"gain": 64, "scale": 420.5, "offset": -1200.0})
    assert device.gain == 64
    assert device.scale == 420.5
    assert device.offset == -1200.0


def test_invalid_gain_raises():
    with pytest.raises(ValueError, match="gain"):
        _make_device(settings={"gain": 100})


def test_zero_scale_raises():
    with pytest.raises(ValueError, match="scale"):
        _make_device(settings={"scale": 0})


def test_gain_pulse_table():
    # gain -> extra PD_SCK pulses after the 24 data bits (datasheet)
    assert GAIN_PULSES == {128: 1, 64: 3, 32: 2}


def test_no_state_attribute():
    # DataRecorder._read_device_state checks hasattr(device, "_state") BEFORE
    # get_state(); an attribute with that name would shadow the real value.
    assert not hasattr(_make_device(), "_state")


# --- Serialization --------------------------------------------------------------


def test_to_dict_from_dict_round_trip():
    device = _make_device(settings={"gain": 32, "scale": 2.5}, name="scale_1")
    data = device.to_dict()
    clone = HX711Device.from_dict(data, _FakeBoard())
    assert clone.device_type == "HX711"
    assert clone.id == device.id
    assert clone.name == "scale_1"
    assert clone.pins == PINS
    assert clone.gain == 32
    assert clone.scale == 2.5


# --- Fake chip harness ----------------------------------------------------------


class FakeHX711:
    """Chip-side fake: a coordinated DOUT/SCK pair speaking the HX711 protocol.

    Queue 24-bit signed samples on ``frames``. DOUT reads busy (1) until a
    frame is queued and ``frame_interval`` has elapsed since the previous
    frame finished, then ready (0). Each SCK rising edge shifts the next bit
    out MSB-first; the pulse total for each completed frame is recorded in
    ``pulse_counts`` (finalized when the driver polls DOUT for the next
    frame). ``slow_pulse_at = N`` stalls the Nth pulse of the current/next
    frame by ``slow_pulse_sleep`` to cross the power-down threshold (one-shot).

    Per the datasheet the 25th pulse drives DOUT back high, so the first DOUT
    read after a frame is clocked always reports high on a live chip — that is
    the rise the driver's liveness check looks for. ``stuck_low = True`` models
    a disconnected DOUT instead: the line reads a permanent low through the
    pin's internal pull-down, which this protocol means "data ready".

    ``cycle = True`` re-appends each consumed frame, so the chip free-runs like
    real hardware instead of exhausting. Prefer it for sampler-driven tests:
    with a finite queue an unlucky spurious glitch (an OS stall crossing the
    SCK-high limit) consumes a frame the fake never replaces, where a real chip
    would simply convert again. Finite-queue tests therefore queue a surplus of
    identical frames rather than the bare minimum; only tests that need exact
    frame accounting queue precisely what they consume.

    Thread safety: the driver's sampler thread drives these methods while the
    test's event loop reads the attributes. No locking — the GIL makes the list
    mutations here safe.
    """

    def __init__(self, frame_interval: float = 0.0, cycle: bool = False):
        self.frames: list[int] = []
        self.pulse_counts: list[int] = []
        self.frames_started = 0
        self.frame_interval = frame_interval
        self.cycle = cycle
        self.stuck_low = False
        self.slow_pulse_at: int | None = None
        # 1-based frame ordinals to stall (vs. slow_pulse_at's one-shot), for
        # tests that need consecutive frames to glitch.
        self.slow_frames: set[int] = set()
        self.slow_pulse_sleep = 150e-6  # how long the slow pulse stalls
        self._shifting = False
        self._bits: list[int] = []
        self._current_bit = 1
        self._pulses = 0
        self._frame_index = 0
        self._frame_done_at = 0.0
        self.dout = _FakeDout(self)
        self.sck = _FakeSck(self)

    def _ready(self) -> bool:
        return bool(self.frames) and (
            time.perf_counter() - self._frame_done_at >= self.frame_interval
        )

    def _begin_frame(self) -> None:
        value = self.frames.pop(0)
        if self.cycle:
            self.frames.append(value)  # free-run: the chip converts again
        self._bits = [((value & 0xFFFFFF) >> (23 - i)) & 1 for i in range(24)]
        self._pulses = 0
        self._frame_index += 1
        # Counts frames the driver STARTED clocking, including ones it never
        # finishes (a stuck-low line never finalizes). Pacing assertions read
        # this rather than pulse_counts, which only records finished frames.
        self.frames_started += 1
        self._shifting = True

    def _finish_frame(self) -> None:
        self.pulse_counts.append(self._pulses)
        self._shifting = False
        self._frame_done_at = time.perf_counter()

    def rising_edge(self) -> None:
        if not self._shifting:
            if not self._ready():
                return  # clocked while busy (e.g. power-down drive): ignored
            self._begin_frame()
        self._pulses += 1
        one_shot = self.slow_pulse_at is not None and self._pulses == self.slow_pulse_at
        per_frame = self._frame_index in self.slow_frames and self._pulses == 3
        if one_shot or per_frame:
            if one_shot:
                self.slow_pulse_at = None
            time.sleep(self.slow_pulse_sleep)  # push the SCK-high window over the limit
        self._current_bit = self._bits.pop(0) if self._bits else 1

    def read_dout(self) -> int:
        if self.stuck_low:
            # Disconnected line: permanent low via the pull-down. The frame
            # bookkeeping still has to run — the driver really did clock 25
            # pulses and really is about to clock 25 more. Skipping it would
            # freeze _shifting True forever, so _begin_frame would never run
            # again and every per-frame counter would stall at 1 no matter how
            # hard the driver spun.
            if self._shifting and self._pulses >= 25:
                self._finish_frame()
            return 0
        if self._shifting:
            if self._pulses >= 25:
                # Driver finished the frame (24 data + >=1 gain pulse) and is
                # reading the line: finalize. The 25th pulse drove DOUT high,
                # so this read is high even if the next conversion is already
                # due — the chip only pulls it low again afterwards.
                self._finish_frame()
                return 1
            return self._current_bit
        return 0 if self._ready() else 1


class _FakeDout:
    def __init__(self, chip: FakeHX711):
        self._chip = chip
        self.closed = False

    @property
    def value(self) -> int:
        return self._chip.read_dout()

    def close(self) -> None:
        self.closed = True


class _FakeSck:
    def __init__(self, chip: FakeHX711):
        self._chip = chip
        self.is_on = False
        self.closed = False

    def on(self) -> None:
        self.is_on = True
        self._chip.rising_edge()

    def off(self) -> None:
        self.is_on = False

    def close(self) -> None:
        self.closed = True


def _finalize(chip: FakeHX711) -> None:
    """Poll DOUT once so the fake finalizes the just-clocked frame."""
    chip.dout.value  # noqa: B018 - side effect: records the pulse count


# --- _read_frame protocol -------------------------------------------------------


@pytest.fixture(autouse=True)
def relaxed_sck_limit(monkeypatch):
    """Raise the SCK-high glitch limit for every test in this module.

    The fake's pulse window spans three Python calls, so the real 60 us limit
    is within reach of an ordinary GC pause or thread preempt on a loaded CI
    runner — every clocked frame would be a flake vector. The limit's *value*
    is a datasheet constant, not behavior under test; what these tests check
    is the surrounding logic (decode, discard, re-prime), which is identical
    at any threshold. Tests that deliberately glitch stall a pulse by
    ``slow_pulse_sleep``, which they set high enough to cross this limit.
    """
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "SCK_HIGH_LIMIT_S", 5e-3)


def test_read_frame_decodes_positive_value():
    chip = FakeHX711()
    chip.frames.append(0x123456)
    device = _make_device()
    assert device._read_frame(chip.dout, chip.sck, device._stop_event) == 0x123456


def test_read_frame_decodes_zero():
    chip = FakeHX711()
    chip.frames.append(0)
    device = _make_device()
    assert device._read_frame(chip.dout, chip.sck, device._stop_event) == 0


def test_read_frame_sign_extends_negative_value():
    chip = FakeHX711()
    chip.frames.append(-12345)
    device = _make_device()
    assert device._read_frame(chip.dout, chip.sck, device._stop_event) == -12345


@pytest.mark.parametrize("gain,pulses", [(128, 25), (32, 26), (64, 27)])
def test_read_frame_pulse_count_selects_gain(gain, pulses):
    chip = FakeHX711()
    chip.frames.append(42)
    device = _make_device(settings={"gain": gain})
    assert device._read_frame(chip.dout, chip.sck, device._stop_event) == 42
    _finalize(chip)
    assert chip.pulse_counts == [pulses]


def test_read_frame_returns_none_when_stopped_while_waiting():
    chip = FakeHX711()  # no frames queued -> DOUT stays busy
    device = _make_device()
    device._stop_event.set()
    assert device._read_frame(chip.dout, chip.sck, device._stop_event) is None


def test_read_frame_raises_glitch_on_slow_pulse():
    from glider.hal.devices.hx711 import _GlitchError

    chip = FakeHX711()
    chip.frames.append(42)
    chip.slow_pulse_at = 5
    chip.slow_pulse_sleep = 0.01  # cross the relaxed_sck_limit fixture's 5 ms
    device = _make_device()
    with pytest.raises(_GlitchError, match="us"):
        device._read_frame(chip.dout, chip.sck, device._stop_event)


@pytest.mark.parametrize("rail", [0x7FFFFF, -0x800000])
def test_read_frame_raises_glitch_on_saturated_value(rail):
    from glider.hal.devices.hx711 import _GlitchError

    chip = FakeHX711()
    chip.frames.append(rail)
    device = _make_device()
    with pytest.raises(_GlitchError, match="saturated"):
        device._read_frame(chip.dout, chip.sck, device._stop_event)


# --- Lifecycle ------------------------------------------------------------------


@pytest.fixture
def fake_chip(monkeypatch):
    """Inject a fake ``gpiozero`` wired to one FakeHX711; return the chip.

    The SCK-high limit is relaxed module-wide by the ``relaxed_sck_limit``
    autouse fixture.
    """
    chip = FakeHX711()
    module = types.ModuleType("gpiozero")
    module.DigitalInputDevice = lambda pin, **kwargs: chip.dout
    module.DigitalOutputDevice = lambda pin, **kwargs: chip.sck
    monkeypatch.setitem(sys.modules, "gpiozero", module)
    return chip


async def _wait_for_sample(device, timeout=2.0):
    """Poll get_state() until the sampler caches a value; fail on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = await device.get_state()
        if state is not None:
            return state
        await asyncio.sleep(0.005)
    raise AssertionError("sampler produced no sample within timeout")


async def test_initialize_starts_sampler_and_caches_sample(fake_chip):
    # Surplus identical frames: a spurious glitch consuming one must not
    # starve the finite fake (see FakeHX711's docstring).
    fake_chip.frames.extend([1000] * 20)
    device = _make_device()
    await device.initialize()
    try:
        assert device.is_initialized
        assert await _wait_for_sample(device) == 1000.0  # scale=1, offset=0
    finally:
        await device.shutdown()


async def test_initialize_without_gpiozero_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpiozero", None)  # forces ImportError
    device = _make_device()
    with pytest.raises(RuntimeError, match="gpiozero"):
        await device.initialize()
    assert not device.is_initialized


async def test_shutdown_stops_sampler_and_closes_pins(fake_chip):
    device = _make_device()
    await device.initialize()
    # No frames queued: the sampler is stuck waiting for data-ready, so this
    # also proves shutdown is not blocked by a disconnected sensor.
    await asyncio.wait_for(device.shutdown(), timeout=2.0)
    assert not device.is_initialized
    # Releasing the pins IS the safe state: an HX711 actuates nothing, and a
    # PD_SCK power-down would lapse the moment close() released the line.
    assert fake_chip.sck.closed
    assert fake_chip.dout.closed


async def test_shutdown_without_initialize_is_safe():
    device = _make_device()
    await device.shutdown()  # must not raise
    assert not device.is_initialized


async def test_reinitialize_after_shutdown(fake_chip):
    device = _make_device()
    await device.initialize()
    await device.shutdown()
    fake_chip.dout.closed = False
    fake_chip.sck.closed = False
    fake_chip.frames.extend([7] * 20)
    await device.initialize()
    try:
        assert await _wait_for_sample(device) == 7.0
    finally:
        await device.shutdown()


async def test_timing_glitch_consumes_next_frame_as_reprime_dummy(fake_chip):
    # Exactly two frames: 111 timing-glitches and is discarded, and 222 is
    # swallowed by the gain re-priming dummy read. So NOTHING is ever cached.
    # This is the assertion that actually pins the dummy read down - without
    # it, 222 would be cached and get_state() would return 222.0. (Asserting
    # a *later* value is cached cannot prove it: the sampler races through
    # these frames in microseconds, so a poll can never observe the transient.)
    fake_chip.frames.extend([111, 222])
    fake_chip.slow_pulse_at = 3
    fake_chip.slow_pulse_sleep = 0.01  # cross the relaxed 5 ms limit
    device = _make_device()
    await device.initialize()
    try:
        await asyncio.sleep(0.1)  # ample time to clock both frames
        assert await device.get_state() is None
        # Both frames were clocked with their full 25 pulses (24 data + 1 gain
        # for the default gain of 128) - the dummy read is a real frame.
        assert fake_chip.pulse_counts == [25, 25]
    finally:
        await device.shutdown()


async def test_glitched_frame_discarded_then_gain_reprimed(fake_chip):
    # Frame 1 glitches (slow pulse) -> discarded; frame 2 is consumed as the
    # gain re-priming dummy read; frame 3 is the first trusted sample. The
    # surplus 333s are identical, so an extra spurious glitch consuming one
    # can neither starve the fake nor change the assertion.
    fake_chip.frames.extend([111, 222] + [333] * 20)
    fake_chip.slow_pulse_at = 3
    fake_chip.slow_pulse_sleep = 0.01  # cross the relaxed 5 ms limit
    device = _make_device()
    await device.initialize()
    try:
        assert await _wait_for_sample(device) == 333.0
    finally:
        await device.shutdown()


async def test_reprime_dummy_that_also_glitches_does_not_trust_the_next_frame(fake_chip):
    # Frame 1 timing-glitches, so the chip may now be reset to A/128 and
    # frame 2 is burned as the re-prime dummy - but frame 2 glitches too, so
    # the chip is STILL unprimed and frame 3 must be burned as another dummy
    # rather than cached. Exactly three frames, so a correct driver caches
    # nothing; swallowing the dummy's own glitch caches frame 3 as if it were
    # configured-gain data when it is really a channel-A/128 conversion.
    fake_chip.frames.extend([111, 222, 333])
    fake_chip.slow_frames = {1, 2}
    fake_chip.slow_pulse_sleep = 0.01  # cross the relaxed 5 ms limit
    device = _make_device(settings={"gain": 32})
    await device.initialize()
    try:
        await asyncio.sleep(0.1)  # ample time to clock all three frames
        assert await device.get_state() is None
        assert fake_chip.pulse_counts == [26, 26, 26]  # gain 32 -> 24 + 2
    finally:
        await device.shutdown()


async def test_saturated_frame_does_not_trigger_reprime_dummy(fake_chip):
    # A saturated reading involves no power-down, so the gain is still primed
    # and the next frame must be trusted directly rather than burned as a
    # dummy. Exactly two frames: the rail discards, then 77 must be cached.
    # If saturation wrongly re-primed, 77 would be swallowed -> None.
    fake_chip.frames.extend([0x7FFFFF, 77])
    device = _make_device()
    await device.initialize()
    try:
        assert await _wait_for_sample(device) == 77.0
    finally:
        await device.shutdown()


async def test_discards_escalate_every_nth_frame(fake_chip, monkeypatch, caplog):
    import logging

    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "CONSECUTIVE_DISCARD_WARN", 2)
    monkeypatch.setattr(hx711_module, "GLITCH_BACKOFF_S", 0.0)
    # Saturated frames glitch regardless of timing, and no longer re-prime, so
    # all four are read as main frames: bad reaches 2 (WARN) and 4 (WARN),
    # then 55 caches and resets the counter. Exactly two warnings - the
    # escalation fires every Nth discard, not once per fault.
    rail = 0x7FFFFF
    fake_chip.frames.extend([rail] * 4 + [55] * 20)
    device = _make_device()
    with caplog.at_level(logging.WARNING, logger="glider.hal.devices.hx711"):
        await device.initialize()
        try:
            assert await _wait_for_sample(device) == 55.0
        finally:
            await device.shutdown()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "consecutive samples discarded" in warnings[0].getMessage()


# --- Reads / calibration ----------------------------------------------------------


async def test_read_applies_offset_and_scale(fake_chip):
    fake_chip.frames.extend([1400] * 20)  # surplus: see FakeHX711's docstring
    device = _make_device(settings={"offset": 400.0, "scale": 100.0})
    await device.initialize()
    try:
        await _wait_for_sample(device)
        assert await device.execute_action("read") == 10.0  # (1400-400)/100
        assert await device.execute_action("read_raw") == 1400
    finally:
        await device.shutdown()


async def test_read_before_first_sample_raises(fake_chip, monkeypatch):
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "READ_WAIT_S", 0.05)
    device = _make_device()
    await device.initialize()
    try:
        with pytest.raises(RuntimeError, match="no sample"):
            await device.read()
        assert await device.get_state() is None  # recorder-safe: no raise
    finally:
        await device.shutdown()


async def test_read_waits_for_a_late_first_sample(fake_chip):
    # At experiment start the sampler has not clocked a frame yet - up to a
    # full period (100 ms at 10 SPS). Raising there would error the first
    # DeviceRead node, and since a node fires its exec output only after the
    # read returns, every downstream node including EndExperiment would never
    # run: the protocol stops mid-flight with the recording left open.
    device = _make_device()
    await device.initialize()
    try:

        async def feed_late():
            await asyncio.sleep(0.05)
            fake_chip.frames.extend([42] * 20)

        task = asyncio.create_task(feed_late())
        try:
            assert await device.read() == 42.0
        finally:
            await task
    finally:
        await device.shutdown()


async def test_read_waits_out_a_transient_stale_cache(fake_chip, monkeypatch):
    # A momentarily stale cache (a briefly overloaded cell rails every frame)
    # must not fail the read outright while the sampler is about to recover.
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "MAX_SAMPLE_AGE_S", 0.03)
    device = _make_device()
    await device.initialize()
    try:
        fake_chip.frames.append(9)
        await _wait_for_sample(device)
        await asyncio.sleep(0.05)  # the cached 9 ages out
        assert await device.get_state() is None  # confirms it is stale

        async def feed_late():
            await asyncio.sleep(0.03)
            fake_chip.frames.extend([11] * 20)

        task = asyncio.create_task(feed_late())
        try:
            assert await device.read() == 11.0  # waited, then read fresh
        finally:
            await task
    finally:
        await device.shutdown()


async def test_read_uninitialized_raises_via_execute_action():
    device = _make_device()
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.execute_action("read")


# --- Liveness: staleness + stuck-low DOUT -----------------------------------------


async def test_disconnected_dout_caches_no_sample(fake_chip):
    # A disconnected DOUT reads a permanent low through its internal pull-down,
    # which this protocol means "data ready" - 24 zero bits decode to a
    # perfectly plausible 0 that passes both the timing and saturation checks.
    # Only the datasheet's 25th-pulse rise distinguishes it from a live chip.
    fake_chip.stuck_low = True
    fake_chip.frames.extend([1234] * 20)
    device = _make_device()
    await device.initialize()
    try:
        await asyncio.sleep(0.1)
        assert await device.get_state() is None  # never a fabricated 0.0
    finally:
        await device.shutdown()


def _stamp(device, raw, age_s):
    """Plant a cached sample of a given age (no real sleeping)."""
    device._latest = (raw, time.perf_counter() - age_s)


async def test_read_raises_when_sample_is_stale(fake_chip, monkeypatch):
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "MAX_SAMPLE_AGE_S", 0.5)
    monkeypatch.setattr(hx711_module, "READ_WAIT_S", 0.05)
    device = _make_device()
    # Initialized with no frames queued, so the sampler parks on data-ready
    # and the stamped sample stays the newest one: read waits out READ_WAIT_S
    # and then reports the age rather than a fabricated value.
    await device.initialize()
    try:
        _stamp(device, 1000, age_s=5.0)
        # The reported age includes the wait the read just spent, so it is
        # 5.0s + READ_WAIT_S rather than exactly 5.0s.
        with pytest.raises(RuntimeError, match=r"last sample is 5\.\ds old"):
            await device.read()
        _stamp(device, 1000, age_s=5.0)
        with pytest.raises(RuntimeError, match="sensor may be disconnected or overloaded"):
            await device.read_raw()
    finally:
        await device.shutdown()


async def test_get_state_returns_none_when_sample_is_stale(fake_chip, monkeypatch):
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "MAX_SAMPLE_AGE_S", 0.5)
    device = _make_device()
    _stamp(device, 1000, age_s=5.0)
    # The recorder gets an empty cell, never a stale number frozen into a
    # flat line in the experiment CSV.
    assert await device.get_state() is None


async def test_fresh_sample_is_not_stale(fake_chip, monkeypatch):
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "MAX_SAMPLE_AGE_S", 0.5)
    device = _make_device(settings={"offset": 100.0, "scale": 2.0})
    _stamp(device, 1000, age_s=0.0)
    assert await device.get_state() == (1000 - 100.0) / 2.0
    assert await device.read() == (1000 - 100.0) / 2.0
    assert await device.read_raw() == 1000


async def test_overloaded_cell_goes_stale_rather_than_reporting_last_weight(fake_chip, monkeypatch):
    # The motivating scenario: a good reading, then the cell is overloaded and
    # every frame rails. Reads must stop reporting the last pre-overload
    # weight rather than freezing it into the record forever.
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "MAX_SAMPLE_AGE_S", 0.1)
    fake_chip.frames.append(500)  # one good frame...
    fake_chip.frames.extend([0x7FFFFF] * 40)  # ...then the cell is overloaded
    device = _make_device()
    await device.initialize()
    try:
        assert await _wait_for_sample(device) == 500.0
        await asyncio.sleep(0.25)  # let the good sample age out
        assert await device.get_state() is None
        with pytest.raises(RuntimeError, match="old"):
            await device.read()
    finally:
        await device.shutdown()


# --- Tare -------------------------------------------------------------------------


@pytest.fixture
def slow_chip(monkeypatch):
    """Fake chip that serves one frame per 20ms, like real 10/80 SPS pacing,
    so tare's fresh-sample gating sees each sample exactly once."""
    chip = FakeHX711(frame_interval=0.02)
    module = types.ModuleType("gpiozero")
    module.DigitalInputDevice = lambda pin, **kwargs: chip.dout
    module.DigitalOutputDevice = lambda pin, **kwargs: chip.sck
    monkeypatch.setitem(sys.modules, "gpiozero", module)
    return chip


@pytest.fixture
def cycling_slow_chip(monkeypatch):
    """``slow_chip`` that free-runs a repeating [10, 30, 30, 30, 100] cycle.

    Cycling is what a real HX711 does — it keeps converting — and it means a
    missed poll window or a spurious glitch can never starve tare of frames.

    Both properties of the cycle's shape are load-bearing; see the median
    test for why.
    """
    chip = FakeHX711(frame_interval=0.02, cycle=True)
    chip.frames.extend([10, 30, 30, 30, 100])
    module = types.ModuleType("gpiozero")
    module.DigitalInputDevice = lambda pin, **kwargs: chip.dout
    module.DigitalOutputDevice = lambda pin, **kwargs: chip.sck
    monkeypatch.setitem(sys.modules, "gpiozero", module)
    return chip


async def test_tare_takes_median_and_updates_offset_and_settings(cycling_slow_chip):
    # The chip cycles [10, 30, 30, 30, 100] forever, like a real HX711
    # free-running under a steady load. The cycle's shape is chosen so the
    # assertion cannot flake; both properties are load-bearing:
    #
    # 1. 30 DOMINATES (3 of the 5 positions). Any 5 consecutive samples are
    #    the same multiset and median to 30 - but tare under CI load may also
    #    SKIP a beat (the sampler overwrites the single _latest slot before
    #    tare polls it), and a skipped draw is not a consecutive one. With 30
    #    in the majority, any draw spanning a skipped beat still carries >=3
    #    thirties and still medians to 30. (A [10, 30, 20, 50, 40] cycle also
    #    medians to 30 when consecutive, but a skipped beat can yield e.g.
    #    [10, 20, 50, 40, 10] -> 20, which flakes.)
    # 2. The outliers are ASYMMETRIC (10 vs 100), so mean = 40 != median = 30
    #    and a mean-based implementation still fails this test.
    #
    # Cycling also makes starvation impossible: the finite-queue version of
    # this test timed out ~1 run in 10 under load.
    device = _make_device(settings={"scale": 10.0})
    notified: list = []
    device.set_settings_changed_callback(notified.append)
    await device.initialize()
    try:
        await _wait_for_sample(device)  # a pre-tare sample tare must exclude
        offset = await device.execute_action("tare")
        assert offset == 30.0
        assert device.offset == 30.0
        assert device.config.settings["offset"] == 30.0
        # The offset must be ANNOUNCED, not just stored: it lives on the
        # device, and only this notification folds it into the session and
        # marks it dirty. Without it, closing GLIDER never prompts to save and
        # the calibration is silently lost.
        assert notified == [device]
        # Offset is applied on the read path, whichever frame is latest.
        raw = await device.execute_action("read_raw")
        assert await device.execute_action("read") == (raw - 30.0) / 10.0
    finally:
        await device.shutdown()


async def test_tare_excludes_stale_cached_sample(fake_chip, monkeypatch):
    # The median test above is too robust to prove exclusion (a wrongly
    # included stale value need not move the median). This is the clean
    # proof: with only a stale sample and no new frames, tare must time out
    # rather than count the stale one five times.
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "TARE_TIMEOUT_S", 0.3)
    fake_chip.frames.append(999)  # one frame, then the chip goes quiet
    device = _make_device()
    await device.initialize()
    try:
        await _wait_for_sample(device)  # 999 cached
        with pytest.raises(RuntimeError, match="tare timed out"):
            await device.tare()  # must NOT count the stale 999
    finally:
        await device.shutdown()


async def test_tare_raises_when_shut_down_mid_collection(slow_chip):
    # tare's under-loop initialized check backs a real concurrency claim:
    # an e-stop during a tare must abort it, not let it run to completion
    # against released pins.
    slow_chip.frames.append(123)
    device = _make_device()
    await device.initialize()
    try:
        await _wait_for_sample(device)
    except AssertionError:
        pass
    tare_task = asyncio.create_task(device.tare())
    await asyncio.sleep(0.05)  # let tare enter its collection loop
    await device.shutdown()
    with pytest.raises(RuntimeError, match="shut down during tare"):
        await tare_task


async def test_tare_times_out_without_samples(fake_chip, monkeypatch):
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "TARE_TIMEOUT_S", 0.1)
    device = _make_device()
    await device.initialize()  # no frames queued -> no samples ever
    try:
        with pytest.raises(RuntimeError, match="tare timed out"):
            await device.tare()
    finally:
        await device.shutdown()


# --- Liveness / robustness --------------------------------------------------------


async def test_shutdown_clears_cached_sample(fake_chip):
    # A torn-down device must not keep answering the recorder from its cache:
    # get_state() never consults _initialized, so a surviving tuple becomes a
    # fabricated post-shutdown row in the experiment CSV.
    fake_chip.frames.extend([1000] * 20)
    device = _make_device()
    await device.initialize()
    await _wait_for_sample(device)
    await device.shutdown()
    assert await device.get_state() is None


async def test_stuck_low_dout_does_not_busy_spin(fake_chip, monkeypatch):
    # A disconnected DOUT reads a permanent low through its pull-down, so
    # _read_frame returns instantly. Without pacing on the glitch path the
    # loop clocks frames as fast as CPython allows and pegs a core.
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "GLITCH_BACKOFF_S", 0.02)
    fake_chip.stuck_low = True
    fake_chip.cycle = True  # the line is dead, but the driver keeps trying
    fake_chip.frames.append(0)
    device = _make_device()
    await device.initialize()
    try:
        await asyncio.sleep(0.2)
        assert await device.get_state() is None  # nothing fabricated
        # ~10 frames fit in 200ms at a 20ms backoff; an unpaced loop clocks
        # tens of thousands. Verified by mutation: deleting the backoff takes
        # this from ~10 to ~40000, so the bound genuinely discriminates rather
        # than passing on a counter that never moves.
        assert fake_chip.frames_started < 100
    finally:
        await device.shutdown()


async def test_sampler_survives_transient_unexpected_error(fake_chip, monkeypatch):
    # One non-glitch error must not kill sampling for the rest of the run.
    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "ERROR_BACKOFF_S", 0.01)
    fake_chip.frames.extend([1234] * 20)
    calls = {"n": 0}
    real_read_dout = fake_chip.read_dout

    def flaky_read_dout():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient GPIO fault")
        return real_read_dout()

    fake_chip.read_dout = flaky_read_dout
    device = _make_device()
    await device.initialize()
    try:
        assert await _wait_for_sample(device) == 1234.0  # recovered
    finally:
        await device.shutdown()


async def test_sampler_gives_up_after_persistent_unexpected_errors(fake_chip, monkeypatch, caplog):
    import logging

    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "ERROR_BACKOFF_S", 0.001)
    monkeypatch.setattr(hx711_module, "MAX_UNEXPECTED_ERRORS", 3)

    def always_raise():
        raise OSError("wedged GPIO")

    fake_chip.read_dout = always_raise
    device = _make_device()
    with caplog.at_level(logging.ERROR, logger="glider.hal.devices.hx711"):
        await device.initialize()
        try:
            await asyncio.sleep(0.15)
            assert device._thread is not None
            assert not device._thread.is_alive()  # gave up rather than spun
        finally:
            await device.shutdown()
    assert any("giving up" in r.getMessage() for r in caplog.records)


async def test_reinitialize_does_not_unmute_a_straggler_thread(fake_chip):
    # A straggler that outlived a timed-out join is muted by the OLD stop
    # event. If initialize() reused and cleared that event, the straggler
    # would wake, race the new sampler into _latest, and log a spurious crash.
    # Cycling: the first sampler would otherwise drain a finite queue in
    # microseconds, leaving the re-initialized one with nothing to read.
    fake_chip.cycle = True
    fake_chip.frames.append(5)
    device = _make_device()
    await device.initialize()
    old_event = device._stop_event
    old_thread = device._thread
    await device.shutdown()
    assert old_event.is_set()

    fake_chip.dout.closed = False
    fake_chip.sck.closed = False
    await device.initialize()
    try:
        # The new sampler must own a different event, leaving the old one set
        # so any straggler stays muted.
        assert device._stop_event is not old_event
        assert old_event.is_set()
        assert device._thread is not old_thread
        assert await _wait_for_sample(device) == 5.0
    finally:
        await device.shutdown()


async def test_discard_warning_repeats_for_a_persistent_fault(fake_chip, monkeypatch, caplog):
    # A permanent fault must not report once and then go quiet for hours.
    import logging

    import glider.hal.devices.hx711 as hx711_module

    monkeypatch.setattr(hx711_module, "CONSECUTIVE_DISCARD_WARN", 2)
    monkeypatch.setattr(hx711_module, "GLITCH_BACKOFF_S", 0.0)
    rail = 0x7FFFFF
    fake_chip.frames.extend([rail] * 200)
    device = _make_device()
    with caplog.at_level(logging.WARNING, logger="glider.hal.devices.hx711"):
        await device.initialize()
        try:
            await asyncio.sleep(0.3)
        finally:
            await device.shutdown()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 2  # escalation repeats rather than firing once


# --- apply_settings (live recalibration) ------------------------------------------


def test_apply_settings_updates_cached_read_path_authority():
    # The edit dialog mutates a live device's settings. The cached attributes
    # are the read path, so updating only config.settings would leave the
    # running device reading at the old calibration while the saved file
    # claims the new one.
    device = _make_device(settings={"scale": 1.0, "offset": 0.0, "gain": 128})
    device.apply_settings({"scale": 420.0, "offset": 50.0, "gain": 32})
    assert device.scale == 420.0
    assert device.offset == 50.0
    assert device.gain == 32
    assert device.config.settings["scale"] == 420.0


def test_apply_settings_is_partial():
    device = _make_device(settings={"scale": 2.0, "offset": 7.0})
    device.apply_settings({"scale": 3.0})
    assert device.scale == 3.0
    assert device.offset == 7.0  # untouched key survives


def test_apply_settings_rejects_zero_scale_without_committing():
    device = _make_device(settings={"scale": 2.0})
    with pytest.raises(ValueError, match="scale"):
        device.apply_settings({"scale": 0})
    # Validation must precede the commit: a rejected edit leaves the live
    # device exactly as it was, not half-applied.
    assert device.scale == 2.0
    assert device.config.settings["scale"] == 2.0


def test_apply_settings_rejects_bad_gain_without_committing():
    device = _make_device(settings={"gain": 128})
    with pytest.raises(ValueError, match="gain"):
        device.apply_settings({"gain": 100})
    assert device.gain == 128
    assert device.config.settings["gain"] == 128


def test_base_device_apply_settings_updates_config():
    # The generic hook the GUI calls for every device type.
    from glider.hal.base_device import DigitalOutputDevice

    device = DigitalOutputDevice(_FakeBoard(), DeviceConfig(pins={"output": 1}, settings={"a": 1}))
    device.apply_settings({"a": 2, "b": 3})
    assert device.config.settings == {"a": 2, "b": 3}


# --- Registry ---------------------------------------------------------------------


def test_registered_in_device_registry():
    import glider.hal.devices  # noqa: F401 - registration side effect

    assert DEVICE_REGISTRY.get("HX711") is HX711Device


def test_create_device_from_dict_builds_hx711():
    import glider.hal.devices  # noqa: F401 - registration side effect
    from glider.hal.base_device import create_device_from_dict

    data = _make_device(name="scale_1").to_dict()
    device = create_device_from_dict(data, _FakeBoard())
    assert isinstance(device, HX711Device)
