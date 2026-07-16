"""HX711 load-cell ADC device.

Reads an HX711 24-bit weigh-scale ADC by bit-banging its two-wire protocol
(DOUT + PD_SCK) directly via gpiozero, bypassing the board abstraction (same
precedent as StepperA4988Device: lazy-import in ``initialize()``, blocking
work off the event loop). A free-running daemon sampler thread paces itself on
the chip's data-ready line and caches the latest validated sample, so
``read``/``get_state()`` never block — the DataRecorder polls the cache at its
own interval.

Protocol (HX711 datasheet): DOUT falls when a conversion is ready; the host
then clocks 25-27 PD_SCK pulses, reading one bit per pulse MSB-first (24-bit
two's complement). The total pulse count selects the NEXT conversion's channel
and gain: 25 = channel A gain 128, 26 = channel B gain 32, 27 = channel A
gain 64. PD_SCK high time must stay under 60 us or the chip powers down
mid-read (and resets to channel A / gain 128). Python on Linux cannot
guarantee that, so every frame is validated (pulse timing + rail saturation)
and glitched frames are discarded; see ``_read_frame``.

Calibration: ``read`` returns ``(raw - offset) / scale``. ``offset`` is set at
runtime by the ``tare`` action (median of the next TARE_SAMPLES fresh
samples); ``scale`` is measured once with a known mass and entered in the
device settings. The cached ``self._offset``/``self._scale`` attributes are
the read-path authority; ``config.settings`` mirrors them for serialization
(``GliderCore.save_session`` syncs settings into the session before saving).
"""

import asyncio
import logging
import statistics
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from glider.hal.base_device import BaseDevice, DeviceConfig

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard

logger = logging.getLogger(__name__)

# gain -> extra PD_SCK pulses after the 24 data bits (per the HX711 datasheet).
GAIN_PULSES: dict[int, int] = {128: 1, 64: 3, 32: 2}

# Datasheet: PD_SCK high > 60 us powers the chip down mid-frame.
SCK_HIGH_LIMIT_S = 60e-6

# 24-bit rails; the datasheet defines these as the out-of-range indicators.
SATURATED = frozenset({0x7FFFFF, -0x800000})

# Consecutive discarded frames before one warning is logged (wiring fault or
# severe CPU starvation; isolated scheduler glitches stay at debug level).
CONSECUTIVE_DISCARD_WARN = 10

# Data-ready poll interval; also bounds shutdown latency while waiting.
READY_POLL_S = 0.001

# A cached sample older than this is not a reading. Generous vs. the chip's
# slowest rate (10 SPS = 100 ms/frame) so ordinary jitter never trips it,
# tight enough that a dead sensor surfaces in about a second.
MAX_SAMPLE_AGE_S = 1.0

TARE_SAMPLES = 5
TARE_TIMEOUT_S = 10.0


class _GlitchError(Exception):
    """A clocked frame failed validation and must be discarded.

    ``powered_down`` marks the timing glitch — the only kind that can have
    powered the chip down mid-frame and reset it to channel A / gain 128, and
    therefore the only kind that needs a dummy re-prime read afterwards.
    """

    def __init__(self, message: str, powered_down: bool = False):
        super().__init__(message)
        self.powered_down = powered_down


class HX711Device(BaseDevice):
    """HX711 24-bit load-cell ADC on two directly-claimed GPIO pins.

    Settings:
    - gain: 128 (channel A, default), 64 (channel A) or 32 (channel B)
    - scale: raw counts per output unit (e.g. counts/gram); non-zero,
        negative allowed (cell mounted in compression). Default 1.0.
    - offset: raw counts at zero load; written by the ``tare`` action.
        Default 0.0.
    """

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        s = config.settings
        gain = int(s.get("gain", 128))
        if gain not in GAIN_PULSES:
            raise ValueError(f"gain must be one of {sorted(GAIN_PULSES)}, got {gain}")
        self._gain = gain
        self._scale = float(s.get("scale", 1.0))
        if self._scale == 0.0:
            raise ValueError("scale must be non-zero")
        self._offset = float(s.get("offset", 0.0))
        # gpiozero handles; owned by the sampler thread after initialize().
        # Deliberately NOT stored in an attribute named ``_pins`` (see
        # stepper_a4988.py — HardwareManager overwrites ``_pins``).
        self._dout: Any = None
        self._sck: Any = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._sample_lock = threading.Lock()
        # Latest validated sample as (raw_counts, perf_counter_timestamp).
        # NOT named ``_state``: DataRecorder._read_device_state returns a
        # ``_state`` attribute in preference to calling get_state().
        self._latest: tuple[int, float] | None = None

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "HX711"

    @property
    def required_pins(self) -> list[str]:
        return ["dout", "sck"]

    @property
    def gain(self) -> int:
        """Configured channel/gain (128 or 64 = channel A, 32 = channel B)."""
        return self._gain

    @property
    def scale(self) -> float:
        """Raw counts per output unit."""
        return self._scale

    @property
    def offset(self) -> float:
        """Raw counts at zero load (updated by ``tare``)."""
        return self._offset

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "read": self.read,
            "read_raw": self.read_raw,
            "tare": self.tare,
        }

    # --- lifecycle ---

    async def initialize(self) -> None:
        """Claim DOUT/SCK via gpiozero and start the sampler thread."""

        def _claim():
            try:
                import gpiozero
            except ImportError as e:
                raise RuntimeError(
                    "gpiozero not installed. Run: pip install 'GLIDER[rpi]' "
                    "(or pip install gpiozero)"
                ) from e
            # Default pull_up=False enables the internal pull-down, which is
            # deliberate: it makes a disconnected DOUT a deterministic
            # stuck-low that _read_frame's liveness check catches every time,
            # where a floating line would read arbitrary noise.
            dout = gpiozero.DigitalInputDevice(self._config.pins["dout"])
            try:
                # SCK low = chip powered up and converting.
                sck = gpiozero.DigitalOutputDevice(self._config.pins["sck"], initial_value=False)
            except Exception:
                # Close what we already claimed; otherwise the DOUT pin stays
                # reserved and the retry fails on DOUT too.
                try:
                    dout.close()
                except Exception:
                    pass
                raise
            return dout, sck

        self._dout, self._sck = await asyncio.to_thread(_claim)
        self._stop_event.clear()
        self._latest = None
        self._thread = threading.Thread(
            target=self._sampler_loop,
            args=(self._dout, self._sck),
            name=f"hx711-sampler-{self._name}",
            daemon=True,
        )
        self._thread.start()
        self._initialized = True
        logger.info("HX711 initialized on pins %s (gain %d)", self._config.pins, self._gain)

    def _sampler_loop(self, dout: Any, sck: Any) -> None:
        """Free-running sampler; sole owner of the GPIO after initialize().

        Paces itself on the chip's data-ready line (10 or 80 SPS, strapped on
        the breakout's RATE pin). Glitched frames are discarded. A *timing*
        glitch may have powered the chip down mid-frame, which resets it to
        channel A / gain 128, so one dummy frame re-primes the configured gain
        before data is trusted again; the other glitch kinds involve no
        power-down, and re-priming after one would throw away a good
        conversion for nothing. Runs until the stop event is set; once it is,
        GPIO errors are expected (shutdown may have closed the handles) and
        exit quietly.
        """
        consecutive_bad = 0
        try:
            while not self._stop_event.is_set():
                try:
                    raw = self._read_frame(dout, sck)
                except _GlitchError as e:
                    consecutive_bad += 1
                    if consecutive_bad == CONSECUTIVE_DISCARD_WARN:
                        logger.warning(
                            "HX711 %s: %d consecutive samples discarded (%s) - "
                            "check wiring / CPU load",
                            self._name,
                            consecutive_bad,
                            e,
                        )
                    else:
                        logger.debug("HX711 %s: discarded sample (%s)", self._name, e)
                    if e.powered_down:
                        # Only a power-down resets the gain to A/128, so only
                        # then is a dummy frame needed to re-prime it.
                        try:
                            self._read_frame(dout, sck)  # dummy read: re-prime gain
                        except _GlitchError:
                            pass
                    continue
                if raw is None:  # stop requested while waiting for data-ready
                    break
                consecutive_bad = 0
                with self._sample_lock:
                    # perf_counter, NOT monotonic: on Windows before Python
                    # 3.13, monotonic() ticks at ~15.6 ms — coarser than a
                    # frame period — which breaks tare's freshness gating.
                    self._latest = (raw, time.perf_counter())
        except Exception:
            if not self._stop_event.is_set():
                logger.exception("HX711 %s: sampler thread crashed", self._name)

    async def shutdown(self) -> None:
        """E-stop safe state (mirrors StepperA4988Device.shutdown ordering).

        Clears ``_initialized`` FIRST so actions queued behind the command
        lock refuse to run, sets the stop event, joins the sampler (bounded -
        the data-ready wait re-checks the event every READY_POLL_S), then
        releases the pins. If the join ever timed out, the straggler thread is
        still safe: its loop swallows GPIO errors once the stop event is set.

        There is deliberately no power-down step. Driving PD_SCK high for
        >60 us powers the chip down, but ``close()`` releases the pin straight
        afterwards and PD_SCK has no pull on typical breakouts — the line
        floats, the (unlatched) power-down lapses, and the chip wakes right
        back up. The hold would be theater. Unlike the stepper's ENABLE line
        there is nothing to make safe: an HX711 is a milliamp ADC that
        actuates nothing, so releasing the pins IS the safe state.
        """
        self._initialized = False
        try:
            self._stop_event.set()
            thread, self._thread = self._thread, None
            dout, self._dout = self._dout, None
            sck, self._sck = self._sck, None

            def _release():
                if thread is not None:
                    thread.join(timeout=1.0)
                    if thread.is_alive():
                        logger.error(
                            "HX711 %s: sampler thread did not exit within 1.0s; "
                            "releasing pins anyway (the straggler exits quietly)",
                            self._name,
                        )
                for dev in (sck, dout):
                    if dev is not None:
                        try:
                            dev.close()
                        except Exception:
                            pass

            await asyncio.to_thread(_release)
        finally:
            self._initialized = False

    def _latest_sample(self) -> tuple[int, float] | None:
        with self._sample_lock:
            return self._latest

    def _fresh_sample(self) -> tuple[int, float] | None:
        """The cached sample, but only while it is still a reading.

        Returns ``None`` when nothing has been sampled yet OR when the cached
        sample has aged past ``MAX_SAMPLE_AGE_S``. Without this the read path
        would report the last good value forever once frames stop being
        accepted — an overloaded cell railing every frame, a severed DOUT, a
        dead sampler thread — and a lab instrument would silently record a
        fabricated constant.
        """
        sample = self._latest_sample()
        if sample is None:
            return None
        if time.perf_counter() - sample[1] > MAX_SAMPLE_AGE_S:
            return None
        return sample

    def _sample_or_raise(self) -> tuple[int, float]:
        """``_fresh_sample`` for the read actions, with a diagnostic raise."""
        sample = self._latest_sample()
        if sample is None:
            raise RuntimeError(f"HX711 {self._name}: no sample received yet")
        age = time.perf_counter() - sample[1]
        if age > MAX_SAMPLE_AGE_S:
            raise RuntimeError(
                f"HX711 {self._name}: last sample is {age:.1f}s old - "
                "sensor may be disconnected or overloaded"
            )
        return sample

    async def get_state(self) -> float | None:
        """Latest calibrated value for the DataRecorder poll.

        ``None`` before the first sample and whenever the cached one is stale,
        so the recorder writes an empty cell — the honest representation of
        "no reading" — rather than a fabricated number. Async because the
        recorder awaits it.
        """
        sample = self._fresh_sample()
        if sample is None:
            return None
        return (sample[0] - self._offset) / self._scale

    # --- protocol (runs in the sampler thread) ---

    def _read_frame(self, dout: Any, sck: Any) -> int | None:
        """Block until data-ready, clock out one sample, validate it.

        Returns the sign-extended raw value, or ``None`` if the stop event was
        set while waiting for data-ready. Raises ``_GlitchError`` when the
        frame fails validation: an SCK-high window that may have crossed the
        60 us power-down threshold (measured conservatively — the window
        includes the GPIO call overhead, so it can only over-report), a DOUT
        line that never rose after the frame (no live chip driving it), or a
        rail-saturated value. Handles are passed in (not read from ``self``)
        so a concurrent ``shutdown()`` nulling the attributes cannot race the
        loop.
        """
        while dout.value:
            if self._stop_event.wait(READY_POLL_S):
                return None

        raw = 0
        max_high = 0.0
        for _ in range(24):
            t0 = time.perf_counter()
            sck.on()
            bit = 1 if dout.value else 0
            sck.off()
            max_high = max(max_high, time.perf_counter() - t0)
            raw = (raw << 1) | bit
        for _ in range(GAIN_PULSES[self._gain]):
            t0 = time.perf_counter()
            sck.on()
            sck.off()
            max_high = max(max_high, time.perf_counter() - t0)

        # Datasheet: the 25th pulse drives DOUT back high. Sample the line now,
        # while it still reflects this frame, and check it below.
        dout_high = bool(dout.value)

        if raw >= 0x800000:  # 24-bit two's complement
            raw -= 0x1000000

        # Timing first: it is the only glitch that implies a power-down, and
        # a stalled thread can also leave the line in an arbitrary state.
        if max_high > SCK_HIGH_LIMIT_S:
            raise _GlitchError(
                f"SCK high {max_high * 1e6:.0f} us exceeded the power-down threshold",
                powered_down=True,
            )
        if not dout_high:
            # A disconnected DOUT reads a permanent low through its internal
            # pull-down, which this protocol means "data ready" — it would
            # otherwise decode into a plausible 0 and spin the loop at 100%
            # of a core with no pacing.
            raise _GlitchError("DOUT stuck low after the frame - no chip driving the line")
        if raw in SATURATED:
            raise _GlitchError(f"saturated reading {raw:#x}")
        return raw

    # --- actions ---

    async def read(self) -> float:
        """Calibrated reading ``(raw - offset) / scale`` from the cached sample.

        Raises if no sample has arrived or the cached one is stale.
        """
        sample = self._sample_or_raise()
        return (sample[0] - self._offset) / self._scale

    async def read_raw(self) -> int:
        """Latest raw signed 24-bit counts from the cached sample.

        Raises if no sample has arrived or the cached one is stale.
        """
        return self._sample_or_raise()[0]

    async def tare(self) -> float:
        """Median of the next TARE_SAMPLES fresh samples becomes the offset.

        Timestamp-gated so the stale cached sample is never counted and each
        conversion counts once. Updates BOTH the cached attribute (the read
        path) and ``config.settings`` (the serialization path — synced into
        the session by ``GliderCore.save_session``). Runs under the
        per-device command lock like every action; needs ~TARE_SAMPLES
        conversion periods (about 0.5 s at 10 SPS).
        """
        if not self._initialized:
            raise RuntimeError(f"HX711 {self._name} is not initialized")
        # perf_counter throughout: it is the clock the sampler stamps samples
        # with (monotonic() is too coarse on Windows before Python 3.13).
        deadline = time.perf_counter() + TARE_TIMEOUT_S
        collected: list[int] = []
        last_ts = time.perf_counter()
        while len(collected) < TARE_SAMPLES:
            await asyncio.sleep(0.01)
            if not self._initialized:
                raise RuntimeError(f"HX711 {self._name} was shut down during tare")
            sample = self._latest_sample()
            if sample is not None and sample[1] > last_ts:
                collected.append(sample[0])
                last_ts = sample[1]
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"HX711 {self._name}: tare timed out after "
                    f"{TARE_TIMEOUT_S:.1f}s "
                    f"({len(collected)}/{TARE_SAMPLES} samples) - "
                    "is the sensor connected?"
                )
        offset = float(statistics.median(collected))
        self._offset = offset
        self._config.settings["offset"] = offset
        logger.info("HX711 %s: tared, offset=%s", self._name, offset)
        return offset

    # --- serialization ---

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "HX711Device":
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance
