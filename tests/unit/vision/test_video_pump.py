"""The rehearsal pump must deliver every frame, in order, and admit when it lags.

Those three are not stylistic. The live feature extractor computes velocity and
acceleration over *unit frame spacing*, so a skipped frame doubles the apparent
displacement across the gap and inflates exactly the kinematics the behavior
model keys on. A pump that quietly dropped frames would produce confident,
wrong behavior -- and a rehearsal exists to be believed.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from glider.vision.video_pump import VideoPump


class _FakeSource:
    """Stands in for VideoFileSource: a fixed number of numbered frames."""

    def __init__(self, count=10, fps=30.0):
        self.frame_count = count
        self.fps = fps
        self.released = False
        self.loaded_path = None
        self._count = count

    def load(self, path):
        self.loaded_path = path
        return True

    def frames(self):
        for index in range(self._count):
            # The frame's own index is written into the pixels, so a test can
            # prove ordering and completeness from what the callback received.
            yield index, np.full((2, 2, 3), index, dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def pump_factory(monkeypatch):
    """Build a pump over a fake source, returning (pump, received, source)."""

    def _make(count=10, fps=30.0, speed=0.0, on_frame=None, **kwargs):
        source = _FakeSource(count=count, fps=fps)
        monkeypatch.setattr("glider.vision.video_pump.VideoFileSource", lambda: source)
        received: list[tuple[int, float]] = []

        def _record(frame, timestamp):
            received.append((int(frame[0, 0, 0]), timestamp))

        pump = VideoPump("clip.mp4", on_frame or _record, speed=speed, **kwargs)
        return pump, received, source

    return _make


def _run_to_completion(pump, timeout=5.0):
    done = threading.Event()
    original = pump._on_finished

    def _finished(stats):
        if original:
            original(stats)
        done.set()

    pump._on_finished = _finished
    assert pump.start()
    assert done.wait(timeout), "the pump never finished"
    pump.stop()


# --- the hard rule: every frame, in order -------------------------------------


def test_every_frame_is_delivered_in_order(pump_factory):
    pump, received, _ = pump_factory(count=25)

    _run_to_completion(pump)

    assert [index for index, _ in received] == list(range(25))


def test_no_frame_is_skipped_even_when_the_consumer_is_slow(pump_factory):
    """The failure this pump exists to avoid. A slow consumer must make the
    rehearsal late, never gap-toothed."""
    slow_calls = []

    def _slow(frame, timestamp):
        time.sleep(0.004)  # far longer than the 1/1000s frame interval below
        slow_calls.append(int(frame[0, 0, 0]))

    pump, _received, _ = pump_factory(count=20, fps=1000.0, speed=1.0, on_frame=_slow)

    _run_to_completion(pump)

    assert slow_calls == list(range(20))


def test_it_reports_lag_rather_than_catching_up_by_skipping(pump_factory):
    def _slow(frame, timestamp):
        time.sleep(0.004)

    pump, _received, _ = pump_factory(count=15, fps=1000.0, speed=1.0, on_frame=_slow)

    _run_to_completion(pump)

    stats = pump.stats()
    assert stats.frames_delivered == 15
    assert stats.max_lag_s > 0, "a pump that could not keep up reported no lag"


def test_an_unpaced_run_reports_no_lag(pump_factory):
    """Lag is meaningless without a schedule; reporting some would be noise."""
    pump, _received, _ = pump_factory(count=10, speed=0.0)

    _run_to_completion(pump)

    assert pump.stats().max_lag_s == 0.0


# --- timestamps ---------------------------------------------------------------


def test_timestamps_are_wall_clock_like_the_camera(pump_factory):
    """CameraManager passes time.time(); anything downstream measuring elapsed
    real time has to see the same kind of number from a rehearsal."""
    before = time.time()
    pump, received, _ = pump_factory(count=5)

    _run_to_completion(pump)

    after = time.time()
    assert all(before <= ts <= after for _, ts in received)


def test_timestamps_increase(pump_factory):
    pump, received, _ = pump_factory(count=10)

    _run_to_completion(pump)

    stamps = [ts for _, ts in received]
    assert stamps == sorted(stamps)


# --- pacing -------------------------------------------------------------------


def test_real_time_pacing_takes_about_the_clip_length(pump_factory):
    """A 10-frame clip at 50 fps is 0.2 s. Real-time mode has to actually
    take that long, or 'does inference keep up' is unanswerable."""
    pump, _received, _ = pump_factory(count=10, fps=50.0, speed=1.0)

    started = time.perf_counter()
    _run_to_completion(pump)
    elapsed = time.perf_counter() - started

    assert 0.12 < elapsed < 0.9, f"took {elapsed:.3f}s for a 0.2s clip"


def test_unpaced_is_much_faster_than_real_time(pump_factory):
    pump, _received, _ = pump_factory(count=40, fps=50.0, speed=0.0)

    started = time.perf_counter()
    _run_to_completion(pump)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.4, f"unpaced run took {elapsed:.3f}s for a 0.8s clip"


def test_speed_multiplies_the_rate(pump_factory):
    pump, _received, _ = pump_factory(count=20, fps=50.0, speed=4.0)

    started = time.perf_counter()
    _run_to_completion(pump)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.35, f"4x of a 0.4s clip took {elapsed:.3f}s"


# --- lifecycle ----------------------------------------------------------------


def test_stopping_midway_stops_promptly(pump_factory):
    """A rehearsal drives real hardware. Stop has to mean stop, not 'after the
    rest of the clip'."""
    seen = threading.Event()

    def _on_frame(frame, timestamp):
        seen.set()

    pump, _received, _ = pump_factory(count=100000, fps=1000.0, speed=1.0, on_frame=_on_frame)
    assert pump.start()
    assert seen.wait(2.0)

    started = time.perf_counter()
    pump.stop()

    assert time.perf_counter() - started < 1.0
    assert not pump.is_running


def test_a_finished_run_reports_finished(pump_factory):
    captured = {}
    pump, _received, _ = pump_factory(count=5, on_finished=lambda s: captured.update(final=s))

    _run_to_completion(pump)

    assert captured["final"].finished is True
    assert captured["final"].frames_delivered == 5
    assert captured["final"].progress == 1.0


def test_an_unreadable_file_fails_to_start(monkeypatch):
    class _Unloadable:
        frame_count = 0
        fps = 30.0

        def load(self, path):
            return False

        def release(self):
            pass

    monkeypatch.setattr("glider.vision.video_pump.VideoFileSource", lambda: _Unloadable())
    pump = VideoPump("missing.mp4", lambda f, t: None)

    assert pump.start() is False
    assert not pump.is_running


def test_starting_twice_is_refused(pump_factory):
    pump, _received, _ = pump_factory(count=100000, fps=1000.0, speed=1.0)
    assert pump.start()
    try:
        assert pump.start() is False
    finally:
        pump.stop()


def test_stop_is_safe_before_start(pump_factory):
    pump, _received, _ = pump_factory()
    pump.stop()  # must not raise
    assert not pump.is_running


def test_a_raising_consumer_ends_the_run_rather_than_gapping_it(pump_factory):
    """Swallowing the error would deliver a stream with holes in it, which is
    the one thing this pump must never produce."""
    delivered = []

    def _explode(frame, timestamp):
        delivered.append(int(frame[0, 0, 0]))
        if len(delivered) == 3:
            raise RuntimeError("consumer is broken")

    pump, _received, _ = pump_factory(count=50, on_frame=_explode)

    _run_to_completion(pump)

    assert delivered == [0, 1, 2], "the pump kept going after its consumer broke"
    assert pump.stats().finished is True


def test_the_source_is_released(pump_factory):
    pump, _received, source = pump_factory(count=5)

    _run_to_completion(pump)

    assert source.released is True
