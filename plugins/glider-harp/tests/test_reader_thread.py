"""The background reader thread: serial bytes in, cache rows out.

Every wait here is a predicate with a deadline. A thread test that asserts
after a fixed sleep passes or fails on how busy the machine was, which is the
one thing these tests must not measure.
"""

import struct
import threading
import time

import pytest

from glider_harp.frames import ChecksumError, FrameError, FrameSplitter, TruncatedFrameError
from glider_harp.reader import HarpReader, RegisterCache

# Generous: every wait is for something the reader does within milliseconds, so
# the only way to spend this budget is a real failure.
_TIMEOUT_S = 2.0


def _wait_until(predicate, timeout=_TIMEOUT_S, interval=0.002):
    """Poll ``predicate`` until it holds or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _frame(msg_type, address, payload_type, payload, port=0xFF):
    body = bytes([msg_type, len(payload) + 4, address, port, payload_type]) + payload
    return body + bytes([sum(body) & 0xFF])


def _event(address, value=1):
    """An untimestamped event frame for ``address``."""
    return _frame(3, address, 0x01, bytes([value]))


def _timestamped_event(address, value=1, seconds=7, micros=500):
    payload = seconds.to_bytes(4, "little") + micros.to_bytes(2, "little") + bytes([value])
    return _frame(3, address, 0x11, payload)


class _FakeSerial:
    """A serial handle a test can drive: hands out queued chunks, then blocks.

    ``read`` returning ``b""`` after a short sleep is what a real port does
    when its timeout expires with nothing on the wire, which is the state the
    idle flush exists for. ``push`` lets a test add bytes after the reader has
    already drained the queue, so a chunk boundary can be placed at a chosen
    moment rather than guessed at with a sleep.
    """

    def __init__(self, chunks=(), idle_sleep=0.005):
        self._chunks = list(chunks)
        self._lock = threading.Lock()
        self._idle_sleep = idle_sleep
        self.is_open = True
        self.timeout = None
        self.reads = 0
        self.error = None

    def push(self, chunk):
        with self._lock:
            self._chunks.append(chunk)

    def drained(self):
        with self._lock:
            return not self._chunks

    def read(self, _n=1):
        with self._lock:
            self.reads += 1
            chunk = self._chunks.pop(0) if self._chunks else None
        if chunk is not None:
            return chunk
        # Paced like a port timeout, so a reader that spins on a failing read
        # cannot flood the log faster than the test can notice.
        time.sleep(self._idle_sleep)
        if self.error is not None:
            raise self.error
        return b""

    def close(self):
        self.is_open = False


class _ManualClock:
    """A stand-in for ``time`` inside the reader, advanced only by the test.

    The reader consults ``time.monotonic()`` for exactly one policy decision --
    has the line been silent for a whole window -- so substituting the module
    makes "inside the window" a fact the test constructs rather than a bet on
    the host scheduler. Only ``monotonic`` is provided, on purpose: a reader
    that grew a real sleep would fail loudly here instead of timing a test.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._now = 0.0

    def monotonic(self):
        with self._lock:
            return self._now

    def advance(self, seconds):
        with self._lock:
            self._now += seconds


class _BufferedFakeSerial:
    """A handle that reports ``in_waiting``, as pyserial does.

    ``_FakeSerial`` deliberately does not, so the reader is exercised against
    the minimum a handle can offer; this one exists to pin what the reader does
    when the port can say how much it is holding.
    """

    def __init__(self, data=b""):
        self._pending = bytearray(data)
        self._lock = threading.Lock()
        self.timeout = None
        self.requested = []

    @property
    def in_waiting(self):
        with self._lock:
            return len(self._pending)

    def read(self, n=1):
        with self._lock:
            self.requested.append(n)
            taken = bytes(self._pending[:n])
            del self._pending[:n]
        if taken:
            return taken
        time.sleep(0.005)
        return b""


@pytest.fixture
def cache():
    return RegisterCache({32: "lick", 33: "poke"})


@pytest.fixture
def reader_for(cache):
    """Build and start readers, stopping every one however the test ends."""
    readers = []

    def make(serial, idle_flush_s=0.5):
        reader = HarpReader(serial, cache, idle_flush_s=idle_flush_s)
        readers.append(reader)
        reader.start()
        return reader

    yield make
    for reader in readers:
        reader.stop()


# --- frames reaching the cache ---


def test_events_reach_the_cache(cache, reader_for):
    stream = _timestamped_event(32, 1) + _timestamped_event(32, 0)
    reader = reader_for(_FakeSerial([stream]))

    assert _wait_until(lambda: cache.peek()["lick_count"] == 2)
    row = cache.snapshot()
    assert row["lick_state"] == 0
    assert row["lick_count"] == 2
    assert row["lick_last_ms"] == pytest.approx((7 + 500 * 32e-6) * 1000.0)
    assert reader.frame_count == 2
    assert (reader.error_count, reader.decode_failures) == (0, 0)


def test_frames_split_across_reads_are_reassembled(cache, reader_for):
    """The pathological read size a loaded serial port really produces."""
    raw = _event(32, 1)
    reader_for(_FakeSerial([bytes([byte]) for byte in raw]))

    assert _wait_until(lambda: cache.peek()["lick_count"] == 1)
    assert cache.peek()["lick_state"] == 1


def test_a_port_that_reports_in_waiting_is_drained_in_one_read(cache, reader_for):
    """A burst is one read, not one per byte.

    A byte-at-a-time loop is correct and costs a syscall per byte; at 115200
    baud on a Pi that is the difference between idle and not. When the handle
    can say how much it is holding, the reader takes it all.
    """
    fake = _BufferedFakeSerial(b"".join(_event(32, 1) for _ in range(5)))
    reader = reader_for(fake)

    assert _wait_until(lambda: cache.peek()["lick_count"] == 5)
    assert reader.error_count == 0
    assert max(fake.requested) > 1


def test_non_event_frames_are_counted_but_not_recorded(cache, reader_for):
    """frame_count answers "is anything arriving"; the cache answers "did it happen"."""
    reader = reader_for(_FakeSerial([_frame(2, 32, 0x01, bytes([1]))]))

    assert _wait_until(lambda: reader.frame_count == 1)
    assert cache.peek()["lick_count"] == 0


# --- failures that must not end the thread ---


def test_a_corrupt_frame_is_counted_not_raised(cache, reader_for):
    corrupt = bytearray(_event(32, 1))
    corrupt[-1] ^= 0xFF
    good = _event(33, 1)
    reader = reader_for(_FakeSerial([bytes(corrupt), good]))

    assert _wait_until(lambda: cache.peek()["poke_count"] == 1)
    assert reader.error_count == 1
    assert cache.peek()["lick_count"] == 0
    assert reader.is_alive()


@pytest.mark.parametrize(
    "raised",
    [FrameError("invalid header"), TruncatedFrameError("prefix"), ChecksumError("corrupt")],
    ids=["frame-error", "truncated", "checksum"],
)
def test_no_decode_failure_ends_the_thread(cache, reader_for, monkeypatch, raised):
    """The failure Task 5's hierarchy exists to prevent, injected at the handler.

    A reader that branched on the exception's *type* too narrowly -- catching
    ``ChecksumError`` alone, say -- lets a plain ``FrameError`` end the thread,
    and the trial then records nothing with no error anywhere. The messages
    cannot be used to tell the cases apart either: upstream validates the
    checksum before the length field, so a truncated frame's text reads
    "Checksum mismatch".

    Injected by replacing the reader's ``decode``, because the splitter only
    returns frames that already decoded -- which is exactly the assumption this
    pins. Patching the name the reader imported is also what makes bypassing
    ``decode`` for ``harp.protocol`` visible: ``HarpParseError`` is not a
    ``ValueError``, so a frame parsed directly would throw past the handler.
    """
    import glider_harp.reader as reader_module

    real_decode = reader_module.decode
    seen = []

    def flaky(raw):
        seen.append(raw)
        if len(seen) == 1:
            raise raised
        return real_decode(raw)

    monkeypatch.setattr(reader_module, "decode", flaky)
    reader = reader_for(_FakeSerial([_event(32, 1), _event(33, 1)]))

    assert _wait_until(lambda: cache.peek()["poke_count"] == 1)
    assert reader.is_alive()
    assert reader.decode_failures == 1
    assert reader.frame_count == 1
    # A decoder disagreeing with the splitter is not a corrupt cable.
    assert reader.error_count == 0


@pytest.mark.parametrize(
    "raised",
    [ValueError("bad payload"), struct.error("unpack requires 4 bytes"), TypeError("not bytes")],
    ids=["value-error", "struct-error", "type-error"],
)
def test_an_error_processing_a_read_does_not_end_the_thread(cache, reader_for, monkeypatch, raised):
    """The other door into a silently empty recording.

    Only the read was guarded once. Anything raised on the way from bytes to
    the cache -- out of the splitter, the decoder, or ``ingest`` -- ended the
    thread with ``failure`` still None and the traceback going to
    ``threading.excepthook`` on stderr, where a packaged GUI shows nobody and
    ``logger`` never sees it. The device then polls a cache that will never
    change again.

    Not reachable while payloads are read as plain little-endian integers. It
    becomes reachable the moment payload decoding grows types of its own,
    which is why these are the exceptions that decoding raises.
    """
    real_ingest = cache.ingest
    seen = []

    def flaky(frame):
        seen.append(frame)
        if len(seen) == 1:
            raise raised
        return real_ingest(frame)

    monkeypatch.setattr(cache, "ingest", flaky)
    reader = reader_for(_FakeSerial([_event(32, 1), _event(33, 1)]))

    assert _wait_until(lambda: cache.peek()["poke_count"] == 1)
    assert reader.is_alive()
    assert reader.failure is None
    assert reader.processing_errors == 1
    # A bug in this package is not corruption on the wire.
    assert (reader.error_count, reader.decode_failures) == (0, 0)


def test_an_error_processing_the_idle_flush_does_not_end_the_thread(cache, reader_for, monkeypatch):
    """The flush reaches the cache by its own path, so it needs its own guard.

    A guard wrapped around the read path alone leaves this one bare, and it is
    the path that runs at the end of every trial.
    """
    events = [_event(32, i % 2) for i in range(36)]
    stalled = bytes([3, 255]) + b"".join(events)
    assert FrameSplitter().feed(stalled) == []

    def always_raises(frame):
        raise ValueError("bad payload")

    monkeypatch.setattr(cache, "ingest", always_raises)
    reader = reader_for(_FakeSerial([stalled]), idle_flush_s=0.05)

    assert _wait_until(lambda: reader.processing_errors >= 1)
    assert reader.idle_flushes >= 1
    assert reader.is_alive()
    assert reader.failure is None


def test_a_read_failure_stops_the_reader(cache, reader_for):
    """A dead port must stop the thread visibly, not propagate and not spin."""
    fake = _FakeSerial([_event(32, 1)], idle_sleep=0.05)
    reader = reader_for(fake)
    assert _wait_until(lambda: cache.peek()["lick_count"] == 1)

    fake.error = OSError("device disconnected")

    assert _wait_until(lambda: not reader.is_alive())
    assert isinstance(reader.failure, OSError)
    # error_count is corruption on the wire, not the state of the port.
    assert reader.error_count == 0
    reader.stop()


# --- the idle flush ---


@pytest.mark.parametrize("message_type", [1, 2, 3], ids=["read", "write", "event"])
def test_the_idle_flush_releases_stranded_frames(cache, reader_for, message_type):
    """The splitter's one residual stall, which only a clock can resolve.

    A noise byte that happens to be a valid message type, followed by a length
    byte claiming a long frame, parks the splitter in "waiting for the rest".
    Ample trailing data always settles it, so this is invisible on a live
    stream -- and permanent if the device falls silent, which is how every
    trial ends.

    All three message types are swept because a single head is a chosen
    constant: nothing about the stall is specific to Event, and a fix that
    happened to key on one of them would pass with only the measured case.
    """
    events = [_event(32, i % 2) for i in range(36)]
    stalled = bytes([message_type, 255]) + b"".join(events)
    # Asserted, not assumed: the point of the input is that the splitter alone
    # emits none of it, which no timing in the reader can be blamed for.
    assert FrameSplitter().feed(stalled) == []

    reader = reader_for(_FakeSerial([stalled]), idle_flush_s=0.05)

    assert _wait_until(lambda: cache.peek()["lick_count"] == 36)
    assert reader.idle_flushes >= 1
    assert reader.error_count == 0


def test_a_stalled_head_is_not_flushed_before_the_window(cache, reader_for):
    """Flushing on the first empty read would invent messages out of payloads.

    This frame's payload contains the bytes of a complete, correctly
    checksummed inner frame for a different register -- payloads are arbitrary
    bytes, so this is a matter of time on a real device. Cut the read in the
    middle of it and a reader that hunted for a boundary as soon as a read came
    back empty emits the inner frame and loses the outer one: one event
    recorded against the wrong register, one event lost.
    """
    inner = _event(32, 7)
    outer = _frame(3, 33, 0x01, bytes([0]) + inner + bytes(4))
    cut = outer.find(inner) + len(inner)
    assert 0 < cut < len(outer)

    fake = _FakeSerial([outer[:cut]])
    reader = reader_for(fake, idle_flush_s=5.0)
    assert _wait_until(fake.drained)

    # Several reads come back empty here -- the moment an over-eager flush
    # fires -- while the window has nowhere near elapsed.
    reads = fake.reads
    assert _wait_until(lambda: fake.reads >= reads + 5)
    fake.push(outer[cut:])

    assert _wait_until(lambda: cache.peek()["poke_count"] == 1)
    assert cache.peek()["lick_count"] == 0
    assert reader.idle_flushes == 0


def test_an_embedded_frame_is_mis_framed_by_a_mid_frame_flush(cache, reader_for):
    """The idle flush's accepted cost, pinned rather than left to a comment.

    Same adversarial input as the test above -- an outer event for ``poke``
    whose payload contains a complete, correctly checksummed inner event for
    ``lick`` -- but with the window short enough that the flush actually fires
    while the outer frame is half-arrived. That test proves the flush does not
    fire early; without this one, nothing exercises it firing at all on a
    payload that can be mis-framed, and the comment describing the damage
    would be the only thing asserting it.

    The damage is deterministic, not the "one offset in 256" a phantom
    normally costs: clearing the splitter's sync belief is precisely what
    disarms its guard against hunting inside an in-flight head, so resync finds
    the embedded frame every time. The real event is lost and a phantom lands
    on the wrong register, with no counter moving to say so.

    Pinned because it is accepted, not fixed: preventing it means judging an
    in-flight head by its payload-type bits, which duplicates the parser
    knowledge ``frames`` isolates and would hold real stalls open. If a future
    change does prevent it, this test should fail and be rewritten to the
    better behaviour -- that is the point of pinning it.
    """
    inner = _event(32, 7)
    outer = _frame(3, 33, 0x01, bytes([0]) + inner + bytes(4))
    cut = outer.find(inner) + len(inner)
    fake = _FakeSerial([outer[:cut]])
    reader = reader_for(fake, idle_flush_s=0.05)

    assert _wait_until(lambda: reader.idle_flushes >= 1)
    fake.push(outer[cut:])

    assert _wait_until(lambda: cache.peek()["lick_count"] == 1)  # the phantom
    row = cache.peek()
    assert row["lick_state"] == 7  # the inner frame's payload, never sent as an event
    assert row["poke_count"] == 0  # the event that really happened, lost
    # Nothing in the counters marks the swap: it is corruption-free framing.
    assert (reader.error_count, reader.decode_failures) == (0, 0)


def test_a_slow_device_is_not_flushed_while_bytes_keep_arriving(cache, reader_for):
    """The window is silence on the line, not elapsed time since the reader started.

    This frame dribbles in over more than twice the idle window, with every gap
    far inside it. A reader that timed the window from anywhere but the last
    byte it read would tear a frame that was simply arriving slowly.
    """
    raw = _frame(3, 32, 0x01, bytes(range(55)))
    assert len(raw) == 61
    fake = _FakeSerial()
    reader = reader_for(fake, idle_flush_s=0.25)

    for byte in raw:
        fake.push(bytes([byte]))
        time.sleep(0.01)

    assert _wait_until(lambda: cache.peek()["lick_count"] == 1)
    assert reader.idle_flushes == 0


def test_a_quiet_link_with_nothing_held_never_flushes(cache, reader_for):
    """Silence alone is not a stall: with an empty buffer there is nothing to release."""
    fake = _FakeSerial([_event(32, 1)])
    reader = reader_for(fake, idle_flush_s=0.02)
    assert _wait_until(lambda: cache.peek()["lick_count"] == 1)

    reads = fake.reads
    assert _wait_until(lambda: fake.reads >= reads + 20)  # many windows' worth
    assert reader.idle_flushes == 0


def test_a_buffer_with_nothing_framable_is_rescanned_once_per_window(reader_for, monkeypatch):
    """A head that never resolves must cost one rescan per window, not one per read.

    These two bytes claim a 257-byte frame that will never arrive, so the flush
    releases nothing and the buffer stays exactly as it was -- the shape a
    device that died mid-frame leaves behind. Rescanning it on every read means
    decoding at every offset several times a second for the rest of the trial.

    The reader's clock is substituted rather than raced. Measured in wall
    time, "many reads well inside one window" was a bet that thirty ~5 ms
    reads finish inside 0.5 s, and a loaded macOS runner loses it: each sleep
    can oversleep past 16 ms, the window genuinely elapses, and the reader's
    correct once-per-window flush read as a failure. With the clock held
    still, a second flush can only be the per-read rescan this test forbids.
    """
    import glider_harp.reader as reader_module

    clock = _ManualClock()
    monkeypatch.setattr(reader_module, "time", clock)

    fake = _FakeSerial([bytes([3, 255])])
    reader = reader_for(fake, idle_flush_s=0.5)

    # A second read has begun, so the loop has finished processing the chunk
    # and the silence being timed started, on the manual clock, at zero.
    assert _wait_until(lambda: fake.reads >= 2)
    clock.advance(0.6)  # one window of silence
    assert _wait_until(lambda: reader.idle_flushes >= 1)

    # Any number of further empty reads inside the frozen window: no rescan.
    reads = fake.reads
    assert _wait_until(lambda: fake.reads >= reads + 30)
    assert reader.idle_flushes == 1

    # And the next window costs exactly one more, however many reads it holds.
    clock.advance(0.6)
    assert _wait_until(lambda: reader.idle_flushes >= 2)
    reads = fake.reads
    assert _wait_until(lambda: fake.reads >= reads + 30)
    assert reader.idle_flushes == 2


# --- what the counters mean ---


@pytest.mark.parametrize("fill", [b"\xff", b"\x00"], ids=["ff", "00"])
def test_error_count_tracks_corruption_not_framing_noise(cache, reader_for, fill):
    """A device already mid-transmission when the port opens is normal, not a fault.

    Conflating framing noise with corruption makes every healthy connect look
    like a failing cable, which is the one thing this count is consulted for.
    Two fills, because one value passing says nothing about the rule: 0xff
    claims a frame longer than the burst and 0x00 one shorter than the minimum,
    so they reach the splitter's "cannot judge yet" and "reject at once" paths
    respectively and still cost no corruption. (Sweeping further is not free:
    at 0x7f the length byte lands a complete, wrongly-checksummed candidate at
    the head -- corruption by the splitter's own definition, counted on
    purpose, and a different case from framing noise.)
    """
    fake = _FakeSerial([fill * 200 + _event(32, 1)])
    reader = reader_for(fake)

    assert _wait_until(lambda: cache.peek()["lick_count"] == 1)
    assert reader.error_count == 0
    assert (reader.resyncs, reader.bytes_discarded) == (1, 200)


def test_the_reader_never_consumes_the_counters(cache, reader_for):
    """``snapshot`` clears counts, so exactly one caller may poll it -- not this thread.

    A reader that read its own cache would silently eat events out of the CSV,
    leaving a recording that is wrong in no visible way.
    """
    fake = _FakeSerial([b"".join(_event(32, 1) for _ in range(3))])
    reader = reader_for(fake, idle_flush_s=0.02)
    assert _wait_until(lambda: cache.peek()["lick_count"] == 3)

    reads = fake.reads
    assert _wait_until(lambda: fake.reads >= reads + 20)  # well past the idle window
    assert cache.peek()["lick_count"] == 3
    assert cache.snapshot()["lick_count"] == 3
    assert reader.frame_count == 3


# --- lifecycle ---


def _live_reader_threads():
    return [t for t in threading.enumerate() if t.name == "harp-reader" and t.is_alive()]


def test_stop_joins_the_thread(cache):
    """``stop`` must leave nothing running; a long read must not defeat it.

    The handle here sits 0.2 s in every empty read, so a ``stop`` that only
    asked the thread to finish would return with it still running. The thread
    itself is checked as well as ``is_alive``, because reporting a thread as
    stopped by forgetting about it satisfies the flag while leaving it reading.
    """
    reader = HarpReader(_FakeSerial(idle_sleep=0.2), cache)
    reader.start()
    assert _wait_until(reader.is_alive)

    assert reader.stop() is True

    assert reader.is_alive() is False
    assert _live_reader_threads() == []
    assert reader.stop() is True  # idempotent


def test_stop_reports_a_thread_it_could_not_join(cache):
    """False means the thread still owns the port, and the caller must not use it.

    Otherwise the only signal is a log line: the caller carries on, writes a
    register, reads the reply back on a short timeout while the live reader
    consumes it, and sees nothing but a timeout it cannot explain. Reachable on
    a loaded Pi at the end of a trial, where the thread may not be scheduled
    through a read plus overhead inside the join.

    The handle here parks in ``read`` far longer than the join allows, which is
    that situation without the load.
    """
    fake = _FakeSerial(idle_sleep=1.0)
    fake.timeout = 3.0
    reader = HarpReader(fake, cache)
    reader.start()
    assert _wait_until(reader.is_alive)

    assert reader.stop(timeout=0.05) is False

    assert reader.is_alive() is True  # and says so, rather than reporting gone
    # The timeout stays borrowed on purpose: a thread still running needs the
    # short one to keep noticing the stop event.
    assert fake.timeout != 3.0
    assert reader.stop(timeout=2.0) is True  # retryable, and it finishes the job
    assert fake.timeout == 3.0


def test_a_failed_thread_start_leaves_the_reader_startable(cache, monkeypatch):
    """Thread exhaustion on a Pi must not strand the port or mask the error.

    Recording the thread before it has started makes the caller's ``stop``
    raise "cannot join thread before it is started" -- the original error lost,
    the borrowed timeout never returned.
    """
    fake = _FakeSerial()
    fake.timeout = 3.0
    reader = HarpReader(fake, cache)
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("can't start"))
    )

    with pytest.raises(RuntimeError, match="can't start"):
        reader.start()

    assert reader.is_alive() is False
    assert fake.timeout == 3.0  # handed back, not stranded
    assert reader.stop() is True  # and does not raise about an unstarted thread

    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="one-shot"):
        reader.start()  # retired by the stop above, not by the failed start


def test_a_reader_whose_start_failed_can_be_started_again(cache, monkeypatch):
    """A failed start leaves nothing behind, so the caller's retry is the fix."""
    fake = _FakeSerial()
    reader = HarpReader(fake, cache)
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("can't start"))
    )
    with pytest.raises(RuntimeError, match="can't start"):
        reader.start()

    monkeypatch.undo()
    reader.start()
    try:
        assert reader.is_alive()
    finally:
        reader.stop()


def test_a_stopped_reader_cannot_be_restarted(cache):
    """``stop`` before ``start`` is harmless in itself, and must retire the reader.

    The stop event is latched, so a reader started after any stop would end on
    its first loop test: alive for microseconds, then quietly gone, leaving an
    empty recording and nothing anywhere saying the reader never ran.
    """
    fake = _FakeSerial()
    fake.timeout = 3.0
    reader = HarpReader(fake, cache)
    assert reader.stop() is True

    with pytest.raises(RuntimeError):
        reader.start()
    assert reader.is_alive() is False
    # A reader that never ran never borrowed anything, so it has nothing to
    # hand back -- restoring regardless would clear a timeout it never took.
    assert fake.timeout == 3.0


@pytest.mark.parametrize("caller_timeout", [3.0, None], ids=["chosen", "unset"])
def test_stop_gives_the_port_its_read_timeout_back(cache, caller_timeout):
    """The reader borrows the timeout; it does not keep it.

    The caller owns this port for writes and register round-trips too. A
    timeout it chose and silently lost does not fail here -- it surfaces much
    later as a read that returned early, on a port nobody remembers editing.
    """
    fake = _FakeSerial(idle_sleep=0.01)
    fake.timeout = caller_timeout
    reader = HarpReader(fake, cache)

    reader.start()
    assert fake.timeout != caller_timeout  # borrowed while running

    reader.stop()
    assert fake.timeout == caller_timeout
    reader.stop()  # idempotent, and does not re-restore
    assert fake.timeout == caller_timeout


def test_the_reader_thread_is_a_daemon(reader_for):
    """A reader that outlived its device must never hold the process open."""
    reader_for(_FakeSerial())
    threads = _live_reader_threads()
    assert threads
    assert all(thread.daemon for thread in threads)


def test_start_is_one_shot(cache):
    """Restarting would resume a splitter still holding the last session's bytes."""
    reader = HarpReader(_FakeSerial(), cache)
    reader.start()
    try:
        with pytest.raises(RuntimeError):
            reader.start()
    finally:
        reader.stop()
    with pytest.raises(RuntimeError):
        reader.start()


@pytest.mark.parametrize(
    "idle_flush_s, ceiling",
    [(0.2, 0.2), (10.0, 1.0)],
    ids=["bounded-by-the-idle-window", "bounded-for-stop"],
)
def test_the_reader_bounds_the_port_read_timeout(reader_for, idle_flush_s, ceiling):
    """Everything the reader does on a silent line happens when a read returns.

    A port opened with ``timeout=None`` blocks in ``read`` forever, so the stop
    event is never seen and the idle window never elapses. The reader takes the
    handle's timeout for the duration rather than trusting whoever opened it.
    """
    fake = _FakeSerial()
    reader_for(fake, idle_flush_s=idle_flush_s)
    assert 0 < fake.timeout <= ceiling


def test_a_non_positive_idle_window_is_rejected(cache):
    """Zero would hunt for a boundary inside every frame that arrived in pieces."""
    with pytest.raises(ValueError):
        HarpReader(_FakeSerial(), cache, idle_flush_s=0)
