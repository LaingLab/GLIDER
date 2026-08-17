"""Harp binary frame decoding."""

import pytest
from harp.protocol import HarpParseError

from glider_harp.frames import (
    ChecksumError,
    FrameError,
    FrameSplitter,
    HarpFrame,
    TruncatedFrameError,
    decode,
)


def _frame(msg_type, address, payload_type, payload, port=0xFF):
    body = bytes([msg_type, len(payload) + 4, address, port, payload_type]) + payload
    return body + bytes([sum(body) & 0xFF])


def _valid():
    return _frame(msg_type=3, address=32, payload_type=0x01, payload=bytes([1]))


def test_decode_untimestamped_event():
    raw = _frame(msg_type=3, address=32, payload_type=0x01, payload=bytes([1]))
    frame = decode(raw)
    assert isinstance(frame, HarpFrame)
    assert frame.message_type == 3
    assert frame.address == 32
    assert frame.payload == bytes([1])
    assert frame.timestamp is None


def test_decode_timestamped_event():
    """Bit 4 of PayloadType means Seconds(U32) + Micros(U16) precede payload."""
    payload = (7).to_bytes(4, "little") + (500).to_bytes(2, "little") + bytes([1])
    raw = _frame(msg_type=3, address=32, payload_type=0x11, payload=payload)
    frame = decode(raw)
    assert frame.payload == bytes([1])
    assert frame.timestamp == pytest.approx(7 + 500 * 32e-6)


def test_bad_checksum_raises():
    raw = bytearray(_frame(3, 32, 0x01, bytes([1])))
    raw[-1] ^= 0xFF
    with pytest.raises(ChecksumError):
        decode(bytes(raw))


def test_decode_preserves_address_port_and_payload_type():
    """Every field read straight off the header, at values distinct from the defaults."""
    raw = _frame(msg_type=1, address=44, payload_type=0x02, payload=bytes([1, 2]), port=2)
    frame = decode(raw)
    assert frame.message_type == 1
    assert frame.address == 44
    assert frame.port == 2
    assert frame.payload_type == 0x02


def test_payload_type_excludes_the_timestamp_flag():
    """payload_type describes the element type alone; 0x10 is carried by timestamp."""
    payload = (7).to_bytes(4, "little") + (500).to_bytes(2, "little") + bytes([1])
    raw = _frame(msg_type=3, address=32, payload_type=0x11, payload=payload)
    frame = decode(raw)
    assert frame.payload_type == 0x01
    assert frame.timestamp is not None


@pytest.mark.parametrize(
    "raw",
    [
        _valid()[:-1],
        _valid()[:-2],
        _valid()[:4],
        bytes([3, 5]),
        bytes([3]),
        b"",
    ],
    ids=["checksum-dropped", "two-bytes-dropped", "four-bytes", "two-garbage", "one-byte", "empty"],
)
def test_truncated_frame_is_retryable(raw):
    """A short read must not look like corruption.

    A reader that treats truncation as a bad checksum discards a good frame
    every time a serial read splits mid-frame, instead of waiting for the rest.
    """
    with pytest.raises(TruncatedFrameError) as excinfo:
        decode(raw)
    assert not isinstance(excinfo.value, ChecksumError)
    assert "checksum" not in str(excinfo.value).lower()


def test_no_partial_read_of_a_realistic_frame_blames_the_checksum():
    """Sweep every prefix of a timestamped frame, as a serial reader would see them.

    The parser upstream validates the checksum before the length field, so
    without our own ordering nearly every prefix here reports a checksum
    mismatch -- swamping Task 8's logs with corruption reports for what are
    ordinary partial reads.
    """
    payload = (7).to_bytes(4, "little") + (500).to_bytes(2, "little") + bytes(range(8))
    raw = _frame(msg_type=3, address=44, payload_type=0x11, payload=payload)
    assert len(raw) == 20

    for end in range(len(raw)):
        with pytest.raises(TruncatedFrameError) as excinfo:
            decode(raw[:end])
        assert "checksum" not in str(excinfo.value).lower(), f"prefix of {end} bytes"

    assert decode(raw).address == 44


def test_no_decode_failure_can_escape_a_reader():
    """Every failure must be caught by both `except FrameError` and `except ValueError`.

    An escaping exception ends the reader thread, and the recording then comes
    back empty with nothing logged -- the failure this hierarchy exists to
    prevent. Both arms are pinned because each is a single word in a base
    class, easily lost in a refactor and silent when wrong.
    """
    for exc_type in (FrameError, TruncatedFrameError, ChecksumError):
        assert issubclass(exc_type, FrameError)
        assert issubclass(exc_type, ValueError)


def test_over_long_buffer_is_not_retryable():
    """Concatenated frames must resync, not wait for bytes that will never help."""
    pair = _valid() + _valid()
    with pytest.raises(FrameError) as excinfo:
        decode(pair)
    assert not isinstance(excinfo.value, TruncatedFrameError)
    assert not isinstance(excinfo.value, ChecksumError)


@pytest.mark.parametrize(
    "raw",
    [
        _frame(msg_type=0, address=32, payload_type=0x01, payload=bytes([1])),
        _frame(msg_type=3, address=32, payload_type=0x03, payload=bytes([1, 2, 3])),
        _frame(msg_type=3, address=32, payload_type=0x21, payload=bytes([1])),
    ],
    ids=["bad-message-type", "bad-payload-size-nibble", "reserved-payload-bit"],
)
def test_upstream_parse_errors_are_translated(raw):
    """Frames that only upstream rejects must still surface as our own type.

    These are structurally sound and correctly checksummed, so they get past
    every local check and reach ``HarpMessage.parse``. That translation is the
    module's whole reason for existing, so it needs a test that reaches it.
    """
    with pytest.raises(FrameError) as excinfo:
        decode(raw)
    assert not isinstance(excinfo.value, HarpParseError)
    assert not isinstance(excinfo.value, TruncatedFrameError)
    # Chained from upstream, proving the frame really did reach the parser
    # rather than being rejected by a local check that happens to agree.
    assert isinstance(excinfo.value.__cause__, HarpParseError)


def _other():
    """A frame distinguishable from ``_valid()`` in both length and contents."""
    return _frame(msg_type=1, address=44, payload_type=0x02, payload=bytes([7, 9]), port=2)


def test_splitter_yields_complete_frames():
    splitter = FrameSplitter()
    a = _frame(3, 32, 0x01, bytes([1]))
    b = _frame(3, 32, 0x01, bytes([0]))
    assert list(splitter.feed(a + b)) == [a, b]


def test_splitter_buffers_partial_frame():
    splitter = FrameSplitter()
    raw = _frame(3, 32, 0x01, bytes([1]))
    assert list(splitter.feed(raw[:3])) == []
    assert list(splitter.feed(raw[3:])) == [raw]
    # Consuming one byte too few here leaves the frame's checksum byte behind,
    # which desynchronises everything after it. Nothing was noise, so a clean
    # split throws nothing away and never loses framing; an off-by-one shows up
    # as the resync it forces. Asserting the yielded frames alone misses it.
    assert (splitter.bytes_discarded, splitter.resyncs) == (0, 0)


def test_splitter_recovers_after_garbage():
    """A desynchronised byte must not wedge the stream permanently."""
    splitter = FrameSplitter()
    raw = _frame(3, 32, 0x01, bytes([1]))
    list(splitter.feed(b"\x00"))
    frames = list(splitter.feed(raw))
    assert raw in frames


def test_feed_takes_effect_when_called_not_when_iterated():
    """Absorbing the chunk must be tied to the call, not to consumption.

    Written as a generator function, ``feed`` would defer its buffer append
    until the caller iterated. A caller that ignored the return value would
    then silently lose the chunk, and two feeds consumed out of order would
    splice the stream backwards -- both invisible until a device produces them.
    """
    splitter = FrameSplitter()
    raw = _valid()
    splitter.feed(raw[:3])  # return value deliberately dropped
    assert list(splitter.feed(raw[3:])) == [raw]

    # Same property from the other side: iterators taken before either is
    # consumed still reflect the order the chunks were fed in, not read in.
    a = _valid()
    b = _other()
    splitter = FrameSplitter()
    stream = a + b
    first = splitter.feed(stream[:3])
    second = splitter.feed(stream[3:])
    assert list(second) == [a, b]
    assert list(first) == []


def test_resync_consumes_the_frame_it_recovers():
    """Recovery must leave the reader positioned after the frame, holding nothing.

    Resync self-heals, so a splitter that consumed the wrong number of bytes
    here still limps along and still yields the right frames -- it just carries
    a stale byte in front of every later frame, which is one coincidence away
    from framing a bogus message. The buffer catches that; the output does not.
    """
    splitter = FrameSplitter()
    a = _valid()
    b = _other()
    assert splitter.feed(b"\x00" + a) == [a]
    assert splitter.bytes_discarded == 1

    # The next frame must come back having cost nothing further. A stale byte
    # left in front of it forces another resync to clear, which the yielded
    # frames alone never reveal because resync recovers either way.
    assert splitter.feed(b) == [b]
    assert (splitter.bytes_discarded, splitter.resyncs) == (1, 1)


def test_splitter_accepts_one_byte_at_a_time():
    """The pathological read size a loaded serial port really produces.

    Order is asserted, not just membership: a splitter that emitted the second
    frame first would still satisfy a set comparison.
    """
    splitter = FrameSplitter()
    a = _valid()
    b = _other()
    assert a != b
    yielded = []
    for byte in a + b:
        yielded.extend(splitter.feed(bytes([byte])))
    assert yielded == [a, b]


def test_corrupt_frame_does_not_consume_its_successor():
    """Resync must land on the next real frame, not swallow it as padding.

    The bytes after a corrupt frame are re-examined from every offset, so the
    good frame behind it survives. A splitter that trusted the corrupt frame's
    length byte and skipped that many bytes would lose the good one.
    """
    splitter = FrameSplitter()
    corrupt = bytearray(_valid())
    corrupt[-1] ^= 0xFF
    good = _other()
    assert list(splitter.feed(bytes(corrupt) + good)) == [good]


def test_splitter_survives_a_chunk_boundary_at_every_offset():
    """Sweep every split of a two-frame buffer, as a serial reader would see them.

    Frames of unequal length are used so the boundary lands at a different
    place within each frame, and so reversed output is detectable.
    """
    a = _valid()
    b = _other()
    stream = a + b
    assert len(a) != len(b)

    for split in range(len(stream) + 1):
        splitter = FrameSplitter()
        yielded = list(splitter.feed(stream[:split]))
        yielded += list(splitter.feed(stream[split:]))
        assert yielded == [a, b], f"split at {split}"


def test_partial_read_does_not_reframe_a_payload_that_looks_like_a_frame():
    """While synchronised, a short read must wait rather than hunt forward.

    This outer frame's payload happens to contain the bytes of a complete,
    correctly checksummed inner frame -- payloads are arbitrary bytes, so this
    is a matter of time on a real device. A splitter that scanned forward
    whenever the head was incomplete would emit the inner frame, then treat
    the rest of the outer frame as garbage: one message invented, one lost.
    """
    inner = _valid()
    outer = _frame(msg_type=3, address=33, payload_type=0x01, payload=bytes([0]) + inner + bytes(4))
    cut = outer.find(inner) + len(inner)
    assert 0 < cut < len(outer)

    splitter = FrameSplitter()
    assert list(splitter.feed(outer[:cut])) == []
    assert list(splitter.feed(outer[cut:])) == [outer]


@pytest.mark.parametrize("noise", [1, 6, 200, 255, 256, 257, 3000])
def test_noise_does_not_strand_the_frames_behind_it(noise):
    """Noise must never park the splitter in "waiting for the rest".

    Every one of these bytes claims a 257-byte frame. Trusting that length at
    an offset nothing has validated made the splitter withhold each frame
    behind it until 257 bytes happened to arrive -- and forever if the device
    went quiet at the end of a trial. Only bursts of a full maximum frame or
    more recovered, so sweeping the size is what catches it: a single large
    value passes while a realistic 200-byte burst swallows the recording.
    """
    splitter = FrameSplitter()
    good = _valid()
    assert splitter.feed(b"\xff" * noise) == []
    assert splitter.feed(good) == [good]


def test_frames_behind_noise_are_not_delayed_into_a_burst():
    """Measured case from review: ten events behind two noise bytes."""
    splitter = FrameSplitter()
    events = [_frame(3, 32, 0x01, bytes([i])) for i in range(10)]
    assert splitter.feed(b"\x00\xc8" + b"".join(events)) == events
    assert splitter.bytes_discarded == 2


def test_scanning_past_noise_does_not_inflate_the_corruption_count():
    """Noise must cost at most one corruption report, not one per offset.

    Resync tries every offset in the buffer, and a one-byte wrapping checksum
    passes by chance at roughly one offset in 256. Counting those would make
    any stretch of garbage read as a failing cable, which is the opposite of
    what Task 8 surfaces the count for.
    """
    splitter = FrameSplitter()
    good = _valid()
    assert splitter.feed(b"\xff" * 3000 + good) == [good]
    # Only the one complete-looking candidate at the head we had reason to
    # trust; the thousands scanned past during resync are framing, not corruption.
    assert splitter.checksum_errors <= 1
    assert splitter.bytes_discarded >= 3000


def test_corruption_is_counted_apart_from_framing_noise():
    """The counters are Task 8's only view of what the splitter swallowed."""
    splitter = FrameSplitter()
    corrupt = bytearray(_valid())
    corrupt[-1] ^= 0xFF
    good = _other()
    assert splitter.feed(bytes(corrupt) + good) == [good]
    assert splitter.checksum_errors == 1
    assert splitter.resyncs == 1
    assert splitter.bytes_discarded == len(corrupt)


def test_a_clean_stream_reports_nothing_swallowed():
    """A healthy stream must leave every counter at zero, or they are useless."""
    splitter = FrameSplitter()
    stream = _valid() + _other() + _valid()
    assert len(splitter.feed(stream)) == 3
    assert (splitter.checksum_errors, splitter.resyncs, splitter.bytes_discarded) == (0, 0, 0)


def test_splitter_does_not_hoard_bytes_it_can_never_frame():
    """A corrupt length byte must not grow the buffer without bound.

    Every byte of this noise claims a 257-byte frame, so a splitter that simply
    waits for each claimed length accumulates the whole stream forever. Once
    more than one maximum frame is held with nothing decodable in it, the
    leading bytes cannot start a frame and must be dropped.
    """
    splitter = FrameSplitter()
    assert list(splitter.feed(b"\xff" * 3000)) == []
    assert len(splitter._buffer) <= 257

    # ...and the buffer is still a working splitter, not merely a small one.
    raw = _valid()
    assert raw in list(splitter.feed(raw))
