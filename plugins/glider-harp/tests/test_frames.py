"""Harp binary frame decoding."""

import pytest
from harp.protocol import HarpParseError

from glider_harp.frames import (
    ChecksumError,
    FrameError,
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
