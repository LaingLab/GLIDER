"""Decode Harp binary frames.

A thin adapter over ``harp.protocol``. That package is a pre-1.0 release
candidate whose API is not frozen -- 0.4.0 and 0.5.0rc1 are entirely different
-- so the rest of ``glider_harp`` depends on the names defined here rather than
on upstream's. An upstream change should touch this file and no other.

The wire format itself is stable:

    MessageType(1) | Length(1) | Address(1) | Port(1) | PayloadType(1)
        | [Seconds(U32) Micros(U16)] | Payload | Checksum(1)
"""

from dataclasses import dataclass

from harp.protocol import HarpMessage, HarpParseError

# Payload-type byte bit signalling that a timestamp precedes the payload.
_TIMESTAMP_FLAG = 0x10

# Offset of the payload-type byte within the frame header.
_PAYLOAD_TYPE_OFFSET = 4

# Offset of the length byte, which counts the bytes following itself.
_LENGTH_OFFSET = 1

# Smallest possible frame: 5 header bytes + 1 checksum byte, with no payload.
_MIN_FRAME_LEN = 6


class FrameError(ValueError):
    """A byte sequence could not be decoded as a Harp frame.

    Raised directly when more bytes cannot help: the buffer is longer than its
    length byte claims, or a header byte is invalid. A reader should resync.
    """


class TruncatedFrameError(FrameError):
    """The buffer is a prefix of a frame that is not fully arrived yet.

    The only retryable failure: a reader should keep accumulating rather than
    discard, because the remaining bytes complete this frame.
    """


class ChecksumError(FrameError):
    """A frame's trailing checksum byte did not match its contents.

    The frame is structurally complete, so the contents are corrupt. A reader
    should resync; waiting for more bytes cannot repair it.
    """


@dataclass(frozen=True)
class HarpFrame:
    """One decoded Harp message.

    ``message_type`` is Read=1, Write=2 or Event=3.

    ``payload_type`` is the payload-type byte with the timestamp flag (0x10)
    masked off, so it always describes the element type alone; whether the
    frame carried a timestamp is expressed by ``timestamp`` instead.

    ``payload`` excludes both the timestamp prefix and the checksum byte.
    """

    message_type: int
    address: int
    port: int
    payload_type: int
    payload: bytes
    timestamp: float | None


def decode(raw: bytes) -> HarpFrame:
    """Decode one complete Harp frame.

    Raises one of three errors, each implying a different remedy:

    * ``TruncatedFrameError`` -- the buffer is a prefix of this frame; wait for
      more bytes.
    * ``ChecksumError`` -- the frame is complete but corrupt; resync.
    * ``FrameError`` -- the buffer is over-long or a header byte is invalid;
      resync. (Base class of the other two, so it catches all three.)
    """
    # Size is established before the checksum so that a short read is never
    # reported as corruption. In a truncated frame the last byte is not the
    # checksum at all, so comparing against it would blame corruption for what
    # is really an incomplete buffer. Truncation stays detectable because
    # slicing bytes off the end cannot alter the length byte.
    #
    # These checks duplicate the parser's, which is deliberate: the parser
    # validates the checksum first and so describes almost every partial read
    # as a checksum mismatch. Owning the order here keeps both the exception
    # type and the message honest for a reader that is still accumulating.
    if len(raw) < _MIN_FRAME_LEN:
        raise TruncatedFrameError(f"Frame too short: {len(raw)} bytes (minimum {_MIN_FRAME_LEN})")

    # Under- and over-length are opposite faults. A short buffer is a frame
    # still arriving; a long one means the caller is misaligned or has
    # concatenated frames, which no amount of further reading will fix.
    expected_len = raw[_LENGTH_OFFSET] + 2
    if len(raw) < expected_len:
        raise TruncatedFrameError(
            f"Frame is {len(raw)} bytes but its length byte implies {expected_len}"
        )
    if len(raw) > expected_len:
        raise FrameError(f"Frame is {len(raw)} bytes but its length byte implies {expected_len}")

    # Corruption is by far the common failure on a serial line, so it gets its
    # own exception type -- a reader resyncs on this but retries on truncation.
    if (sum(raw[:-1]) & 0xFF) != raw[-1]:
        raise ChecksumError("Frame checksum does not match its contents")

    try:
        message = HarpMessage.parse(raw)
    except HarpParseError as exc:
        raise FrameError(str(exc)) from exc

    return HarpFrame(
        message_type=int(message.message_type),
        address=message.address,
        port=message.port,
        payload_type=raw[_PAYLOAD_TYPE_OFFSET] & ~_TIMESTAMP_FLAG,
        payload=bytes(message.payload),
        timestamp=message.timestamp,
    )
