"""Split and decode Harp binary frames.

A thin adapter over ``harp.protocol``. That package is a pre-1.0 release
candidate whose API is not frozen -- 0.4.0 and 0.5.0rc1 are entirely different
-- so the rest of ``glider_harp`` depends on the names defined here rather than
on upstream's. An upstream change should touch this file and no other.

The wire format itself is stable:

    MessageType(1) | Length(1) | Address(1) | Port(1) | PayloadType(1)
        | [Seconds(U32) Micros(U16)] | Payload | Checksum(1)
"""

from collections.abc import Iterator
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

# Largest possible frame: the length byte is a u8 and counts the bytes after
# itself, so no frame can exceed 255 + 2 bytes however corrupt it looks.
_MAX_FRAME_LEN = 257


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


class FrameSplitter:
    """Cut a serial byte stream into complete raw frames.

    A read from a serial port returns whatever bytes happened to be buffered:
    half a frame, six frames, or a frame preceded by noise from a device that
    was already mid-transmission when the port opened. ``feed`` absorbs any of
    those and yields only whole frames, holding the remainder for next time.

    It yields raw ``bytes``, not ``HarpFrame``. Splitting and decoding are
    separate jobs, and a caller that decodes what it receives here can tell a
    corrupt frame (``ChecksumError``) apart from a framing failure, which it
    could not do if this class silently dropped both.

    Not thread-safe: one splitter belongs to one reader.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        # Whether the head of the buffer is believed to be a frame boundary.
        # It is not once a frame there fails to decode, and that distinction
        # decides whether an incomplete head means "wait" or "keep hunting".
        self._synced = True

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        """Absorb one read and return every frame it completed.

        Frames come out in stream order. Bytes that do not yet form a frame are
        retained, so a frame split across any number of reads still arrives.
        """
        self._buffer += chunk
        frames: list[bytes] = []
        while True:
            frame = self._take_frame()
            if frame is None:
                break
            frames.append(frame)
        # Deliberately not a generator function. Making this ``yield`` would
        # defer the buffer append above until the caller iterated, so a caller
        # that ignored the return value would silently drop the chunk, and two
        # feeds consumed out of order would splice the stream out of order.
        # Returning an iterator over an already-built list keeps the buffer
        # update tied to the call rather than to the consumption.
        return iter(frames)

    def _take_frame(self) -> bytes | None:
        """Consume and return the next frame, or None if none is available."""
        frame = self._frame_at(0)
        if frame is not None:
            del self._buffer[: len(frame)]
            self._synced = True
            return frame

        if self._synced and self._head_may_still_complete():
            # An ordinary short read. Scanning forward here would be actively
            # wrong: a later frame could complete first and be emitted out of
            # order, discarding the frame already under way.
            return None

        self._synced = False
        return self._resync()

    def _resync(self) -> bytes | None:
        """Hunt for the next frame boundary after a failure at the head.

        The head is known not to start a frame, so every subsequent offset is a
        candidate. Offsets are tried in order, which keeps output in stream
        order and drops as few bytes as possible.
        """
        for offset in range(1, len(self._buffer)):
            frame = self._frame_at(offset)
            if frame is not None:
                del self._buffer[: offset + len(frame)]
                self._synced = True
                return frame
        self._discard_bytes_that_cannot_start_a_frame()
        return None

    def _frame_at(self, offset: int) -> bytes | None:
        """Return the complete valid frame starting at ``offset``, else None.

        None covers both ways an offset can fail to be a frame start: too few
        bytes held to judge yet, and a complete candidate that decoding
        rejects. A length byte claiming an impossible size needs no check of
        its own -- it yields a candidate under the minimum frame length, which
        ``decode`` rejects on its first statement, before reading any content.
        """
        available = len(self._buffer) - offset
        if available < _MIN_FRAME_LEN:
            return None
        size = self._buffer[offset + _LENGTH_OFFSET] + 2
        if available < size:
            return None
        candidate = bytes(self._buffer[offset : offset + size])
        try:
            decode(candidate)
        except FrameError:
            return None
        return candidate

    def _head_may_still_complete(self) -> bool:
        """Whether the head of the buffer could be a frame that is still arriving."""
        if len(self._buffer) < _MIN_FRAME_LEN:
            return True
        size = self._buffer[_LENGTH_OFFSET] + 2
        return size >= _MIN_FRAME_LEN and len(self._buffer) < size

    def _discard_bytes_that_cannot_start_a_frame(self) -> None:
        """Bound the buffer after a resync that found nothing.

        A corrupt length byte claiming 257 bytes would otherwise have the
        splitter accumulate the stream forever waiting for a frame that does
        not exist. Any offset with a whole maximum frame behind it has been
        tested conclusively and rejected, so only the trailing bytes -- the
        ones a real frame could still be growing into -- are worth keeping.
        """
        keep = _MAX_FRAME_LEN - 1
        if len(self._buffer) > keep:
            del self._buffer[: len(self._buffer) - keep]
