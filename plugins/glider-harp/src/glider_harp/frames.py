"""Split and decode Harp binary frames.

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

# Largest possible frame: the length byte is a u8 and counts the bytes after
# itself, so no frame can exceed 255 + 2 bytes however corrupt it looks.
_MAX_FRAME_LEN = 257

# The only message types the wire format defines. Used to judge whether a byte
# could begin a frame at all, which is the cheapest way to tell a real short
# read from noise that merely claims to be a long frame.
_MESSAGE_TYPES = frozenset((1, 2, 3))  # Read, Write, Event


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

    It returns raw ``bytes``, not ``HarpFrame``: splitting and decoding are
    separate jobs, and the caller decodes what it gets back. Only frames that
    already decoded cleanly are returned, so nothing a caller receives will
    fail to decode a second time. What the splitter rejected is reported
    through counters instead, since a silently dropped byte is indistinguishable
    from a quiet device:

    * ``checksum_errors`` -- frames that arrived complete at a known boundary
      but were corrupt. This is the count that means "bad cable", as distinct
      from the framing noise below. Counted once per corrupt frame, not once
      per read, and only while framing is held: a burst of garbage that never
      resynchronises registers one error, because the splitter cannot know how
      many frames were in it.
    * ``resyncs`` -- times framing was lost and had to be recovered.
    * ``bytes_discarded`` -- bytes thrown away without ever forming a frame.

    Known limitation -- a stall the splitter cannot fix alone. Noise is only
    rejected here by its message-type byte, so garbage that happens to begin
    with 1, 2 or 3 and claims a long frame still parks the splitter in "waiting
    for the rest". Measured at ~1.1% of resyncs (3/256 for the message type,
    times 251/256 for a length byte claiming more than the minimum).

    That wait is bounded in bytes but not in time. At most 255 further bytes
    settle it, so a live stream always recovers -- across all 768 head
    combinations, none stalls once ample data follows. But if the device falls
    silent first, the frames behind it are held indefinitely: a head of
    ``(3, 255)`` ahead of 36 events emits none of them and holds 254 bytes
    until something more arrives, whereupon all 37 flush at once.

    Closing this properly would mean validating payload-type bits here, which
    duplicates the parser knowledge this module exists to isolate. The real
    mitigation belongs to the reader: flush on an idle timeout, since only the
    reader knows the device has gone quiet.

    Not thread-safe: one splitter belongs to one reader.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        # Whether the head of the buffer is believed to be a frame boundary.
        # It is not once a frame there fails to decode, and that distinction
        # decides whether an incomplete head means "wait" or "keep hunting".
        self._synced = True
        self.checksum_errors = 0
        self.resyncs = 0
        self.bytes_discarded = 0

    def feed(self, chunk: bytes) -> list[bytes]:
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
        # A list rather than a generator, and deliberately not a generator
        # function: making this ``yield`` would defer the buffer append above
        # until the caller iterated, so a caller that ignored the return value
        # would silently drop the chunk, and two feeds consumed out of order
        # would splice the stream backwards. Returning the built list keeps the
        # buffer update tied to the call, and leaves the reader a value it can
        # take len() of and test for emptiness.
        return frames

    def _take_frame(self) -> bytes | None:
        """Consume and return the next frame, or None if none is available."""
        candidate = self._candidate_at(0)
        if candidate is not None:
            try:
                decode(candidate)
            except ChecksumError:
                # Counted only at a boundary we had positive reason to trust,
                # which means offset 0 *and* still synced. The same error
                # raised while hunting through noise in ``_resync`` means "not
                # a frame start", not "corrupt frame", and counting those would
                # bury the real corruption signal.
                #
                # The ``_synced`` half is what makes this count once per
                # corrupt frame rather than once per read. A corrupt frame that
                # resync cannot yet see past stays at the head, so every later
                # feed re-decodes it -- and an idle poller calling ``feed(b"")``
                # would otherwise inflate the count without limit.
                if self._synced:
                    self.checksum_errors += 1
            except FrameError:
                pass
            else:
                del self._buffer[: len(candidate)]
                self._synced = True
                return candidate

        elif self._synced and self._head_may_still_complete():
            # An ordinary short read. Scanning forward here would be actively
            # wrong: a later frame could complete first and be emitted out of
            # order, discarding the frame already under way.
            return None

        if self._synced:
            self.resyncs += 1
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
                self.bytes_discarded += offset
                self._synced = True
                return frame
        self._discard_bytes_that_cannot_start_a_frame()
        return None

    def _frame_at(self, offset: int) -> bytes | None:
        """Return the complete valid frame starting at ``offset``, else None.

        Validation is a full ``decode``, not just a checksum test, and the
        caller decodes the result again -- about 20x the cost of a cheaper
        check. That is a deliberate trade: the checksum is a single wrapping
        byte sum, so roughly one in 256 random offsets passes it, and during a
        resync we try every offset in the buffer. Accepting those would emit
        phantom frames assembled from noise, which is far worse for an
        experiment record than the microseconds saved. Revisit only if
        profiling shows this on a hot path.
        """
        candidate = self._candidate_at(offset)
        if candidate is None:
            return None
        try:
            decode(candidate)
        except FrameError:
            return None
        return candidate

    def _candidate_at(self, offset: int) -> bytes | None:
        """The bytes a frame at ``offset`` would occupy, if that many are held.

        None means too few bytes to judge yet. A length byte claiming an
        impossible size needs no check of its own -- it yields a candidate
        under the minimum frame length, which ``decode`` rejects on its first
        statement, before reading any content.
        """
        available = len(self._buffer) - offset
        if available < _MIN_FRAME_LEN:
            return None
        size = self._buffer[offset + _LENGTH_OFFSET] + 2
        if available < size:
            return None
        return bytes(self._buffer[offset : offset + size])

    def _head_may_still_complete(self) -> bool:
        """Whether the head of the buffer could be a frame that is still arriving.

        The message-type byte is checked before the length byte is believed.
        Without that, any noise byte claiming a 257-byte frame would park the
        splitter in "waiting for the rest" and withhold every frame behind it
        until that many bytes happened to arrive -- or forever, if the device
        goes quiet at the end of a trial. Trusting a length byte at an
        unvalidated offset is what makes the wait unbounded.
        """
        if not self._buffer:
            return True
        if self._buffer[0] not in _MESSAGE_TYPES:
            return False
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
            dropped = len(self._buffer) - keep
            del self._buffer[:dropped]
            self.bytes_discarded += dropped
