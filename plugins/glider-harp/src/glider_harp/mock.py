"""A Harp device with a fake port, for tests and for a bench with no hardware.

The substitution is at the **serial handle**, not at ``harp.device.ITransport``,
and the difference decides how much of the real stack a test exercises.

``ITransport`` is a ``@runtime_checkable`` Protocol and a fake one is about
fifteen lines, so it is the tempting seam. But nothing in ``glider_harp`` is
built on ``harp.device``: the read path is ``HarpReader`` -> ``FrameSplitter``
-> ``decode`` -> ``RegisterCache``, and the write path is ``encode`` straight
onto a handle. Faking ``ITransport`` would substitute a layer this package does
not use, and would need a *second* fake for the register round-trips, which are
plain writes and reads on the same handle rather than transport operations.

The handle, by contrast, is the one thing ``HarpDevice`` gets from the OS.
``HarpReader`` documents its whole requirement as ``read()`` plus a settable
``timeout`` (and an optional ``in_waiting``), which is a smaller surface than
``ITransport``'s four methods. Substituting it leaves every line of framing,
decoding, cache and thread lifecycle running exactly as it does against real
hardware -- only the bytes are ours. One fake, and it drives both halves.

What the fake does:

* answers ``WhoAmI`` and ``OperationControl`` reads, and remembers what
  ``OperationControl`` was written -- so "did initialize() actually take the
  device out of Standby" is a question a test can ask;
* replays a supplied list of event frames once ``start_replay`` is called, one
  per read, and then blocks like an idle line;
* signals when the replay has been **consumed**, not merely handed over. See
  ``_top_up``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any

from glider.hal.base_device import DeviceConfig
from glider_harp.device import (
    OPERATION_CONTROL,
    WHO_AM_I,
    HarpDevice,
    _as_who_am_i,
)
from glider_harp.frames import (
    MESSAGE_READ,
    MESSAGE_WRITE,
    FrameSplitter,
    decode,
    encode,
)

logger = logging.getLogger(__name__)

# Payload-type name by the length of the payload, for building a reply to an
# arbitrary register read. Only the unsigned widths, which is what a fake is
# ever asked for.
_TYPE_BY_LENGTH = {1: "U8", 2: "U16", 4: "U32", 8: "U64"}

# Wire payload-type code -> name, derived by asking the codec rather than by
# writing the table out: a second copy of it here could disagree with the one
# inside ``harp.protocol``, and an echo built with the wrong type is exactly
# the sort of fake-only bug that makes a suite agree with itself.
_TYPE_BY_CODE = {
    decode(encode(MESSAGE_READ, 0, name)).payload_type: name for name in _TYPE_BY_LENGTH.values()
}

# What a device that has never been configured tends to hold in
# OperationControl: heartbeat and operation LED on, mode bits clear -- i.e.
# Standby. Non-zero on purpose, so a mode change that clobbered the rest of the
# register instead of preserving it is visible.
DEFAULT_OPERATION_CONTROL = 0xC0

# How long one idle read blocks. Short enough that a test does not wait on it,
# long enough not to spin a core.
_IDLE_READ_S = 0.01


class FakeHarpPort:
    """A ``serial.Serial`` stand-in that speaks just enough Harp.

    Thread-safe: the reader thread reads while the event loop writes, which is
    the arrangement the real handle is used in.
    """

    def __init__(
        self,
        frames: Iterable[bytes] = (),
        who_am_i: int | None = None,
        operation_control: int = DEFAULT_OPERATION_CONTROL,
        registers: Mapping[int, bytes] | None = None,
    ) -> None:
        # ``who_am_i`` of None means "answer with whatever the schema declares";
        # MockHarpDevice fills it in when the port is opened.
        self.who_am_i = who_am_i
        self.operation_control = operation_control
        self.timeout: Any = None
        self.closed = False
        # Every frame the device was sent, in order, for tests that care what
        # was written rather than what came back.
        self.writes: list[bytes] = []
        # Extra read replies, as address -> payload bytes.
        self.registers: dict[int, bytes] = dict(registers or {})

        self._frames = [bytes(frame) for frame in frames]
        self._pending = bytearray()
        self._requests = FrameSplitter()
        self._lock = threading.Lock()
        self._replaying = False
        self._replayed = threading.Event()

    # --- the handle surface HarpReader and HarpDevice actually use ---

    @property
    def in_waiting(self) -> int:
        with self._lock:
            self._top_up()
            return len(self._pending)

    def read(self, size: int = 1) -> bytes:
        with self._lock:
            self._top_up()
            if self._pending:
                chunk = bytes(self._pending[:size])
                del self._pending[:size]
                return chunk
        # Idle: block a little and come back empty, like a real port at its
        # read timeout. Outside the lock, so a writer is never held up by a
        # reader waiting on a silent line.
        timeout = self.timeout
        time.sleep(_IDLE_READ_S if timeout is None else min(float(timeout), _IDLE_READ_S))
        return b""

    def write(self, data: bytes) -> int:
        data = bytes(data)
        with self._lock:
            if self.closed:
                raise OSError("FakeHarpPort: the port is closed")
            self.writes.append(data)
            # Fed through the real splitter so a request split across writes
            # is reassembled the way the device would.
            for raw in self._requests.feed(data):
                self._handle_request(raw)
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        with self._lock:
            self.closed = True

    # --- replay ---

    def start_replay(self) -> None:
        """Begin handing out the event frames.

        Explicit, rather than starting the moment ``OperationControl`` goes
        Active, because between that write and the reader thread there is still
        a confirming read: events queued in that window would be swallowed by
        the round-trip's own splitter and never reach the cache. On real
        hardware they would be, too -- a handful of events at the instant of
        connection is a real and accepted loss -- but a test that wants to
        count what it replayed should not have to model it.
        """
        with self._lock:
            self._replaying = True
            if not self._frames:
                self._replayed.set()

    def wait_for_replay(self, timeout: float = 2.0) -> bool:
        """Block until every replayed frame has reached the cache.

        A ``threading.Event`` wait, not a sleep and not an ``await``: ``harp``
        is threaded, so the thing being waited for happens on the reader
        thread, and the only sound way to observe it from the event loop is a
        synchronisation primitive that both sides agree on. Returns whether the
        replay completed within ``timeout``.
        """
        return self._replayed.wait(timeout)

    def _top_up(self) -> None:
        """Queue the next replayed frame. Called with the lock held.

        This is also where the replay is declared finished, and the placement
        is the whole reason it can be trusted. The reader loop is: read a
        chunk, feed it to the splitter, ingest every frame that came out, go
        round again. So a *later* call for more bytes can only happen once the
        previous chunk has been ingested -- which makes "the reader asked again
        and there is nothing left" a genuine happens-before for "the last frame
        is in the cache". Setting the event when the last frame was handed
        *over* would signal a frame the reader had not looked at yet, and a
        test asserting on counts would fail once in a while on a loaded
        machine.

        **Precondition, and it is the whole argument's foundation: while a
        replay is running, only the reader thread may call ``read`` or
        ``in_waiting``.** The happens-before is inherited from *that* loop's
        shape, so it says nothing about any other caller. A second thread
        draining the handle satisfies the event without any frame reaching the
        cache at all -- ``wait_for_replay()`` then returns ``True`` beside a
        count of zero, which reads as a device that recorded nothing rather
        than as a test that broke its own fixture. Nothing in this package does
        that (``HarpDevice`` does every round-trip before ``start()`` or after
        ``stop()``, for its own reasons), so this is a rule for whoever adds
        the next thing, not a defect.
        """
        if self._pending or not self._replaying:
            return
        if self._frames:
            self._pending += self._frames.pop(0)
        else:
            self._replayed.set()

    # --- the device side ---

    def _handle_request(self, raw: bytes) -> None:
        """Answer one frame the host sent. Called with the lock held."""
        frame = decode(raw)
        if frame.message_type == MESSAGE_WRITE:
            if frame.address == OPERATION_CONTROL and frame.payload:
                self.operation_control = frame.payload[0]
            else:
                self.registers[frame.address] = frame.payload
            # Echoed like a real device, though nothing in HarpDevice waits for
            # it -- a write during a recording must not read the port at all.
            type_name = _TYPE_BY_CODE.get(frame.payload_type, "U8")
            self._pending += encode(MESSAGE_WRITE, frame.address, type_name, frame.payload)
            return
        if frame.message_type == MESSAGE_READ:
            reply = self._reply_for(frame.address)
            if reply is not None:
                self._pending += reply

    def _reply_for(self, address: int) -> bytes | None:
        """The Read reply for one address, or None for a register we do not have."""
        if address == WHO_AM_I:
            return encode(
                MESSAGE_READ, WHO_AM_I, "U16", int(self.who_am_i or 0).to_bytes(2, "little")
            )
        if address == OPERATION_CONTROL:
            return encode(MESSAGE_READ, OPERATION_CONTROL, "U8", bytes([self.operation_control]))
        payload = self.registers.get(address)
        if payload is None:
            return None
        type_name = _TYPE_BY_LENGTH.get(len(payload))
        if type_name is None:
            return None
        return encode(MESSAGE_READ, address, type_name, payload)


class MockHarpDevice(HarpDevice):
    """A ``HarpDevice`` wired to a ``FakeHarpPort``.

    Everything above the handle is the real thing: the same lifecycle, the same
    round-trips, the same reader thread and register cache. Only ``_open_port``
    differs.
    """

    def __init__(
        self,
        board: Any,
        config: DeviceConfig,
        name: str | None = None,
        *,
        frames: Iterable[bytes] = (),
        who_am_i: int | None = None,
        operation_control: int = DEFAULT_OPERATION_CONTROL,
        registers: Mapping[int, bytes] | None = None,
    ) -> None:
        super().__init__(board, config, name)
        self.port_handle = FakeHarpPort(
            frames=frames,
            who_am_i=who_am_i,
            operation_control=operation_control,
            registers=registers,
        )

    @property
    def device_type(self) -> str:
        return "MockHarp"

    def _open_port(self) -> Any:
        # Called after the schema is loaded, so a fake left to answer "whatever
        # the schema says" can be filled in here rather than in every test.
        handle = self.port_handle
        if handle.who_am_i is None:
            handle.who_am_i = _as_who_am_i(self._schema.get("whoAmI")) or 0
        handle.closed = False
        return handle

    async def initialize(self) -> None:
        await super().initialize()
        # Only once the round-trips are done and the reader owns the port; see
        # FakeHarpPort.start_replay.
        self.port_handle.start_replay()

    @property
    def operation_control(self) -> int:
        """What the fake device currently holds in register 10."""
        return self.port_handle.operation_control

    def wait_for_replay(self, timeout: float = 2.0) -> bool:
        """Block until the replayed events have reached the cache."""
        return self.port_handle.wait_for_replay(timeout)
