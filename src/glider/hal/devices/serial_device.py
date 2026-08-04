"""Generic serial/UART device.

Talks to any serial peripheral (a serial scale, GPS, syringe pump, custom
firmware, or an OS-paired Bluetooth-Classic/SPP module) by port + line
settings, without needing a dedicated device class per gadget. One device
instance owns one ``serial.Serial`` connection.

Two usage modes, chosen by the ``stream`` setting:

- request/response (``stream=False``, default): ``write`` sends a command,
  ``read_line`` reads one framed reply, ``query`` does both. Each call does its
  own blocking I/O off the event loop via ``asyncio.to_thread``.
- streaming (``stream=True``): a background daemon reader thread continuously
  reads framed lines and caches the latest one, so ``read``/``get_state`` never
  block and the DataRecorder can poll a free-running sensor at its own interval
  (the same pattern HX711Device uses for its sampler).

Framing: lines are delimited by the configured ``terminator`` (default ``\n``);
reads use ``Serial.read_until`` so a custom terminator (e.g. ``\r``) is honored.
Cross-platform via pyserial (Windows COMx, macOS /dev/cu.*, Linux /dev/tty*).
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from glider.hal.base_device import BaseDevice, DeviceConfig

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard

logger = logging.getLogger(__name__)

# pyserial byte-size / parity / stop-bit values accepted directly by
# serial.Serial (ints for bytesize/stopbits, single chars for parity).
_VALID_BYTESIZE = (5, 6, 7, 8)
_VALID_PARITY = ("N", "E", "O", "M", "S")
_VALID_STOPBITS = (1, 2)

# A cached streamed sample older than this is not a reading (mirrors HX711).
MAX_SAMPLE_AGE_S = 5.0
# How long a streaming ``read`` waits for the first fresh line before giving up.
READ_WAIT_S = 1.0

# The streaming reader's per-read timeout is clamped to this window, independent
# of the user's ``timeout`` setting, so the reader always re-checks its stop
# event within MAX_READER_TIMEOUT_S (bounding shutdown latency and preventing a
# thread leak on a large user timeout) and never busy-loops on timeout=0.
MIN_READER_TIMEOUT_S = 0.05
MAX_READER_TIMEOUT_S = 1.0


class GenericSerialDevice(BaseDevice):
    """A serial/UART peripheral on one host serial port.

    Settings:
    - port: serial port path/name (e.g. "/dev/ttyUSB0", "COM3"). Required.
    - baudrate: bits per second (default 9600).
    - bytesize: data bits, one of 5/6/7/8 (default 8).
    - parity: "N"/"E"/"O"/"M"/"S" (default "N").
    - stopbits: 1 or 2 (default 1).
    - timeout: read timeout in seconds (default 1.0).
    - terminator: line delimiter for reads/writes (default "\\n").
    - encoding: text codec for commands and replies (default "utf-8").
    - stream: when True, run a background reader that caches the latest line
        for read/get_state (default False).
    """

    # Rendered by the hardware panel's schema form when adding/editing a device.
    SETTINGS_SCHEMA = [
        {
            "key": "port",
            "label": "Port",
            "type": "str",
            "default": "",
            "help": "Serial port, e.g. /dev/ttyUSB0 or COM3. Use Scan on the board to list ports.",
        },
        {
            "key": "baudrate",
            "label": "Baud rate",
            "type": "enum",
            "default": 9600,
            "choices": [
                [9600, "9600"],
                [19200, "19200"],
                [38400, "38400"],
                [57600, "57600"],
                [115200, "115200"],
                [230400, "230400"],
            ],
        },
        {
            "key": "bytesize",
            "label": "Data bits",
            "type": "enum",
            "default": 8,
            "choices": [[8, "8"], [7, "7"], [6, "6"], [5, "5"]],
        },
        {
            "key": "parity",
            "label": "Parity",
            "type": "enum",
            "default": "N",
            "choices": [["N", "None"], ["E", "Even"], ["O", "Odd"]],
        },
        {
            "key": "stopbits",
            "label": "Stop bits",
            "type": "enum",
            "default": 1,
            "choices": [[1, "1"], [2, "2"]],
        },
        {
            "key": "timeout",
            "label": "Read timeout (s)",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 60.0,
            "decimals": 2,
        },
        {
            "key": "terminator",
            "label": "Line terminator",
            "type": "enum",
            "default": "\n",
            "choices": [["\n", "LF (\\n)"], ["\r", "CR (\\r)"], ["\r\n", "CRLF (\\r\\n)"]],
        },
        {"key": "encoding", "label": "Encoding", "type": "str", "default": "utf-8"},
        {
            "key": "stream",
            "label": "Stream (background read)",
            "type": "bool",
            "default": False,
            "help": "Continuously read framed lines in the background so the recorder can log them.",
        },
    ]

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        parsed = self._parse_settings(config.settings)
        self._port = parsed["port"]
        self._baudrate = parsed["baudrate"]
        self._bytesize = parsed["bytesize"]
        self._parity = parsed["parity"]
        self._stopbits = parsed["stopbits"]
        self._timeout = parsed["timeout"]
        self._terminator = parsed["terminator"]
        self._encoding = parsed["encoding"]
        self._stream = parsed["stream"]

        self._serial: Any = None  # serial.Serial handle, opened in initialize()
        # Serializes direct-handle actions (write / non-streaming read_line) with
        # each other AND with shutdown()'s close, so the port is never closed out
        # from under an in-flight action (emergency stop bypasses the command
        # lock). The streaming reader thread is a separate concern -- it is
        # stopped and joined before close; it and write() may touch the handle
        # concurrently, which pyserial permits (one reader + one writer).
        self._port_lock = asyncio.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._sample_lock = threading.Lock()
        # Latest streamed line as (text, perf_counter_timestamp). NOT named
        # ``_state`` (DataRecorder reads a ``_state`` attribute in preference to
        # calling get_state()).
        self._latest: tuple[str, float] | None = None

    # --- settings ---

    @staticmethod
    def _parse_settings(settings: dict[str, Any]) -> dict[str, Any]:
        """Validate a settings dict; the single place these are interpreted."""
        port = str(settings.get("port", "")).strip()
        baudrate = int(settings.get("baudrate", 9600))
        if baudrate <= 0:
            raise ValueError(f"baudrate must be positive, got {baudrate}")
        bytesize = int(settings.get("bytesize", 8))
        if bytesize not in _VALID_BYTESIZE:
            raise ValueError(f"bytesize must be one of {_VALID_BYTESIZE}, got {bytesize}")
        parity = str(settings.get("parity", "N")).upper()
        if parity not in _VALID_PARITY:
            raise ValueError(f"parity must be one of {_VALID_PARITY}, got {parity!r}")
        stopbits = int(settings.get("stopbits", 1))
        if stopbits not in _VALID_STOPBITS:
            raise ValueError(f"stopbits must be one of {_VALID_STOPBITS}, got {stopbits}")
        timeout = float(settings.get("timeout", 1.0))
        if timeout < 0:
            raise ValueError(f"timeout must be >= 0, got {timeout}")
        terminator = str(settings.get("terminator", "\n")) or "\n"
        encoding = str(settings.get("encoding", "utf-8")) or "utf-8"
        stream = bool(settings.get("stream", False))
        return {
            "port": port,
            "baudrate": baudrate,
            "bytesize": bytesize,
            "parity": parity,
            "stopbits": stopbits,
            "timeout": timeout,
            "terminator": terminator,
            "encoding": encoding,
            "stream": stream,
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        """Adopt edited settings (validated first).

        Every serial setting affects the open connection or the framing/stream
        behavior, none of which can change safely under a live handle (e.g.
        flipping ``stream`` would desync the running reader thread from the
        mode flag). So while the device is initialized the edit is only recorded
        to ``config.settings`` (for the saved file) and takes effect on the next
        initialize(); the live caches stay consistent with the open port. Before
        init, the caches are refreshed immediately. Validation runs against the
        merged result first, so a rejected edit leaves the device untouched.
        """
        parsed = self._parse_settings({**self._config.settings, **settings})
        self._config.settings.update(settings)
        if self._initialized:
            logger.info("GenericSerial %s: settings saved; reconnect to apply", self._name)
            return
        self._port = parsed["port"]
        self._baudrate = parsed["baudrate"]
        self._bytesize = parsed["bytesize"]
        self._parity = parsed["parity"]
        self._stopbits = parsed["stopbits"]
        self._timeout = parsed["timeout"]
        self._terminator = parsed["terminator"]
        self._encoding = parsed["encoding"]
        self._stream = parsed["stream"]

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "GenericSerial"

    @property
    def required_pins(self) -> list[str]:
        # Serial is a transport; no GPIO pins are allocated.
        return []

    @property
    def port(self) -> str:
        """Configured serial port path/name."""
        return self._port

    @property
    def baudrate(self) -> int:
        """Configured baud rate."""
        return self._baudrate

    @property
    def is_streaming(self) -> bool:
        """Whether a background reader thread caches the latest line."""
        return self._stream

    @property
    def actions(self) -> dict[str, Callable]:
        return {
            "write": self.write,
            "read_line": self.read_line,
            "query": self.query,
            "read": self.read,
        }

    # --- lifecycle ---

    async def initialize(self) -> None:
        """Open the serial port (lazy-import pyserial in a worker thread)."""
        if not self._port:
            raise ValueError("GenericSerial: 'port' setting is required")

        def _open():
            try:
                import serial
            except ImportError as e:
                raise RuntimeError(
                    "pyserial not installed. Run: pip install pyserial " "(or reinstall GLIDER)."
                ) from e
            return serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=self._bytesize,
                parity=self._parity,
                stopbits=self._stopbits,
                timeout=self._timeout,
            )

        self._serial = await asyncio.to_thread(_open)

        if self._stream:
            # Bound the reader's blocking read so it observes the stop event
            # within MAX_READER_TIMEOUT_S regardless of the user's timeout (which
            # is meaningful for request/response reads but would, at 0, busy-loop
            # the reader, and at 60 would leak it past shutdown's join). The
            # streaming read_line() path uses the cache, not this handle timeout.
            reader_timeout = max(MIN_READER_TIMEOUT_S, min(self._timeout, MAX_READER_TIMEOUT_S))
            self._serial.timeout = reader_timeout
            stop_event = threading.Event()  # fresh event per (re)init, like HX711
            self._stop_event = stop_event
            self._latest = None
            self._thread = threading.Thread(
                target=self._reader_loop,
                args=(self._serial, stop_event),
                name=f"serial-reader-{self._name}",
                daemon=True,
            )
            self._thread.start()

        self._initialized = True
        logger.info(
            "GenericSerial initialized on %s @ %d (stream=%s)",
            self._port,
            self._baudrate,
            self._stream,
        )

    async def shutdown(self) -> None:
        """Stop the reader, wait out in-flight action I/O, then close the port.

        Order matters: clear ``_initialized`` so queued actions refuse to run;
        stop and join the reader thread so it stops touching the handle (its
        blocking read is bounded to <= MAX_READER_TIMEOUT_S, so the join
        reliably succeeds); then take the port lock so an in-flight
        ``write()``/``read_line()`` finishes before ``close()`` releases the fd.
        Wrapped so a failing close() still clears state and drops the cache.
        """
        self._initialized = False
        try:
            self._stop_event.set()
            thread, self._thread = self._thread, None
            if thread is not None:
                await asyncio.to_thread(self._join_reader, thread)
            async with self._port_lock:
                ser, self._serial = self._serial, None
                if ser is not None:
                    try:
                        await asyncio.to_thread(ser.close)
                    except Exception as e:  # close is best-effort
                        logger.warning("GenericSerial %s: error during close: %s", self._name, e)
        finally:
            self._initialized = False
            with self._sample_lock:
                self._latest = None

    def _join_reader(self, thread: threading.Thread) -> None:
        thread.join(timeout=2.0)
        if thread.is_alive():
            logger.error(
                "GenericSerial %s: reader thread did not exit within 2.0s; "
                "closing the port anyway",
                self._name,
            )

    # --- streaming reader (runs in the reader thread) ---

    def _reader_loop(self, ser: Any, stop_event: threading.Event) -> None:
        """Free-running line reader; caches the latest framed line.

        Paces itself on ``read_until`` (which returns on the terminator or the
        port timeout, so it re-checks the stop event at least every ``timeout``
        seconds). Once the stop event is set, serial errors are expected
        (shutdown may have closed the handle) and the loop exits quietly.
        """
        term = self._terminator.encode(self._encoding)
        while not stop_event.is_set():
            try:
                raw = ser.read_until(term)
            except Exception:
                if stop_event.is_set():
                    break  # expected: shutdown closed the handle
                logger.exception("GenericSerial %s: reader error", self._name)
                if stop_event.wait(0.5):
                    break
                continue
            if not raw:
                continue  # timed out with no data; loop re-checks the stop event
            if not raw.endswith(term):
                # read_until returned before the terminator (an inter-byte gap
                # exceeded the read timeout): a partial/truncated frame. Discard
                # it rather than caching a fabricated value.
                logger.debug("GenericSerial %s: discarded partial frame %r", self._name, raw)
                continue
            text = self._decode(raw)
            if text:
                with self._sample_lock:
                    self._latest = (text, time.perf_counter())

    def _decode(self, raw: bytes) -> str:
        """Decode bytes and strip the trailing terminator/whitespace."""
        text = raw.decode(self._encoding, errors="replace")
        return text.rstrip("\r\n").rstrip()

    def _fresh_sample(self) -> tuple[str, float] | None:
        with self._sample_lock:
            sample = self._latest
        if sample is None:
            return None
        if time.perf_counter() - sample[1] > MAX_SAMPLE_AGE_S:
            return None
        return sample

    async def get_state(self) -> str | None:
        """Latest streamed line for the DataRecorder poll (None if stale/absent).

        Non-blocking, so the recorder is never held up. Returns None on a
        non-streaming device (nothing is cached).
        """
        sample = self._fresh_sample()
        return sample[0] if sample is not None else None

    # --- actions ---

    def _require_open(self) -> Any:
        if not self._initialized or self._serial is None:
            raise RuntimeError(f"GenericSerial {self._name} is not initialized")
        return self._serial

    @staticmethod
    def _blocking_write(ser: Any, buf: bytes) -> None:
        ser.write(buf)
        ser.flush()

    async def write(self, *args: Any) -> None:
        """Send a command followed by the configured terminator.

        Variadic and comma-joins its args (like ``BLEDevice.write``), so the
        node layer's comma-split constant round-trips: ``write("SET", 1, 2)`` ->
        ``"SET,1,2"``. The terminator is appended only if not already present,
        so both ``"MEAS?"`` and ``"MEAS?\\n"`` frame correctly.
        """
        if not args or all(a is None for a in args):
            raise ValueError("write requires a value")
        payload = ",".join(str(a) for a in args if a is not None)
        if not payload.endswith(self._terminator):
            payload += self._terminator
        buf = payload.encode(self._encoding)
        async with self._port_lock:
            ser = self._require_open()  # re-checked under the lock (see shutdown)
            await asyncio.to_thread(self._blocking_write, ser, buf)

    async def read_line(self) -> str:
        """Read one framed line and return it decoded.

        On a streaming device the reader thread owns the port, so this returns
        the latest cached line. On a non-streaming device it reads directly and
        requires the terminator to arrive within the read timeout -- a partial
        read (terminator not seen) raises rather than returning a truncated,
        fabricated value.
        """
        if self._stream:
            sample = await self._await_fresh_sample()
            return sample[0]
        term = self._terminator.encode(self._encoding)
        async with self._port_lock:
            ser = self._require_open()  # re-checked under the lock (see shutdown)
            raw = await asyncio.to_thread(ser.read_until, term)
        if not raw.endswith(term):
            raise RuntimeError(
                f"GenericSerial {self._name}: incomplete read (no terminator "
                f"within {self._timeout:.2f}s)"
            )
        return self._decode(raw)

    async def query(self, *args: Any) -> str:
        """Write a command, then read one framed reply (request/response)."""
        await self.write(*args)
        return await self.read_line()

    async def read(self) -> str:
        """Primary read: the latest streamed line, or a direct read_line."""
        return await self.read_line()

    async def _await_fresh_sample(self) -> tuple[str, float]:
        """Wait up to READ_WAIT_S for a fresh streamed line; raise if none comes."""
        deadline = time.perf_counter() + READ_WAIT_S
        while True:
            sample = self._fresh_sample()
            if sample is not None:
                return sample
            if not self._initialized:
                raise RuntimeError(f"GenericSerial {self._name} is not initialized")
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    f"GenericSerial {self._name}: no line received within "
                    f"{READ_WAIT_S:.1f}s - is the device sending data?"
                )
            await asyncio.sleep(0.01)

    # --- serialization ---

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "GenericSerialDevice":
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance
