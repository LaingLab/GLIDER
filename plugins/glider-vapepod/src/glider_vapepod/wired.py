"""VapePod over a USB cable instead of Bluetooth.

The same pod and the same commands as :class:`~glider_vapepod.device.VapePodDevice`,
sent as lines over the XIAO's USB serial port. Built on GLIDER's
:class:`~glider.hal.devices.serial_device.GenericSerialDevice`, so it goes on a
``serial`` board.

The wire, both ways one line each, ended by ``\\n``:

- to the pod: ``off``, ``pump,<percent>``, ``puff,<ms>,<percent>``, ``status``
- from the pod: ``atom=<percent> pump=<percent>`` on every change and once a
  second, or a note starting with ``# `` (a refusal, the boot banner, ...)

**The heartbeat.** The firmware turns everything off when a wired host goes
quiet for 3 s (or closes the port). This device sends ``status`` every
``HEARTBEAT_S`` from its own thread, so a busy GUI thread can't let the pump
drop out; a GLIDER that dies or is closed stops the heartbeat, and the pod
switches itself off.

**A pulled cable** is not reconnected here: the port is gone, so the next
command raises (a puff never goes missing silently) and the heartbeat logs it.
Reconnect the device from the Hardware panel once the cable is back.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from glider.hal.base_device import DeviceConfig
from glider.hal.devices.serial_device import GenericSerialDevice
from glider_vapepod.device import ACTION_ARGS_SCHEMA, MAX_PUFF_MS, whole_number

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard

logger = logging.getLogger(__name__)

# Seeed's USB IDs for the XIAO nRF52840 running a sketch (not its bootloader,
# which is 0x0045). Used to find the pod when no port is set.
XIAO_VID = 0x2886
XIAO_PIDS = (0x8045,)

# How often the heartbeat goes out. The firmware's HOST_TIMEOUT_MS is 3000, so
# two can go missing before the pod switches off.
HEARTBEAT_S = 1.0

# How long initialize() waits for the pod's first status line. An ESP32
# DevKitC resets when the port opens and takes about a second to boot.
FIRST_STATUS_S = 3.0

# The courtesy "off" on shutdown; see VapePodDevice.shutdown.
OFF_ON_SHUTDOWN_S = 0.75

STATUS_PREFIX = "atom="
NOTE_PREFIX = "#"


def find_pod_port() -> str:
    """The port of the one XIAO nRF52840 plugged in, found by its USB IDs.

    Raises if there is none, or more than one (then the port has to be set).
    """
    from serial.tools import list_ports

    ports = [p.device for p in list_ports.comports() if p.vid == XIAO_VID and p.pid in XIAO_PIDS]
    if len(ports) == 1:
        return ports[0]
    if not ports:
        raise ConnectionError(
            "VapePod USB: no XIAO nRF52840 found on USB. Plug it in, or set the port "
            "(an ESP32 pod always needs the port set)."
        )
    raise ConnectionError(
        f"VapePod USB: more than one XIAO nRF52840 on USB ({', '.join(ports)}); "
        "set the port of the one to use"
    )


class VapePodUSBDevice(GenericSerialDevice):
    """The side pod over USB: timed puffs, and its pump on or off.

    Settings:
    - port: serial port. Leave blank to find a XIAO nRF52840 by its USB IDs.
    """

    SETTINGS_SCHEMA = [
        {
            "key": "port",
            "label": "Port",
            "type": "str",
            "default": "",
            "help": (
                "Leave blank to find the XIAO by itself. Otherwise e.g. /dev/ttyACM0 "
                "(Pi), COM5 (Windows), or a stable /dev/serial/by-id/... path."
            ),
        },
    ]

    ACTION_ARGS_SCHEMA = ACTION_ARGS_SCHEMA

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        # The pod's framing, not the user's: lines, streamed, so the status the
        # pod pushes is always cached for read() and the recorder. USB CDC
        # ignores the baud rate; 115200 is what an ESP32's UART bridge needs.
        config.settings.setdefault("baudrate", 115200)
        config.settings["terminator"] = "\n"
        config.settings["stream"] = True
        config.settings.setdefault("timeout", 0.5)
        super().__init__(board, config, name)
        # Held across every write, from the heartbeat thread and from the event
        # loop alike, so two lines can never interleave on the wire.
        self._write_lock = threading.Lock()
        self._heartbeat: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()
        # A blank port means "find it", every time: a replugged XIAO can come
        # back as a different /dev/ttyACM*.
        self._find_port = not self._port

    def apply_settings(self, settings: dict[str, Any]) -> None:
        """Keep the pod's framing, and re-decide whether the port is looked up."""
        settings = {**settings, "terminator": "\n", "stream": True}
        super().apply_settings(settings)
        if not self._initialized:
            self._find_port = not self._port

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "VapePodUSB"

    @property
    def actions(self) -> dict[str, Callable]:
        """The pod's command set, as on Bluetooth. ``read`` returns its status."""
        return {
            "puff": self.puff,
            "pump_on": self.pump_on,
            "pump_off": self.pump_off,
            "pump": self.pump,
            "off": self.off,
            "read": self.read,
            "write": self.write,
        }

    # --- actions ---

    async def puff(self, duration_ms: Any, power: Any = 100) -> None:
        """Run the atomizer for ``duration_ms`` at ``power`` %, then stop.

        The pod times the puff itself, so this returns once the line is sent.
        """
        ms = whole_number(duration_ms, "duration_ms", 1, MAX_PUFF_MS)
        percent = whole_number(power, "power", 1, 100)
        await self.write("puff", ms, percent)

    async def pump(self, percent: Any) -> None:
        """Run the pump at ``percent`` % (0 is off) until told otherwise."""
        await self.write("pump", whole_number(percent, "percent", 0, 100))

    async def pump_on(self) -> None:
        """Run the pump at full speed."""
        await self.write("pump", 100)

    async def pump_off(self) -> None:
        """Stop the pump."""
        await self.write("pump", 0)

    async def off(self) -> None:
        """Turn the atomizer and the pump off."""
        await self.write("off")

    # --- wire ---

    def _blocking_write(self, ser: Any, buf: bytes) -> None:
        # An instance method here (static in the base, called through self), so
        # it can take the lock the heartbeat thread shares.
        with self._write_lock:
            ser.write(buf)
            ser.flush()

    def _decode(self, raw: bytes) -> str:
        """Pass status lines to the cache; log the pod's notes and drop them.

        Runs on the base class's reader thread, which caches whatever this
        returns and skips an empty string.
        """
        text = super()._decode(raw)
        if text.startswith(STATUS_PREFIX):
            return text
        if text.startswith(NOTE_PREFIX):
            message = text.lstrip("#").strip()
            if "refused" in message or "lost" in message or "dropped" in message:
                logger.warning("VapePod USB %s: pod says: %s", self._name, message)
            else:
                logger.info("VapePod USB %s: pod says: %s", self._name, message)
        elif text:
            logger.debug("VapePod USB %s: ignored line %r", self._name, text)
        return ""

    # --- heartbeat (its own thread) ---

    def _heartbeat_loop(self, ser: Any, stop: threading.Event) -> None:
        line = b"status\n"
        failing = False
        while not stop.wait(HEARTBEAT_S):
            try:
                self._blocking_write(ser, line)
            except Exception as e:  # noqa: BLE001 - keep beating; report once
                if stop.is_set():
                    break  # shutdown closed the port under us
                if not failing:
                    logger.error(
                        "VapePod USB %s: heartbeat failed (%s); the pod switches itself "
                        "off within 3 s. Reconnect it from the Hardware panel.",
                        self._name,
                        e,
                    )
                failing = True
            else:
                if failing:
                    logger.warning("VapePod USB %s: heartbeat going through again", self._name)
                failing = False

    # --- lifecycle ---

    async def initialize(self) -> None:
        """Open the port, set the pod to off, and wait for its status.

        A port that opens but never sends a status is not a VapePod (or runs
        firmware older than 0.3.0, which takes no commands over USB), and is
        closed again with an error rather than left looking connected.
        """
        if self._find_port:
            self._port = await asyncio.to_thread(find_pod_port)
            logger.info("VapePod USB %s: found the pod on %s", self._name, self._port)
        await super().initialize()
        try:
            stop = threading.Event()
            self._heartbeat_stop = stop
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                args=(self._serial, stop),
                name=f"vapepod-heartbeat-{self._name}",
                daemon=True,
            )
            self._heartbeat.start()
            # A known state, and the first heartbeat. An ESP32 that is still
            # booting misses it, but it starts from off anyway.
            await self.write("off")
            await self._await_first_status()
        except BaseException:
            await self.shutdown()
            raise

    async def _await_first_status(self) -> None:
        deadline = time.perf_counter() + FIRST_STATUS_S
        while self._fresh_sample() is None:
            if time.perf_counter() >= deadline:
                raise ConnectionError(
                    f"VapePod USB {self._name}: no status from {self._port} within "
                    f"{FIRST_STATUS_S:.0f} s. Is it a VapePod running firmware 0.3.0 or later?"
                )
            await asyncio.sleep(0.02)

    async def shutdown(self) -> None:
        """Stop the heartbeat, turn the pod off, then close the port.

        The pod would switch off on its own once the port closes (XIAO) or the
        heartbeat stops (ESP32, 3 s later); the ``off`` makes emergency stop,
        End Experiment and app quit stop it at once. Best-effort and bounded,
        like the Bluetooth driver's.
        """
        self._heartbeat_stop.set()
        thread, self._heartbeat = self._heartbeat, None
        if thread is not None:
            await asyncio.to_thread(thread.join, HEARTBEAT_S + 1.0)
        try:
            if self._initialized and self._serial is not None:
                try:
                    await asyncio.wait_for(self.write("off"), timeout=OFF_ON_SHUTDOWN_S)
                except Exception as e:  # noqa: BLE001 - teardown must not raise
                    logger.warning(
                        "VapePod USB %s: could not send 'off' before closing (%s); "
                        "it turns off when the port closes",
                        self._name,
                        e,
                    )
        finally:
            await super().shutdown()
