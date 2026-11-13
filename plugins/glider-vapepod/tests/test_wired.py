"""Tests for VapePodUSBDevice: the pod over a USB cable.

A fake ``serial`` module stands in for pyserial. Its port answers every good
command line with a status line, the way the firmware does, and can be told to
say nothing (a port that is not a VapePod) or to slip in a note.
"""

import asyncio
import logging
import queue
import re
import sys
import time
import types
from pathlib import Path

import pytest

import glider_vapepod.wired as wired
from glider.hal.base_device import DeviceConfig
from glider_vapepod.device import (
    ACTION_ARGS_SCHEMA,
    MAX_ATOM_PERCENT,
    MAX_PUFF_MS,
    VapePodDevice,
)
from glider_vapepod.wired import VapePodUSBDevice

FIRMWARE_DIR = Path(__file__).parents[1] / "firmware"
SKETCHES = [
    FIRMWARE_DIR / "vapepod" / "vapepod.ino",
    FIRMWARE_DIR / "vapepod_esp32" / "vapepod_esp32.ino",
]


class FakePort:
    """A pyserial ``Serial`` that behaves like the pod's firmware."""

    def __init__(self, pod, **kwargs):
        self.pod = pod
        self.kwargs = kwargs
        self.timeout = kwargs.get("timeout")
        self.is_open = True
        self.written: list[bytes] = []
        self._replies: queue.Queue[bytes] = queue.Queue()
        self.pump = 0

    # writes
    def write(self, buf):
        if self.pod.get("write_error"):
            raise self.pod["write_error"]
        self.written.append(bytes(buf))
        line = bytes(buf).decode().strip()
        if self.pod.get("silent"):
            return len(buf)
        if line.startswith("pump,"):
            self.pump = int(line.split(",")[1])
        elif line == "off":
            self.pump = 0
        for note in self.pod.pop("notes", []):
            self._replies.put(f"# {note}\n".encode())
        self._replies.put(f"atom=0 pump={self.pump}\n".encode())
        return len(buf)

    def flush(self):
        pass

    # reads
    def read_until(self, term=b"\n"):
        try:
            return self._replies.get(timeout=min(self.timeout or 0.05, 0.05))
        except queue.Empty:
            return b""

    def reset_input_buffer(self):
        pass

    def cancel_read(self):
        pass

    def cancel_write(self):
        pass

    def close(self):
        self.is_open = False

    @property
    def lines(self) -> list[str]:
        return [b.decode() for b in self.written]


@pytest.fixture
def pod(monkeypatch):
    """A fake pyserial; returns a dict holding the ports it opened and faults."""
    state: dict = {"ports": [], "comports": []}

    def make_port(**kwargs):
        port = FakePort(state, **kwargs)
        state["ports"].append(port)
        state["port"] = port
        return port

    serial_mod = types.ModuleType("serial")
    serial_mod.Serial = make_port
    tools = types.ModuleType("serial.tools")
    list_ports = types.ModuleType("serial.tools.list_ports")
    list_ports.comports = lambda: state["comports"]
    tools.list_ports = list_ports
    serial_mod.tools = tools
    monkeypatch.setitem(sys.modules, "serial", serial_mod)
    monkeypatch.setitem(sys.modules, "serial.tools", tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", list_ports)
    monkeypatch.setattr(wired, "FIRST_STATUS_S", 0.5)
    return state


def _comport(device, vid=wired.XIAO_VID, pid=0x8045):
    return types.SimpleNamespace(device=device, vid=vid, pid=pid)


class _FakeBoard:
    id = "serial"


def _make(settings=None):
    config = DeviceConfig(pins={}, settings=dict(settings or {}))
    return VapePodUSBDevice(_FakeBoard(), config, name="pod")


@pytest.fixture
async def device(pod):
    dev = _make({"port": "/dev/ttyACM0"})
    await dev.initialize()
    yield dev
    await dev.shutdown()


def _commands(port) -> list[str]:
    """What was sent, heartbeats left out."""
    return [line for line in port.lines if line != "status\n"]


# --- identity -----------------------------------------------------------------


def test_device_type():
    assert _make().device_type == "VapePodUSB"


def test_same_actions_and_panel_arguments_as_bluetooth():
    usb = _make()
    assert set(usb.actions) == {"puff", "pump_on", "pump_off", "pump", "off", "read", "write"}
    assert VapePodUSBDevice.ACTION_ARGS_SCHEMA is ACTION_ARGS_SCHEMA
    assert VapePodDevice.ACTION_ARGS_SCHEMA is ACTION_ARGS_SCHEMA


def test_framing_is_forced_and_saved():
    dev = _make({"port": "COM5", "stream": False, "terminator": "\r"})
    assert dev.is_streaming
    assert dev._terminator == "\n"
    assert dev._config.settings["stream"] is True
    assert dev.baudrate == 115200


def test_requires_no_pins():
    assert _make().required_pins == []


# --- connecting ---------------------------------------------------------------


async def test_initialize_sets_off_and_reads_the_status(device, pod):
    assert _commands(pod["port"])[0] == "off\n"
    assert await device.read() == "atom=0 pump=0"


async def test_a_port_that_never_answers_is_refused_and_closed(pod):
    pod["silent"] = True
    dev = _make({"port": "/dev/ttyACM0"})
    with pytest.raises(ConnectionError, match="no status"):
        await dev.initialize()
    assert not pod["port"].is_open
    assert not dev.is_initialized
    assert dev._heartbeat is None


async def test_a_blank_port_finds_the_xiao(pod):
    pod["comports"] = [
        _comport("/dev/ttyAMA0", vid=None, pid=None),
        _comport("/dev/ttyACM1"),
    ]
    dev = _make()
    await dev.initialize()
    try:
        assert pod["port"].kwargs["port"] == "/dev/ttyACM1"
        # Still blank in the saved file, so a replug that renumbers it is found.
        assert dev._config.settings.get("port", "") == ""
    finally:
        await dev.shutdown()


async def test_a_blank_port_is_looked_up_again_on_each_connect(pod):
    pod["comports"] = [_comport("/dev/ttyACM0")]
    dev = _make()
    await dev.initialize()
    await dev.shutdown()
    pod["comports"] = [_comport("/dev/ttyACM1")]
    await dev.initialize()
    try:
        assert pod["port"].kwargs["port"] == "/dev/ttyACM1"
    finally:
        await dev.shutdown()


async def test_editing_the_port_switches_the_lookup_on_and_off(pod):
    dev = _make({"port": "/dev/ttyACM0"})
    dev.apply_settings({"port": ""})
    assert dev._find_port
    dev.apply_settings({"port": "COM7"})
    assert not dev._find_port and dev.port == "COM7"


async def test_an_explicit_port_skips_the_lookup(pod):
    pod["comports"] = [_comport("/dev/ttyACM9")]
    dev = _make({"port": "/dev/ttyACM0"})
    await dev.initialize()
    try:
        assert pod["port"].kwargs["port"] == "/dev/ttyACM0"
    finally:
        await dev.shutdown()


@pytest.mark.parametrize(
    "comports, message",
    [
        ([], "no XIAO"),
        ([_comport("/dev/ttyACM0"), _comport("/dev/ttyACM1")], "more than one"),
        ([_comport("/dev/ttyUSB0", vid=0x10C4, pid=0xEA60)], "no XIAO"),  # an ESP32's bridge
        ([_comport("/dev/ttyACM0", pid=0x0045)], "no XIAO"),  # the bootloader
    ],
)
async def test_the_lookup_needs_exactly_one_xiao(pod, comports, message):
    pod["comports"] = comports
    with pytest.raises(ConnectionError, match=message):
        await _make().initialize()


# --- commands -----------------------------------------------------------------


async def test_puff_sends_duration_and_power(device, pod):
    await device.puff(3000, 15)
    assert _commands(pod["port"])[-1] == "puff,3000,15\n"


async def test_puff_defaults_to_the_power_cap_and_takes_whole_floats(device, pod):
    await device.puff(2500.0)
    assert _commands(pod["port"])[-1] == f"puff,2500,{MAX_ATOM_PERCENT}\n"


@pytest.mark.parametrize("bad", [MAX_ATOM_PERCENT + 1, 100])
async def test_a_puff_above_the_power_cap_sends_nothing(device, pod, bad):
    before = list(_commands(pod["port"]))
    with pytest.raises(ValueError, match="power"):
        await device.puff(1000, bad)
    assert _commands(pod["port"]) == before


@pytest.mark.parametrize("bad", [0, -1, MAX_PUFF_MS + 1, 1.5, "long", None])
async def test_a_bad_puff_sends_nothing(device, pod, bad):
    before = list(_commands(pod["port"]))
    with pytest.raises(ValueError, match="duration_ms"):
        await device.puff(bad)
    assert _commands(pod["port"]) == before


@pytest.mark.parametrize(
    "call, expected",
    [
        (lambda d: d.pump(55), "pump,55\n"),
        (lambda d: d.pump_on(), "pump,100\n"),
        (lambda d: d.pump_off(), "pump,0\n"),
        (lambda d: d.off(), "off\n"),
    ],
)
async def test_pump_and_off_commands(device, pod, call, expected):
    await call(device)
    assert _commands(pod["port"])[-1] == expected


async def test_read_tracks_the_pods_status(device, pod):
    await device.pump(70)
    deadline = time.perf_counter() + 1.0
    while await device.read() != "atom=0 pump=70":
        assert time.perf_counter() < deadline
        await asyncio.sleep(0.01)


async def test_a_puff_with_the_cable_gone_raises(device, pod):
    pod["write_error"] = OSError("device disconnected")
    with pytest.raises(OSError):
        await device.puff(1000)


async def test_the_pods_notes_are_logged_not_read_as_status(device, pod, caplog):
    pod["notes"] = ["refused: puff,99999"]
    with caplog.at_level(logging.WARNING, logger="glider_vapepod.wired"):
        await device.pump(20)
        deadline = time.perf_counter() + 1.0
        while "refused" not in caplog.text:
            assert time.perf_counter() < deadline
            await asyncio.sleep(0.01)
    assert (await device.read()).startswith("atom=")


# --- heartbeat ----------------------------------------------------------------


async def test_the_heartbeat_keeps_going_without_any_commands(pod, monkeypatch):
    monkeypatch.setattr(wired, "HEARTBEAT_S", 0.02)
    dev = _make({"port": "/dev/ttyACM0"})
    await dev.initialize()
    try:
        await asyncio.sleep(0.2)
        assert pod["port"].lines.count("status\n") >= 3
    finally:
        await dev.shutdown()


async def test_the_heartbeat_outruns_the_firmwares_timeout():
    for sketch in SKETCHES:
        source = sketch.read_text(encoding="utf-8")
        timeout = re.search(r"HOST_TIMEOUT_MS = (\d+);", source)
        assert timeout is not None, sketch.name
        # Two beats may go missing before the pod gives up on the host.
        assert int(timeout.group(1)) / 1000 >= 3 * wired.HEARTBEAT_S, sketch.name


def test_the_firmware_status_line_is_what_the_driver_caches():
    for sketch in SKETCHES:
        source = sketch.read_text(encoding="utf-8")
        assert '"atom=%u pump=%u"' in source, sketch.name
        assert wired.STATUS_PREFIX == "atom="


async def test_a_failing_heartbeat_is_logged_once(pod, monkeypatch, caplog):
    monkeypatch.setattr(wired, "HEARTBEAT_S", 0.02)
    dev = _make({"port": "/dev/ttyACM0"})
    await dev.initialize()
    try:
        with caplog.at_level(logging.ERROR, logger="glider_vapepod.wired"):
            pod["write_error"] = OSError("gone")
            await asyncio.sleep(0.2)
        assert caplog.text.count("heartbeat failed") == 1
    finally:
        pod.pop("write_error")
        await dev.shutdown()


# --- shutdown -----------------------------------------------------------------


async def test_shutdown_sends_off_stops_the_heartbeat_and_closes(pod, monkeypatch):
    monkeypatch.setattr(wired, "HEARTBEAT_S", 0.02)
    dev = _make({"port": "/dev/ttyACM0"})
    await dev.initialize()
    await dev.pump(100)
    await dev.shutdown()
    port = pod["port"]
    assert _commands(port)[-1] == "off\n"
    assert not port.is_open
    sent = len(port.written)
    await asyncio.sleep(0.1)
    assert len(port.written) == sent  # nothing after close


async def test_shutdown_survives_a_dead_port(pod):
    dev = _make({"port": "/dev/ttyACM0"})
    await dev.initialize()
    pod["write_error"] = OSError("gone")
    await dev.shutdown()  # must not raise
    assert not pod["port"].is_open


async def test_status_changes_are_logged_once_each(device, pod, caplog):
    with caplog.at_level(logging.INFO, logger="glider_vapepod.wired"):
        await device.pump(40)
        await device.pump(40)  # same status again: not logged twice
        deadline = time.perf_counter() + 1.0
        while "pump=40" not in caplog.text:
            assert time.perf_counter() < deadline
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
    assert caplog.text.count("pod status: atom=0 pump=40") == 1
