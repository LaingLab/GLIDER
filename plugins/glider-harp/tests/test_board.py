"""``HarpBoard`` -- the transport the hardware panel attaches Harp devices to.

A transport board has no GPIO and opens no ports, so there is very little of it;
what there is exists to fail *early and legibly*. Two things are worth pinning:

* Pin operations must raise rather than quietly do nothing. A board that
  accepts ``write_digital`` and returns is a flow graph that runs green and
  drives no hardware.
* ``connect()`` must reject a mis-resolved Harp stack. ``harp`` depends on
  ``harp-protocol`` with no lower bound, so a plain install can pair a 0.5.x
  ``harp`` with an incompatible 0.4.0 ``harp-protocol`` and report success --
  ``harp.protocol`` still imports, and the damage only shows up much later as a
  missing name deep inside a register build. Both halves of that are tested:
  the module missing outright, and the module present but wrong.

``serial.tools.list_ports`` is mocked throughout, so nothing here depends on
what happens to be plugged into the machine running the suite.
"""

from __future__ import annotations

import sys
import types

import pytest

from glider.hal.base_board import BoardConnectionState
from glider_harp.board import HarpBoard


class _FakePort:
    def __init__(self, device, description):
        self.device = device
        self.description = description


@pytest.fixture
def fake_ports(monkeypatch):
    """Replace the host's port enumeration with a fixed list."""
    from serial.tools import list_ports

    ports: list[_FakePort] = []
    monkeypatch.setattr(list_ports, "comports", lambda: list(ports))
    return ports


# --- identity ------------------------------------------------------------


def test_board_identity():
    board = HarpBoard()
    assert board.board_type == "harp"
    assert board.name == "Harp"


def test_capabilities_declare_no_pins():
    """The GUI filters pin dropdowns off this; a non-empty map would offer pins
    that do not exist."""
    board = HarpBoard()
    assert board.capabilities.pins == {}
    assert board.capabilities.name == "Harp"


# --- connect / disconnect ------------------------------------------------


async def test_connect_marks_the_transport_ready():
    board = HarpBoard()
    assert await board.connect() is True
    assert board.is_connected
    assert board.state is BoardConnectionState.CONNECTED


async def test_disconnect_returns_the_board_to_disconnected():
    board = HarpBoard()
    await board.connect()
    assert board.is_connected

    await board.disconnect()
    assert not board.is_connected
    assert board.state is BoardConnectionState.DISCONNECTED


async def test_connect_fails_when_harp_protocol_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "harp.protocol", None)

    board = HarpBoard()
    with pytest.raises(RuntimeError) as excinfo:
        await board.connect()

    message = str(excinfo.value)
    assert "harp-protocol" in message
    assert not board.is_connected
    assert board.state is BoardConnectionState.ERROR


async def test_connect_fails_when_harp_protocol_is_the_wrong_version(monkeypatch):
    """The failure mode a version check on the module alone would miss.

    0.4.0 imports as ``harp.protocol`` perfectly happily; it just does not
    define the names the 0.5.x codec is written against.
    """
    stub = types.ModuleType("harp.protocol")  # no HarpMessage
    monkeypatch.setitem(sys.modules, "harp.protocol", stub)

    board = HarpBoard()
    with pytest.raises(RuntimeError) as excinfo:
        await board.connect()

    message = str(excinfo.value)
    assert "HarpMessage" in message, "the message must name what was missing"
    assert "harp-protocol" in message
    assert board.state is BoardConnectionState.ERROR


async def test_connect_fails_when_pyserial_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)

    board = HarpBoard()
    with pytest.raises(RuntimeError) as excinfo:
        await board.connect()

    assert "pyserial" in str(excinfo.value)
    assert board.state is BoardConnectionState.ERROR


# --- scan ----------------------------------------------------------------


async def test_scan_returns_description_port_pairs(fake_ports):
    """Label first, identifier second -- the order ``SerialBoard`` and
    ``BLEBoard`` use, and the one the hardware panel unpacks."""
    fake_ports.append(_FakePort("COM3", "Harp Behavior (FTDI)"))
    fake_ports.append(_FakePort("/dev/ttyUSB0", "USB Serial"))

    results = await HarpBoard.scan()

    assert results == [
        ("Harp Behavior (FTDI)", "COM3"),
        ("USB Serial", "/dev/ttyUSB0"),
    ]


async def test_scan_agrees_with_serial_board_on_which_half_is_the_identifier(fake_ports):
    """Drift insurance, not coverage: the order mutant is already killed above.

    This pins one thing only -- that the identifier is in the same position as
    ``SerialBoard`` puts it -- because that is the half whose reversal is
    silent. Asserting full equality would also pin the fallback rule, the
    ``strip()``, the absence of filtering and the tuple arity, and core has no
    test for ``SerialBoard.scan()`` at all. A legitimate core change (say,
    disambiguating duplicate descriptions) would then break nothing in core and
    surface here as a list-inequality in a plugin, pointing at neither the
    change nor the convention.
    """
    from glider.hal.boards.serial_board import SerialBoard

    fake_ports.append(_FakePort("COM3", "Harp Behavior (FTDI)"))

    assert [ident for _, ident in await HarpBoard.scan()] == [
        ident for _, ident in await SerialBoard.scan()
    ]


async def test_scan_falls_back_to_the_port_when_there_is_no_description(fake_ports):
    """An empty label in a combo box is a row the operator cannot pick."""
    fake_ports.append(_FakePort("COM7", ""))
    fake_ports.append(_FakePort("COM8", None))

    assert await HarpBoard.scan() == [("COM7", "COM7"), ("COM8", "COM8")]


async def test_scan_returns_an_empty_list_when_nothing_is_attached(fake_ports):
    assert await HarpBoard.scan() == []


async def test_scan_does_not_filter_out_ports_that_do_not_look_like_harp(fake_ports):
    """Harp boards enumerate as generic FTDI/CDC adapters; filtering on the
    label would hide real hardware with no way to get it back."""
    fake_ports.append(_FakePort("COM4", "USB Serial Device"))

    assert await HarpBoard.scan() == [("USB Serial Device", "COM4")]


async def test_scan_surfaces_an_enumeration_failure(monkeypatch):
    """Swallowing this would report "no devices found" for a broken driver."""
    from serial.tools import list_ports

    def _boom():
        raise OSError("the serial subsystem is unavailable")

    monkeypatch.setattr(list_ports, "comports", _boom)

    with pytest.raises(OSError, match="serial subsystem"):
        await HarpBoard.scan()


# --- pin operations ------------------------------------------------------

PIN_OPERATIONS = [
    ("set_pin_mode", lambda b: b.set_pin_mode(1, None)),
    ("write_digital", lambda b: b.write_digital(1, True)),
    ("read_digital", lambda b: b.read_digital(1)),
    ("write_analog", lambda b: b.write_analog(1, 128)),
    ("read_analog", lambda b: b.read_analog(1)),
]


@pytest.mark.parametrize("op,call", PIN_OPERATIONS, ids=[name for name, _ in PIN_OPERATIONS])
async def test_every_pin_operation_raises(op, call):
    board = HarpBoard()
    with pytest.raises(NotImplementedError) as excinfo:
        await call(board)

    message = str(excinfo.value)
    assert "Harp" in message and "no GPIO" in message
    assert op in message, "the message must name the operation that was attempted"


@pytest.mark.parametrize("op,call", PIN_OPERATIONS, ids=[name for name, _ in PIN_OPERATIONS])
async def test_pin_operations_raise_even_once_connected(op, call):
    """A connected transport is still a transport: connecting must not make a
    pin write look available."""
    board = HarpBoard()
    await board.connect()
    with pytest.raises(NotImplementedError):
        await call(board)


async def test_the_generic_pin_dispatchers_raise_too():
    """``write_pin``/``read_pin`` are BaseBoard's own entry points, and the node
    layer uses them rather than the typed methods."""
    from glider.hal.base_board import PinType

    board = HarpBoard()
    with pytest.raises(NotImplementedError):
        await board.write_pin(1, PinType.DIGITAL, True)
    with pytest.raises(NotImplementedError):
        await board.read_pin(1, PinType.ANALOG)


# --- reporting a device's broken link -----------------------------------


def test_a_transport_failure_shows_as_an_error_state_and_reaches_listeners():
    """Harp devices own their own ports, so the board never touches the thing
    that fails and cannot notice a pulled cable itself.

    It is nonetheless where one has to surface: the board's state is what the
    hardware panel shows, and its error callbacks are what ``HardwareManager``
    wires its own listeners to. Without this a device whose reader thread died
    has nowhere at all to say so.
    """
    board = HarpBoard()
    seen: list[Exception] = []
    board.register_error_callback(seen.append)

    failure = OSError("the device has been disconnected")
    board.report_transport_failure(failure)

    # ERROR, not DISCONNECTED: the transport did not shut down, it broke, and
    # those want different responses from whoever is watching.
    assert board.state is BoardConnectionState.ERROR
    assert seen == [failure]
