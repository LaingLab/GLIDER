import asyncio
import sys
import threading
import types
from unittest.mock import patch

import pytest

from glider.hal.base_board import BoardConnectionState, PinMode, PinType
from glider.hal.boards.telemetrix_board import TelemetrixBoard


@pytest.fixture
def mock_telemetrix():
    with patch("telemetrix_aio.telemetrix_aio.TelemetrixAIO") as mock:
        yield mock


@pytest.mark.asyncio
async def test_telemetrix_board_init():
    """Test TelemetrixBoard initialization."""
    board = TelemetrixBoard(port="COM3", board_type="uno")
    assert board.name == "Arduino Uno"
    assert board.board_type == "telemetrix"
    assert not board.is_connected


@pytest.mark.asyncio
async def test_telemetrix_board_capabilities():
    """Test board capabilities."""
    board = TelemetrixBoard()
    caps = board.capabilities
    assert caps.supports_analog
    assert 14 in caps.pins  # A0
    assert caps.pins[14].description == "A0"


@pytest.mark.asyncio
async def test_connect_and_disconnect_offload_thread_start_stop(monkeypatch):
    """TelemetrixThread.start()/stop() block for seconds; they must run off-loop."""
    import glider.hal.boards.telemetrix_board as tb

    fake_mod = types.ModuleType("telemetrix_aio")
    fake_mod.telemetrix_aio = types.SimpleNamespace(TelemetrixAIO=object)
    monkeypatch.setitem(sys.modules, "telemetrix_aio", fake_mod)

    threads = {}

    class FakeThread:
        def start(self, port, sleep_tune=0.0001):
            threads["start"] = threading.current_thread()

        def stop(self):
            threads["stop"] = threading.current_thread()

        @property
        def telemetrix(self):
            return object()

        @property
        def is_running(self):
            return True

    monkeypatch.setattr(tb, "TelemetrixThread", FakeThread)
    loop_thread = threading.current_thread()

    board = tb.TelemetrixBoard(port="COM3")
    assert await board.connect() is True
    assert threads["start"] is not loop_thread

    await board.disconnect()
    assert threads["stop"] is not loop_thread


@pytest.mark.asyncio
async def test_disconnect_waits_for_in_flight_connect(monkeypatch):
    """connect()/disconnect() now await mid-body (asyncio.to_thread), losing
    the implicit event-loop serialization they used to have. An overlapping
    disconnect() must not stop() a thread that connect() is mid-start() on,
    nor clobber the freshly started thread handle."""
    import glider.hal.boards.telemetrix_board as tb

    fake_mod = types.ModuleType("telemetrix_aio")
    fake_mod.telemetrix_aio = types.SimpleNamespace(TelemetrixAIO=object)
    monkeypatch.setitem(sys.modules, "telemetrix_aio", fake_mod)

    started = threading.Event()
    release = threading.Event()
    events = []

    class FakeThread:
        def start(self, port, sleep_tune=0.0001):
            events.append("start-begin")
            started.set()
            release.wait(timeout=5)
            events.append("start-end")

        def stop(self):
            events.append("stop")

        @property
        def telemetrix(self):
            return object()

        @property
        def is_running(self):
            return True

    monkeypatch.setattr(tb, "TelemetrixThread", FakeThread)
    board = tb.TelemetrixBoard(port="COM3")

    connect_task = asyncio.create_task(board.connect())
    await asyncio.to_thread(started.wait, 5)

    disconnect_task = asyncio.create_task(board.disconnect())
    await asyncio.sleep(0.05)
    # disconnect must be blocked on the connect lock, not interleaving.
    assert "stop" not in events

    release.set()
    assert await connect_task is True
    await disconnect_task

    assert events == ["start-begin", "start-end", "stop"]
    assert board._telemetrix_thread is None
    assert board.state == BoardConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_set_pin_mode_offloads_serial_call_to_thread():
    board = TelemetrixBoard(port="COM3")
    record = {"names": [], "threads": []}

    class FakeThread:
        @property
        def is_running(self):
            return True

        @property
        def telemetrix(self):
            return object()

        def call_method(self, name, *args, **kwargs):
            record["names"].append(name)
            record["threads"].append(threading.current_thread())

    board._telemetrix_thread = FakeThread()
    board._set_state(BoardConnectionState.CONNECTED)
    loop_thread = threading.current_thread()

    await board.set_pin_mode(3, PinMode.OUTPUT, PinType.DIGITAL)

    assert record["names"] == ["set_pin_mode_digital_output"]
    assert record["threads"][0] is not loop_thread


@pytest.mark.asyncio
async def test_dead_thread_triggers_auto_reconnect():
    """A mid-session worker-thread death must engage auto-reconnect."""
    board = TelemetrixBoard(port="COM3", auto_reconnect=True)

    class DeadThread:
        @property
        def is_running(self):
            return False

        @property
        def telemetrix(self):
            return None

    board._telemetrix_thread = DeadThread()
    board._state = BoardConnectionState.CONNECTED
    # Pre-drop cached pin state: stale once the thread dies (the Arduino
    # resets on the next serial open), so detection must clear it.
    board._pin_modes[5] = PinMode.OUTPUT
    board._pin_values[5] = True
    board._pwm_pins_forced_low.add(5)
    board._analog_map[14] = 0

    try:
        assert board.is_connected is False
        assert board._reconnect_task is not None
        assert board.state == BoardConnectionState.RECONNECTING
        assert board._pin_modes == {}
        assert board._pin_values == {}
        assert board._pwm_pins_forced_low == set()
        assert board._analog_map == {}
    finally:
        task = board._reconnect_task
        board.stop_reconnect()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_disconnect_clears_pin_state_caches():
    board = TelemetrixBoard(port="COM3")
    board._pin_modes[5] = PinMode.OUTPUT
    board._pin_values[5] = True
    board._pwm_pins_forced_low.add(5)
    board._analog_map[14] = 0

    await board.disconnect()

    assert board._pin_modes == {}
    assert board._pin_values == {}
    assert board._pwm_pins_forced_low == set()
    assert board._analog_map == {}
