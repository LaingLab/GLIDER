"""A device's link state, separate from whether it has been set up.

``_initialized`` was doing two jobs — "has been configured" and "is
reachable" — and the second was a lie the moment a link dropped. These cover
the default derivation: a pin-based device has no link of its own and is
exactly as connected as its board.
"""

from glider.hal.base_board import BoardConnectionState, ConnectionState
from glider.hal.base_device import BaseDevice, DeviceConfig


class _FakeBoard:
    def __init__(self, connected=True):
        self.id = "fake_board"
        self.is_connected = connected


class _PinDevice(BaseDevice):
    """A device with no link of its own — the default case."""

    @property
    def device_type(self):
        return "PinThing"

    @property
    def actions(self):
        return {}

    async def initialize(self):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False

    @classmethod
    def from_dict(cls, data, board):
        return cls(board, DeviceConfig())


def _device(connected=True):
    return _PinDevice(_FakeBoard(connected), DeviceConfig())


def test_connection_state_is_the_board_enum():
    """One vocabulary, not two: the strip's mapping already speaks it."""
    assert ConnectionState is BoardConnectionState


def test_pin_device_owns_no_link():
    assert _device().owns_link is False


def test_uninitialized_device_is_disconnected():
    assert _device().link_state is ConnectionState.DISCONNECTED


async def test_initialized_device_on_connected_board_is_connected():
    device = _device(connected=True)
    await device.initialize()
    assert device.link_state is ConnectionState.CONNECTED


async def test_initialized_device_on_dead_board_is_disconnected():
    device = _device(connected=True)
    await device.initialize()
    device.board.is_connected = False
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_shutdown_returns_to_disconnected():
    device = _device()
    await device.initialize()
    await device.shutdown()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_poll_link_is_a_no_op_by_default():
    """The supervisor calls this on every device; the default must be cheap."""
    device = _device()
    assert await device.poll_link() is None


def test_link_state_callback_fires_with_the_device():
    device = _device()
    seen = []
    device.set_link_state_callback(seen.append)
    device._notify_link_state()
    assert seen == [device]


def test_link_state_callback_can_be_cleared():
    device = _device()
    seen = []
    device.set_link_state_callback(seen.append)
    device.set_link_state_callback(None)
    device._notify_link_state()
    assert seen == []


def test_a_raising_callback_does_not_escape():
    """A broken GUI listener must not take a hardware state change down."""
    device = _device()

    def _boom(_dev):
        raise RuntimeError("listener exploded")

    device.set_link_state_callback(_boom)
    device._notify_link_state()  # must not raise
