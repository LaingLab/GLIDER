"""HardwareManager's device link channel.

Mirrors the board channel (on_connection_change) so the GUI has one shape to
learn, and wires each device in _track_device -- the same chokepoint that
already wires the settings hook, so no creation path can register a device
without it.
"""

from glider.core.hardware_manager import LINK_POLL_INTERVAL_S, HardwareManager
from glider.hal.base_board import ConnectionState
from glider.hal.base_device import BaseDevice, DeviceConfig


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


class _LinkDevice(BaseDevice):
    """A device that owns a link, so the supervisor polls it."""

    def __init__(self, board, config, name=None):
        super().__init__(board, config, name)
        self._link = ConnectionState.DISCONNECTED
        self.polls = 0

    @property
    def device_type(self):
        return "FakeLink"

    @property
    def actions(self):
        return {}

    @property
    def owns_link(self):
        return True

    @property
    def link_state(self):
        return self._link

    async def poll_link(self):
        self.polls += 1

    def set_link(self, state):
        self._link = state
        self._notify_link_state()

    async def initialize(self):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False

    @classmethod
    def from_dict(cls, data, board):
        return cls(board, DeviceConfig())


class _PinDevice(BaseDevice):
    """No link of its own; the supervisor must skip it."""

    def __init__(self, board, config, name=None):
        super().__init__(board, config, name)
        self.polls = 0

    @property
    def device_type(self):
        return "FakePin"

    @property
    def actions(self):
        return {}

    async def poll_link(self):
        self.polls += 1

    async def initialize(self):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False

    @classmethod
    def from_dict(cls, data, board):
        return cls(board, DeviceConfig())


def _manager_with(device, device_id="dev1"):
    manager = HardwareManager()
    manager._track_device(device_id, device)
    return manager


def test_the_poll_interval_is_two_seconds():
    assert LINK_POLL_INTERVAL_S == 2.0


def test_a_device_link_change_reaches_a_listener():
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)
    seen = []
    manager.on_device_connection_change(lambda dev_id, state: seen.append((dev_id, state)))
    device.set_link(ConnectionState.RECONNECTING)
    assert seen == [("dev1", ConnectionState.RECONNECTING)]


def test_every_listener_is_told():
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)
    first, second = [], []
    manager.on_device_connection_change(lambda d, s: first.append(s))
    manager.on_device_connection_change(lambda d, s: second.append(s))
    device.set_link(ConnectionState.ERROR)
    assert first == second == [ConnectionState.ERROR]


def test_a_raising_listener_does_not_stop_the_others():
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)
    survived = []

    def _boom(_dev_id, _state):
        raise RuntimeError("listener exploded")

    manager.on_device_connection_change(_boom)
    manager.on_device_connection_change(lambda d, s: survived.append(s))
    device.set_link(ConnectionState.CONNECTED)
    assert survived == [ConnectionState.CONNECTED]


def test_a_device_tracked_before_the_listener_is_still_wired():
    """Registration order must not decide whether a device is heard."""
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)  # tracked first
    seen = []
    manager.on_device_connection_change(lambda d, s: seen.append(s))  # listener second
    device.set_link(ConnectionState.DISCONNECTED)
    assert seen == [ConnectionState.DISCONNECTED]


async def test_a_sweep_polls_a_link_owning_device():
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)
    await manager.poll_device_links()
    assert device.polls == 1


async def test_a_sweep_skips_a_pin_device():
    """Polling a derived link_state is pure overhead on every rig with GPIO."""
    device = _PinDevice(_FakeBoard(), DeviceConfig(), name="led")
    manager = _manager_with(device)
    await manager.poll_device_links()
    assert device.polls == 0


async def test_a_failing_poll_does_not_stop_the_sweep():
    good = _LinkDevice(_FakeBoard(), DeviceConfig(), name="good")
    bad = _LinkDevice(_FakeBoard(), DeviceConfig(), name="bad")

    async def _boom():
        raise OSError("adapter went away")

    bad.poll_link = _boom
    manager = HardwareManager()
    manager._track_device("bad", bad)
    manager._track_device("good", good)
    await manager.poll_device_links()
    assert good.polls == 1


async def test_the_supervisor_starts_and_stops():
    manager = HardwareManager()
    manager.start_link_supervisor()
    assert manager._link_supervisor is not None
    await manager.stop_link_supervisor()
    assert manager._link_supervisor is None


async def test_starting_twice_keeps_one_task():
    manager = HardwareManager()
    manager.start_link_supervisor()
    first = manager._link_supervisor
    manager.start_link_supervisor()
    assert manager._link_supervisor is first
    await manager.stop_link_supervisor()


async def test_stopping_when_not_started_is_quiet():
    await HardwareManager().stop_link_supervisor()  # must not raise


async def test_manager_shutdown_stops_the_supervisor():
    manager = HardwareManager()
    manager.start_link_supervisor()
    await manager.shutdown()
    assert manager._link_supervisor is None
