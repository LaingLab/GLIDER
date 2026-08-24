"""HardwareManager's device link channel.

Mirrors the board channel (on_connection_change) so the GUI has one shape to
learn, and wires each device in _track_device -- the same chokepoint that
already wires the settings hook, so no creation path can register a device
without it.
"""

import asyncio
import warnings

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


def test_starting_with_no_running_loop_is_quiet():
    """MainWindow wires signals in __init__, before qasync starts the loop.

    The supervisor coroutine must not even be constructed when there is no
    loop to run it on: building it and then discarding it (because
    create_task raises before it can be awaited) is exactly what leaves an
    unawaited-coroutine warning behind at garbage collection.
    """
    manager = HardwareManager()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manager.start_link_supervisor()  # must not raise
    assert manager._link_supervisor is None
    assert not any("never awaited" in str(w.message) for w in caught)


# --- the supervisor must not be able to die quietly ---------------------------
#
# The poll backstop is the mechanism, not a belt: CoreBluetooth and WinRT drop
# the disconnect callback often enough that "GLIDER says connected when it
# isn't" is reproducible without it. One plugin device used to be able to end
# link supervision for the whole session, in silence.


class _RudeDevice(_LinkDevice):
    """owns_link raises -- which a plugin's arbitrary property can."""

    @property
    def owns_link(self):
        raise RuntimeError("this plugin's owns_link is broken")


async def test_a_raising_owns_link_does_not_stop_the_sweep():
    rude = _RudeDevice(_FakeBoard(), DeviceConfig(), name="rude")
    good = _LinkDevice(_FakeBoard(), DeviceConfig(), name="good")
    manager = HardwareManager()
    manager._track_device("rude", rude)
    manager._track_device("good", good)

    await manager.poll_device_links()  # must not raise

    assert good.polls == 1


async def test_a_raising_owns_link_does_not_end_supervision(monkeypatch):
    """The whole point: the supervisor task survives to poll again."""
    monkeypatch.setattr("glider.core.hardware_manager.LINK_POLL_INTERVAL_S", 0)
    rude = _RudeDevice(_FakeBoard(), DeviceConfig(), name="rude")
    manager = HardwareManager()
    manager._track_device("rude", rude)

    manager.start_link_supervisor()
    for _ in range(20):
        await asyncio.sleep(0)

    assert not manager._link_supervisor.done()
    await manager.stop_link_supervisor()


async def test_a_sweep_that_raises_outright_does_not_end_supervision(monkeypatch):
    """Belt for the sweep itself, not just for one device inside it."""
    monkeypatch.setattr("glider.core.hardware_manager.LINK_POLL_INTERVAL_S", 0)
    manager = HardwareManager()
    sweeps = []

    async def _boom():
        sweeps.append(1)
        raise RuntimeError("the sweep itself exploded")

    manager.poll_device_links = _boom
    manager.start_link_supervisor()
    for _ in range(40):
        await asyncio.sleep(0)

    assert len(sweeps) > 1, "the supervisor must keep polling after a failed sweep"
    assert not manager._link_supervisor.done()
    await manager.stop_link_supervisor()


async def test_the_supervisor_task_reports_its_exceptions():
    """Nothing awaits the supervisor, so a crash in it needs the done-callback.

    remove_done_callback returns how many it removed, which is the only way to
    ask a Task what is attached to it.
    """
    from glider.core.async_utils import log_task_exception

    manager = HardwareManager()
    manager.start_link_supervisor()
    assert manager._link_supervisor.remove_done_callback(log_task_exception) == 1
    await manager.stop_link_supervisor()
