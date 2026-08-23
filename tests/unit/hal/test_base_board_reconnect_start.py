"""BaseBoard.start_reconnect against a context with no running event loop.

``TelemetrixBoard.is_connected`` -- a *property* -- calls this, so it is read
from wherever a status readout happens to be built, including before qasync
starts the loop. The old shape published RECONNECTING, then built the coroutine,
then let ``create_task`` raise: the board was left wearing RECONNECTING with
nothing retrying it, and the discarded coroutine warned at garbage collection.
Same fix, and the same ordering, as HardwareManager.start_link_supervisor.
"""

import asyncio
import warnings

from glider.hal.base_board import BaseBoard, BoardCapabilities, BoardConnectionState


class _FakeBoard(BaseBoard):
    """Just enough board to reach start_reconnect."""

    @property
    def name(self) -> str:
        return "Fake"

    @property
    def board_type(self) -> str:
        return "Fake"

    @property
    def capabilities(self) -> BoardCapabilities:
        return BoardCapabilities(name="Fake", pins={})

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def set_pin_mode(self, *a, **k) -> None:
        return None

    async def write_digital(self, pin: int, value: bool) -> None:
        return None

    async def read_digital(self, pin: int) -> bool:
        return False

    async def write_analog(self, pin: int, value: int) -> None:
        return None

    async def read_analog(self, pin: int) -> int:
        return 0


def test_starting_with_no_running_loop_is_quiet():
    board = _FakeBoard(auto_reconnect=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        board.start_reconnect()  # must not raise
    assert board._reconnect_task is None
    assert not any("never awaited" in str(w.message) for w in caught)


def test_a_board_with_no_loop_is_not_left_saying_reconnecting():
    """The word has to mean something: nothing is retrying this board."""
    board = _FakeBoard(auto_reconnect=True)
    board.start_reconnect()
    assert board.state is BoardConnectionState.DISCONNECTED


async def test_starting_on_a_loop_still_arms_the_retry():
    board = _FakeBoard(auto_reconnect=True)
    board.start_reconnect()
    assert board._reconnect_task is not None
    assert board.state is BoardConnectionState.RECONNECTING
    board.stop_reconnect()


async def test_the_retry_task_reports_its_exceptions():
    """Nothing awaits the task, so a crash in it needs the done-callback.

    remove_done_callback returns how many it removed, which is the only way to
    ask a Task what is attached to it.
    """
    from glider.core.async_utils import log_task_exception

    board = _FakeBoard(auto_reconnect=True)
    board.start_reconnect()
    assert board._reconnect_task.remove_done_callback(log_task_exception) == 1
    board.stop_reconnect()
    await asyncio.sleep(0)
