from glider.core.hardware_manager import HardwareManager
from glider.hal.mock_board import MockBoard


def test_no_boards_returns_false():
    assert HardwareManager().is_any_board_connected() is False


def test_connected_mock_board_returns_true():
    # MockBoard.__init__ sets state to CONNECTED, so a fresh board is connected.
    manager = HardwareManager()
    manager._boards["b1"] = MockBoard()
    assert manager.is_any_board_connected() is True


async def test_registered_but_disconnected_board_returns_false():
    # Must explicitly disconnect — MockBoard auto-connects in its constructor.
    manager = HardwareManager()
    board = MockBoard()
    await board.disconnect()
    manager._boards["b1"] = board
    assert manager.is_any_board_connected() is False
