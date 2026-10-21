from unittest.mock import MagicMock

import pytest

from glider.gui.runner.device_grid import DeviceCardGrid

pytestmark = pytest.mark.usefixtures("qtbot")

_EMPTY = "No devices yet. Add a board on the Dashboard › Hardware tab."


def _device(name):
    device = MagicMock()
    device.device_type = "DigitalOutput"
    device._state = True
    device.link_state = None
    device.name = name
    return device


def _grid(qtbot, core, n=3):
    devices = {f"dev_{i}": _device(f"dev_{i}") for i in range(n)}
    core.hardware_manager.devices = devices
    core.hardware_manager.get_device = devices.get
    grid = DeviceCardGrid(core)
    qtbot.addWidget(grid)
    grid.refresh_devices()
    return grid


def test_cards_lay_out_in_two_columns(qtbot, mock_core):
    grid = _grid(qtbot, mock_core, 3)
    positions = [
        grid._grid.getItemPosition(grid._grid.indexOf(c))[:2] for c in grid.cards().values()
    ]
    assert positions == [(0, 0), (0, 1), (1, 0)]
    assert list(grid.cards()) == ["dev_0", "dev_1", "dev_2"]


def test_refresh_rebuilds_without_leaving_stale_cards(qtbot, mock_core):
    grid = _grid(qtbot, mock_core, 3)
    del mock_core.hardware_manager.devices["dev_1"]
    grid.refresh_devices()
    assert list(grid.cards()) == ["dev_0", "dev_2"]
    positions = [
        grid._grid.getItemPosition(grid._grid.indexOf(c))[:2] for c in grid.cards().values()
    ]
    assert positions == [(0, 0), (0, 1)]


def test_empty_state_names_the_hardware_tab(qtbot, mock_core):
    grid = _grid(qtbot, mock_core, 0)
    assert grid._empty.text() == _EMPTY
    assert grid._empty.isVisibleTo(grid)
    assert grid.cards() == {}
    grid.refresh_devices()  # idempotent
    mock_core.hardware_manager.devices = {"d": _device("d")}
    grid.refresh_devices()
    assert not grid._empty.isVisibleTo(grid)


def test_cards_keep_their_state_labels_and_property(qtbot, mock_core):
    grid = _grid(qtbot, mock_core, 1)
    card = grid.cards()["dev_0"]
    assert card.property("deviceCard") is True
    assert card._state_label.text() == "HIGH"
    assert card._ready_label.text() == "Unknown"


def test_state_poll_runs_only_while_live(qtbot, mock_core):
    grid = _grid(qtbot, mock_core, 1)
    card = next(iter(grid.cards().values()))
    grid.set_live(True)
    assert grid._poll.isActive()
    assert grid._poll.interval() > 0

    mock_core.hardware_manager.devices["dev_0"]._state = False
    grid.set_live(False)  # one last repaint on the way out
    assert not grid._poll.isActive()
    assert card._state_label.text() == "LOW"


def test_the_scroll_area_is_vertical_only(qtbot, mock_core):
    from PyQt6.QtCore import Qt

    grid = _grid(qtbot, mock_core, 1)
    assert grid._scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
