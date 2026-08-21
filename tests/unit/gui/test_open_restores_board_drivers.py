"""Opening a file must rebuild the boards the file actually describes.

File > Open guessed the driver as ``"telemetrix" if driver_type == "arduino"
else "pigpio"``, so every board that was neither Arduino nor Pi came back as a
Raspberry Pi GPIO board. A Bluetooth adapter carrying a Maimu stimulator, or a
serial board carrying a GenericSerial device, was silently reparented onto a
driver that cannot talk to it -- and the devices on it then failed to
initialise, so a closed-loop experiment stopped working after a reload with
nothing saying why.

It also dropped ``board_type``, ``auto_reconnect`` and board settings, all of
which the files in ``examples/`` carry.
"""

from __future__ import annotations

import pytest

from glider.core.experiment_session import BoardConfig, DeviceConfig, ExperimentSession
from glider.core.glider_core import GliderCore
from glider.hal.boards.ble_board import BLEBoard
from glider.hal.boards.pi_gpio_board import PiGPIOBoard
from glider.hal.boards.serial_board import SerialBoard
from glider.hal.boards.telemetrix_board import TelemetrixBoard


@pytest.fixture
def core() -> GliderCore:
    c = GliderCore()
    c._session = ExperimentSession()
    return c


def _with_board(core: GliderCore, driver_type: str, **kwargs) -> GliderCore:
    core.session.add_board(BoardConfig(id="b1", driver_type=driver_type, **kwargs))
    return core


@pytest.mark.parametrize(
    ("driver_type", "expected"),
    [
        ("arduino", TelemetrixBoard),
        ("raspberry_pi", PiGPIOBoard),
        ("bluetooth", BLEBoard),
        ("serial", SerialBoard),
    ],
)
def test_every_driver_type_is_rebuilt_as_itself(core, driver_type, expected):
    _with_board(core, driver_type)

    core.populate_hardware_from_session()

    board = core.hardware_manager.get_board("b1")
    assert board is not None, f"{driver_type} board was not created"
    assert isinstance(
        board, expected
    ), f"{driver_type} came back as {type(board).__name__}, not {expected.__name__}"


def test_board_type_survives(core):
    """The examples all carry board_type; add_board had no way to set it."""
    _with_board(core, "arduino", board_type="uno")

    core.populate_hardware_from_session()

    assert core.hardware_manager.get_board("b1")._board_type == "uno"


def test_auto_reconnect_survives(core):
    _with_board(core, "arduino", auto_reconnect=True)

    core.populate_hardware_from_session()

    assert core.hardware_manager.get_board("b1")._auto_reconnect is True


def test_a_ble_device_lands_on_its_bluetooth_board(core):
    """The regression that matters: a Maimu reopened from a saved experiment."""
    _with_board(core, "bluetooth")
    core.session.add_device(
        DeviceConfig(
            id="stim1",
            device_type="Maimu",
            name="Stimulator",
            board_id="b1",
            pins={},
            settings={"address": "AA:BB:CC:DD:EE:FF"},
        )
    )

    core.populate_hardware_from_session()

    device = core.hardware_manager.get_device("stim1")
    assert device is not None, "the Maimu was not recreated"
    assert isinstance(
        device.board, BLEBoard
    ), f"the Maimu was parented to {type(device.board).__name__}"


def test_reopening_replaces_rather_than_stacks(core):
    """Opening a second file must not leave the first one's hardware behind."""
    _with_board(core, "bluetooth")
    core.populate_hardware_from_session()

    core.session.clear()
    core.session.add_board(BoardConfig(id="b2", driver_type="arduino"))
    core.populate_hardware_from_session()

    assert set(core.hardware_manager.boards) == {"b2"}


def test_one_bad_board_does_not_stop_the_others(core):
    """A file naming a driver this install does not have must still open."""
    core.session.add_board(BoardConfig(id="bad", driver_type="does_not_exist"))
    core.session.add_board(BoardConfig(id="b1", driver_type="arduino"))

    core.populate_hardware_from_session()

    assert core.hardware_manager.get_board("b1") is not None
    assert core.hardware_manager.get_board("bad") is None
