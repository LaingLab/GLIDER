"""GliderCore.save_session must sync runtime-mutated live device settings
(e.g. an HX711 tare offset) into the session's DeviceConfig copies before
serializing. The session keeps independent copies (see
ExperimentSession.add_device) and would otherwise silently drop them.
"""

from glider.core.experiment_session import DeviceConfig as SessionDeviceConfig
from glider.core.experiment_session import ExperimentSession
from glider.core.glider_core import GliderCore
from glider.hal.base_device import DeviceConfig, DigitalOutputDevice
from glider.hal.mock_board import MockBoard


def _core_with_device():
    core = GliderCore()
    session = core.new_session()
    board = MockBoard()
    board._id = "b1"
    core.hardware_manager._boards["b1"] = board
    device = DigitalOutputDevice(board, DeviceConfig(pins={"output": 13}, settings={"offset": 0.0}))
    core.hardware_manager._devices["dev1"] = device
    session.add_device(
        SessionDeviceConfig(
            id="dev1",
            device_type="DigitalOutput",
            name="d",
            board_id="b1",
            pins={"output": 13},
            settings={"offset": 0.0},
        )
    )
    return core, session, device


def test_save_session_syncs_live_device_settings(temp_dir):
    core, session, device = _core_with_device()
    device.config.settings["offset"] = 1234.5  # runtime mutation (like tare)

    path = core.save_session(str(temp_dir / "t.glider"))

    assert session.get_device("dev1").settings == {"offset": 1234.5}
    loaded = ExperimentSession.load(path)
    assert loaded.get_device("dev1").settings == {"offset": 1234.5}


async def test_create_device_does_not_alias_session_settings():
    # create_device used to hand the session's own settings dict to the
    # device. Sharing it across the serialization boundary makes the
    # save_session sync's inequality guard compare a dict to itself, so it
    # never fires and never marks the session dirty.
    core = GliderCore()
    session = core.new_session()
    board = MockBoard()
    board._id = "b1"
    core.hardware_manager._boards["b1"] = board
    cfg = SessionDeviceConfig(
        id="dev1",
        device_type="DigitalOutput",
        name="d",
        board_id="b1",
        pins={"output": 13},
        settings={"offset": 0.0},
    )
    session.add_device(cfg)

    device = await core.hardware_manager.create_device(cfg)

    assert device.config.settings == cfg.settings
    assert device.config.settings is not cfg.settings
    device.config.settings["offset"] = 5.0
    assert cfg.settings["offset"] == 0.0  # the session copy is untouched


async def test_device_settings_change_marks_session_dirty(temp_dir):
    # A tare mutates only the device. Without a notification the session stays
    # clean, so closing GLIDER never prompts to save and the offset is lost.
    core = GliderCore()
    session = core.new_session()
    board = MockBoard()
    board._id = "b1"
    core.hardware_manager._boards["b1"] = board
    cfg = SessionDeviceConfig(
        id="dev1",
        device_type="DigitalOutput",
        name="d",
        board_id="b1",
        pins={"output": 13},
        settings={"offset": 0.0},
    )
    session.add_device(cfg)
    device = await core.hardware_manager.create_device(cfg)
    core.save_session(str(temp_dir / "t.glider"))
    assert not session.is_dirty

    device.config.settings["offset"] = 4500.0
    device._notify_settings_changed()

    assert session.is_dirty  # the close prompt will now appear
    assert session.get_device("dev1").settings["offset"] == 4500.0


def test_save_session_ignores_devices_absent_from_session(temp_dir):
    core, session, device = _core_with_device()
    extra = DigitalOutputDevice(
        core.hardware_manager._boards["b1"],
        DeviceConfig(pins={"output": 7}, settings={"q": 1}),
    )
    core.hardware_manager._devices["ghost"] = extra  # never added to session

    core.save_session(str(temp_dir / "t.glider"))  # must not raise

    assert session.get_device("ghost") is None
