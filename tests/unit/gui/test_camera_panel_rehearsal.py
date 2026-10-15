"""The rehearsal has to be indistinguishable from a camera, downstream.

The whole value of playing a recording through the live path is that nothing
downstream treats it specially: the same classifier runs, the same nodes fire,
the same hardware is written to. If the frames took a different route -- or
arrived with a different shape of timestamp -- a green rehearsal would prove
nothing about the live run it is standing in for.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glider.gui.panels.camera_panel import CameraPanel, FrameData

pytestmark = pytest.mark.usefixtures("qtbot")


def _panel():
    """A panel with only what the rehearsal path touches."""
    panel = CameraPanel.__new__(CameraPanel)  # skip the heavy __init__
    panel._rehearsal_pump = None
    panel._behavior_running = True
    return panel


def test_a_rehearsal_frame_goes_down_the_camera_funnel(qtbot):
    """_frame_received is what the camera emits and what _handle_frame_input
    listens on. A rehearsal that used any other route would bypass the
    behavior worker, which is the entire point of running one."""
    panel = _panel()
    received: list[FrameData] = []
    panel._frame_received = SimpleNamespace(emit=received.append)

    frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    panel._on_rehearsal_frame(frame, 1234.5)

    assert len(received) == 1
    assert isinstance(received[0], FrameData)
    assert received[0].timestamp == 1234.5
    assert int(received[0].frame[0, 0, 0]) == 7


def test_the_frame_is_copied(qtbot):
    """The pump reuses decoder buffers; handing the original across a thread
    boundary would let it change under the consumer."""
    panel = _panel()
    received: list[FrameData] = []
    panel._frame_received = SimpleNamespace(emit=received.append)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    panel._on_rehearsal_frame(frame, time.time())
    frame[:] = 99  # the decoder moves on

    assert int(received[0].frame[0, 0, 0]) == 0


def test_it_does_not_require_an_open_camera(qtbot):
    """_on_frame drops everything unless _preview_active. A rehearsal has to
    work on a machine with no camera at all -- which is most of them."""
    panel = _panel()
    panel._preview_active = False
    received: list[FrameData] = []
    panel._frame_received = SimpleNamespace(emit=received.append)

    panel._on_rehearsal_frame(np.zeros((2, 2, 3), dtype=np.uint8), 1.0)

    assert len(received) == 1


# --- the end-to-end claim -----------------------------------------------------


async def test_a_recording_drives_real_hardware(qtbot, monkeypatch, tmp_path):
    """The claim the feature makes: play a video, the stimulator fires.

    Everything between the frames and the GATT write is the real thing -- the
    live signal bus, the Behavior Input node, the exec connection, the device.
    Only the two genuine externals are faked: pose/behavior inference (which
    would need a trained model and weights) and bleak.
    """
    import sys
    from unittest.mock import MagicMock

    written: list[bytes] = []

    class _FakeClient:
        # *a, **k absorb kwargs the real BleakClient accepts and BLEDevice
        # passes -- e.g. disconnected_callback -- that this fake has no use
        # for, so it keeps matching the real constructor's signature.
        def __init__(self, address, *a, **k):
            self.address = address
            self.is_connected = False

        async def connect(self):
            self.is_connected = True

        async def disconnect(self):
            self.is_connected = False

        async def write_gatt_char(self, char, data, response=False):
            written.append(bytes(data))

    bleak = MagicMock(name="bleak")
    bleak.BleakClient = _FakeClient

    async def _present(address, timeout=10.0):
        return address  # stands in for the BLEDevice bleak returns; BleakClient takes either

    bleak.BleakScanner.find_device_by_address = _present
    monkeypatch.setitem(sys.modules, "bleak", bleak)

    from glider.core.flow_engine import FlowEngine
    from glider.core.hardware_manager import HardwareManager
    from glider.core.live_signals import BehaviorEvent, LiveSignalBus
    from glider.hal.mock_board import MockBoard
    from glider.nodes.vision import register_behavior_nodes

    maimu = pytest.importorskip(
        "glider_maimu", reason="glider-maimu is not installed; the stimulus half needs it"
    )

    HardwareManager.register_driver("mock", MockBoard)
    hardware = HardwareManager()
    hardware._boards["b1"] = MockBoard()
    monkeypatch.setitem(
        __import__("glider.hal.base_device", fromlist=["DEVICE_REGISTRY"]).DEVICE_REGISTRY,
        "Maimu",
        maimu.MaimuDevice,
    )
    hardware.add_device_multi_pin(
        "stim", "Maimu", "b1", pins={}, name="Stim", settings={"address": "AA:BB"}
    )
    await hardware.initialize_device("stim")

    bus = LiveSignalBus()
    engine = FlowEngine(hardware)
    engine.set_live_signals(bus)
    register_behavior_nodes(engine)
    monkeypatch.setitem(FlowEngine._node_registry, "Maimu", maimu.MaimuNode)

    watcher = engine.create_node(node_id="w", node_type="BehaviorInput", position=(0.0, 0.0))
    watcher.target_behavior = "freezing"
    watcher.min_frames = 3
    stim = engine.create_node(node_id="s", node_type="Maimu", position=(1.0, 0.0), device_id="stim")
    stim.mode = "pulse"
    stim.period_ms = 500
    stim.pulse_width_ms = 5
    stim.count = 20
    stim.intensity_pct = 100
    engine.create_connection(
        connection_id="c",
        from_node_id="w",
        from_output=2,
        to_node_id="s",
        to_input=0,
        connection_type="exec",
    )
    await engine.start()

    import asyncio

    deadline = asyncio.get_running_loop().time() + 2.0
    while not watcher._subscribed and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)

    # The stand-in for pose + behavior inference: every frame of this clip is a
    # freeze, which is what the recording would contain.
    panel = _panel()
    panel._frame_received = SimpleNamespace(
        emit=lambda fd: bus.publish_behavior(BehaviorEvent(behavior="freezing"))
    )

    for _ in range(5):
        panel._on_rehearsal_frame(np.zeros((2, 2, 3), dtype=np.uint8), time.time())

    deadline = asyncio.get_running_loop().time() + 2.0
    while not written and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    await engine.stop()

    assert written == [b"500,5,20,100"], "a recorded freeze did not reach the stimulator"


# --- which video does it play? ------------------------------------------------


def _panel_with_source(loaded_path=None):
    panel = CameraPanel.__new__(CameraPanel)
    panel._rehearsal_pump = None
    panel._behavior_running = True
    panel._video_source = SimpleNamespace(
        is_loaded=loaded_path is not None,
        path=loaded_path,
    )
    panel._rehearse_btn = SimpleNamespace(_text="", setText=lambda t: None)
    return panel


def test_it_plays_the_video_already_loaded_in_the_panel(qtbot):
    """Prompting for a second file let a rehearsal run against a different
    video than the one on screen -- the one the zones were drawn over."""
    panel = _panel_with_source("/clips/session12.mp4")

    assert panel._rehearsal_video() == "/clips/session12.mp4"


def test_it_prompts_only_when_nothing_is_loaded(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    asked = []

    def _ask(*args, **kwargs):
        asked.append(True)
        return ("", "")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(_ask))
    panel = _panel_with_source(None)

    assert panel._rehearsal_video() is None
    assert asked, "it should have asked for a file when none was loaded"


def test_a_chosen_file_is_loaded_into_the_panel_too(qtbot, monkeypatch):
    """So the scrub bar, zone drawing and preview all refer to what is playing,
    rather than the pump quietly running something the panel knows nothing of."""
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("/clips/new.mp4", ""))
    )
    panel = _panel_with_source(None)
    applied = []
    panel._apply_video = lambda p: (applied.append(Path(p)), True)[1]

    assert panel._rehearsal_video() == "/clips/new.mp4"
    # Compared as paths: Path() normalises separators, and the point is that the
    # panel was told about the same file, not how the OS spells it.
    assert applied == [Path("/clips/new.mp4")]


def test_a_file_that_will_not_load_is_not_rehearsed(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("/clips/broken.mp4", ""))
    )
    panel = _panel_with_source(None)
    panel._apply_video = lambda p: False

    assert panel._rehearsal_video() is None


def test_the_button_names_the_video_it_would_play(qtbot):
    """A button reading 'Rehearse from video...' next to a loaded clip is a
    question; one reading 'Rehearse session12.mp4' is an answer."""
    panel = _panel_with_source("/clips/session12.mp4")
    labels = []
    panel._rehearse_btn = SimpleNamespace(setText=labels.append)

    panel._update_rehearse_label()

    assert labels == ["Rehearse session12.mp4"]


def test_the_button_asks_when_nothing_is_loaded(qtbot):
    panel = _panel_with_source(None)
    labels = []
    panel._rehearse_btn = SimpleNamespace(setText=labels.append)

    panel._update_rehearse_label()

    assert labels == ["Rehearse from video…"]


def test_the_label_is_left_alone_while_a_rehearsal_runs(qtbot):
    """It reads 'Stop rehearsal' then, and relabelling it mid-run would take
    away the only way to stop hardware that is being driven."""
    panel = _panel_with_source("/clips/session12.mp4")
    panel._rehearsal_pump = SimpleNamespace(is_running=True)
    labels = []
    panel._rehearse_btn = SimpleNamespace(setText=labels.append)

    panel._update_rehearse_label()

    assert labels == []


# --- what it needs before it can run -------------------------------------------


def test_rehearsal_does_not_need_a_behavior_model(qtbot):
    """A tracker, a Zone Input node and a stimulator are a complete experiment
    with no behavior classifier in it. Rehearse used to wait for a behavior
    model to load, so that experiment could not be rehearsed at all -- and it
    sat inside the Live Behavior box, which said the same thing."""
    from PyQt6.QtWidgets import QGroupBox

    from glider.vision.camera_manager import CameraManager
    from glider.vision.cv_processor import CVProcessor

    panel = CameraPanel(CameraManager(), CVProcessor())
    qtbot.addWidget(panel)

    assert panel._rehearse_btn.isEnabled()
    behavior = next(g for g in panel.findChildren(QGroupBox) if g.title() == "Live Behavior")
    assert not behavior.isAncestorOf(panel._rehearse_btn)


# --- what the researcher sees and what gets tracked ---------------------------


def _real_panel(qtbot):
    from glider.vision.camera_manager import CameraManager
    from glider.vision.cv_processor import CVProcessor

    panel = CameraPanel(CameraManager(), CVProcessor())
    qtbot.addWidget(panel)
    return panel


def test_rehearsal_frames_reach_the_preview_without_a_camera(qtbot):
    """The preview used to drop every frame unless a camera preview was open,
    which in video-file mode it never is -- so the video sat on frame 0 while
    the rehearsal ran blind behind it."""
    panel = _real_panel(qtbot)
    assert not panel._preview_active
    panel._rehearsal_pump = SimpleNamespace(is_running=True, stop=lambda: None)

    frame = np.full((4, 4, 3), 9, dtype=np.uint8)
    panel._process_frame_on_main_thread(FrameData(frame=frame, timestamp=1.0))

    assert panel._last_frame is frame


def test_starting_a_rehearsal_readies_computer_vision(qtbot, monkeypatch):
    """Zone enter/exit comes out of the CV processor, which was only ever
    initialized when a camera preview started. Uninitialized, every rehearsal
    frame skipped CV, so a Zone Input node never fired."""
    import glider.vision.video_pump as video_pump

    class _Pump:
        def __init__(self, *a, **k):
            pass

        def start(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(video_pump, "VideoPump", _Pump)
    panel = _real_panel(qtbot)
    panel._rehearsal_video = lambda: "/clips/session12.mp4"
    panel._cv_enabled_cb.setChecked(True)
    assert not panel._cv_processor.is_initialized

    panel._start_rehearsal()

    assert panel._cv_processor.is_initialized


class _QuietPump:
    started: list[bool] = []

    def __init__(self, *a, **k):
        pass

    def start(self):
        _QuietPump.started.append(True)
        return True

    def stop(self):
        pass


def test_a_model_that_will_not_load_stops_the_rehearsal_not_glider(qtbot, monkeypatch):
    """A pose backend re-raises its model's own error, and GLIDER installs no
    excepthook -- so it escaping the Rehearse click aborted the whole app. It
    is reported instead, and no hardware is driven."""
    from PyQt6.QtWidgets import QMessageBox

    import glider.vision.video_pump as video_pump
    from glider.vision.pose.spec import PoseModelError

    _QuietPump.started = []
    monkeypatch.setattr(video_pump, "VideoPump", _QuietPump)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a: warned.append(a[2])))
    panel = _real_panel(qtbot)
    panel._rehearsal_video = lambda: "/clips/session12.mp4"
    panel._cv_enabled_cb.setChecked(True)

    def _fail():
        raise PoseModelError("enter the keypoint names")

    monkeypatch.setattr(panel._cv_processor, "initialize", _fail)

    requested = []
    panel.rehearsal_requested.connect(requested.append)

    panel._start_rehearsal()

    assert warned == ["enter the keypoint names"]
    assert requested == [], "the experiment must not be started"
    assert _QuietPump.started == []
    assert panel._rehearsal_pump is None


def _rehearse_with_bus(qtbot, monkeypatch, bus):
    import glider.vision.video_pump as video_pump

    monkeypatch.setattr(video_pump, "VideoPump", _QuietPump)
    panel = _real_panel(qtbot)
    panel.set_live_signals(bus)
    assert panel.begin_rehearsal("/clips/session12.mp4")
    return panel._rehearse_status.text()


def test_it_says_when_no_node_is_listening(qtbot, monkeypatch):
    """An experiment with no Zone Input or Behavior Input node plays the video
    to nobody, which looked exactly like the rehearsal being broken."""
    from glider.core.live_signals import LiveSignalBus

    assert "is listening" in _rehearse_with_bus(qtbot, monkeypatch, LiveSignalBus())


def test_it_does_not_nag_once_a_node_is_listening(qtbot, monkeypatch):
    from glider.core.live_signals import LiveSignalBus

    bus = LiveSignalBus()
    bus.subscribe_zones(lambda states: None)

    assert "is listening" not in _rehearse_with_bus(qtbot, monkeypatch, bus)


# --- the experiment it drives --------------------------------------------------


def test_rehearse_asks_the_host_to_start_the_experiment_first(qtbot, monkeypatch):
    """Playing straight away, before the experiment's nodes were listening,
    is what made the first rehearsal do nothing."""
    import glider.vision.video_pump as video_pump

    _QuietPump.started = []
    monkeypatch.setattr(video_pump, "VideoPump", _QuietPump)
    panel = _real_panel(qtbot)
    panel._rehearsal_video = lambda: "/clips/session12.mp4"
    requested = []
    panel.rehearsal_requested.connect(requested.append)

    panel._start_rehearsal()

    assert requested == ["/clips/session12.mp4"]
    assert _QuietPump.started == []


def test_the_end_of_a_rehearsal_is_announced(qtbot):
    """So the host can stop the experiment the rehearsal started."""
    from glider.vision.video_pump import PumpStats

    panel = _real_panel(qtbot)
    ended = []
    panel.rehearsal_ended.connect(lambda: ended.append(True))

    panel._on_rehearsal_finished_main(PumpStats(10, 10, 0.0, 0.0, True))

    assert ended == [True]


@pytest.mark.parametrize("state, stops", [("READY", True), ("ERROR", True), ("RUNNING", False)])
def test_a_rehearsal_stops_with_its_experiment(qtbot, state, stops):
    """Once the experiment stops its nodes stop listening; the video would
    play on to nobody."""
    panel = _real_panel(qtbot)
    stopped = []
    panel._rehearsal_pump = SimpleNamespace(stop=lambda: stopped.append(True))

    panel.on_experiment_state(state)

    assert bool(stopped) is stops
