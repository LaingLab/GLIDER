"""The rehearsal has to be indistinguishable from a camera, downstream.

The whole value of playing a recording through the live path is that nothing
downstream treats it specially: the same classifier runs, the same nodes fire,
the same hardware is written to. If the frames took a different route -- or
arrived with a different shape of timestamp -- a green rehearsal would prove
nothing about the live run it is standing in for.
"""

from __future__ import annotations

import time
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
        def __init__(self, address):
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
    stim.duration_s = 10
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

    assert written == [b"500,10"], "a recorded freeze did not reach the stimulator"
