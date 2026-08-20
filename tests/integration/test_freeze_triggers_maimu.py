"""Closed loop: the animal freezes, the Maimu stimulates.

This is the whole point of live behavior classification in GLIDER -- a stimulus
delivered because of what the animal is doing, not because a timer expired. The
chain has four hops and each was built separately, so this drives all of them at
once and asserts on the bytes that reach the peripheral:

    classifier frame -> LiveSignalBus -> Behavior Input node
                     -> exec connection -> Maimu node -> MaimuDevice -> GATT write

Nothing here is stubbed except the two ends that are genuinely external: the
classifier (a plain label per frame, which is exactly what
``LiveBehaviorClassifier.classify_frame`` yields) and bleak.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from glider.core.flow_engine import FlowEngine
from glider.core.hardware_manager import HardwareManager
from glider.core.live_signals import BehaviorEvent, LiveSignalBus
from glider.hal.mock_board import MockBoard
from glider.nodes.hardware import register_hardware_nodes
from glider.nodes.vision import register_behavior_nodes

FREEZING = "freezing"
MIN_FRAMES = 3
PERIOD_MS = 500
DURATION_S = 10


class _FakeClient:
    """Minimal async BleakClient stand-in that records what was written."""

    def __init__(self, address):
        self.address = address
        self.is_connected = False
        self.written: list[bytes] = []

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def write_gatt_char(self, char, data, response=False):
        self.written.append(bytes(data))


@pytest.fixture
def fake_bleak(monkeypatch):
    from unittest.mock import MagicMock

    created = {}

    def make_client(address, *a, **k):
        created["client"] = _FakeClient(address)
        return created["client"]

    module = MagicMock(name="bleak")
    module.BleakClient = make_client
    monkeypatch.setitem(sys.modules, "bleak", module)
    return created


async def _wait_for(predicate, timeout=2.0):
    """Let the loop run until predicate() is true, or fail loudly."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


async def _rig(fake_bleak):
    """Build the whole chain and return (bus, engine, fake client)."""
    HardwareManager.register_driver("mock", MockBoard)
    hardware = HardwareManager()
    hardware._boards["b1"] = MockBoard()
    hardware.add_device_multi_pin(
        "stim1",
        "Maimu",
        "b1",
        pins={},
        name="Stimulator",
        settings={"address": "AA:BB:CC:DD:EE:FF"},
    )
    await hardware.initialize_device("stim1")

    bus = LiveSignalBus()
    engine = FlowEngine(hardware)
    engine.set_live_signals(bus)
    register_behavior_nodes(engine)
    register_hardware_nodes(engine)

    watcher = engine.create_node(node_id="watch", node_type="BehaviorInput", position=(0.0, 0.0))
    watcher.target_behavior = FREEZING
    watcher.min_frames = MIN_FRAMES

    stim = engine.create_node(
        node_id="stim", node_type="Maimu", position=(1.0, 0.0), device_id="stim1"
    )
    stim.mode = "pulse"
    stim.period_ms = PERIOD_MS
    stim.duration_s = DURATION_S
    assert stim.device is not None, "precondition: the Maimu node did not bind its device"

    # "On Enter" is output 2; the Maimu node's only input is exec.
    engine.create_connection(
        connection_id="c1",
        from_node_id="watch",
        from_output=2,
        to_node_id="stim",
        to_input=0,
        connection_type="exec",
    )

    # FlowEngine.start() schedules each continuous node's start() as a task and
    # returns before it has run, so the watcher is not on the bus yet. Frames
    # published in that window are dropped -- harmless at 30 fps, where the next
    # frame is 33 ms away, but a test that publishes immediately would race it.
    await engine.start()
    assert await _wait_for(lambda: watcher._subscribed), "watcher never subscribed"
    return bus, engine, fake_bleak["client"]


def _frame(label: str, index: int) -> BehaviorEvent:
    return BehaviorEvent(behavior=label, frame_index=index)


async def test_freezing_triggers_a_maimu_pulse(fake_bleak):
    """The headline: enough consecutive freezing frames stimulate the animal."""
    bus, engine, client = await _rig(fake_bleak)
    try:
        for i in range(MIN_FRAMES):
            bus.publish_behavior(_frame(FREEZING, i))

        fired = await _wait_for(lambda: client.written)
        assert fired, "freezing never reached the stimulator"
        assert client.written == [b"500,10"]
    finally:
        await engine.stop()


async def test_one_stray_frame_does_not_stimulate(fake_bleak):
    """Frame-wise classification is noisy. A single misclassified frame firing
    hardware is the failure mode min_frames exists to prevent."""
    bus, engine, client = await _rig(fake_bleak)
    try:
        bus.publish_behavior(_frame("rearing", 0))
        bus.publish_behavior(_frame(FREEZING, 1))  # the stray
        bus.publish_behavior(_frame("rearing", 2))

        await asyncio.sleep(0.1)
        assert client.written == []
    finally:
        await engine.stop()


async def test_a_sustained_freeze_stimulates_once_not_per_frame(fake_bleak):
    """On Enter is an edge, not a level. Ten frames of freezing is one bout."""
    bus, engine, client = await _rig(fake_bleak)
    try:
        for i in range(10):
            bus.publish_behavior(_frame(FREEZING, i))

        assert await _wait_for(lambda: client.written)
        await asyncio.sleep(0.1)
        assert client.written == [b"500,10"], "the stimulus repeated within one bout"
    finally:
        await engine.stop()


async def test_a_second_freeze_bout_stimulates_again(fake_bleak):
    """Enter, leave, enter again -- two bouts, two stimuli."""
    bus, engine, client = await _rig(fake_bleak)
    try:
        for i in range(MIN_FRAMES):
            bus.publish_behavior(_frame(FREEZING, i))
        assert await _wait_for(lambda: len(client.written) == 1)

        for i in range(MIN_FRAMES):  # leave
            bus.publish_behavior(_frame("rearing", 10 + i))
        for i in range(MIN_FRAMES):  # and freeze again
            bus.publish_behavior(_frame(FREEZING, 20 + i))

        assert await _wait_for(lambda: len(client.written) == 2)
        assert client.written == [b"500,10", b"500,10"]
    finally:
        await engine.stop()


async def test_a_stopped_flow_does_not_stimulate(fake_bleak):
    """Stopping the experiment must stop the closed loop, not just the display."""
    bus, engine, client = await _rig(fake_bleak)
    await engine.stop()

    for i in range(MIN_FRAMES * 2):
        bus.publish_behavior(_frame(FREEZING, i))

    await asyncio.sleep(0.1)
    assert client.written == [], "a stopped flow still drove the hardware"


async def test_the_warmup_label_does_not_stimulate(fake_bleak):
    """The classifier emits an empty label until its feature buffer fills."""
    bus, engine, client = await _rig(fake_bleak)
    try:
        for i in range(MIN_FRAMES * 2):
            bus.publish_behavior(_frame("", i))

        await asyncio.sleep(0.1)
        assert client.written == []
    finally:
        await engine.stop()
