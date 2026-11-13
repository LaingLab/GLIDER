"""A dropped link: the pump comes back as it was, a puff never does.

The firmware turns both outputs off the moment the link drops. On reconnect the
pump goes back to the speed the experiment last asked for, and with nothing
asked the pod is told ``off``, so it is always in a known state. A puff is a
dose: one cut short is reported, one asked for while the link is down raises,
and neither is ever sent late.
"""

import asyncio
import logging
import types

import pytest

from glider.hal.base_board import ConnectionState
from glider.hal.base_device import DeviceConfig
from glider.hal.devices.ble_device import BLEDevice
from glider_vapepod.device import VapePodDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch):
    monkeypatch.setattr(BLEDevice, "RECONNECT_BASE_S", 0.001)
    monkeypatch.setattr(BLEDevice, "RECONNECT_MAX_BACKOFF_S", 0.001)


@pytest.fixture
def clock(monkeypatch):
    """The device's view of time, moved by hand: [now] in seconds."""
    now = [1000.0]
    monkeypatch.setattr(
        "glider_vapepod.device.time", types.SimpleNamespace(perf_counter=lambda: now[0])
    )
    return now


async def _initialized():
    config = DeviceConfig(settings={"address": "11:22:33:44:55:66"})
    device = VapePodDevice(_FakeBoard(), config, name="pod")
    await device.initialize()
    return device


async def _settle(device):
    for _ in range(500):
        if device.link_state is ConnectionState.CONNECTED:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0)
    raise AssertionError("never reconnected")


async def _first_write_on_the_new_link(device):
    for _ in range(200):
        if device._client.written:
            return device._client.written
        await asyncio.sleep(0)
    raise AssertionError("nothing was written after the reconnect")


async def _drop_and_stay_down(device, fake_bleak):
    fake_bleak["connect_error"] = OSError("out of range")
    fake_bleak["client"].drop()
    for _ in range(20):
        await asyncio.sleep(0)
    assert device.link_state is ConnectionState.RECONNECTING


# --- what the reconnect puts back ---------------------------------------------


async def test_with_nothing_asked_the_reconnect_writes_off(fake_bleak):
    device = await _initialized()
    fake_bleak["client"].drop()
    await _settle(device)
    assert await _first_write_on_the_new_link(device) == [b"off"]
    await device.shutdown()


@pytest.mark.parametrize("speed", [100, 40])
async def test_the_pump_comes_back_at_its_speed(fake_bleak, speed):
    device = await _initialized()
    await device.pump(speed)
    fake_bleak["client"].drop()
    await _settle(device)
    assert await _first_write_on_the_new_link(device) == [f"pump,{speed}".encode()]
    await device.shutdown()


@pytest.mark.parametrize("stop", ["pump_off", "off"])
async def test_a_pump_turned_off_stays_off(fake_bleak, stop):
    device = await _initialized()
    await device.pump_on()
    await getattr(device, stop)()
    fake_bleak["client"].drop()
    await _settle(device)
    assert await _first_write_on_the_new_link(device) == [b"off"]
    await device.shutdown()


async def test_a_puff_is_never_resumed(fake_bleak, clock):
    device = await _initialized()
    await device.pump_on()
    await device.puff(8000)
    clock[0] += 2
    fake_bleak["client"].drop()
    await _settle(device)
    for _ in range(50):
        await asyncio.sleep(0)
    assert device._client.written == [b"pump,100"]
    await device.shutdown()


async def test_a_puff_cut_short_is_reported(fake_bleak, clock, caplog):
    device = await _initialized()
    await device.puff(8000)
    clock[0] += 2
    with caplog.at_level(logging.WARNING, logger="glider_vapepod.device"):
        fake_bleak["client"].drop()
        await _settle(device)
    assert any("mid-puff" in r.message for r in caplog.records)
    await device.shutdown()


async def test_a_puff_that_had_finished_is_not_reported(fake_bleak, clock, caplog):
    device = await _initialized()
    await device.puff(3000)
    clock[0] += 5
    with caplog.at_level(logging.WARNING, logger="glider_vapepod.device"):
        fake_bleak["client"].drop()
        await _settle(device)
    assert not any("mid-puff" in r.message for r in caplog.records)
    await device.shutdown()


async def test_shutting_down_mid_puff_is_not_reported(fake_bleak, caplog):
    """Stopping the puff is what a shutdown is for."""
    device = await _initialized()
    await device.puff(8000)
    with caplog.at_level(logging.WARNING, logger="glider_vapepod.device"):
        await device.shutdown()
    assert not any("mid-puff" in r.message for r in caplog.records)


# --- commands while the link is down ------------------------------------------


async def test_a_puff_with_the_link_down_raises_and_sends_nothing(fake_bleak):
    device = await _initialized()
    await _drop_and_stay_down(device, fake_bleak)

    with pytest.raises(ConnectionError, match="puff not delivered"):
        await asyncio.wait_for(device.puff(3000), timeout=0.5)

    fake_bleak["connect_error"] = None
    await _settle(device)
    for _ in range(50):
        await asyncio.sleep(0)
    assert all(b"puff" not in w for c in fake_bleak["clients"] for w in c.written)
    await device.shutdown()


async def test_pump_commands_during_a_drop_return_at_once_and_the_newest_wins(fake_bleak):
    device = await _initialized()
    await _drop_and_stay_down(device, fake_bleak)

    await asyncio.wait_for(device.pump_on(), timeout=0.5)
    await asyncio.wait_for(device.off(), timeout=0.5)
    await asyncio.wait_for(device.pump(70), timeout=0.5)
    assert all(client.written == [] for client in fake_bleak["clients"][1:])

    fake_bleak["connect_error"] = None
    await _settle(device)
    assert await _first_write_on_the_new_link(device) == [b"pump,70"]
    await device.shutdown()


async def test_a_pump_command_whose_write_dies_with_the_link_is_sent_on_reconnect(fake_bleak):
    device = await _initialized()
    original = device._client

    async def _dies(char, data, response=False):
        original.is_connected = False  # the link went down under this write
        raise OSError("link dropped mid-write")

    original.write_gatt_char = _dies

    await asyncio.wait_for(device.pump(55), timeout=0.5)
    await _settle(device)
    assert await _first_write_on_the_new_link(device) == [b"pump,55"]
    await device.shutdown()


async def test_a_puff_whose_write_dies_with_the_link_raises(fake_bleak):
    device = await _initialized()
    original = device._client

    async def _dies(char, data, response=False):
        original.is_connected = False
        raise OSError("link dropped mid-write")

    original.write_gatt_char = _dies

    with pytest.raises(OSError):
        await asyncio.wait_for(device.puff(3000), timeout=0.5)
    await _settle(device)
    assert await _first_write_on_the_new_link(device) == [b"off"]
    await device.shutdown()


async def test_shutdown_forgets_the_pump(fake_bleak):
    device = await _initialized()
    await device.pump_on()
    await device.shutdown()
    assert device._pump_percent == 0
