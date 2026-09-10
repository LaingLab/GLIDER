"""Tests for the Maimu BLE stimulator device.

bleak is lazy-imported inside the connect path, so these inject a fake ``bleak``
module with an async fake client -- the same shape as test_ble_device_full.py.
The fake records an ordered event log, because the behaviour that matters most
here is that shutdown sends "off" *before* it drops the link.
"""

import asyncio
import sys

import pytest

from glider.hal.base_device import DeviceConfig
from glider_maimu.device import (
    DEFAULT_SERVICE_UUID,
    DEFAULT_WRITE_CHAR_UUID,
    MaimuDevice,
)


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"


class _FakeClient:
    """Minimal async BleakClient stand-in with an ordered event log."""

    def __init__(self, address):
        self.address = address
        self.is_connected = False
        self.events = []  # ("write", b"...") / ("disconnect",) in call order
        self.write_error = None
        self.write_hangs = False

    @property
    def written(self):
        return [payload for kind, payload in self.events if kind == "write"]

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.events.append(("disconnect", None))
        self.is_connected = False

    async def write_gatt_char(self, char, data, response=False):
        if self.write_hangs:
            await asyncio.sleep(30)
        if self.write_error:
            raise self.write_error
        self.events.append(("write", bytes(data)))
        self.last_write_char = char
        self.last_response = response


@pytest.fixture
def fake_bleak(monkeypatch):
    """Inject a fake ``bleak``; yield (module, created-client holder)."""
    from unittest.mock import MagicMock

    created = {}

    def make_client(address, *a, **k):
        # A failed write makes BLEDevice._with_retry drop the client and
        # reconnect, so faults have to live on the holder, not on one instance
        # -- otherwise the retry silently succeeds on a fresh client.
        client = _FakeClient(address)
        client.write_error = created.get("write_error")
        client.write_hangs = created.get("write_hangs", False)
        created["client"] = client
        created.setdefault("clients", []).append(client)
        return client

    module = MagicMock(name="bleak")
    module.BleakClient = make_client

    resolved = MagicMock()
    resolved.address = "AA:BB:CC:DD:EE:FF"

    async def find_device_by_name(name, timeout=8.0):
        created["scanned_name"] = name
        return resolved

    module.BleakScanner.find_device_by_name = find_device_by_name

    monkeypatch.setitem(sys.modules, "bleak", module)
    return module, created


def _fault(created, **kwargs):
    """Arm a fault on every client -- the live one and any reconnect after it."""
    created.update(kwargs)
    for client in created.get("clients", []):
        for key, value in kwargs.items():
            setattr(client, key, value)


def _make_device(settings=None, name=None):
    config = DeviceConfig(pins={}, settings=dict(settings or {}))
    return MaimuDevice(_FakeBoard(), config, name=name or "maimu")


async def _initialized(settings=None, *, name=None):
    device = _make_device(settings or {"address": "11:22:33:44:55:66"}, name=name)
    await device.initialize()
    return device


# --- identity / config --------------------------------------------------------


def test_device_type_is_maimu():
    assert _make_device().device_type == "Maimu"


def test_actions_are_the_command_set():
    assert set(_make_device().actions) == {"on", "off", "pulse", "write"}


def test_read_is_not_offered():
    """The peripheral has no read characteristic; offering it would only fail."""
    assert "read" not in _make_device().actions


def test_requires_no_pins():
    assert _make_device().required_pins == []


def test_uuid_defaults_are_applied():
    device = _make_device({"address": "x"})
    assert device._write_char == DEFAULT_WRITE_CHAR_UUID
    assert device._service_uuid == DEFAULT_SERVICE_UUID


def test_uuid_defaults_are_persisted_to_settings():
    """A file saved from this device carries the UUIDs, not just the runtime cache."""
    device = _make_device({"address": "x"})
    assert device._config.settings["write_char_uuid"] == DEFAULT_WRITE_CHAR_UUID
    assert device._config.settings["service_uuid"] == DEFAULT_SERVICE_UUID


def test_explicit_uuid_overrides_default():
    device = _make_device({"address": "x", "write_char_uuid": "custom-uuid"})
    assert device._write_char == "custom-uuid"


def test_from_dict_applies_defaults():
    device = MaimuDevice.from_dict(
        {"name": "stim", "config": {"settings": {"address": "x"}}}, _FakeBoard()
    )
    assert isinstance(device, MaimuDevice)
    assert device._write_char == DEFAULT_WRITE_CHAR_UUID


# --- lifecycle ----------------------------------------------------------------


async def test_initialize_requires_address_or_name():
    device = _make_device({})
    with pytest.raises(ValueError, match="address"):
        await device.initialize()


async def test_initialize_connects(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    assert device.is_initialized
    assert created["client"].is_connected


async def test_address_resolved_from_advertised_name(fake_bleak):
    _module, created = fake_bleak
    await _initialized({"name": "Maimu-01"})
    assert created["scanned_name"] == "Maimu-01"
    assert created["client"].address == "AA:BB:CC:DD:EE:FF"


# --- commands -----------------------------------------------------------------


async def test_on_writes_on(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    await device.on()
    assert created["client"].written == [b"on"]


async def test_off_writes_off(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    await device.off()
    assert created["client"].written == [b"off"]


async def test_pulse_writes_all_four_fields(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    await device.pulse(50, 4, 4, 100)
    assert created["client"].written == [b"50,4,4,100"]


async def test_pulse_writes_whole_numbers_from_floats(fake_bleak):
    """Property spinboxes and math nodes deliver floats; "50.0,4.0,..." would be
    rejected by the firmware's strict parser."""
    _module, created = fake_bleak
    device = await _initialized()
    await device.pulse(50.0, 4.0, 4.0, 100.0)
    assert created["client"].written == [b"50,4,4,100"]


async def test_count_of_zero_is_allowed(fake_bleak):
    """0 means run until stopped -- the only field with a floor of 0."""
    _module, created = fake_bleak
    device = await _initialized()
    await device.pulse(50, 4, 0, 100)
    assert created["client"].written == [b"50,4,0,100"]


async def test_intensity_of_zero_is_allowed(fake_bleak):
    """Armed but dark is a legitimate control condition, not an error."""
    _module, created = fake_bleak
    device = await _initialized()
    await device.pulse(50, 4, 4, 0)
    assert created["client"].written == [b"50,4,4,0"]


async def test_command_goes_to_the_configured_characteristic(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    await device.on()
    assert created["client"].last_write_char == DEFAULT_WRITE_CHAR_UUID


async def test_write_uses_response_for_a_write_only_characteristic(fake_bleak):
    """The Maimu declares WRITE only, so writes must be acknowledged."""
    _module, created = fake_bleak
    device = await _initialized()
    client = created["client"]
    char = type("Char", (), {"properties": ["write"]})()
    client.services = type("Svc", (), {"get_characteristic": lambda self, u: char})()
    await device.on()
    assert client.last_response is True


@pytest.mark.parametrize("bad", [0, -1, 0.5, "fast", None])
async def test_pulse_rejects_invalid_period(fake_bleak, bad):
    device = await _initialized()
    with pytest.raises(ValueError, match="period_ms"):
        await device.pulse(bad, 4, 4, 100)


@pytest.mark.parametrize("bad", [0, -1, 2.5, "wide", None])
async def test_pulse_rejects_invalid_width(fake_bleak, bad):
    device = await _initialized()
    with pytest.raises(ValueError, match="pulse_width_ms"):
        await device.pulse(50, bad, 4, 100)


@pytest.mark.parametrize("bad", [-1, 1.5, "many", None, 65536])
async def test_pulse_rejects_invalid_count(fake_bleak, bad):
    device = await _initialized()
    with pytest.raises(ValueError, match="count"):
        await device.pulse(50, 4, bad, 100)


@pytest.mark.parametrize("bad", [-1, 0.5, "bright", None, 101])
async def test_pulse_rejects_invalid_intensity(fake_bleak, bad):
    device = await _initialized()
    with pytest.raises(ValueError, match="intensity_pct"):
        await device.pulse(50, 4, 4, bad)


async def test_pulse_rejects_a_width_wider_than_its_period(fake_bleak):
    """The firmware rejects it too, silently. Raising here is what makes it
    visible to the researcher instead of a device that did nothing."""
    device = await _initialized()
    with pytest.raises(ValueError, match="pulse_width_ms"):
        await device.pulse(50, 51, 4, 100)


async def test_a_width_equal_to_its_period_is_allowed(fake_bleak):
    """width == period is how continuous-at-intensity is spelled."""
    _module, created = fake_bleak
    device = await _initialized()
    await device.pulse(100, 100, 0, 40)
    assert created["client"].written == [b"100,100,0,40"]


async def test_a_width_equal_to_its_period_needs_count_zero(fake_bleak):
    """500,500,1,40 reads as one bounded 500 ms pulse and is not.

    A train with no gap has nothing to end on, so the firmware would light the
    implant for the rest of the session. Continuous light has exactly one
    spelling -- count = 0 -- and this is the guard that enforces it.
    """
    _module, created = fake_bleak
    device = await _initialized()
    with pytest.raises(ValueError, match="pulse_width_ms"):
        await device.pulse(500, 500, 1, 40)
    assert created["client"].written == []


async def test_a_bounded_single_pulse_is_still_expressible(fake_bleak):
    """Nothing is lost: period > width with count = 1 is one bounded pulse."""
    _module, created = fake_bleak
    device = await _initialized()
    await device.pulse(1000, 500, 1, 40)
    assert created["client"].written == [b"1000,500,1,40"]


async def test_rejected_pulse_writes_nothing(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    with pytest.raises(ValueError):
        await device.pulse(0, 4, 4, 100)
    assert created["client"].written == []


async def test_execute_action_routes_to_pulse(fake_bleak):
    """The node calls through execute_action, not the method directly."""
    _module, created = fake_bleak
    device = await _initialized()
    await device.execute_action("pulse", 250, 5, 10, 80)
    assert created["client"].written == [b"250,5,10,80"]


# --- shutdown -----------------------------------------------------------------


async def test_shutdown_sends_off_before_disconnecting(fake_bleak):
    """The firmware runs a train autonomously -- a bare disconnect would leave
    the stimulator running."""
    _module, created = fake_bleak
    device = await _initialized()
    await device.pulse(50, 4, 0, 100)
    client = created["client"]

    await device.shutdown()

    assert client.events == [
        ("write", b"50,4,0,100"),
        ("write", b"off"),
        ("disconnect", None),
    ]
    assert not client.is_connected
    assert not device.is_initialized


async def test_shutdown_disconnects_even_when_off_fails(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    _fault(created, write_error=RuntimeError("peripheral vanished"))

    await device.shutdown()

    # The live client is disconnected even though no "off" ever landed.
    assert ("disconnect", None) in created["client"].events
    assert created["client"].written == []
    assert not device.is_initialized
    assert device._client is None


async def test_shutdown_disconnects_even_when_off_hangs(fake_bleak, monkeypatch):
    """A wedged write must not eat the emergency-stop budget."""
    import glider_maimu.device as maimu_module

    monkeypatch.setattr(maimu_module, "OFF_ON_SHUTDOWN_S", 0.05)
    _module, created = fake_bleak
    device = await _initialized()
    _fault(created, write_hangs=True)

    await asyncio.wait_for(device.shutdown(), timeout=2.0)

    assert ("disconnect", None) in created["client"].events
    assert not device.is_initialized
    assert device._client is None


async def test_shutdown_before_initialize_is_safe():
    device = _make_device({"address": "x"})
    await device.shutdown()  # must not raise, and must not try to write
    assert not device.is_initialized


async def test_a_shutdown_with_no_link_says_so(fake_bleak, caplog):
    """Mid-reconnect there is no client to write 'off' over.

    The warning already existed for a write that *failed*; a write that never
    happened is the same news to the operator -- a stimulator left in an
    unknown state -- and used to be reported as nothing at all.
    """
    import logging

    device = await _initialized()
    device._client = None  # what a reconnect attempt in flight looks like

    with caplog.at_level(logging.WARNING, logger="glider_maimu.device"):
        await device.shutdown()

    assert any("may still be running" in r.message for r in caplog.records)
    assert not device.is_initialized


async def test_a_shutdown_before_initialize_says_nothing(caplog):
    """A device nothing has ever commanded is not news."""
    import logging

    device = _make_device({"address": "x"})

    with caplog.at_level(logging.WARNING, logger="glider_maimu.device"):
        await device.shutdown()

    assert caplog.records == []
