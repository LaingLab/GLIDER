"""DataRecorder expands devices that declare sub-columns.

A device may contribute several CSV columns by returning names from
``state_columns()`` (see ``BaseDevice.state_columns``). Devices that
return ``None`` — every device that existed before the hook — must keep
producing exactly the column they always did.
"""

from __future__ import annotations

from glider.core.data_recorder import DataRecorder


class _FakeDevice:
    """Minimal stand-in: a controllable device_type and no board.

    The real ``MultiDevice`` fake in ``tests/unit/hal/test_state_columns.py``
    needs a board and hard-codes its ``device_type``; these tests need to
    vary both, so they use a local fake keyed under an arbitrary device id.
    """

    def __init__(self, device_type: str, columns: list[str] | None = None):
        self.device_type = device_type
        self._columns = columns

    def state_columns(self) -> list[str] | None:
        return self._columns


class _LegacyDevice:
    """A device predating the hook: no ``state_columns`` attribute at all."""

    def __init__(self, device_type: str):
        self.device_type = device_type


class _RaisingDevice:
    """A device whose ``state_columns()`` misbehaves."""

    device_type = "broken"

    def state_columns(self) -> list[str] | None:
        raise RuntimeError("boom")


class _FakeHardwareManager:
    def __init__(self, devices):
        self.devices = devices


def _recorder_for(devices) -> DataRecorder:
    """Build a recorder over a fixed device mapping."""
    return DataRecorder(_FakeHardwareManager(devices))


def test_single_column_device_unchanged():
    """A device returning None keeps the existing column name."""
    recorder = _recorder_for({"lever": _FakeDevice("button")})
    assert recorder._get_device_columns() == ["lever:button"]


def test_multi_column_device_expands():
    """A device declaring sub-columns emits one column per name."""
    recorder = _recorder_for({"lick": _FakeDevice("harp", columns=["state", "count", "last_ms"])})
    assert recorder._get_device_columns() == [
        "lick:state",
        "lick:count",
        "lick:last_ms",
    ]


def test_mixed_devices_keep_order():
    """Single- and multi-column devices coexist in declaration order."""
    recorder = _recorder_for(
        {
            "lever": _FakeDevice("button"),
            "lick": _FakeDevice("harp", columns=["state", "count"]),
        }
    )
    assert recorder._get_device_columns() == [
        "lever:button",
        "lick:state",
        "lick:count",
    ]


def test_device_without_state_columns_attribute():
    """A device that never heard of the hook still gets its column."""
    recorder = _recorder_for({"relay": _LegacyDevice("digital_output")})
    assert recorder._get_device_columns() == ["relay:digital_output"]


def test_failing_state_columns_falls_back_to_device_type():
    """A raising state_columns() degrades to the single-column name."""
    recorder = _recorder_for({"flaky": _RaisingDevice()})
    assert recorder._get_device_columns() == ["flaky:broken"]
