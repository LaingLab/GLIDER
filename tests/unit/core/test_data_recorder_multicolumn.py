"""DataRecorder expands devices that declare sub-columns.

A device may contribute several CSV columns by returning names from
``state_columns()`` (see ``BaseDevice.state_columns``). Devices that
return ``None`` — every device that existed before the hook — must keep
producing exactly the column they always did.
"""

from __future__ import annotations

import csv
import time

import pytest

from glider.analysis._io import parse_csv
from glider.core.data_recorder import DataRecorder


class _FakeDevice:
    """Minimal stand-in: a controllable device_type and no board.

    The real ``MultiDevice`` fake in ``tests/unit/hal/test_state_columns.py``
    needs a board and hard-codes its ``device_type``; these tests need to
    vary both, so they use a local fake keyed under an arbitrary device id.
    """

    def __init__(self, device_type: str, columns: list[str] | None = None, state=None):
        self.device_type = device_type
        self._columns = columns
        # Deliberately *not* named ``_state`` or ``_value``:
        # ``_read_device_state`` probes those attribute names directly, and
        # this fake must be read through ``get_state()``.
        self._payload = state

    def state_columns(self) -> list[str] | None:
        return self._columns

    async def get_state(self):
        return self._payload


class _LegacyDevice:
    """A device predating the hook: no ``state_columns`` attribute at all."""

    def __init__(self, device_type: str):
        self.device_type = device_type


class _RaisingDevice:
    """A device whose ``state_columns()`` misbehaves."""

    device_type = "broken"

    def state_columns(self) -> list[str] | None:
        raise RuntimeError("boom")


class _ShadowedStateDevice:
    """A multi-column device that also happens to carry a scalar ``_state``.

    No device does this today, but nothing stops one: ``_state`` is a
    perfectly ordinary name for an internal cache. The recorder must go by
    the declared ``state_columns()`` contract rather than by which
    attribute it happens to probe first.
    """

    device_type = "harp"

    def __init__(self):
        self._state = 0  # internal scalar; must never reach the CSV

    def state_columns(self) -> list[str] | None:
        return ["state"]

    async def get_state(self):
        return {"state": 1}


class _FakeHardwareManager:
    def __init__(self, devices):
        self.devices = devices
        self.boards: dict = {}


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


# --- Malformed state_columns() returns ---------------------------------
#
# A device may come from a third-party package (the Harp driver does), so
# core validates the value rather than trusting anything that didn't
# raise. The bare-string case is the dangerous one: a str is iterable, so
# "state" would silently expand into one column per character.


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("state", id="bare_string"),
        pytest.param([1, 2], id="non_string_entries"),
        pytest.param(["a", "a"], id="duplicate_names"),
        pytest.param(["a", ""], id="empty_name"),
    ],
)
def test_malformed_state_columns_degrades_to_single_column(bad):
    """Anything but a list of unique non-empty strings is refused."""
    recorder = _recorder_for({"lick": _FakeDevice("harp", columns=bad)})
    assert recorder._get_device_columns() == ["lick:harp"]
    assert recorder._degraded_devices == ["lick"]


def test_empty_state_columns_falls_back_without_a_warning():
    """``[]`` means "no sub-columns", so the single-column path applies.

    The contract says a single-column device returns ``None``, never
    ``[]``, but ``[]`` is unambiguous and harmless — it asks for nothing.
    Treating it as degraded would put a warning in the CSV for a device
    that is in fact recording its scalar fine.
    """
    recorder = _recorder_for({"lick": _FakeDevice("harp", columns=[])})
    assert recorder._get_device_columns() == ["lick:harp"]
    assert recorder._degraded_devices == []


# --- Row values --------------------------------------------------------


async def _device_cells(recorder: DataRecorder) -> list[str]:
    """Build one row and return just its device cells.

    ``_build_row`` prefixes four fixed cells (frame, timestamp,
    elapsed_ms, flow_elapsed_ms) and appends zone cells; these recorders
    have no zone configuration, so everything past the fourth cell is a
    device cell.
    """
    recorder._device_columns = recorder._get_device_columns()
    row = await recorder._build_row(None, "2026-01-01T00:00:00", 0.0)
    return row[4:]


@pytest.mark.asyncio
async def test_multi_column_state_lands_in_matching_cells():
    """Each sub-column cell gets its own key from the state dict."""
    recorder = _recorder_for(
        {"lick": _FakeDevice("harp", columns=["state", "count"], state={"state": 1, "count": 3})}
    )
    assert await _device_cells(recorder) == ["1", "3"]


@pytest.mark.asyncio
async def test_missing_sub_key_writes_an_empty_cell():
    """A key the device didn't report is empty, not a crash or a repeat."""
    recorder = _recorder_for(
        {"lick": _FakeDevice("harp", columns=["state", "count"], state={"state": 0})}
    )
    assert await _device_cells(recorder) == ["0", ""]


@pytest.mark.asyncio
async def test_single_column_device_still_writes_its_scalar():
    """The pre-existing single-column path is untouched by the dict branch."""
    recorder = _recorder_for({"lever": _FakeDevice("button", state=True)})
    assert await _device_cells(recorder) == ["1"]


@pytest.mark.asyncio
async def test_state_attribute_does_not_shadow_get_state_for_multi_column():
    """A multi-column device is read through get_state(), never ``_state``.

    ``_read_device_state`` probes ``_state`` before ``get_state()``, so
    without an explicit guard a device holding both would hand the
    recorder a scalar and every one of its cells would come out empty.
    """
    device = _ShadowedStateDevice()
    recorder = _recorder_for({"lick": device})
    assert await recorder._read_device_state(device) == {"state": 1}
    assert await _device_cells(recorder) == ["1"]


# --- Degraded-device metadata ------------------------------------------


def _metadata_lines(recorder: DataRecorder, tmp_path) -> list[str]:
    """Write just the metadata block and return the file's lines."""
    path = tmp_path / "metadata.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        recorder._writer = csv.writer(fh)
        recorder._write_metadata("degraded_test")
    recorder._writer = None
    return path.read_text(encoding="utf-8").splitlines()


def test_degraded_device_is_reported_in_metadata(tmp_path):
    """A device whose state_columns() raised records nothing all session.

    The only signal today is a log line nobody reads during a run, so the
    CSV itself has to say so.
    """
    recorder = _recorder_for({"flaky": _RaisingDevice()})
    lines = _metadata_lines(recorder, tmp_path)

    warnings = [line for line in lines if line.startswith("# WARNING")]
    assert warnings == ["# WARNING,flaky,state_columns() failed; recorded as single column"], lines


def test_healthy_devices_produce_no_warning(tmp_path):
    """Nothing is emitted for devices that behaved."""
    recorder = _recorder_for(
        {
            "lever": _FakeDevice("button"),
            "lick": _FakeDevice("harp", columns=["state", "count"]),
        }
    )
    assert [line for line in _metadata_lines(recorder, tmp_path) if "WARNING" in line] == []


def test_warning_row_survives_the_analysis_reader(tmp_path):
    """The real reader must still parse a file carrying a warning row.

    Asserting on raw line text only proves the string was written; this
    pins the contract that actually matters. The device id here contains
    a comma — ids are free-text, and a comma in an interpolated cell
    would make csv quote it, pushing the quote to the front of the line
    where it stops being a comment row.
    """
    recorder = _recorder_for({"fl,aky": _RaisingDevice(), "lever": _FakeDevice("button")})
    path = tmp_path / "metadata.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        recorder._writer = csv.writer(fh)
        recorder._write_metadata("degraded_test")
    recorder._writer = None

    warning_line = next(
        line for line in path.read_text(encoding="utf-8").splitlines() if "WARNING" in line
    )
    assert warning_line.startswith("#"), warning_line

    metadata, df = parse_csv(path)
    # `_parse_metadata_header` splits on the first comma rather than
    # csv-parsing, so the quoted id cell keeps its quotes here; what
    # matters is that the row was read as metadata at all, and that it
    # names the device.
    assert "fl,aky" in metadata["WARNING"]
    assert "recorded as single column" in metadata["WARNING"]
    assert metadata["Experiment Name"] == "degraded_test"
    # The header row survived: comment rows were skipped, not consumed.
    assert list(df.columns) == ["frame", "timestamp", "elapsed_ms", "flow_elapsed_ms"] + [
        "fl,aky:broken",
        "lever:button",
    ]


# --- Device-reported recording warnings --------------------------------


class _WarningDevice(_FakeDevice):
    """A device that knows one of its own columns can never change."""

    def __init__(self, warnings):
        super().__init__("harp", columns=["lick_state", "lick_count"], state=None)
        self._warnings = warnings

    def recording_warnings(self) -> list[str]:
        return self._warnings


class _BadWarningDevice(_FakeDevice):
    """A third-party device whose warnings hook misbehaves.

    A bare string is iterable, so an unvalidated hook would write it out
    one character per row and bury the metadata block.
    """

    def __init__(self, warnings):
        super().__init__("harp", columns=["a"], state=None)
        self._warnings = warnings

    def recording_warnings(self):
        if isinstance(self._warnings, Exception):
            raise self._warnings
        return self._warnings


def test_a_device_reported_warning_reaches_the_csv(tmp_path):
    """``derive`` logs a recorded register that emits no events, and nobody
    reads a log during an overnight run. The recording has to carry it."""
    recorder = _recorder_for({"harp1": _WarningDevice(["register Foo will never change"])})
    lines = _metadata_lines(recorder, tmp_path)

    assert [line for line in lines if line.startswith("# WARNING")] == [
        "# WARNING,harp1,register Foo will never change"
    ], lines


def test_several_warnings_from_one_device_each_get_a_row(tmp_path):
    recorder = _recorder_for({"harp1": _WarningDevice(["first thing", "second thing"])})
    lines = _metadata_lines(recorder, tmp_path)
    assert [line for line in lines if line.startswith("# WARNING")] == [
        "# WARNING,harp1,first thing",
        "# WARNING,harp1,second thing",
    ], lines


def test_a_device_reported_warning_survives_the_analysis_reader(tmp_path):
    """Cell 0 is a fixed literal for the same reason the degraded row's is:
    csv quotes any cell holding a comma, and a quote at the front of the line
    stops it being a comment row."""
    recorder = _recorder_for({"ha,rp": _WarningDevice(["Foo, which emits nothing, is recorded"])})
    path = tmp_path / "metadata.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        recorder._writer = csv.writer(fh)
        recorder._write_metadata("warning_test")
    recorder._writer = None

    warning_line = next(
        line for line in path.read_text(encoding="utf-8").splitlines() if "WARNING" in line
    )
    assert warning_line.startswith("#"), warning_line

    metadata, df = parse_csv(path)
    assert "ha,rp" in metadata["WARNING"]
    assert metadata["Experiment Name"] == "warning_test"
    assert list(df.columns) == ["frame", "timestamp", "elapsed_ms", "flow_elapsed_ms"] + [
        "ha,rp:lick_state",
        "ha,rp:lick_count",
    ]


@pytest.mark.parametrize(
    "warnings",
    ["a bare string", ["", "ok"], [123], RuntimeError("boom"), None],
    ids=["string", "empty-entry", "non-string", "raises", "none"],
)
def test_a_malformed_warnings_hook_costs_nothing(tmp_path, warnings):
    """Devices arrive from third-party packages. A hook that misbehaves must
    not take the metadata block with it."""
    recorder = _recorder_for({"harp1": _BadWarningDevice(warnings)})
    lines = _metadata_lines(recorder, tmp_path)
    assert [line for line in lines if "WARNING" in line] == [], lines
    assert any(line.startswith("# Experiment Name") for line in lines)


def test_a_device_without_the_hook_is_left_alone(tmp_path):
    """Every device that predates the hook."""
    recorder = _recorder_for({"lever": _LegacyDevice("button")})
    assert [line for line in _metadata_lines(recorder, tmp_path) if "WARNING" in line] == []


# --- Warnings a device only learns during the run -----------------------


class _LateWarningDevice(_FakeDevice):
    """A device that learns something is wrong after recording has started.

    The motivating case is a serial link that dies mid-session: the header
    block was written before the first row, so it cannot carry this, and the
    resulting CSV is the plausible kind of wrong — a column that repeats one
    value for three hours reads exactly like a subject that stopped
    responding.
    """

    def __init__(self, warnings_before, warnings_after):
        super().__init__("harp", columns=["lick_state"], state={"lick_state": 1})
        self._warnings = list(warnings_before)
        self._after = list(warnings_after)

    def fail(self):
        self._warnings = self._after

    def recording_warnings(self) -> list[str]:
        return list(self._warnings)


async def test_a_warning_raised_during_the_run_reaches_the_csv(tmp_path):
    device = _LateWarningDevice([], ["the reader thread stopped; rows after this are not readings"])
    recorder = _recorder_for({"harp1": device})
    recorder.set_camera_driven(True)
    recorder.set_output_directory(tmp_path)

    path = await recorder.start("late")
    await recorder.record_at_frame(0, time.time())
    device.fail()  # the cable is pulled here
    await recorder.record_at_frame(1, time.time())
    await recorder.stop()

    warnings = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("# WARN")
    ]
    assert warnings == [
        "# WARNING,harp1,the reader thread stopped; rows after this are not readings"
    ], path.read_text(encoding="utf-8")


async def test_a_warning_already_in_the_header_is_not_repeated_in_the_footer(tmp_path):
    """A static warning is reported once, not once at each end of the file."""
    device = _LateWarningDevice(["a column that cannot change"], ["a column that cannot change"])
    recorder = _recorder_for({"harp1": device})
    recorder.set_camera_driven(True)
    recorder.set_output_directory(tmp_path)

    path = await recorder.start("static")
    await recorder.record_at_frame(0, time.time())
    await recorder.stop()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [ln for ln in lines if ln.startswith("# WARN")] == [
        "# WARNING,harp1,a column that cannot change"
    ], lines


async def test_the_footer_warning_does_not_disturb_the_analysis_reader(tmp_path):
    """It lands after the data rows, so the reader must skip it as a comment
    rather than parse it as one."""
    device = _LateWarningDevice([], ["the reader thread stopped mid-run"])
    recorder = _recorder_for({"harp1": device})
    recorder.set_camera_driven(True)
    recorder.set_output_directory(tmp_path)

    path = await recorder.start("late")
    await recorder.record_at_frame(0, time.time())
    device.fail()
    await recorder.stop()

    _, df = parse_csv(path)
    assert list(df.columns)[-1] == "harp1:lick_state"
    assert len(df) >= 1
    assert "the reader thread stopped mid-run" in path.read_text(encoding="utf-8")
