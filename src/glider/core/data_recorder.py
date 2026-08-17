"""
Data Recorder - Records experiment data to CSV files.

Provides timestamped logging of all device states during experiment execution.
"""

import asyncio
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from glider.core.experiment_session import ExperimentSession
    from glider.core.hardware_manager import HardwareManager
    from glider.vision.cv_processor import CVProcessor
    from glider.vision.zones import ZoneConfiguration

logger = logging.getLogger(__name__)


class DataRecorder:
    """
    Records experiment data to CSV files.

    Creates timestamped CSV files with device states sampled at regular intervals.
    """

    DEFAULT_SAMPLE_INTERVAL = 0.1  # 100ms default sampling

    def __init__(
        self,
        hardware_manager: "HardwareManager",
        output_dir: Path | None = None,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    ):
        """
        Initialize the data recorder.

        Args:
            hardware_manager: The hardware manager to read device states from
            output_dir: Directory to save CSV files (defaults to current directory)
            sample_interval: Time between samples in seconds
        """
        self._hardware_manager = hardware_manager
        self._output_dir = output_dir or Path.cwd()
        self._sample_interval = sample_interval

        self._recording = False
        self._file: Any | None = None
        self._writer: csv.writer | None = None
        self._file_path: Path | None = None
        self._start_time: datetime | None = None
        # Unix-time origin against which all `elapsed_ms` cells are
        # computed. If `set_session_epoch` is called before `start()`,
        # that value is used (so every recorder in the session shares t=0
        # and elapsed_ms is joinable across files). Otherwise it falls
        # back to this recorder's own start_time on `start()`.
        self._start_timestamp: float = 0.0
        self._session_epoch_override: float | None = None
        # Flow-relative timing anchor. See tracking_logger.TrackingDataLogger
        # for the full rationale: ``elapsed_ms`` measures time from the
        # recorder's session epoch (joinable across recorders), but the
        # analyst wants time from the *flow* boundary so sensor traces
        # line up at t=0 with StartExperiment. Set once by GliderCore the
        # instant flow_engine.start() returns; rows before are empty in
        # the flow_elapsed_ms column, rows after carry the offset.
        self._flow_anchor: float | None = None
        self._sample_task: asyncio.Task | None = None
        self._device_columns: list[str] = []
        # Devices whose state_columns() raised or returned something
        # malformed. They fall back to a single column, and
        # _read_device_state hits the same failure on every sample and
        # yields None — so the device records nothing but empty cells for
        # the whole session. _write_metadata says so in the CSV itself.
        self._degraded_devices: list[str] = []
        self._zone_columns: list[str] = []
        self._zone_config: ZoneConfiguration | None = None
        self._cv_processor: CVProcessor | None = None
        # When True, the recorder is driven by per-frame record_at_frame() calls
        # from the camera pipeline; the internal asyncio.sleep sampling loop is
        # not created. Must be set before start() to take effect.
        self._camera_driven: bool = False

    @property
    def is_recording(self) -> bool:
        """Whether recording is currently active."""
        return self._recording

    @property
    def file_path(self) -> Path | None:
        """Path to the current recording file."""
        return self._file_path

    @property
    def sample_interval(self) -> float:
        """Current sample interval in seconds."""
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, value: float) -> None:
        """Set the sample interval (minimum 0.01 seconds)."""
        self._sample_interval = max(0.01, value)

    @property
    def is_camera_driven(self) -> bool:
        """Whether this recorder is driven by camera frame ticks."""
        return self._camera_driven

    def set_session_epoch(self, epoch: float) -> None:
        """
        Anchor this recorder's ``elapsed_ms`` column to a shared Unix
        timestamp instead of its own ``_start_time``.

        Used by ``GliderCore.start_experiment`` to give every recorder
        (data, events, tracking) the same t=0 so the elapsed_ms columns
        are directly comparable across files. Must be called before
        ``start()``.

        Args:
            epoch: Unix timestamp (seconds, float) to use as t=0.
        """
        if self._recording:
            logger.warning("set_session_epoch called while recording; takes effect on next start()")
        self._session_epoch_override = float(epoch)

    def set_flow_anchor(self, timestamp: float) -> None:
        """
        Anchor ``flow_elapsed_ms`` to a flow-boundary wall-clock timestamp.

        Mirrors ``TrackingDataLogger.set_flow_anchor`` so the device-state
        CSV carries the same flow-relative time column as the tracking
        CSV — analysts can plot sensor traces and tracking traces against
        the same t=0 (flow start) without computing per-file offsets.
        Rows written before this call get an empty ``flow_elapsed_ms``
        cell; rows after get ``(timestamp - anchor) * 1000``.

        Args:
            timestamp: Unix timestamp (seconds, float) marking flow t=0.
        """
        self._flow_anchor = float(timestamp)

    def set_camera_driven(self, enabled: bool) -> None:
        """
        Configure whether this recorder writes one row per camera frame
        (driven by record_at_frame()) instead of running its own timer loop.

        Must be set before start() to take effect. When enabled, the internal
        sampling task is not created and rows are only written when the
        camera pipeline calls record_at_frame(frame_no, frame_ts).

        Args:
            enabled: True to use camera-frame ticks; False (default) to use
                the timer loop with sample_interval.
        """
        if self._recording:
            logger.warning("set_camera_driven called while recording; takes effect on next start()")
        self._camera_driven = enabled

    def _generate_filename(self, experiment_name: str) -> str:
        """Generate a filename with experiment name, date, and time."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize experiment name for filename
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in experiment_name)
        safe_name = safe_name.strip().replace(" ", "_")
        if not safe_name:
            safe_name = "experiment"
        return f"{safe_name}_{timestamp}.csv"

    def set_output_directory(self, path: Path) -> None:
        """Set the directory for saving CSV recording files.

        Args:
            path: Directory path for output files
        """
        self._output_dir = path

    def set_zone_configuration(self, zone_config: "ZoneConfiguration") -> None:
        """Set zone configuration for zone state logging."""
        self._zone_config = zone_config

    def set_cv_processor(self, cv_processor: "CVProcessor") -> None:
        """Set CV processor for reading zone states."""
        self._cv_processor = cv_processor

    def _get_device_columns(self) -> list[str]:
        """
        Get list of device column names.

        A device may contribute several columns by returning names from
        ``state_columns()``; those become ``{device_id}:{name}``. Devices
        returning ``None`` keep the historical ``{device_id}:{device_type}``.

        The return value is validated, not merely trusted not to raise:
        devices can arrive from third-party packages, and a bare string
        (say ``"state"``) is iterable, so it would otherwise expand into
        one column per character. Anything that isn't ``None`` or a list
        of unique non-empty strings is refused.

        Also refreshes ``_degraded_devices`` with the ids whose
        ``state_columns()`` raised or misbehaved, for ``_write_metadata``
        to report.
        """
        columns = []
        self._degraded_devices = []
        for device_id, device in self._hardware_manager.devices.items():
            sub_columns = None
            if hasattr(device, "state_columns"):
                try:
                    sub_columns = device.state_columns()
                except Exception:
                    logger.exception("state_columns() failed for device %s", device_id)
                    sub_columns = None
                    self._degraded_devices.append(device_id)
                if sub_columns is not None and (
                    not isinstance(sub_columns, list)
                    or not all(isinstance(n, str) and n for n in sub_columns)
                    or len(set(sub_columns)) != len(sub_columns)
                ):
                    logger.error(
                        "state_columns() returned %r for device %s", sub_columns, device_id
                    )
                    sub_columns = None
                    self._degraded_devices.append(device_id)
            if sub_columns:
                columns.extend(f"{device_id}:{name}" for name in sub_columns)
            else:
                device_type = getattr(device, "device_type", "unknown")
                columns.append(f"{device_id}:{device_type}")
        return columns

    def _get_zone_columns(self) -> list[str]:
        """Get list of zone column names."""
        if not self._zone_config or not self._zone_config.zones:
            return []
        return [f"zone:{zone.name}" for zone in self._zone_config.zones]

    async def _read_device_state(self, device) -> Any:
        """Read the current state of a device."""
        try:
            # Multi-column devices must be read through get_state(), which
            # returns a dict; `_state` (if present) is a scalar and would
            # silently win the checks below.
            sub_columns = device.state_columns() if hasattr(device, "state_columns") else None
            if sub_columns and hasattr(device, "get_state"):
                return await device.get_state()

            # Try different methods to get device state
            if hasattr(device, "_state"):
                return device._state
            if hasattr(device, "get_state"):
                return await device.get_state()
            if hasattr(device, "read"):
                return await device.read()
            if hasattr(device, "_angle"):  # For ServoDevice
                return device._angle
            if hasattr(device, "_position"):  # For MotorGovernorDevice
                return device._position
            if hasattr(device, "_value"):
                return device._value
            return None
        except Exception as e:
            logger.debug(f"Could not read state for device: {e}")
            return None

    async def _sample_devices(self) -> dict[str, Any]:
        """Sample all device states."""
        states = {}
        for device_id, device in self._hardware_manager.devices.items():
            state = await self._read_device_state(device)
            states[device_id] = state
        return states

    def _write_metadata(
        self, experiment_name: str, session: Optional["ExperimentSession"] = None
    ) -> None:
        """Write metadata header to the CSV file."""
        if self._writer is None:
            return

        # Resolve columns up front: the device section below reports any
        # device that degraded while doing so, and the header row at the
        # bottom of this method consumes the result.
        self._device_columns = self._get_device_columns()
        self._zone_columns = self._get_zone_columns()

        # Write metadata section
        self._writer.writerow(["# GLIDER Experiment Data"])
        self._writer.writerow(["# Experiment Name", experiment_name])
        self._writer.writerow(
            ["# Start Time", self._start_time.isoformat() if self._start_time else ""]
        )
        self._writer.writerow(["# Sample Interval (s)", self._sample_interval])

        # Write experiment metadata if session available
        if session and session.metadata:
            metadata = session.metadata
            if metadata.protocol:
                self._writer.writerow(["# Protocol", metadata.protocol])
            if metadata.experiment_type:
                self._writer.writerow(["# Experiment Type", metadata.experiment_type])
            if metadata.experimenter:
                self._writer.writerow(["# Experimenter", metadata.experimenter])
            if metadata.lab:
                self._writer.writerow(["# Lab", metadata.lab])
            if metadata.project:
                self._writer.writerow(["# Project", metadata.project])

            # Write active subject info
            active_subject = metadata.get_active_subject()
            if active_subject:
                self._writer.writerow([])
                self._writer.writerow(["# Active Subject"])
                self._writer.writerow(["# Subject ID", active_subject.subject_id])
                if active_subject.name:
                    self._writer.writerow(["# Subject Name", active_subject.name])
                if active_subject.group:
                    self._writer.writerow(["# Group", active_subject.group])
                if active_subject.sex:
                    self._writer.writerow(["# Sex", active_subject.sex])
                if active_subject.age:
                    self._writer.writerow(["# Age", active_subject.age])
                if active_subject.weight:
                    self._writer.writerow(["# Weight", active_subject.weight])
                if active_subject.strain:
                    self._writer.writerow(["# Strain", active_subject.strain])
                if active_subject.solution:
                    self._writer.writerow(["# Solution", active_subject.solution])
                if active_subject.concentration:
                    self._writer.writerow(["# Concentration", active_subject.concentration])
                if active_subject.dose:
                    self._writer.writerow(["# Dose", active_subject.dose])
                if active_subject.route:
                    self._writer.writerow(["# Route", active_subject.route])

        self._writer.writerow([])

        # Write board info
        self._writer.writerow(["# Boards"])
        for board_id, board in self._hardware_manager.boards.items():
            board_type = getattr(board, "board_type", "unknown")
            connected = getattr(board, "is_connected", False)
            self._writer.writerow(
                ["#", board_id, board_type, "Connected" if connected else "Disconnected"]
            )
        self._writer.writerow([])

        # Write device info
        self._writer.writerow(["# Devices"])
        for device_id, device in self._hardware_manager.devices.items():
            device_type = getattr(device, "device_type", "unknown")
            board = getattr(device, "board", None)
            board_id = board.id if board else "none"
            config = getattr(device, "_config", None)
            pins = config.pins if config else {}
            pin_str = ", ".join(f"{k}={v}" for k, v in pins.items())
            self._writer.writerow(["#", device_id, device_type, f"board={board_id}", pin_str])
        # A device whose state_columns() failed is recorded as a single
        # column, and _read_device_state hits that same failure on every
        # sample and yields None — so the device writes nothing but empty
        # cells for the whole session. Say so here: the log line it also
        # emits is not something anyone watches mid-run.
        #
        # Cell 0 is a fixed literal, never interpolated. Readers key off
        # the leading `#` to skip comment rows, and csv quotes any cell
        # containing a comma — so an id like "fl,aky" in cell 0 would
        # push the quote to the front of the line and break them.
        for device_id in self._degraded_devices:
            self._writer.writerow(
                ["# WARNING", device_id, "state_columns() failed; recorded as single column"]
            )
        self._writer.writerow([])

        # Write column headers. `frame` is the canonical camera frame index
        # (matches TrackingDataLogger's `frame` column for one-key joins
        # against the video and tracking CSV). It is empty for rows written
        # by the timer-driven sampling loop (i.e., headless / no-camera
        # sessions, or before the first frame arrives). ``flow_elapsed_ms``
        # is empty until ``set_flow_anchor()`` is called (pre-flow
        # samples), then carries time-since-flow-start so this CSV's
        # sensor traces align at t=0 with the tracking CSV and runner.
        # (Both column lists were resolved at the top of this method.)
        headers = (
            ["frame", "timestamp", "elapsed_ms", "flow_elapsed_ms"]
            + self._device_columns
            + self._zone_columns
        )
        self._writer.writerow(headers)

    async def start(
        self,
        experiment_name: str = "experiment",
        session: Optional["ExperimentSession"] = None,
    ) -> Path:
        """
        Start recording data.

        Args:
            experiment_name: Name of the experiment for the filename
            session: Optional experiment session for additional metadata

        Returns:
            Path to the created CSV file
        """
        if self._recording:
            logger.warning("Recording already in progress")
            return self._file_path

        # Generate filename and create file
        filename = self._generate_filename(experiment_name)
        self._file_path = self._output_dir / filename
        self._start_time = datetime.now()
        # elapsed_ms is computed against either a session-wide epoch
        # (so multiple recorders share t=0) or, as a fallback, this
        # recorder's own start_time. Both `_record_sample` (timer) and
        # `record_at_frame` (camera) use this single value.
        self._start_timestamp = (
            self._session_epoch_override
            if self._session_epoch_override is not None
            else self._start_time.timestamp()
        )
        # Reset the flow anchor so a previous run's flow boundary doesn't
        # leak into a new recording. GliderCore will call set_flow_anchor()
        # again once flow_engine.start() returns.
        self._flow_anchor = None

        # Open file and create CSV writer
        self._file = open(self._file_path, "w", newline="", encoding="utf-8")
        try:
            self._writer = csv.writer(self._file)

            # Write metadata and headers
            self._write_metadata(experiment_name, session)
            self._file.flush()

            # Start sampling task only when the recorder is timer-driven.
            # In camera-driven mode the per-frame record_at_frame() calls
            # from the camera pipeline are the sole source of rows, so we
            # skip creating the timer task entirely.
            self._recording = True
            if not self._camera_driven:
                from glider.core.async_utils import log_task_exception

                self._sample_task = asyncio.create_task(self._sampling_loop())
                self._sample_task.add_done_callback(log_task_exception)
        except Exception:
            self._file.close()
            self._file = None
            self._writer = None
            raise

        logger.info(f"Started recording to {self._file_path}")
        return self._file_path

    async def _sampling_loop(self) -> None:
        """Main sampling loop."""
        while self._recording:
            try:
                await self._record_sample()
            except Exception as e:
                logger.error(f"Error recording sample: {e}")

            await asyncio.sleep(self._sample_interval)

    async def _build_row(
        self,
        frame: int | None,
        iso_timestamp: str,
        elapsed_ms: float,
        flow_elapsed_cell: str = "",
    ) -> list[str]:
        """
        Build a CSV row for one device-state sample.

        Shared by the timer-driven path (`_record_sample`) and the
        frame-driven path (`record_at_frame`) so both write identical
        column layouts.

        Args:
            frame: Camera frame index, or None for timer-driven rows (which
                write an empty string in the `frame` column).
            iso_timestamp: ISO 8601 timestamp string for the row.
            elapsed_ms: Milliseconds since the recorder started.
            flow_elapsed_cell: Pre-formatted ``flow_elapsed_ms`` cell
                value. Empty string for samples taken before
                ``set_flow_anchor()`` is called; ``"{ms:.1f}"`` for
                samples after. Callers compute it because they have the
                raw timestamp; ``_build_row`` just injects.

        Returns:
            List of stringified cell values, in column order.
        """
        states = await self._sample_devices()
        row: list[str] = [
            "" if frame is None else str(frame),
            iso_timestamp,
            f"{elapsed_ms:.1f}",
            flow_elapsed_cell,
        ]

        # Device states (in column order). A column is either
        # "{device_id}:{device_type}" for single-column devices or
        # "{device_id}:{sub_column}" for multi-column ones; the dict branch
        # below distinguishes them by what get_state() returned.
        for col in self._device_columns:
            device_id, _, sub_key = col.partition(":")
            state = states.get(device_id)
            if isinstance(state, dict):
                state = state.get(sub_key)
            if state is None:
                row.append("")
            elif isinstance(state, bool):
                row.append("1" if state else "0")
            elif isinstance(state, float):
                row.append(f"{state:.4f}")
            else:
                row.append(str(state))

        # Zone states (in column order). Zones are pixel-space inferences,
        # so they're only meaningful when a CV processor + zone config are
        # both attached; otherwise we emit empty cells.
        zone_states = {}
        if self._cv_processor and self._zone_config:
            zone_states = self._cv_processor.get_zone_states()
        for col in self._zone_columns:
            zone_name = col.replace("zone:", "")
            state_value = ""
            for _zone_id, zone_state in zone_states.items():
                if zone_state.zone_name == zone_name:
                    state_value = "1" if zone_state.occupied else "0"
                    break
            row.append(state_value)

        return row

    async def _record_sample(self) -> None:
        """Record a single sample of all device states (timer-driven path)."""
        if not self._recording or self._writer is None:
            return

        now = datetime.now()
        now_ts = now.timestamp()
        # Compute elapsed_ms against the same `_start_timestamp` used by
        # `record_at_frame` so the two write paths share an epoch (and the
        # column is joinable to other recorders when a session epoch is set).
        elapsed_ms = (now_ts - self._start_timestamp) * 1000 if self._start_timestamp else 0
        flow_elapsed_cell = (
            f"{(now_ts - self._flow_anchor) * 1000:.1f}" if self._flow_anchor is not None else ""
        )
        iso = now.isoformat(timespec="milliseconds")
        row = await self._build_row(
            frame=None,
            iso_timestamp=iso,
            elapsed_ms=elapsed_ms,
            flow_elapsed_cell=flow_elapsed_cell,
        )

        self._writer.writerow(row)
        self._file.flush()

    async def record_at_frame(self, frame_no: int, frame_ts: float) -> None:
        """
        Record a row anchored to a camera frame.

        Called from the camera pipeline (see CameraPanel) once per processed
        frame, immediately after TrackingDataLogger.log_frame. The two CSVs
        then share the same `frame` key and the same Unix timestamp, so
        joining device states to tracking output (and to MP4 frame indices)
        is a one-key operation rather than an interpolation.

        Args:
            frame_no: Canonical camera frame index for this session (the
                value of TrackingDataLogger.frame_count after its log_frame
                call for the same frame).
            frame_ts: Unix timestamp (seconds, float) of the frame as
                captured by CameraManager (time.time() taken at the moment
                of frame readback).
        """
        if not self._recording or self._writer is None:
            return

        elapsed_ms = (frame_ts - self._start_timestamp) * 1000 if self._start_timestamp else 0
        flow_elapsed_cell = (
            f"{(frame_ts - self._flow_anchor) * 1000:.1f}" if self._flow_anchor is not None else ""
        )
        iso = datetime.fromtimestamp(frame_ts).isoformat(timespec="milliseconds")
        row = await self._build_row(
            frame=frame_no,
            iso_timestamp=iso,
            elapsed_ms=elapsed_ms,
            flow_elapsed_cell=flow_elapsed_cell,
        )

        self._writer.writerow(row)
        self._file.flush()

    async def stop(self) -> Path | None:
        """
        Stop recording and close the file.

        Returns:
            Path to the completed CSV file, or None if not recording
        """
        if not self._recording:
            return None

        # Cancel sampling task
        if self._sample_task:
            self._sample_task.cancel()
            try:
                await self._sample_task
            except asyncio.CancelledError:
                pass
            self._sample_task = None

        # Write final sample while _recording is still True
        try:
            await self._record_sample()
        except Exception:
            pass

        self._recording = False

        # Write footer
        if self._writer:
            end_time = datetime.now()
            duration = (end_time - self._start_time).total_seconds() if self._start_time else 0
            self._writer.writerow([])
            self._writer.writerow(["# End Time", end_time.isoformat()])
            self._writer.writerow(["# Duration (s)", f"{duration:.2f}"])

        # Close file
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

        file_path = self._file_path
        logger.info(f"Stopped recording. Data saved to {file_path}")

        return file_path

    async def record_event(self, event_name: str, details: str = "") -> None:
        """
        Record a custom event in the data file.

        Args:
            event_name: Name of the event
            details: Additional details
        """
        if not self._recording or self._writer is None:
            return

        now = datetime.now()
        elapsed_ms = (now - self._start_time).total_seconds() * 1000 if self._start_time else 0

        # Write event as a comment row
        self._writer.writerow([f"# EVENT: {event_name}", f"{elapsed_ms:.1f}ms", details])
        self._file.flush()
