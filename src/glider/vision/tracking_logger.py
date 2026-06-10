"""
Tracking Data Logger - Log CV results to CSV.

Logs computer vision tracking data to CSV file synchronized
with experiment timestamps, matching DataRecorder output format.
Includes real-world distance calculations when calibration is available.
"""

import csv
import logging
import math
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Frames between os.fsync() calls. ~10s at 30fps — small enough to bound
# crash loss, large enough to avoid hammering SD cards on a Raspberry Pi.
_FSYNC_INTERVAL_FRAMES = 300

if TYPE_CHECKING:
    from glider.core.experiment_session import ExperimentSession
    from glider.vision.calibration import CameraCalibration
    from glider.vision.cv_processor import TrackedObject
    from glider.vision.zones import ZoneConfiguration

logger = logging.getLogger(__name__)


class TrackingDataLogger:
    """
    Logs computer vision results to CSV file.

    Output format matches DataRecorder pattern with columns:
    frame, timestamp, elapsed_ms, object_id, class, x, y, w, h, confidence,
    center_x, center_y, distance_px, distance_mm, cumulative_mm, zone_ids,
    behavioral_state, velocity_px_frame
    """

    # Number of consecutive write failures tolerated before we give up on
    # the recording. One transient EIO during a four-hour overnight run
    # shouldn't abort the experiment; ten in a row mean the disk is gone.
    _MAX_CONSECUTIVE_WRITE_ERRORS = 5

    def __init__(self, output_dir: Path | None = None):
        """
        Initialize the tracking logger.

        Args:
            output_dir: Directory for output files, or None for current directory
        """
        self._output_dir = Path(output_dir) if output_dir else Path.cwd()
        self._file = None
        self._writer = None
        self._file_path: Path | None = None
        self._partial_path: Path | None = None  # .partial.csv during recording
        self._start_time: datetime | None = None
        # See DataRecorder.set_session_epoch — when set, elapsed_ms in
        # this file is anchored to the shared session epoch instead of
        # this logger's own start_time, so tracking rows are joinable to
        # device-state and event rows on the elapsed_ms column.
        self._start_timestamp: float = 0.0
        self._session_epoch_override: float | None = None
        # Flow-relative timing anchor. ``elapsed_ms`` measures time from
        # the shared session epoch (recorder/event/tracking are all
        # joinable on it), but the analyst usually wants time from the
        # *flow* boundary, not the recorder-start boundary. ``_flow_anchor``
        # is the wall-clock instant the flow began; frames logged before
        # the anchor is set get an empty ``flow_elapsed_ms`` cell
        # (pre-flow setup / camera warmup); frames after get
        # ``(timestamp - anchor) * 1000``. Set by GliderCore the moment
        # ``flow_engine.start()`` returns.
        self._flow_anchor: float | None = None
        self._frame_count = 0
        self._recording = False
        self._calibration: CameraCalibration | None = None
        self._zone_config: ZoneConfiguration | None = None
        self._frame_width: int = 0
        self._frame_height: int = 0
        # Track previous positions and cumulative distances for each object
        self._prev_positions: dict[int, tuple[float, float]] = {}
        self._cumulative_distances: dict[int, float] = {}
        # Periodic fsync so a power-loss or crash only loses a bounded window
        # of tracking data instead of whatever the OS happens to be buffering.
        self._frames_since_fsync = 0
        # Write-failure tracking. The tracking CSV is the primary scientific
        # artifact; previous code's bare writerow calls would silently abort
        # the log on a disk-full / USB-unplug / encoding error, while
        # `is_recording` continued to return True to the UI. We now count
        # consecutive failures and surface them.
        self._write_error_count: int = 0
        self._failed: bool = False
        self._writer_error_callback: Callable[[Exception], None] | None = None

    @property
    def is_recording(self) -> bool:
        """Whether logging is active and write-healthy."""
        return self._recording and not self._failed

    @property
    def is_failed(self) -> bool:
        """True if the writer has aborted due to repeated I/O failures."""
        return self._failed

    def set_writer_error_callback(self, callback: Callable[[Exception], None] | None) -> None:
        """Register a callback invoked on terminal write failure.

        Mirrors ``VideoRecorder.set_error_callback`` so the UI can react
        when the tracking CSV stream dies (typically disk-full / USB
        unplug). The callback fires once when the consecutive-error cap
        is hit, after which ``is_recording`` returns False.
        """
        self._writer_error_callback = callback

    def _on_write_error(self, exc: Exception, *, context: str) -> bool:
        """Handle a write exception. Returns True if recording aborted.

        Increments the consecutive-failure counter; on the Nth failure
        (per ``_MAX_CONSECUTIVE_WRITE_ERRORS``) sets ``_failed=True``,
        notifies the callback, and the caller should stop writing.
        """
        self._write_error_count += 1
        logger.warning(
            "TrackingDataLogger %s failed (%d/%d): %s",
            context,
            self._write_error_count,
            self._MAX_CONSECUTIVE_WRITE_ERRORS,
            exc,
        )
        if self._write_error_count >= self._MAX_CONSECUTIVE_WRITE_ERRORS:
            logger.error(
                "TrackingDataLogger giving up after %d consecutive write errors; "
                "tracking output for this experiment is incomplete",
                self._write_error_count,
            )
            self._failed = True
            self._recording = False
            if self._writer_error_callback is not None:
                try:
                    self._writer_error_callback(exc)
                except Exception:
                    logger.exception("TrackingDataLogger error callback raised")
            return True
        return False

    @property
    def file_path(self) -> Path | None:
        """Path to the current/last log file."""
        return self._file_path

    @property
    def frame_count(self) -> int:
        """Number of frames logged."""
        return self._frame_count

    def set_output_directory(self, path: Path) -> None:
        """
        Set the output directory for log files.

        Args:
            path: Directory path
        """
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def set_session_epoch(self, epoch: float) -> None:
        """
        Anchor ``elapsed_ms`` against a shared Unix timestamp instead of
        this logger's own ``start_time``. Must be called before ``start()``.
        Lets the tracking CSV join cleanly to the device-state CSV and
        event log on the elapsed_ms column.
        """
        if self._recording:
            logger.warning("set_session_epoch called while recording; takes effect on next start()")
        self._session_epoch_override = float(epoch)

    def set_flow_anchor(self, timestamp: float) -> None:
        """
        Anchor ``flow_elapsed_ms`` to a flow-boundary wall-clock timestamp.

        ``elapsed_ms`` measures time from the recorder's session epoch
        (shared across all recorders so rows are joinable), but the
        analyst usually wants time from the *flow* boundary so an
        ethogram / raster plot lines up with t=0 at the moment
        StartExperiment fired. Call this once at flow start with the
        Unix timestamp of that instant. Frames logged before the call
        get an empty ``flow_elapsed_ms`` cell (pre-flow setup / camera
        warmup); frames after get ``(timestamp - anchor) * 1000``.

        Args:
            timestamp: Unix timestamp (seconds, float) marking flow t=0.
        """
        self._flow_anchor = float(timestamp)

    def set_calibration(self, calibration: "CameraCalibration") -> None:
        """
        Set calibration for real-world distance calculations.

        Args:
            calibration: CameraCalibration instance
        """
        self._calibration = calibration

    def set_zone_configuration(self, zone_config: "ZoneConfiguration") -> None:
        """
        Set zone configuration for logging zone occupancy.

        Args:
            zone_config: ZoneConfiguration instance
        """
        self._zone_config = zone_config

    def _get_zones_for_point(self, center_x: float, center_y: float) -> str:
        """
        Get comma-separated list of zone names containing a point.

        Args:
            center_x: X coordinate in pixels
            center_y: Y coordinate in pixels

        Returns:
            Comma-separated string of zone names, or empty string
        """
        if not self._zone_config or not self._zone_config.zones:
            return ""

        if self._frame_width == 0 or self._frame_height == 0:
            return ""

        zone_names = self._zone_config.get_zone_names_for_point(
            center_x / self._frame_width, center_y / self._frame_height
        )
        return ",".join(zone_names)

    def set_frame_size(self, width: int, height: int) -> None:
        """
        Set frame dimensions for distance calculations.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self._frame_width = width
        self._frame_height = height

    def _fsync_if_due(self, force: bool = False) -> None:
        """
        Flush Python buffers and, on an interval (or when forced), fsync the
        file descriptor so bytes reach the disk.

        fsync is relatively expensive — we call it every
        ``_FSYNC_INTERVAL_FRAMES`` frames so the worst-case crash loss is
        bounded to roughly that many frames of tracking data.
        """
        if self._file is None:
            return
        try:
            self._file.flush()
        except Exception:
            logger.exception("TrackingDataLogger: flush failed")
            return
        self._frames_since_fsync += 1
        if force or self._frames_since_fsync >= _FSYNC_INTERVAL_FRAMES:
            try:
                os.fsync(self._file.fileno())
            except OSError:
                # fsync can fail on some filesystems (e.g., certain network
                # mounts); the flush above is still useful, so swallow.
                logger.debug("TrackingDataLogger: os.fsync failed", exc_info=True)
            self._frames_since_fsync = 0

    def _generate_filename(self, experiment_name: str) -> str:
        """
        Generate filename for tracking data.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Formatted filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize name
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in experiment_name)
        safe_name = safe_name.strip().replace(" ", "_") or "experiment"
        return f"{safe_name}_{timestamp}_tracking.csv"

    async def start(
        self,
        experiment_name: str = "experiment",
        session: "ExperimentSession | None" = None,
    ) -> Path:
        """
        Start logging tracking data.

        Args:
            experiment_name: Name for the log file
            session: Optional experiment session for additional metadata

        Returns:
            Path to the log file being created
        """
        if self._recording:
            logger.warning("Tracking logger already recording")
            return self._file_path

        # Generate filename
        filename = self._generate_filename(experiment_name)
        self._file_path = self._output_dir / filename
        self._start_time = datetime.now()
        self._start_timestamp = (
            self._session_epoch_override
            if self._session_epoch_override is not None
            else self._start_time.timestamp()
        )
        self._frame_count = 0

        # Reset the flow anchor so a previous run's flow boundary doesn't
        # leak into a new recording. The orchestrator (GliderCore) will
        # call set_flow_anchor() once the flow engine actually starts.
        self._flow_anchor = None

        # Clear tracking state
        self._prev_positions.clear()
        self._cumulative_distances.clear()

        # Open file and create writer.  If any subsequent write raises (e.g.,
        # metadata serialization error), close the file and clear state so the
        # caller does not see a recording session with a leaked fd.
        self._file = open(self._file_path, "w", newline="", encoding="utf-8")
        try:
            self._writer = csv.writer(self._file)
            self._write_header_and_metadata(experiment_name, session)
        except Exception:
            self._file.close()
            self._file = None
            self._writer = None
            raise

        self._recording = True
        logger.info(f"Started tracking log: {self._file_path}")
        return self._file_path

    def _write_header_and_metadata(self, experiment_name: str, session) -> None:
        """Write all header / metadata rows. Raises if any write fails."""
        # Write metadata header
        self._writer.writerow(["# GLIDER Tracking Data"])
        self._writer.writerow(["# Experiment", experiment_name])
        self._writer.writerow(["# Start Time", self._start_time.isoformat()])

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

        # Write calibration info if available
        if self._calibration and self._calibration.is_calibrated:
            self._writer.writerow(["# Pixels/mm", f"{self._calibration.pixels_per_mm:.4f}"])
            self._writer.writerow(
                [
                    "# Calibration Resolution",
                    f"{self._calibration.calibration_width}x{self._calibration.calibration_height}",
                ]
            )
        else:
            self._writer.writerow(["# Calibration", "Not calibrated (distances in pixels)"])

        self._writer.writerow([])

        # Write column headers. ``flow_elapsed_ms`` is empty for frames
        # captured before ``set_flow_anchor()`` is called (pre-flow
        # setup), and ``(timestamp - flow_anchor) * 1000`` afterward —
        # giving downstream ethogram / raster scripts a column they can
        # plot directly with t=0 aligned to flow start.
        self._writer.writerow(
            [
                "frame",
                "timestamp",
                "elapsed_ms",
                "flow_elapsed_ms",
                "object_id",
                "class",
                "x",
                "y",
                "w",
                "h",
                "confidence",
                "center_x",
                "center_y",
                "distance_px",
                "distance_mm",
                "cumulative_mm",
                "zone_ids",
                "behavioral_state",
                "velocity_px_frame",
            ]
        )
        # Force fsync on the header row so the file is durable on disk before
        # any frame data is written.
        self._fsync_if_due(force=True)

    def _safe_writerow(self, row: list, *, context: str = "writerow") -> bool:
        """Write a CSV row, swallowing/escalating I/O errors.

        Returns True on success. On failure, increments the consecutive-
        error counter and, on the Nth failure, marks the logger ``_failed``
        and stops recording. Subsequent calls to ``log_frame`` will short-
        circuit at the ``is_recording`` guard.
        """
        if self._writer is None or self._failed:
            return False
        try:
            self._writer.writerow(row)
        except Exception as e:
            self._on_write_error(e, context=context)
            return False
        # Success — reset the consecutive-failure counter so transient
        # errors don't accumulate over hours of healthy writes.
        if self._write_error_count > 0:
            self._write_error_count = 0
        return True

    def log_frame(
        self,
        timestamp: float,
        tracked_objects: list["TrackedObject"],
        motion_detected: bool = False,
        motion_area: float = 0.0,
    ) -> None:
        """
        Log tracking data for a single frame.

        Args:
            timestamp: Frame timestamp (Unix time)
            tracked_objects: List of tracked objects
            motion_detected: Whether motion was detected
            motion_area: Area percentage with motion
        """
        if not self.is_recording or self._writer is None:
            return

        self._frame_count += 1

        # Debug: Log when we actually write data
        if tracked_objects or motion_detected:
            if self._frame_count <= 5 or self._frame_count % 100 == 0:
                logger.debug(
                    f"Logging data: frame={self._frame_count}, "
                    f"objects={len(tracked_objects)}, motion={motion_detected}"
                )
        elapsed_ms = (timestamp - self._start_timestamp) * 1000
        # Time relative to flow start (StartExperiment fired). Empty until
        # GliderCore calls set_flow_anchor(); subsequent frames carry the
        # flow-aligned millisecond offset analysts can plot directly.
        flow_elapsed_cell = (
            f"{(timestamp - self._flow_anchor) * 1000:.1f}" if self._flow_anchor is not None else ""
        )
        iso_timestamp = datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds")

        # Log each tracked object
        for obj in tracked_objects:
            x, y, w, h = obj.bbox

            # Calculate center position
            center_x = x + w / 2
            center_y = y + h / 2

            # Calculate distance from previous position
            distance_px = 0.0
            distance_mm = 0.0
            cumulative_mm = 0.0

            if obj.track_id in self._prev_positions:
                prev_x, prev_y = self._prev_positions[obj.track_id]
                dx = center_x - prev_x
                dy = center_y - prev_y
                distance_px = math.sqrt(dx * dx + dy * dy)

                # Convert to mm if calibrated
                if self._calibration and self._calibration.is_calibrated:
                    distance_mm = self._calibration.pixels_to_mm(
                        distance_px, self._frame_width, self._frame_height
                    )
                else:
                    distance_mm = distance_px  # Use pixels if not calibrated

                # Update cumulative distance
                if obj.track_id not in self._cumulative_distances:
                    self._cumulative_distances[obj.track_id] = 0.0
                self._cumulative_distances[obj.track_id] += distance_mm
                cumulative_mm = self._cumulative_distances[obj.track_id]
            else:
                # First time seeing this object
                self._cumulative_distances[obj.track_id] = 0.0

            # Update previous position
            self._prev_positions[obj.track_id] = (center_x, center_y)

            # Get zone IDs for this object's position
            zone_ids = self._get_zones_for_point(center_x, center_y)

            # Get behavioral state and velocity from tracked object
            behavioral_state = getattr(obj, "behavioral_state", "unknown")
            velocity = getattr(obj, "velocity", 0.0)

            if not self._safe_writerow(
                [
                    self._frame_count,
                    iso_timestamp,
                    f"{elapsed_ms:.1f}",
                    flow_elapsed_cell,
                    obj.track_id,
                    obj.class_name,
                    x,
                    y,
                    w,
                    h,
                    f"{obj.confidence:.3f}",
                    f"{center_x:.1f}",
                    f"{center_y:.1f}",
                    f"{distance_px:.2f}",
                    f"{distance_mm:.2f}",
                    f"{cumulative_mm:.2f}",
                    zone_ids,
                    behavioral_state,
                    f"{velocity:.2f}",
                ],
                context="log_frame[object]",
            ):
                return  # write failure; logger may have flagged _failed

        # Log motion event if no objects but motion detected
        if not tracked_objects and motion_detected:
            self._safe_writerow(
                [
                    self._frame_count,
                    iso_timestamp,
                    f"{elapsed_ms:.1f}",
                    flow_elapsed_cell,
                    -1,  # No object ID for motion-only
                    "motion",
                    0,
                    0,
                    0,
                    0,  # No bbox
                    f"{motion_area:.3f}",
                    "",
                    "",
                    "",
                    "",
                    "",  # Empty distance fields
                    "",  # Empty zone_ids
                    "",  # Empty behavioral_state
                    "",  # Empty velocity
                ],
                context="log_frame[motion]",
            )

        # Log periodic heartbeat frames when no activity (every 30 seconds)
        # This helps confirm tracking is running even with no detections
        if not tracked_objects and not motion_detected:
            # Log a heartbeat every ~900 frames (30 seconds at 30fps)
            if self._frame_count == 1 or self._frame_count % 900 == 0:
                self._writer.writerow(
                    [
                        self._frame_count,
                        iso_timestamp,
                        f"{elapsed_ms:.1f}",
                        flow_elapsed_cell,
                        -1,
                        "heartbeat",
                        0,
                        0,
                        0,
                        0,
                        "0.000",
                        "",
                        "",
                        "",
                        "",
                        "",  # Empty distance fields
                        "",  # Empty zone_ids
                        "",  # Empty behavioral_state
                        "",  # Empty velocity
                    ]
                )

        self._fsync_if_due()

    def log_event(self, event_name: str, details: str = "") -> None:
        """
        Log a custom event.

        Args:
            event_name: Name of the event
            details: Optional details
        """
        if not self._recording or self._writer is None:
            return

        timestamp = datetime.now()
        elapsed_ms = (timestamp.timestamp() - self._start_timestamp) * 1000
        iso_timestamp = timestamp.isoformat(timespec="milliseconds")

        self._writer.writerow(
            [
                f"# EVENT: {event_name}",
                iso_timestamp,
                f"{elapsed_ms:.1f}",
                "",
                details,
                "",
                "",
                "",
                "",
                "",
            ]
        )
        # Events are often the thing the user most wants preserved across a
        # crash (trial marks, manual annotations), so force fsync.
        self._fsync_if_due(force=True)

    async def stop(self, flow_duration_s: float | None = None) -> Path | None:
        """
        Stop logging and close file.

        Args:
            flow_duration_s: Optional authoritative flow duration to record
                in the footer. When provided, the ``# Duration (s)`` line is
                the *flow's* logical duration (start-of-flow to
                end-of-flow), not the recorder's own ``start_time`` →
                ``stop_time`` wall-clock. This anchors the file's
                ``Duration`` field to the same source-of-truth as the
                runner timer (see ``GliderCore.last_flow_duration_s``) and
                prevents teardown latency from inflating the reported
                experiment duration. When ``None``, falls back to the
                recorder's own wall-clock duration for backwards
                compatibility.

        Returns:
            Path to the saved log file
        """
        if not self._recording:
            return None

        self._recording = False

        # Write footer
        if self._writer and self._file:
            end_time = datetime.now()
            recording_duration = (
                (end_time - self._start_time).total_seconds() if self._start_time else 0
            )
            # ``# Duration (s)`` is the operator's truth-of-record for
            # "how long was the experiment." Prefer the flow-anchored
            # value (insensitive to teardown latency); record the
            # recorder's own clock separately for diagnostics.
            display_duration = (
                flow_duration_s if flow_duration_s is not None else recording_duration
            )

            self._writer.writerow([])
            self._writer.writerow(["# End Time", end_time.isoformat()])
            self._writer.writerow(["# Duration (s)", f"{display_duration:.2f}"])
            if flow_duration_s is not None:
                # Keep the recorder-wall-clock value for diagnostics so a
                # mismatch (large teardown latency, missed frames) is
                # visible in the file without having to re-derive it.
                self._writer.writerow(["# Recording Duration (s)", f"{recording_duration:.2f}"])
            self._writer.writerow(["# Total Frames", self._frame_count])
            # Final fsync so the footer is guaranteed durable before close.
            self._fsync_if_due(force=True)

        # Close file
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

        saved_path = self._file_path
        logger.info(f"Stopped tracking log. Saved to {saved_path}")
        return saved_path

    def get_statistics(self) -> dict:
        """
        Get logging statistics.

        Returns:
            Dictionary with logging stats
        """
        duration = 0.0
        if self._start_time and self._recording:
            duration = (datetime.now() - self._start_time).total_seconds()

        return {
            "recording": self._recording,
            "file_path": str(self._file_path) if self._file_path else None,
            "frame_count": self._frame_count,
            "duration": duration,
            "start_time": self._start_time.isoformat() if self._start_time else None,
        }
