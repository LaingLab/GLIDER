"""
Tracking Data Logger - Log CV results to CSV.

Logs computer vision tracking data to CSV file synchronized
with experiment timestamps, matching DataRecorder output format.
Includes real-world distance calculations when calibration is available.
"""

import csv
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

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
        self._start_time: datetime | None = None
        self._start_timestamp: float = 0.0
        self._frame_count = 0
        self._recording = False
        self._calibration: CameraCalibration | None = None
        self._zone_config: ZoneConfiguration | None = None
        self._frame_width: int = 0
        self._frame_height: int = 0
        # Track previous positions and cumulative distances for each object
        self._prev_positions: dict[int, tuple[float, float]] = {}
        self._cumulative_distances: dict[int, float] = {}

    @property
    def is_recording(self) -> bool:
        """Whether logging is active."""
        return self._recording

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
        self._start_timestamp = self._start_time.timestamp()
        self._frame_count = 0

        # Clear tracking state
        self._prev_positions.clear()
        self._cumulative_distances.clear()

        # Open file and create writer
        self._file = open(self._file_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

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

        # Write column headers
        self._writer.writerow(
            [
                "frame",
                "timestamp",
                "elapsed_ms",
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
        self._file.flush()

        self._recording = True
        logger.info(f"Started tracking log: {self._file_path}")
        return self._file_path

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
        if not self._recording or self._writer is None:
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

            self._writer.writerow(
                [
                    self._frame_count,
                    iso_timestamp,
                    f"{elapsed_ms:.1f}",
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
                ]
            )

        # Log motion event if no objects but motion detected
        if not tracked_objects and motion_detected:
            self._writer.writerow(
                [
                    self._frame_count,
                    iso_timestamp,
                    f"{elapsed_ms:.1f}",
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
                ]
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

        self._file.flush()

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
        self._file.flush()

    async def stop(self) -> Path | None:
        """
        Stop logging and close file.

        Returns:
            Path to the saved log file
        """
        if not self._recording:
            return None

        self._recording = False

        # Write footer
        if self._writer and self._file:
            end_time = datetime.now()
            duration = (end_time - self._start_time).total_seconds() if self._start_time else 0

            self._writer.writerow([])
            self._writer.writerow(["# End Time", end_time.isoformat()])
            self._writer.writerow(["# Duration (s)", f"{duration:.2f}"])
            self._writer.writerow(["# Total Frames", self._frame_count])

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
