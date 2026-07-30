"""
Camera Calibration - Pixel to real-world measurement conversion.

Allows users to draw measurement lines on camera view and assign
real-world distances for accurate tracking distance calculations.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LengthUnit(Enum):
    """Supported length units."""

    MILLIMETERS = "mm"
    CENTIMETERS = "cm"
    METERS = "m"
    INCHES = "in"
    FEET = "ft"

    @classmethod
    def conversion_to_mm(cls, unit: "LengthUnit") -> float:
        """Get conversion factor to millimeters."""
        factors = {
            cls.MILLIMETERS: 1.0,
            cls.CENTIMETERS: 10.0,
            cls.METERS: 1000.0,
            cls.INCHES: 25.4,
            cls.FEET: 304.8,
        }
        return factors.get(unit, 1.0)


@dataclass
class CalibrationLine:
    """A measurement line for camera calibration."""

    # Pixel coordinates (normalized 0-1 for resolution independence)
    start_x: float
    start_y: float
    end_x: float
    end_y: float

    # Real-world measurement
    length: float
    unit: LengthUnit = LengthUnit.MILLIMETERS

    # Display properties
    name: str = ""
    color: tuple[int, int, int] = (0, 255, 0)  # BGR for OpenCV

    @property
    def pixel_length(self) -> float:
        """Calculate pixel length of line (in normalized coords)."""
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        return math.sqrt(dx * dx + dy * dy)

    @property
    def length_mm(self) -> float:
        """Get length in millimeters."""
        return self.length * LengthUnit.conversion_to_mm(self.unit)

    def get_pixel_coords(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized coords to pixel coords for given resolution."""
        return (
            int(self.start_x * width),
            int(self.start_y * height),
            int(self.end_x * width),
            int(self.end_y * height),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "start_x": self.start_x,
            "start_y": self.start_y,
            "end_x": self.end_x,
            "end_y": self.end_y,
            "length": self.length,
            "unit": self.unit.value,
            "name": self.name,
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationLine":
        """Create from dictionary."""
        return cls(
            start_x=data["start_x"],
            start_y=data["start_y"],
            end_x=data["end_x"],
            end_y=data["end_y"],
            length=data["length"],
            unit=LengthUnit(data.get("unit", "mm")),
            name=data.get("name", ""),
            color=tuple(data.get("color", [0, 255, 0])),
        )


@dataclass
class CameraCalibration:
    """
    Camera calibration data for pixel-to-real-world conversion.

    Stores measurement lines and calculates pixels-per-unit ratios.
    Supports saving/loading calibration to/from JSON files.
    """

    lines: list[CalibrationLine] = field(default_factory=list)

    # Camera resolution at time of calibration
    calibration_width: int = 0
    calibration_height: int = 0

    @property
    def is_calibrated(self) -> bool:
        """Whether calibration has been performed."""
        return len(self.lines) > 0 and self.calibration_width > 0

    @property
    def pixels_per_mm(self) -> float:
        """
        Calculate average pixels per millimeter from all calibration lines.

        Lines carrying no scale information (zero pixel length, or no real-world
        length) are excluded from the average rather than averaged in as a 0,
        which would silently deflate the scale for every other line.

        Returns 0 if not calibrated.
        """
        if not self.lines or self.calibration_width == 0:
            return 0.0

        ratios = []
        for line in self.lines:
            x1, y1, x2, y2 = line.get_pixel_coords(self.calibration_width, self.calibration_height)
            pixel_dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            mm_dist = line.length_mm
            if pixel_dist > 0 and mm_dist > 0:
                ratios.append(pixel_dist / mm_dist)

        return sum(ratios) / len(ratios) if ratios else 0.0

    def add_line(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        length: float,
        unit: LengthUnit = LengthUnit.MILLIMETERS,
        name: str = "",
        resolution: tuple[int, int] = None,
    ) -> CalibrationLine:
        """
        Add a calibration line.

        Args:
            start: Start point in pixels
            end: End point in pixels
            length: Real-world length
            unit: Length unit
            name: Optional name for the line
            resolution: Current camera resolution (width, height)

        Returns:
            The created CalibrationLine
        """
        if resolution:
            width, height = resolution
        else:
            width = self.calibration_width or 640
            height = self.calibration_height or 480

        # Store calibration resolution
        if self.calibration_width == 0:
            self.calibration_width = width
            self.calibration_height = height

        # Normalize coordinates
        line = CalibrationLine(
            start_x=start[0] / width,
            start_y=start[1] / height,
            end_x=end[0] / width,
            end_y=end[1] / height,
            length=length,
            unit=unit,
            name=name or f"Line {len(self.lines) + 1}",
        )
        self.lines.append(line)
        logger.info(f"Added calibration line: {line.name} = {length} {unit.value}")
        return line

    def remove_line(self, index: int) -> bool:
        """Remove a calibration line by index."""
        if 0 <= index < len(self.lines):
            removed = self.lines.pop(index)
            logger.info(f"Removed calibration line: {removed.name}")
            return True
        return False

    def clear(self) -> None:
        """Clear all calibration lines."""
        self.lines.clear()
        self.calibration_width = 0
        self.calibration_height = 0
        logger.info("Cleared camera calibration")

    def pixels_to_mm(
        self, pixels: float, current_width: int = None, current_height: int = None
    ) -> float:
        """
        Convert pixel distance to millimeters.

        Args:
            pixels: Distance in pixels
            current_width: Current frame width (for scaling if resolution changed)
            current_height: Current frame height

        Returns:
            Distance in millimeters, or pixels if not calibrated
        """
        ppm = self.pixels_per_mm
        if ppm == 0:
            return pixels  # Return raw pixels if not calibrated

        # Scale for resolution differences
        if current_width and self.calibration_width:
            scale = current_width / self.calibration_width
            ppm *= scale

        return pixels / ppm

    def mm_to_pixels(self, mm: float, current_width: int = None) -> float:
        """
        Convert millimeters to pixel distance.

        Args:
            mm: Distance in millimeters
            current_width: Current frame width (for scaling)

        Returns:
            Distance in pixels
        """
        ppm = self.pixels_per_mm
        if ppm == 0:
            return mm

        if current_width and self.calibration_width:
            scale = current_width / self.calibration_width
            ppm *= scale

        return mm * ppm

    def pixel_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Calculate pixel distance between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    def real_distance(
        self,
        p1: tuple[int, int],
        p2: tuple[int, int],
        current_width: int = None,
        current_height: int = None,
    ) -> float:
        """
        Calculate real-world distance between two pixel points.

        Args:
            p1: First point in pixels
            p2: Second point in pixels
            current_width: Current frame width
            current_height: Current frame height

        Returns:
            Distance in millimeters
        """
        pixel_dist = self.pixel_distance(p1, p2)
        return self.pixels_to_mm(pixel_dist, current_width, current_height)

    def save(self, path: Path) -> None:
        """
        Save calibration to JSON file.

        Args:
            path: File path to save to
        """
        data = {
            "calibration_width": self.calibration_width,
            "calibration_height": self.calibration_height,
            "lines": [line.to_dict() for line in self.lines],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved calibration to {path}")

    def load(self, path: Path) -> bool:
        """
        Load calibration from JSON file.

        Args:
            path: File path to load from

        Returns:
            True if loaded successfully
        """
        try:
            with open(path) as f:
                data = json.load(f)

            self.calibration_width = data.get("calibration_width", 0)
            self.calibration_height = data.get("calibration_height", 0)
            self.lines = [
                CalibrationLine.from_dict(line_data) for line_data in data.get("lines", [])
            ]

            logger.info(f"Loaded calibration from {path}: {len(self.lines)} lines")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration from {path}: {e}")
            return False

    def to_dict(self) -> dict:
        """Convert to dictionary for embedding in session."""
        return {
            "calibration_width": self.calibration_width,
            "calibration_height": self.calibration_height,
            "lines": [line.to_dict() for line in self.lines],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraCalibration":
        """Create from dictionary."""
        cal = cls()
        cal.calibration_width = data.get("calibration_width", 0)
        cal.calibration_height = data.get("calibration_height", 0)
        cal.lines = [CalibrationLine.from_dict(line_data) for line_data in data.get("lines", [])]
        return cal

    def get_info(self) -> dict:
        """Get calibration information."""
        return {
            "is_calibrated": self.is_calibrated,
            "num_lines": len(self.lines),
            "pixels_per_mm": self.pixels_per_mm,
            "calibration_resolution": (self.calibration_width, self.calibration_height),
            "lines": [
                {
                    "name": line.name,
                    "length": line.length,
                    "unit": line.unit.value,
                }
                for line in self.lines
            ],
        }
