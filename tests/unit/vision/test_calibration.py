"""
Tests for glider.vision.calibration module.

Tests camera calibration and pixel-to-real-world conversion.
"""

import pytest

from glider.vision.calibration import (
    CalibrationLine,
    CameraCalibration,
    LengthUnit,
)


class TestLengthUnit:
    """Tests for LengthUnit enum."""

    def test_unit_values(self):
        """Test LengthUnit enum values."""
        assert LengthUnit.MILLIMETERS.value == "mm"
        assert LengthUnit.CENTIMETERS.value == "cm"
        assert LengthUnit.METERS.value == "m"
        assert LengthUnit.INCHES.value == "in"
        assert LengthUnit.FEET.value == "ft"

    def test_conversion_to_mm(self):
        """Test conversion factors to millimeters."""
        assert LengthUnit.conversion_to_mm(LengthUnit.MILLIMETERS) == 1.0
        assert LengthUnit.conversion_to_mm(LengthUnit.CENTIMETERS) == 10.0
        assert LengthUnit.conversion_to_mm(LengthUnit.METERS) == 1000.0
        assert LengthUnit.conversion_to_mm(LengthUnit.INCHES) == 25.4
        assert LengthUnit.conversion_to_mm(LengthUnit.FEET) == 304.8


class TestCalibrationLine:
    """Tests for CalibrationLine dataclass."""

    def test_creation(self):
        """Test CalibrationLine creation."""
        line = CalibrationLine(
            start_x=0.1,
            start_y=0.5,
            end_x=0.9,
            end_y=0.5,
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
        )

        assert line.start_x == 0.1
        assert line.start_y == 0.5
        assert line.end_x == 0.9
        assert line.end_y == 0.5
        assert line.length == 100.0
        assert line.unit == LengthUnit.MILLIMETERS

    def test_default_color(self):
        """Test CalibrationLine default color."""
        line = CalibrationLine(
            start_x=0.0,
            start_y=0.0,
            end_x=1.0,
            end_y=1.0,
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
        )

        assert line.color == (0, 255, 0)  # Default green

    def test_custom_color(self):
        """Test CalibrationLine with custom color."""
        line = CalibrationLine(
            start_x=0.0,
            start_y=0.0,
            end_x=1.0,
            end_y=1.0,
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            color=(255, 0, 0),
        )

        assert line.color == (255, 0, 0)

    def test_pixel_length_property(self):
        """Test pixel_length property (normalized)."""
        # Horizontal line spanning 80% of normalized width
        line = CalibrationLine(
            start_x=0.1,
            start_y=0.5,
            end_x=0.9,
            end_y=0.5,
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
        )

        assert abs(line.pixel_length - 0.8) < 0.001

    def test_length_mm_property(self):
        """Test length_mm property converts to mm."""
        # 10 centimeters = 100 millimeters
        line = CalibrationLine(
            start_x=0.0, start_y=0.5, end_x=1.0, end_y=0.5, length=10.0, unit=LengthUnit.CENTIMETERS
        )

        assert line.length_mm == 100.0

    def test_get_pixel_coords(self):
        """Test converting normalized coords to pixel coords."""
        line = CalibrationLine(
            start_x=0.1,
            start_y=0.2,
            end_x=0.9,
            end_y=0.8,
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
        )

        # For a 640x480 frame
        x1, y1, x2, y2 = line.get_pixel_coords(640, 480)

        assert x1 == int(0.1 * 640)  # 64
        assert y1 == int(0.2 * 480)  # 96
        assert x2 == int(0.9 * 640)  # 576
        assert y2 == int(0.8 * 480)  # 384

    def test_to_dict(self):
        """Test CalibrationLine serialization."""
        line = CalibrationLine(
            start_x=0.1,
            start_y=0.2,
            end_x=0.9,
            end_y=0.8,
            length=150.0,
            unit=LengthUnit.CENTIMETERS,
            color=(255, 128, 0),
            name="Test Line",
        )

        data = line.to_dict()

        assert data["start_x"] == 0.1
        assert data["start_y"] == 0.2
        assert data["end_x"] == 0.9
        assert data["end_y"] == 0.8
        assert data["length"] == 150.0
        assert data["unit"] == "cm"
        assert data["color"] == [255, 128, 0]

    def test_from_dict(self):
        """Test CalibrationLine deserialization."""
        data = {
            "start_x": 0.25,
            "start_y": 0.25,
            "end_x": 0.75,
            "end_y": 0.75,
            "length": 50.0,
            "unit": "in",
            "color": [0, 0, 255],
        }

        line = CalibrationLine.from_dict(data)

        assert line.start_x == 0.25
        assert line.end_x == 0.75
        assert line.length == 50.0
        assert line.unit == LengthUnit.INCHES
        assert line.color == (0, 0, 255)

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CalibrationLine(
            start_x=0.33,
            start_y=0.44,
            end_x=0.55,
            end_y=0.66,
            length=77.7,
            unit=LengthUnit.FEET,
            color=(100, 150, 200),
            name="Roundtrip Test",
        )

        restored = CalibrationLine.from_dict(original.to_dict())

        assert restored.start_x == original.start_x
        assert restored.start_y == original.start_y
        assert restored.end_x == original.end_x
        assert restored.end_y == original.end_y
        assert restored.length == original.length
        assert restored.unit == original.unit
        assert restored.color == original.color


class TestCameraCalibration:
    """Tests for CameraCalibration class."""

    def test_creation(self):
        """Test CameraCalibration creation."""
        calibration = CameraCalibration()

        assert calibration.lines == []
        assert calibration.calibration_width == 0
        assert calibration.calibration_height == 0

    def test_is_calibrated_property(self):
        """Test is_calibrated property."""
        calibration = CameraCalibration()
        assert calibration.is_calibrated is False

        calibration.add_line(
            start=(0, 240),
            end=(640, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )
        assert calibration.is_calibrated is True

    def test_add_line(self):
        """Test adding a calibration line."""
        calibration = CameraCalibration()

        line = calibration.add_line(
            start=(64, 240),
            end=(576, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )

        assert len(calibration.lines) == 1
        assert calibration.calibration_width == 640
        assert calibration.calibration_height == 480
        # Line should be normalized
        assert abs(line.start_x - 0.1) < 0.01
        assert abs(line.end_x - 0.9) < 0.01

    def test_remove_line(self):
        """Test removing a calibration line."""
        calibration = CameraCalibration()
        calibration.add_line(
            start=(0, 240),
            end=(640, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )

        result = calibration.remove_line(0)

        assert result is True
        assert len(calibration.lines) == 0

    def test_clear(self):
        """Test clearing all calibration lines."""
        calibration = CameraCalibration()

        for i in range(3):
            calibration.add_line(
                start=(0, i * 100),
                end=(640, i * 100),
                length=100.0,
                unit=LengthUnit.MILLIMETERS,
                resolution=(640, 480),
            )

        assert len(calibration.lines) == 3

        calibration.clear()

        assert len(calibration.lines) == 0
        assert calibration.calibration_width == 0

    def test_pixels_per_mm(self):
        """Test calculating pixels per mm from calibration line."""
        calibration = CameraCalibration()

        # Line spanning full width at 640 pixels = 100mm
        calibration.add_line(
            start=(0, 240),
            end=(640, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )

        ppm = calibration.pixels_per_mm
        # 640 pixels / 100 mm = 6.4 pixels per mm
        assert abs(ppm - 6.4) < 0.1

    def test_pixels_to_mm(self):
        """Test converting pixel distance to millimeters."""
        calibration = CameraCalibration()

        # Set up calibration: 640 pixels = 100mm
        calibration.add_line(
            start=(0, 240),
            end=(640, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )

        # 64 pixels should be ~10mm
        distance = calibration.pixels_to_mm(64)
        assert abs(distance - 10.0) < 1.0

    def test_mm_to_pixels(self):
        """Test converting millimeters to pixel distance."""
        calibration = CameraCalibration()

        calibration.add_line(
            start=(0, 240),
            end=(640, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )

        # 10mm should be ~64 pixels
        pixels = calibration.mm_to_pixels(10.0)
        assert abs(pixels - 64) < 1

    def test_to_dict(self):
        """Test CameraCalibration serialization."""
        calibration = CameraCalibration()
        calibration.add_line(
            start=(0, 240),
            end=(640, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )

        data = calibration.to_dict()

        assert "lines" in data
        assert len(data["lines"]) == 1
        assert "calibration_width" in data
        assert "calibration_height" in data

    def test_from_dict(self):
        """Test CameraCalibration deserialization."""
        data = {
            "lines": [
                {
                    "start_x": 0.0,
                    "start_y": 0.5,
                    "end_x": 1.0,
                    "end_y": 0.5,
                    "length": 200.0,
                    "unit": "cm",
                    "color": [0, 255, 0],
                    "name": "Test",
                }
            ],
            "calibration_width": 640,
            "calibration_height": 480,
        }

        calibration = CameraCalibration.from_dict(data)

        assert len(calibration.lines) == 1
        assert calibration.lines[0].length == 200.0
        assert calibration.lines[0].unit == LengthUnit.CENTIMETERS
        assert calibration.calibration_width == 640

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = CameraCalibration()
        original.add_line(
            start=(64, 108),
            end=(512, 432),
            length=50.0,
            unit=LengthUnit.INCHES,
            resolution=(640, 480),
        )

        restored = CameraCalibration.from_dict(original.to_dict())

        assert len(restored.lines) == len(original.lines)
        assert restored.lines[0].length == original.lines[0].length
        assert restored.calibration_width == original.calibration_width

    def test_uncalibrated_returns_raw_pixels(self):
        """Test that uncalibrated conversion returns raw pixels."""
        calibration = CameraCalibration()

        # No calibration - should return input value
        result = calibration.pixels_to_mm(100)
        assert result == 100

    def test_pixel_distance(self):
        """Test pixel distance calculation."""
        calibration = CameraCalibration()

        dist = calibration.pixel_distance((0, 0), (3, 4))
        assert dist == 5.0  # 3-4-5 triangle

    def test_real_distance(self):
        """Test real distance calculation between points."""
        calibration = CameraCalibration()
        calibration.add_line(
            start=(0, 0),
            end=(100, 0),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(100, 100),
        )

        # Distance of 10 pixels should be 10mm (1 pixel = 1mm at this calibration)
        dist = calibration.real_distance((0, 0), (10, 0))
        assert abs(dist - 10.0) < 1.0


class TestDegenerateLines:
    """A zero-length line must not drag the averaged scale toward zero."""

    def test_zero_length_line_is_ignored(self):
        calibration = CameraCalibration()
        calibration.add_line(
            start=(0, 240),
            end=(640, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )
        # Both endpoints identical: carries no scale information.
        calibration.add_line(
            start=(100, 100),
            end=(100, 100),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )
        # Still 6.4, not 3.2.
        assert calibration.pixels_per_mm == pytest.approx(6.4, abs=0.1)

    def test_only_degenerate_lines_is_uncalibrated(self):
        calibration = CameraCalibration()
        calibration.add_line(
            start=(10, 10),
            end=(10, 10),
            length=50.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )
        assert calibration.pixels_per_mm == 0.0

    def test_zero_real_length_line_is_ignored(self):
        """The other exclusion branch: real pixels, but no real-world length."""
        calibration = CameraCalibration()
        calibration.add_line(
            start=(0, 240),
            end=(640, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )
        # A line the operator drew but never assigned a measurement to.
        calibration.add_line(
            start=(0, 100),
            end=(320, 100),
            length=0.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )
        assert calibration.pixels_per_mm == pytest.approx(6.4, abs=0.1)
