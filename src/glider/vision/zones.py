"""
Zone System - Define and track zones within the camera view.

Allows users to define named regions (rectangle, circle, polygon) on the
camera preview. Zones can be used as inputs in the node graph to trigger
events when objects enter/exit or occupy zones.
"""

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ZoneShape(Enum):
    """Supported zone shapes."""

    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"


@dataclass
class Zone:
    """
    A named zone within the camera view.

    Zones are defined using normalized coordinates (0-1) for resolution
    independence.
    """

    id: str
    name: str
    shape: ZoneShape
    # For rectangles: [(x1,y1), (x2,y2)] top-left and bottom-right
    # For circles: [(cx,cy), (rx,ry)] center and radius point
    # For polygons: list of vertices
    vertices: list[tuple[float, float]]
    color: tuple[int, int, int] = (0, 255, 0)  # BGR for OpenCV

    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point (normalized coords) is inside the zone.

        Args:
            x: X coordinate (0-1)
            y: Y coordinate (0-1)

        Returns:
            True if point is inside the zone
        """
        if not self.vertices:
            return False

        if self.shape == ZoneShape.RECTANGLE:
            if len(self.vertices) < 2:
                return False
            x1, y1 = self.vertices[0]
            x2, y2 = self.vertices[1]
            # Normalize order
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            return min_x <= x <= max_x and min_y <= y <= max_y

        elif self.shape == ZoneShape.CIRCLE:
            if len(self.vertices) < 2:
                return False
            cx, cy = self.vertices[0]  # Center
            rx, ry = self.vertices[1]  # Point on radius
            # Calculate radius
            radius = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            # Check distance from center
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            return dist <= radius

        elif self.shape == ZoneShape.POLYGON:
            if len(self.vertices) < 3:
                return False
            # Use cv2.pointPolygonTest with pixel coordinates
            # Scale to a reasonable size for the test
            scale = 10000
            pts = np.array(
                [(int(v[0] * scale), int(v[1] * scale)) for v in self.vertices], dtype=np.int32
            )
            test_pt = (int(x * scale), int(y * scale))
            result = cv2.pointPolygonTest(pts, test_pt, measureDist=False)
            return result >= 0

        return False

    def contains_point_pixels(self, px: int, py: int, width: int, height: int) -> bool:
        """
        Check if a pixel coordinate is inside the zone.

        Args:
            px: X coordinate in pixels
            py: Y coordinate in pixels
            width: Frame width
            height: Frame height

        Returns:
            True if point is inside the zone
        """
        if width == 0 or height == 0:
            return False
        # Convert to normalized coordinates
        norm_x = px / width
        norm_y = py / height
        return self.contains_point(norm_x, norm_y)

    def get_pixel_vertices(self, width: int, height: int) -> list[tuple[int, int]]:
        """
        Convert normalized vertices to pixel coordinates.

        Args:
            width: Frame width
            height: Frame height

        Returns:
            List of (x, y) pixel coordinates
        """
        return [(int(v[0] * width), int(v[1] * height)) for v in self.vertices]

    def get_bounding_rect(self) -> tuple[float, float, float, float]:
        """
        Get bounding rectangle in normalized coordinates.

        Returns:
            (x, y, width, height) tuple
        """
        if not self.vertices:
            return (0, 0, 0, 0)

        if self.shape == ZoneShape.CIRCLE:
            cx, cy = self.vertices[0]
            if len(self.vertices) > 1:
                rx, ry = self.vertices[1]
                radius = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            else:
                radius = 0
            return (cx - radius, cy - radius, radius * 2, radius * 2)

        # For rectangle and polygon, compute bounding box
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def get_center(self) -> tuple[float, float]:
        """Get the center point of the zone in normalized coordinates."""
        if not self.vertices:
            return (0.5, 0.5)

        if self.shape == ZoneShape.CIRCLE:
            return self.vertices[0]

        # For rectangle and polygon, use centroid
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "shape": self.shape.value,
            "vertices": self.vertices,
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Zone":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Zone"),
            shape=ZoneShape(data.get("shape", "rectangle")),
            vertices=[tuple(v) for v in data.get("vertices", [])],
            color=tuple(data.get("color", [0, 255, 0])),
        )


@dataclass
class ZoneState:
    """Current state of a zone during tracking."""

    zone_id: str
    zone_name: str
    occupied: bool = False
    object_count: int = 0
    object_ids: set[int] = field(default_factory=set)

    # Event flags (reset each frame)
    entered: bool = False  # Object entered this frame
    exited: bool = False  # Object exited this frame


@dataclass
class ZoneConfiguration:
    """
    Configuration containing all zones for an experiment.

    Stores zones and the resolution they were configured at.
    """

    zones: list[Zone] = field(default_factory=list)
    config_width: int = 0
    config_height: int = 0

    def add_zone(self, zone: Zone) -> None:
        """Add a zone to the configuration."""
        self.zones.append(zone)
        logger.info(f"Added zone: {zone.name} ({zone.shape.value})")

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID."""
        for i, zone in enumerate(self.zones):
            if zone.id == zone_id:
                removed = self.zones.pop(i)
                logger.info(f"Removed zone: {removed.name}")
                return True
        return False

    def get_zone(self, zone_id: str) -> Zone | None:
        """Get a zone by ID."""
        for zone in self.zones:
            if zone.id == zone_id:
                return zone
        return None

    def get_zone_by_name(self, name: str) -> Zone | None:
        """Get a zone by name."""
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None

    def clear(self) -> None:
        """Clear all zones."""
        self.zones.clear()
        self.config_width = 0
        self.config_height = 0
        logger.info("Cleared all zones")

    def point_in_zones(self, x: float, y: float) -> list[str]:
        """
        Get list of zone IDs containing a point (normalized coords).

        Args:
            x: X coordinate (0-1)
            y: Y coordinate (0-1)

        Returns:
            List of zone IDs containing the point
        """
        return [zone.id for zone in self.zones if zone.contains_point(x, y)]

    def point_in_zones_pixels(self, px: int, py: int, width: int, height: int) -> list[str]:
        """
        Get list of zone IDs containing a pixel point.

        Args:
            px: X coordinate in pixels
            py: Y coordinate in pixels
            width: Frame width
            height: Frame height

        Returns:
            List of zone IDs containing the point
        """
        if width == 0 or height == 0:
            return []
        norm_x = px / width
        norm_y = py / height
        return self.point_in_zones(norm_x, norm_y)

    def get_zone_names_for_point(self, x: float, y: float) -> list[str]:
        """
        Get list of zone names containing a point (normalized coords).

        Args:
            x: X coordinate (0-1)
            y: Y coordinate (0-1)

        Returns:
            List of zone names containing the point
        """
        return [zone.name for zone in self.zones if zone.contains_point(x, y)]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "zones": [zone.to_dict() for zone in self.zones],
            "config_width": self.config_width,
            "config_height": self.config_height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneConfiguration":
        """Create from dictionary."""
        config = cls()
        config.zones = [Zone.from_dict(z) for z in data.get("zones", [])]
        config.config_width = data.get("config_width", 0)
        config.config_height = data.get("config_height", 0)
        return config

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved zone configuration to {path}")

    def load(self, path: Path) -> bool:
        """Load configuration from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)

            self.zones = [Zone.from_dict(z) for z in data.get("zones", [])]
            self.config_width = data.get("config_width", 0)
            self.config_height = data.get("config_height", 0)

            logger.info(f"Loaded {len(self.zones)} zones from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load zone configuration: {e}")
            return False


class ZoneTracker:
    """
    Tracks object occupancy in zones across frames.

    Maintains state for each zone including occupancy, object count,
    and enter/exit events.
    """

    def __init__(self):
        """Initialize the zone tracker."""
        self._zone_config: ZoneConfiguration | None = None
        self._zone_states: dict[str, ZoneState] = {}
        self._prev_zone_objects: dict[str, set[int]] = {}  # zone_id -> object IDs

    def set_zone_configuration(self, config: ZoneConfiguration) -> None:
        """Set the zone configuration to track."""
        self._zone_config = config
        self._zone_states.clear()
        self._prev_zone_objects.clear()

        # Initialize state for each zone
        for zone in config.zones:
            self._zone_states[zone.id] = ZoneState(zone_id=zone.id, zone_name=zone.name)
            self._prev_zone_objects[zone.id] = set()

    def update(
        self, tracked_objects: list[Any], frame_width: int, frame_height: int
    ) -> dict[str, ZoneState]:
        """
        Update zone states based on tracked objects.

        Args:
            tracked_objects: List of TrackedObject instances
            frame_width: Current frame width
            frame_height: Current frame height

        Returns:
            Dictionary of zone_id -> ZoneState
        """
        if self._zone_config is None:
            return {}

        # Build current zone occupancy
        current_zone_objects: dict[str, set[int]] = {
            zone.id: set() for zone in self._zone_config.zones
        }

        # Check each tracked object against each zone
        for obj in tracked_objects:
            # Get object center
            if hasattr(obj, "centroid"):
                cx, cy = obj.centroid
            elif hasattr(obj, "bbox"):
                x, y, w, h = obj.bbox
                cx = x + w / 2
                cy = y + h / 2
            else:
                continue

            track_id = getattr(obj, "track_id", id(obj))

            # Check against each zone
            for zone in self._zone_config.zones:
                if zone.contains_point_pixels(int(cx), int(cy), frame_width, frame_height):
                    current_zone_objects[zone.id].add(track_id)

        # Update zone states with enter/exit events
        for zone in self._zone_config.zones:
            zone_id = zone.id
            current_objects = current_zone_objects[zone_id]
            prev_objects = self._prev_zone_objects.get(zone_id, set())

            # Calculate enter/exit
            entered_objects = current_objects - prev_objects
            exited_objects = prev_objects - current_objects

            # Update state
            state = self._zone_states[zone_id]
            state.object_ids = current_objects.copy()
            state.object_count = len(current_objects)
            state.occupied = len(current_objects) > 0
            state.entered = len(entered_objects) > 0
            state.exited = len(exited_objects) > 0

            # Update previous state for next frame
            self._prev_zone_objects[zone_id] = current_objects.copy()

            # Log events
            if state.entered:
                logger.debug(f"Zone '{zone.name}': object(s) entered - {entered_objects}")
            if state.exited:
                logger.debug(f"Zone '{zone.name}': object(s) exited - {exited_objects}")

        return self._zone_states.copy()

    def get_zone_states(self) -> dict[str, ZoneState]:
        """Get current zone states."""
        return self._zone_states.copy()

    def get_zone_state(self, zone_id: str) -> ZoneState | None:
        """Get state for a specific zone."""
        return self._zone_states.get(zone_id)

    def reset(self) -> None:
        """Reset all zone states."""
        for state in self._zone_states.values():
            state.occupied = False
            state.object_count = 0
            state.object_ids.clear()
            state.entered = False
            state.exited = False
        self._prev_zone_objects = {zone_id: set() for zone_id in self._prev_zone_objects}


def draw_zones(
    frame: np.ndarray, zone_config: ZoneConfiguration, alpha: float = 0.3, show_labels: bool = True
) -> np.ndarray:
    """
    Draw zones on a frame with semi-transparent fill.

    Args:
        frame: Input frame (BGR)
        zone_config: Zone configuration
        alpha: Transparency for fill (0-1)
        show_labels: Whether to show zone names

    Returns:
        Frame with zones drawn
    """
    if not zone_config.zones:
        return frame

    output = frame.copy()
    h, w = frame.shape[:2]

    # Create overlay for semi-transparent fill
    overlay = frame.copy()

    for zone in zone_config.zones:
        color = zone.color
        pixel_verts = zone.get_pixel_vertices(w, h)

        if zone.shape == ZoneShape.RECTANGLE:
            if len(pixel_verts) >= 2:
                x1, y1 = pixel_verts[0]
                x2, y2 = pixel_verts[1]
                # Draw filled rectangle on overlay
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                # Draw border on output
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        elif zone.shape == ZoneShape.CIRCLE:
            if len(pixel_verts) >= 2:
                cx, cy = pixel_verts[0]
                rx, ry = pixel_verts[1]
                radius = int(math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2))
                # Draw filled circle on overlay
                cv2.circle(overlay, (cx, cy), radius, color, -1)
                # Draw border on output
                cv2.circle(output, (cx, cy), radius, color, 2)

        elif zone.shape == ZoneShape.POLYGON:
            if len(pixel_verts) >= 3:
                pts = np.array(pixel_verts, dtype=np.int32)
                # Draw filled polygon on overlay
                cv2.fillPoly(overlay, [pts], color)
                # Draw border on output
                cv2.polylines(output, [pts], True, color, 2)

        # Draw label
        if show_labels:
            center = zone.get_center()
            label_x = int(center[0] * w)
            label_y = int(center[1] * h)

            # Draw label background
            label = zone.name
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                output,
                (label_x - 2, label_y - label_h - 4),
                (label_x + label_w + 2, label_y + baseline),
                color,
                -1,
            )
            # Draw label text
            cv2.putText(
                output, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

    # Blend overlay with output for semi-transparent fill
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output
