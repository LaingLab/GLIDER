"""
Behavioral Analysis Module - Detect behavioral states from tracking data.

Analyzes centroid movement patterns to classify behavioral states:
- FREEZE: Complete stillness for sustained duration
- IMMOBILE: Minor movements (grooming, adjusting)
- MOVING: Normal locomotion
- DARTING: Rapid escape/flight response
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class BehaviorState(Enum):
    """Behavioral state classifications."""

    UNKNOWN = auto()
    FREEZE = auto()
    IMMOBILE = auto()
    MOVING = auto()
    DARTING = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class BehaviorSettings:
    """Configuration for behavior detection thresholds."""

    enabled: bool = True
    freeze_threshold: float = 1.0  # Max pixels/frame for freeze
    immobile_threshold: float = 5.0  # Max pixels/frame for immobile
    dart_threshold: float = 50.0  # Min pixels/frame for darting
    freeze_duration: int = 15  # Frames required to confirm freeze
    smoothing_window: int = 5  # Frames for velocity smoothing

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "freeze_threshold": self.freeze_threshold,
            "immobile_threshold": self.immobile_threshold,
            "dart_threshold": self.dart_threshold,
            "freeze_duration": self.freeze_duration,
            "smoothing_window": self.smoothing_window,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BehaviorSettings":
        return cls(
            enabled=data.get("enabled", True),
            freeze_threshold=data.get("freeze_threshold", 1.0),
            immobile_threshold=data.get("immobile_threshold", 5.0),
            dart_threshold=data.get("dart_threshold", 50.0),
            freeze_duration=data.get("freeze_duration", 15),
            smoothing_window=data.get("smoothing_window", 5),
        )


@dataclass
class ObjectBehaviorState:
    """Tracks behavioral state for a single object."""

    current_state: BehaviorState = BehaviorState.UNKNOWN
    velocity: float = 0.0  # Current smoothed velocity in pixels/frame
    low_movement_frames: int = 0  # Consecutive frames below freeze threshold
    velocities: list[float] = field(default_factory=list)  # Recent velocity history


class BehaviorAnalyzer:
    """
    Analyzes object tracking data to determine behavioral states.

    Uses centroid trail history to calculate velocity and classify
    behavior based on configurable thresholds.
    """

    def __init__(self, settings: BehaviorSettings | None = None):
        """
        Initialize the behavior analyzer.

        Args:
            settings: Behavior detection settings, uses defaults if None
        """
        self._settings = settings or BehaviorSettings()
        # Track state per object ID
        self._object_states: dict[int, ObjectBehaviorState] = {}

    @property
    def settings(self) -> BehaviorSettings:
        """Get current settings."""
        return self._settings

    @settings.setter
    def settings(self, value: BehaviorSettings) -> None:
        """Update settings."""
        self._settings = value

    def update_settings(self, settings: BehaviorSettings) -> None:
        """
        Update behavior detection settings.

        Args:
            settings: New settings to apply
        """
        self._settings = settings

    def analyze(
        self,
        track_id: int,
        trail_history: list[tuple[int, int]],
    ) -> tuple[BehaviorState, float]:
        """
        Analyze trail history to determine current behavioral state.

        Args:
            track_id: Unique identifier for the tracked object
            trail_history: List of (x, y) centroid positions, newest last

        Returns:
            Tuple of (BehaviorState, velocity in pixels/frame)
        """
        if not self._settings.enabled:
            return BehaviorState.UNKNOWN, 0.0

        # Get or create state tracker for this object
        if track_id not in self._object_states:
            self._object_states[track_id] = ObjectBehaviorState()
        obj_state = self._object_states[track_id]

        # Need at least 2 points to calculate velocity
        if len(trail_history) < 2:
            return BehaviorState.UNKNOWN, 0.0

        # Calculate instantaneous velocity (distance between last two points)
        x1, y1 = trail_history[-2]
        x2, y2 = trail_history[-1]
        dx = x2 - x1
        dy = y2 - y1
        instant_velocity = math.sqrt(dx * dx + dy * dy)

        # Add to velocity history
        obj_state.velocities.append(instant_velocity)

        # Keep only the smoothing window size
        window = self._settings.smoothing_window
        if len(obj_state.velocities) > window:
            obj_state.velocities = obj_state.velocities[-window:]

        # Calculate smoothed velocity (average over window)
        smoothed_velocity = sum(obj_state.velocities) / len(obj_state.velocities)
        obj_state.velocity = smoothed_velocity

        # Classify behavior based on velocity thresholds
        state = self._classify_state(smoothed_velocity, obj_state)
        obj_state.current_state = state

        return state, smoothed_velocity

    def _classify_state(self, velocity: float, obj_state: ObjectBehaviorState) -> BehaviorState:
        """
        Classify behavioral state based on velocity.

        Args:
            velocity: Smoothed velocity in pixels/frame
            obj_state: Object's state tracker

        Returns:
            Classified BehaviorState
        """
        settings = self._settings

        # Check for darting (highest priority - sudden fast movement)
        if velocity >= settings.dart_threshold:
            obj_state.low_movement_frames = 0
            return BehaviorState.DARTING

        # Check for freeze (requires sustained low movement)
        if velocity < settings.freeze_threshold:
            obj_state.low_movement_frames += 1
            if obj_state.low_movement_frames >= settings.freeze_duration:
                return BehaviorState.FREEZE
            # Not yet frozen, but very low movement
            return BehaviorState.IMMOBILE

        # Reset freeze counter when movement detected
        obj_state.low_movement_frames = 0

        # Check for immobile (low but not freeze-level movement)
        if velocity < settings.immobile_threshold:
            return BehaviorState.IMMOBILE

        # Normal movement
        return BehaviorState.MOVING

    def get_state(self, track_id: int) -> tuple[BehaviorState, float]:
        """
        Get the current behavioral state for an object.

        Args:
            track_id: Object's track ID

        Returns:
            Tuple of (BehaviorState, velocity) or (UNKNOWN, 0.0) if not tracked
        """
        if track_id in self._object_states:
            state = self._object_states[track_id]
            return state.current_state, state.velocity
        return BehaviorState.UNKNOWN, 0.0

    def remove_object(self, track_id: int) -> None:
        """
        Remove tracking data for an object.

        Args:
            track_id: Object's track ID to remove
        """
        self._object_states.pop(track_id, None)

    def clear(self) -> None:
        """Clear all tracked object states."""
        self._object_states.clear()

    def get_state_color(self, state: BehaviorState) -> tuple[int, int, int]:
        """
        Get BGR color for a behavioral state.

        Args:
            state: The behavioral state

        Returns:
            BGR color tuple for OpenCV
        """
        colors = {
            BehaviorState.UNKNOWN: (128, 128, 128),  # Gray
            BehaviorState.FREEZE: (255, 0, 0),  # Blue
            BehaviorState.IMMOBILE: (0, 255, 255),  # Yellow
            BehaviorState.MOVING: (0, 255, 0),  # Green
            BehaviorState.DARTING: (0, 0, 255),  # Red
        }
        return colors.get(state, (128, 128, 128))
