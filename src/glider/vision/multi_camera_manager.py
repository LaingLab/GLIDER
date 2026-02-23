"""
Multi-Camera Manager - Coordinate multiple cameras for simultaneous capture.

Manages multiple CameraManager instances, allowing recording from
several cameras at once while designating one as the primary
camera for CV processing.
"""

import logging
import threading
from collections.abc import Callable

import numpy as np

from glider.vision.camera_manager import CameraInfo, CameraManager, CameraSettings

logger = logging.getLogger(__name__)


class MultiCameraManager:
    """
    Coordinates multiple CameraManager instances for simultaneous capture.

    Features:
    - Manage multiple cameras with unique IDs
    - Designate one camera as "primary" for CV processing
    - Start/stop streaming on all cameras
    - Per-camera frame callbacks
    """

    def __init__(self):
        """Initialize the multi-camera manager."""
        self._cameras: dict[str, CameraManager] = {}
        self._camera_settings: dict[str, CameraSettings] = {}
        self._primary_camera_id: str | None = None
        self._enabled = False
        self._lock = threading.Lock()

        # Callbacks for each camera
        self._frame_callbacks: dict[str, list[Callable[[str, np.ndarray, float], None]]] = {}

    @property
    def enabled(self) -> bool:
        """Whether multi-camera mode is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable multi-camera mode."""
        self._enabled = value

    @property
    def primary_camera_id(self) -> str | None:
        """ID of the primary camera for CV processing."""
        return self._primary_camera_id

    @property
    def primary_camera(self) -> CameraManager | None:
        """Get the primary camera instance."""
        if self._primary_camera_id and self._primary_camera_id in self._cameras:
            return self._cameras[self._primary_camera_id]
        return None

    @property
    def cameras(self) -> dict[str, CameraManager]:
        """Get all camera instances."""
        return self._cameras.copy()

    @property
    def camera_count(self) -> int:
        """Number of connected cameras."""
        return len(self._cameras)

    def get_camera(self, camera_id: str) -> CameraManager | None:
        """Get a specific camera by ID."""
        return self._cameras.get(camera_id)

    def get_camera_settings(self, camera_id: str) -> CameraSettings | None:
        """Get settings for a specific camera."""
        return self._camera_settings.get(camera_id)

    @staticmethod
    def enumerate_all_cameras(max_cameras: int = 10) -> list[CameraInfo]:
        """
        Enumerate all available camera devices.

        Args:
            max_cameras: Maximum number of camera indices to check

        Returns:
            List of available cameras
        """
        return CameraManager.enumerate_cameras(max_cameras)

    def camera_id_from_index(self, camera_index: int) -> str:
        """Generate a camera ID from a camera index."""
        return f"cam_{camera_index}"

    def add_camera(self, camera_id: str, settings: CameraSettings) -> bool:
        """
        Add and connect a new camera.

        Args:
            camera_id: Unique identifier for this camera
            settings: Camera settings to apply

        Returns:
            True if camera was added and connected successfully
        """
        with self._lock:
            if camera_id in self._cameras:
                logger.warning(f"Camera {camera_id} already exists")
                return False

            # Create new camera manager
            camera = CameraManager()
            if not camera.connect(settings):
                logger.error(f"Failed to connect camera {camera_id}")
                return False

            self._cameras[camera_id] = camera
            self._camera_settings[camera_id] = settings
            self._frame_callbacks[camera_id] = []

            # Register internal frame callback to route to external callbacks
            camera.on_frame(lambda frame, ts, cid=camera_id: self._on_camera_frame(cid, frame, ts))

            # Set as primary if first camera
            if self._primary_camera_id is None:
                self._primary_camera_id = camera_id
                logger.info(f"Camera {camera_id} set as primary")

            logger.info(f"Added camera {camera_id} (index {settings.camera_index})")
            return True

    def remove_camera(self, camera_id: str) -> None:
        """
        Disconnect and remove a camera.

        Args:
            camera_id: ID of camera to remove
        """
        with self._lock:
            if camera_id not in self._cameras:
                return

            camera = self._cameras[camera_id]
            if camera.is_streaming:
                camera.stop_streaming()
            camera.disconnect()

            del self._cameras[camera_id]
            del self._camera_settings[camera_id]
            del self._frame_callbacks[camera_id]

            # Update primary if needed
            if self._primary_camera_id == camera_id:
                if self._cameras:
                    self._primary_camera_id = next(iter(self._cameras.keys()))
                    logger.info(f"Primary camera changed to {self._primary_camera_id}")
                else:
                    self._primary_camera_id = None

            logger.info(f"Removed camera {camera_id}")

    def remove_all_cameras(self) -> None:
        """Disconnect and remove all cameras."""
        camera_ids = list(self._cameras.keys())
        for camera_id in camera_ids:
            self.remove_camera(camera_id)

    def set_primary_camera(self, camera_id: str) -> bool:
        """
        Designate a camera as primary for CV processing.

        Args:
            camera_id: ID of camera to set as primary

        Returns:
            True if camera was set as primary
        """
        with self._lock:
            if camera_id not in self._cameras:
                logger.error(f"Camera {camera_id} not found")
                return False

            old_primary = self._primary_camera_id
            self._primary_camera_id = camera_id
            logger.info(f"Primary camera changed from {old_primary} to {camera_id}")
            return True

    def start_streaming(self, camera_id: str) -> bool:
        """
        Start streaming on a specific camera.

        Args:
            camera_id: ID of camera to start

        Returns:
            True if streaming started
        """
        camera = self._cameras.get(camera_id)
        if camera is None:
            return False
        return camera.start_streaming()

    def stop_streaming(self, camera_id: str) -> None:
        """
        Stop streaming on a specific camera.

        Args:
            camera_id: ID of camera to stop
        """
        camera = self._cameras.get(camera_id)
        if camera:
            camera.stop_streaming()

    def start_all_streaming(self) -> dict[str, bool]:
        """
        Start streaming on all connected cameras.

        Returns:
            Dictionary of camera_id -> success
        """
        results = {}
        for camera_id, camera in self._cameras.items():
            results[camera_id] = camera.start_streaming()
        return results

    def stop_all_streaming(self) -> None:
        """Stop streaming on all cameras."""
        for camera in self._cameras.values():
            camera.stop_streaming()

    def is_camera_streaming(self, camera_id: str) -> bool:
        """Check if a specific camera is streaming."""
        camera = self._cameras.get(camera_id)
        return camera.is_streaming if camera else False

    def any_streaming(self) -> bool:
        """Check if any camera is streaming."""
        return any(cam.is_streaming for cam in self._cameras.values())

    def all_streaming(self) -> bool:
        """Check if all cameras are streaming."""
        return all(cam.is_streaming for cam in self._cameras.values()) if self._cameras else False

    def on_frame(self, camera_id: str, callback: Callable[[str, np.ndarray, float], None]) -> None:
        """
        Register callback for frames from a specific camera.

        Args:
            camera_id: Camera to receive frames from
            callback: Function called with (camera_id, frame, timestamp)
        """
        if camera_id in self._frame_callbacks:
            self._frame_callbacks[camera_id].append(callback)

    def on_primary_frame(self, callback: Callable[[str, np.ndarray, float], None]) -> None:
        """
        Register callback for frames from the primary camera only.

        The callback will follow the primary camera if it changes.

        Args:
            callback: Function called with (camera_id, frame, timestamp)
        """
        # Store callback to be called only when frame is from primary
        self._primary_frame_callback = callback

    def remove_frame_callback(self, camera_id: str, callback: Callable) -> None:
        """Remove a frame callback for a specific camera."""
        if camera_id in self._frame_callbacks:
            if callback in self._frame_callbacks[camera_id]:
                self._frame_callbacks[camera_id].remove(callback)

    def _on_camera_frame(self, camera_id: str, frame: np.ndarray, timestamp: float) -> None:
        """
        Internal handler for camera frames.

        Routes frames to registered callbacks.
        """
        # Call per-camera callbacks
        for callback in self._frame_callbacks.get(camera_id, []):
            try:
                callback(camera_id, frame, timestamp)
            except Exception as e:
                logger.error(f"Frame callback error for {camera_id}: {e}")

        # Call primary frame callback if this is the primary camera
        if camera_id == self._primary_camera_id and hasattr(self, "_primary_frame_callback"):
            try:
                self._primary_frame_callback(camera_id, frame, timestamp)
            except Exception as e:
                logger.error(f"Primary frame callback error: {e}")

    def get_frame(self, camera_id: str) -> tuple[np.ndarray, float] | None:
        """
        Get the latest frame from a specific camera.

        Args:
            camera_id: ID of camera

        Returns:
            Tuple of (frame, timestamp) or None
        """
        camera = self._cameras.get(camera_id)
        return camera.get_frame() if camera else None

    def get_all_frames(self) -> dict[str, tuple[np.ndarray, float]]:
        """
        Get the latest frame from all cameras.

        Returns:
            Dictionary of camera_id -> (frame, timestamp)
        """
        frames = {}
        for camera_id, camera in self._cameras.items():
            frame_data = camera.get_frame()
            if frame_data:
                frames[camera_id] = frame_data
        return frames

    def get_camera_fps(self, camera_id: str) -> float:
        """Get current FPS for a specific camera."""
        camera = self._cameras.get(camera_id)
        return camera.current_fps if camera else 0.0

    def get_camera_resolution(self, camera_id: str) -> tuple[int, int] | None:
        """Get resolution for a specific camera."""
        settings = self._camera_settings.get(camera_id)
        return settings.resolution if settings else None

    def is_primary(self, camera_id: str) -> bool:
        """Check if a camera is the primary camera."""
        return camera_id == self._primary_camera_id

    def shutdown(self) -> None:
        """Shutdown all cameras."""
        self.stop_all_streaming()
        self.remove_all_cameras()
        logger.info("Multi-camera manager shutdown complete")
