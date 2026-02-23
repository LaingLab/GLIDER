"""
Multi-Video Recorder - Record video from multiple cameras simultaneously.

Records video from all connected cameras to separate files,
with the primary camera optionally having an annotated video
with tracking overlays.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from glider.vision.frame_writer import FrameWriterThread
from glider.vision.video_recorder import RecordingState, VideoFormat

if TYPE_CHECKING:
    from glider.vision.multi_camera_manager import MultiCameraManager

logger = logging.getLogger(__name__)


class MultiVideoRecorder:
    """
    Records video from multiple cameras simultaneously.

    Each camera records to its own file. The primary camera
    can optionally have a second annotated video with
    tracking overlays.

    File naming convention:
    - Primary camera: {experiment}_{timestamp}.mp4
    - Primary annotated: {experiment}_{timestamp}_annotated.mp4
    - Secondary cameras: {experiment}_{timestamp}_cam{N}.mp4
    """

    def __init__(self, multi_camera_manager: "MultiCameraManager"):
        """
        Initialize the multi-video recorder.

        Args:
            multi_camera_manager: MultiCameraManager instance to record from
        """
        self._multi_cam = multi_camera_manager
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._writer_threads: dict[str, FrameWriterThread] = {}
        self._annotated_writer: cv2.VideoWriter | None = None
        self._annotated_writer_thread: FrameWriterThread | None = None
        self._file_paths: dict[str, Path] = {}
        self._annotated_file_path: Path | None = None
        self._frame_counts: dict[str, int] = {}
        self._frames_dropped: dict[str, int] = {}
        self._annotated_frame_count = 0
        self._state = RecordingState.IDLE
        self._output_dir: Path = Path.cwd()
        self._start_time: datetime | None = None
        self._video_format = VideoFormat()
        self._lock = threading.Lock()
        self._frame_callbacks_registered: dict[str, bool] = {}
        self._record_annotated = False
        self._recording_fps: dict[str, float] = {}
        self._buffer_size: int | None = None

    @property
    def is_recording(self) -> bool:
        """Whether video is currently being recorded."""
        return self._state == RecordingState.RECORDING

    @property
    def state(self) -> RecordingState:
        """Current recording state."""
        return self._state

    @property
    def file_paths(self) -> dict[str, Path]:
        """Paths to all video files being recorded."""
        return self._file_paths.copy()

    @property
    def annotated_file_path(self) -> Path | None:
        """Path to the annotated video file (primary camera only)."""
        return self._annotated_file_path

    @property
    def primary_file_path(self) -> Path | None:
        """Path to the primary camera's video file."""
        primary_id = self._multi_cam.primary_camera_id
        if primary_id and primary_id in self._file_paths:
            return self._file_paths[primary_id]
        return None

    def set_output_directory(self, path: Path) -> None:
        """Set the output directory for video files."""
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Multi-video output directory: {self._output_dir}")

    def set_video_format(self, codec: str = "mp4v", extension: str = ".mp4") -> None:
        """Set the video codec and file extension."""
        self._video_format = VideoFormat(codec=codec, extension=extension)
        logger.debug(f"Video format: {codec} ({extension})")

    def _generate_filename(self, base_name: str, camera_id: str, is_annotated: bool = False) -> str:
        """
        Generate filename for a camera's video.

        Args:
            base_name: Base experiment name
            camera_id: Camera identifier
            is_annotated: Whether this is for annotated video

        Returns:
            Formatted filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in base_name)
        safe_name = safe_name.strip().replace(" ", "_") or "experiment"

        primary_id = self._multi_cam.primary_camera_id

        if is_annotated:
            return f"{safe_name}_{timestamp}_annotated{self._video_format.extension}"
        elif camera_id == primary_id:
            # Primary camera uses simple naming for backwards compatibility
            return f"{safe_name}_{timestamp}{self._video_format.extension}"
        else:
            # Secondary cameras include camera identifier
            # Extract camera number from camera_id (e.g., "cam_1" -> "1")
            cam_num = camera_id.replace("cam_", "")
            return f"{safe_name}_{timestamp}_cam{cam_num}{self._video_format.extension}"

    async def start(
        self, experiment_name: str = "experiment", record_annotated: bool = False
    ) -> dict[str, Path]:
        """
        Start recording video from all connected cameras.

        Args:
            experiment_name: Name for the video files
            record_annotated: Whether to also record annotated video (primary only)

        Returns:
            Dictionary of camera_id -> file path
        """
        if self._state == RecordingState.RECORDING:
            logger.warning("Recording already in progress")
            return self._file_paths

        self._record_annotated = record_annotated
        self._start_time = datetime.now()
        self._file_paths.clear()
        self._frame_counts.clear()
        self._frames_dropped.clear()
        self._annotated_frame_count = 0

        fourcc = cv2.VideoWriter_fourcc(*self._video_format.codec)

        # Build FrameWriterThread kwargs
        fwt_kwargs = {}
        if self._buffer_size is not None:
            fwt_kwargs["max_queue_size"] = self._buffer_size

        with self._lock:
            # Create writer for each camera
            for camera_id, camera in self._multi_cam.cameras.items():
                settings = self._multi_cam.get_camera_settings(camera_id)
                if settings is None:
                    continue

                # Determine FPS
                actual_fps = camera.current_fps
                recording_fps = actual_fps if actual_fps > 1.0 else settings.fps
                self._recording_fps[camera_id] = recording_fps

                # Generate filename
                filename = self._generate_filename(experiment_name, camera_id)
                file_path = self._output_dir / filename

                # Create writer
                writer = cv2.VideoWriter(str(file_path), fourcc, recording_fps, settings.resolution)

                if not writer.isOpened():
                    logger.error(f"Failed to create video writer for {camera_id}: {file_path}")
                    continue

                self._writers[camera_id] = writer
                self._file_paths[camera_id] = file_path
                self._frame_counts[camera_id] = 0
                self._frames_dropped[camera_id] = 0

                # Wrap in FrameWriterThread
                fwt = FrameWriterThread(writer, **fwt_kwargs)
                fwt.start()
                self._writer_threads[camera_id] = fwt

                # Register frame callback
                if not self._frame_callbacks_registered.get(camera_id, False):
                    self._multi_cam.on_frame(camera_id, self._on_frame)
                    self._frame_callbacks_registered[camera_id] = True

                logger.info(f"Recording {camera_id} to {file_path} at {recording_fps:.1f} fps")

            # Create annotated writer for primary camera
            primary_id = self._multi_cam.primary_camera_id
            if record_annotated and primary_id and primary_id in self._multi_cam.cameras:
                primary_settings = self._multi_cam.get_camera_settings(primary_id)
                if primary_settings:
                    annotated_filename = self._generate_filename(
                        experiment_name, primary_id, is_annotated=True
                    )
                    self._annotated_file_path = self._output_dir / annotated_filename

                    recording_fps = self._recording_fps.get(primary_id, primary_settings.fps)
                    self._annotated_writer = cv2.VideoWriter(
                        str(self._annotated_file_path),
                        fourcc,
                        recording_fps,
                        primary_settings.resolution,
                    )

                    if self._annotated_writer.isOpened():
                        self._annotated_writer_thread = FrameWriterThread(
                            self._annotated_writer, **fwt_kwargs
                        )
                        self._annotated_writer_thread.start()
                        logger.info(f"Recording annotated video to {self._annotated_file_path}")
                    else:
                        logger.warning("Failed to create annotated video writer")
                        self._annotated_writer = None

        self._state = RecordingState.RECORDING
        logger.info(f"Started multi-camera recording ({len(self._writers)} cameras)")
        return self._file_paths

    def _on_frame(self, camera_id: str, frame: np.ndarray, timestamp: float) -> None:
        """
        Handle incoming frame for recording.

        Enqueues the frame for async writing by the per-camera FrameWriterThread.

        Args:
            camera_id: ID of camera that sent frame
            frame: Frame from camera
            timestamp: Frame timestamp
        """
        if self._state != RecordingState.RECORDING:
            return

        fwt = self._writer_threads.get(camera_id)
        if fwt is not None:
            if fwt.enqueue(frame.copy()):
                self._frame_counts[camera_id] = self._frame_counts.get(camera_id, 0) + 1
            else:
                self._frames_dropped[camera_id] = self._frames_dropped.get(camera_id, 0) + 1
                dropped = self._frames_dropped[camera_id]
                if dropped == 1 or dropped % 100 == 0:
                    logger.warning(
                        f"MultiVideoRecorder: {camera_id} frame dropped "
                        f"(total dropped: {dropped})"
                    )

    def write_annotated_frame(self, frame: np.ndarray) -> bool:
        """
        Write an annotated frame (primary camera only).

        This should be called with frames that have been processed
        with tracking overlays.

        Args:
            frame: Annotated frame with overlays

        Returns:
            True if frame was written (enqueued)
        """
        if self._state != RecordingState.RECORDING:
            return False

        if not self._record_annotated:
            return False

        if self._annotated_writer_thread is not None:
            if self._annotated_writer_thread.enqueue(frame.copy()):
                self._annotated_frame_count += 1
                return True
        return False

    async def stop(self) -> dict[str, Path]:
        """
        Stop recording and finalize all video files.

        Returns:
            Dictionary of camera_id -> saved file path
        """
        if self._state not in (RecordingState.RECORDING, RecordingState.PAUSED):
            return {}

        self._state = RecordingState.FINALIZING

        # Drain all per-camera writer threads before releasing writers
        for camera_id, fwt in self._writer_threads.items():
            fwt.stop(timeout=30.0)
            dropped = self._frames_dropped.get(camera_id, 0)
            if dropped > 0:
                logger.warning(f"MultiVideoRecorder: {camera_id} total frames dropped: {dropped}")
        self._writer_threads.clear()

        if self._annotated_writer_thread is not None:
            self._annotated_writer_thread.stop(timeout=30.0)
            self._annotated_writer_thread = None

        with self._lock:
            # Release all writers
            for camera_id, writer in self._writers.items():
                if writer is not None:
                    writer.release()
                    logger.info(
                        f"Saved {camera_id} video: {self._file_paths.get(camera_id)} "
                        f"({self._frame_counts.get(camera_id, 0)} frames)"
                    )

            if self._annotated_writer is not None:
                self._annotated_writer.release()
                self._annotated_writer = None
                logger.info(
                    f"Saved annotated video: {self._annotated_file_path} "
                    f"({self._annotated_frame_count} frames)"
                )

            self._writers.clear()

        saved_paths = self._file_paths.copy()
        self._state = RecordingState.IDLE

        duration = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        logger.info(f"Stopped multi-camera recording ({len(saved_paths)} files, {duration:.1f}s)")

        return saved_paths

    async def pause(self) -> None:
        """Pause recording (frames will be skipped)."""
        if self._state == RecordingState.RECORDING:
            self._state = RecordingState.PAUSED
            logger.info("Multi-camera recording paused")

    async def resume(self) -> None:
        """Resume recording after pause."""
        if self._state == RecordingState.PAUSED:
            self._state = RecordingState.RECORDING
            logger.info("Multi-camera recording resumed")

    def get_frame_count(self, camera_id: str) -> int:
        """Get frame count for a specific camera."""
        return self._frame_counts.get(camera_id, 0)

    def get_total_frame_count(self) -> int:
        """Get total frames recorded across all cameras."""
        return sum(self._frame_counts.values())

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def set_buffer_size(self, max_frames: int) -> None:
        """
        Set the frame buffer size for writer threads.

        Must be called before start(). Has no effect on an active recording.

        Args:
            max_frames: Maximum number of frames to buffer per camera.
        """
        self._buffer_size = max_frames

    def get_recording_info(self) -> dict:
        """Get information about the current recording."""
        buffer_depths = {}
        buffer_maxes = {}
        for camera_id, fwt in self._writer_threads.items():
            buffer_depths[camera_id] = fwt.queue_depth
            buffer_maxes[camera_id] = fwt.max_queue_size

        return {
            "state": self._state.name,
            "cameras": list(self._file_paths.keys()),
            "file_paths": {k: str(v) for k, v in self._file_paths.items()},
            "annotated_file_path": (
                str(self._annotated_file_path) if self._annotated_file_path else None
            ),
            "frame_counts": self._frame_counts.copy(),
            "frames_dropped": self._frames_dropped.copy(),
            "annotated_frame_count": self._annotated_frame_count,
            "record_annotated": self._record_annotated,
            "duration": self.duration,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "format": self._video_format.to_dict(),
            "buffer_depths": buffer_depths,
            "buffer_maxes": buffer_maxes,
        }
