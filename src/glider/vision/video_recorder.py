"""
Video Recorder - Record video synced with experiment lifecycle.

Records video to file synchronized with experiment start/stop,
generating filenames that match the DataRecorder pattern.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from glider.vision.frame_writer import FrameWriterThread

if TYPE_CHECKING:
    from glider.vision.camera_manager import CameraManager

logger = logging.getLogger(__name__)


class RecordingState(Enum):
    """State of the video recording."""

    IDLE = auto()
    RECORDING = auto()
    PAUSED = auto()
    FINALIZING = auto()


@dataclass
class VideoFormat:
    """Video output format configuration."""

    codec: str = "mp4v"  # OpenCV fourcc code
    extension: str = ".mp4"
    quality: int = 95  # For JPEG-based codecs

    def to_dict(self) -> dict:
        return {
            "codec": self.codec,
            "extension": self.extension,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoFormat":
        return cls(
            codec=data.get("codec", "mp4v"),
            extension=data.get("extension", ".mp4"),
            quality=data.get("quality", 95),
        )


class VideoRecorder:
    """
    Records video synchronized with experiment lifecycle.

    Integrates with GliderCore to automatically start/stop recording
    when experiments run. Saves video files alongside CSV data files
    with matching timestamps.

    Supports dual recording: raw video and annotated video with tracking overlays.
    """

    def __init__(self, camera_manager: "CameraManager"):
        """
        Initialize the video recorder.

        Args:
            camera_manager: CameraManager instance to capture frames from
        """
        self._camera = camera_manager
        self._writer: Optional[cv2.VideoWriter] = None
        self._annotated_writer: Optional[cv2.VideoWriter] = None
        self._writer_thread: Optional[FrameWriterThread] = None
        self._annotated_writer_thread: Optional[FrameWriterThread] = None
        self._state = RecordingState.IDLE
        self._output_dir: Path = Path.cwd()
        self._file_path: Optional[Path] = None
        self._annotated_file_path: Optional[Path] = None
        self._start_time: Optional[datetime] = None
        self._frame_count = 0
        self._annotated_frame_count = 0
        self._recording_fps = 30.0  # Actual FPS used for recording
        self._video_format = VideoFormat()
        self._lock = threading.Lock()
        self._frame_callback_registered = False
        self._record_annotated = False  # Whether to also record annotated video
        self._buffer_size: Optional[int] = None  # None = use default

    @property
    def is_recording(self) -> bool:
        """Whether video is currently being recorded."""
        return self._state == RecordingState.RECORDING

    @property
    def is_paused(self) -> bool:
        """Whether recording is paused."""
        return self._state == RecordingState.PAUSED

    @property
    def state(self) -> RecordingState:
        """Current recording state."""
        return self._state

    @property
    def file_path(self) -> Optional[Path]:
        """Path to the current/last raw video file."""
        return self._file_path

    @property
    def annotated_file_path(self) -> Optional[Path]:
        """Path to the current/last annotated video file."""
        return self._annotated_file_path

    @property
    def record_annotated(self) -> bool:
        """Whether to record annotated video with tracking overlays."""
        return self._record_annotated

    @record_annotated.setter
    def record_annotated(self, value: bool) -> None:
        """Set whether to record annotated video."""
        self._record_annotated = value

    @property
    def frame_count(self) -> int:
        """Number of frames recorded."""
        return self._frame_count

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def set_output_directory(self, path: Path) -> None:
        """
        Set the output directory for video files.

        Args:
            path: Directory path for video output
        """
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Video output directory: {self._output_dir}")

    def set_video_format(self, codec: str = "mp4v", extension: str = ".mp4") -> None:
        """
        Set the video codec and file extension.

        Args:
            codec: OpenCV fourcc codec code (e.g., "mp4v", "XVID", "MJPG")
            extension: File extension (e.g., ".mp4", ".avi")
        """
        self._video_format = VideoFormat(codec=codec, extension=extension)
        logger.debug(f"Video format: {codec} ({extension})")

    def _generate_filename(self, experiment_name: str) -> str:
        """
        Generate filename matching DataRecorder pattern.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Formatted filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize name
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in experiment_name)
        safe_name = safe_name.strip().replace(" ", "_") or "experiment"
        return f"{safe_name}_{timestamp}{self._video_format.extension}"

    async def start(
        self, experiment_name: str = "experiment", record_annotated: bool = False
    ) -> Path:
        """
        Start recording video.

        Args:
            experiment_name: Name for the video file
            record_annotated: Whether to also record annotated video with overlays

        Returns:
            Path to the raw video file being created
        """
        if self._state == RecordingState.RECORDING:
            logger.warning("Recording already in progress")
            return self._file_path

        self._record_annotated = record_annotated

        # Generate filename for raw video
        filename = self._generate_filename(experiment_name)
        self._file_path = self._output_dir / filename
        self._start_time = datetime.now()
        self._frame_count = 0
        self._annotated_frame_count = 0

        # Get camera settings for video writer
        settings = self._camera.settings
        fourcc = cv2.VideoWriter_fourcc(*self._video_format.codec)

        # Use measured FPS if camera is already streaming, otherwise use configured FPS
        # This helps when actual frame rate is lower due to processing overhead
        actual_fps = self._camera.current_fps
        recording_fps = actual_fps if actual_fps > 1.0 else settings.fps
        self._recording_fps = recording_fps
        logger.debug(
            f"Recording at {recording_fps:.1f} fps (measured: {actual_fps:.1f}, configured: {settings.fps})"
        )

        # Build FrameWriterThread kwargs
        fwt_kwargs = {}
        if self._buffer_size is not None:
            fwt_kwargs["max_queue_size"] = self._buffer_size

        with self._lock:
            # Create raw video writer
            self._writer = cv2.VideoWriter(
                str(self._file_path), fourcc, recording_fps, settings.resolution
            )

            if not self._writer.isOpened():
                logger.error(f"Failed to create video writer: {self._file_path}")
                self._writer = None
                raise RuntimeError(f"Failed to create video file: {self._file_path}")

            self._writer_thread = FrameWriterThread(self._writer, **fwt_kwargs)
            self._writer_thread.start()

            # Create annotated video writer if requested
            if self._record_annotated:
                annotated_filename = self._generate_filename(f"{experiment_name}_annotated")
                self._annotated_file_path = self._output_dir / annotated_filename
                self._annotated_writer = cv2.VideoWriter(
                    str(self._annotated_file_path), fourcc, recording_fps, settings.resolution
                )
                if not self._annotated_writer.isOpened():
                    logger.warning(
                        f"Failed to create annotated video writer: {self._annotated_file_path}"
                    )
                    self._annotated_writer = None
                else:
                    self._annotated_writer_thread = FrameWriterThread(
                        self._annotated_writer, **fwt_kwargs
                    )
                    self._annotated_writer_thread.start()
                    logger.info(f"Recording annotated video to {self._annotated_file_path}")

        # Register frame callback
        if not self._frame_callback_registered:
            self._camera.on_frame(self._on_frame)
            self._frame_callback_registered = True

        self._state = RecordingState.RECORDING
        logger.info(f"Started recording to {self._file_path} at {recording_fps:.1f} fps")
        return self._file_path

    def _on_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Handle incoming frame for recording (raw video).

        Enqueues the frame for async writing by the FrameWriterThread.

        Args:
            frame: Frame from camera
            timestamp: Frame timestamp
        """
        if self._state != RecordingState.RECORDING:
            return

        if self._writer_thread is not None:
            if self._writer_thread.enqueue(frame.copy()):
                self._frame_count += 1
            else:
                dropped = self._writer_thread.frames_dropped
                if dropped == 1 or dropped % 100 == 0:
                    logger.warning(f"VideoRecorder: frame dropped (total dropped: {dropped})")

    def write_annotated_frame(self, frame: np.ndarray) -> bool:
        """
        Write an annotated frame (with tracking overlays) to the annotated video.

        This should be called with frames that have been processed by CVProcessor
        with overlays drawn.

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
            else:
                dropped = self._annotated_writer_thread.frames_dropped
                if dropped == 1 or dropped % 100 == 0:
                    logger.warning(
                        f"VideoRecorder: annotated frame dropped (total dropped: {dropped})"
                    )
        return False

    async def stop(self) -> Optional[Path]:
        """
        Stop recording and finalize video files.

        Returns:
            Path to the saved raw video file, or None if not recording
        """
        if self._state not in (RecordingState.RECORDING, RecordingState.PAUSED):
            return None

        self._state = RecordingState.FINALIZING

        # Drain writer threads (flushes buffered frames) before releasing writers
        if self._writer_thread is not None:
            self._writer_thread.stop(timeout=30.0)
            self._writer_thread = None

        if self._annotated_writer_thread is not None:
            self._annotated_writer_thread.stop(timeout=30.0)
            self._annotated_writer_thread = None

        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None

            if self._annotated_writer is not None:
                self._annotated_writer.release()
                self._annotated_writer = None

        saved_path = self._file_path
        duration = self.duration

        # Calculate actual FPS and fix video if needed
        if duration > 0 and self._frame_count > 0:
            actual_fps = self._frame_count / duration
            fps_drift = abs(actual_fps - self._recording_fps) / self._recording_fps

            # If FPS drift is more than 10%, re-encode with correct FPS
            if fps_drift > 0.1:
                logger.info(
                    f"FPS drift detected: recorded at {self._recording_fps:.1f} fps, "
                    f"actual {actual_fps:.1f} fps. Re-encoding..."
                )
                self._fix_video_fps(self._file_path, actual_fps)
                if self._record_annotated and self._annotated_file_path:
                    annotated_actual_fps = self._annotated_frame_count / duration
                    self._fix_video_fps(self._annotated_file_path, annotated_actual_fps)

        self._state = RecordingState.IDLE

        logger.info(
            f"Stopped recording. Saved to {saved_path} "
            f"({self._frame_count} frames, {duration:.1f}s)"
        )
        if self._record_annotated and self._annotated_file_path:
            logger.info(
                f"Annotated video saved to {self._annotated_file_path} "
                f"({self._annotated_frame_count} frames)"
            )
        return saved_path

    def _fix_video_fps(self, video_path: Path, correct_fps: float) -> bool:
        """
        Re-encode a video with the correct FPS.

        Args:
            video_path: Path to the video file
            correct_fps: The correct FPS to use

        Returns:
            True if successful
        """
        try:
            temp_path = video_path.with_suffix(".temp.mp4")

            # Read original video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video for FPS fix: {video_path}")
                return False

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*self._video_format.codec)

            # Create new writer with correct FPS
            writer = cv2.VideoWriter(str(temp_path), fourcc, correct_fps, (width, height))
            if not writer.isOpened():
                logger.error("Failed to create temp video writer")
                cap.release()
                return False

            # Copy all frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            cap.release()
            writer.release()

            # Replace original with fixed version
            import os

            os.replace(str(temp_path), str(video_path))
            logger.info(f"Fixed video FPS to {correct_fps:.1f}")
            return True

        except Exception as e:
            logger.error(f"Failed to fix video FPS: {e}")
            return False

    async def pause(self) -> None:
        """Pause recording (frames will be skipped)."""
        if self._state == RecordingState.RECORDING:
            self._state = RecordingState.PAUSED
            logger.info("Recording paused")

    async def resume(self) -> None:
        """Resume recording after pause."""
        if self._state == RecordingState.PAUSED:
            self._state = RecordingState.RECORDING
            logger.info("Recording resumed")

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Manually write a frame to the video.

        Useful for writing processed/annotated frames.

        Args:
            frame: Frame to write

        Returns:
            True if frame was written (enqueued)
        """
        if self._state != RecordingState.RECORDING:
            return False

        if self._writer_thread is not None:
            if self._writer_thread.enqueue(frame.copy()):
                self._frame_count += 1
                return True
        return False

    def set_buffer_size(self, max_frames: int) -> None:
        """
        Set the frame buffer size for the writer thread.

        Must be called before start(). Has no effect on an active recording.

        Args:
            max_frames: Maximum number of frames to buffer.
        """
        self._buffer_size = max_frames

    def get_recording_info(self) -> dict:
        """
        Get information about the current recording.

        Returns:
            Dictionary with recording details
        """
        info = {
            "state": self._state.name,
            "file_path": str(self._file_path) if self._file_path else None,
            "annotated_file_path": (
                str(self._annotated_file_path) if self._annotated_file_path else None
            ),
            "frame_count": self._frame_count,
            "annotated_frame_count": self._annotated_frame_count,
            "record_annotated": self._record_annotated,
            "duration": self.duration,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "format": self._video_format.to_dict(),
            "frames_dropped": 0,
            "buffer_depth": 0,
            "buffer_max": 0,
        }
        if self._writer_thread is not None:
            info["frames_dropped"] = self._writer_thread.frames_dropped
            info["buffer_depth"] = self._writer_thread.queue_depth
            info["buffer_max"] = self._writer_thread.max_queue_size
        return info
