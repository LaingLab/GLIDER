"""
Multi-Video Recorder - Record video from multiple cameras simultaneously.

Records video from all connected cameras to separate files,
with the primary camera optionally having an annotated video
with tracking overlays.

Cameras can also be started and stopped one at a time
(:meth:`MultiVideoRecorder.start_camera` / :meth:`MultiVideoRecorder.stop_camera`).
A run lasts from the first camera starting until the last one stops.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from glider.vision.frame_writer import FrameWriterThread
from glider.vision.video_recorder import (
    ACTIVE_STATES,
    FrameClock,
    RecordingState,
    VideoFormat,
    correct_video_fps,
    open_video_writer,
)

if TYPE_CHECKING:
    from glider.vision.multi_camera_manager import MultiCameraManager

logger = logging.getLogger(__name__)


@dataclass
class _ClosingAnnotated:
    """The annotated writer, detached from the recorder so it can be finished."""

    writer_thread: FrameWriterThread | None
    writer: cv2.VideoWriter | None
    path: Path | None
    frames: int


@dataclass
class _ClosingTake:
    """Everything needed to finish one camera's file, detached from the recorder.

    Captured the moment the camera stops, so the file can be finished on a
    worker thread while the other cameras carry on recording -- and so a new
    run resetting the recorder's counters cannot change what this file's frame
    rate is corrected to.
    """

    camera_id: str
    writer_thread: FrameWriterThread
    writer: cv2.VideoWriter | None
    path: Path | None
    recorded_fps: float
    clock: FrameClock | None
    frames: int
    dropped: int
    annotated: _ClosingAnnotated | None = None


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
        self._clocks: dict[str, FrameClock] = {}
        self._record_annotated = False
        self._recording_fps: dict[str, float] = {}
        #: camera_id -> filename-safe operator label. Empty until something
        #: names the cameras; falls back to the camera index, as before.
        self._camera_labels: dict[str, str] = {}
        self._buffer_size: int | None = None
        self._writer_error: BaseException | None = None
        #: When each camera's current (or last) take started and stopped.
        #: Per camera, because cameras no longer necessarily start together.
        self._camera_started: dict[str, datetime] = {}
        self._camera_stopped: dict[str, datetime] = {}
        #: Cameras stopped on their own whose files are still being finished.
        self._saving: dict[str, asyncio.Future] = {}

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

    @property
    def recording_camera_ids(self) -> list[str]:
        """Cameras with a file open right now."""
        return sorted(self._writer_threads)

    @property
    def saving_camera_ids(self) -> list[str]:
        """Cameras stopped on their own whose files are still being finished."""
        return sorted(self._saving)

    def is_camera_recording(self, camera_id: str) -> bool:
        """Whether this camera has a file open right now."""
        return camera_id in self._writer_threads

    def camera_duration(self, camera_id: str) -> float:
        """How long this camera has been recording, or how long its last take ran.

        The run's :attr:`duration` counts from whichever camera started first,
        which says nothing about a camera started ten minutes in.
        """
        started = self._camera_started.get(camera_id)
        if started is None:
            return 0.0
        stopped = None if camera_id in self._writer_threads else self._camera_stopped.get(camera_id)
        return ((stopped or datetime.now()) - started).total_seconds()

    def set_output_directory(self, path: Path) -> None:
        """Set the output directory for video files."""
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Multi-video output directory: {self._output_dir}")

    def set_video_format(self, codec: str = "avc1", extension: str = ".mp4") -> None:
        """Set the video codec and file extension.

        If the codec is unavailable in the local OpenCV build, recording falls
        back to ``mp4v`` automatically.
        """
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
            # Secondary cameras carry the operator's name for the camera when
            # there is one. "_cam5" identifies an enumeration order, which
            # answers nothing a month later about which arena was recorded --
            # and enumeration order is not stable across replugging.
            label = self._camera_labels.get(camera_id)
            if label:
                return f"{safe_name}_{timestamp}_{label}{self._video_format.extension}"
            cam_num = camera_id.replace("cam_", "")
            return f"{safe_name}_{timestamp}_cam{cam_num}{self._video_format.extension}"

    def _unique_path(self, path: Path) -> Path:
        """*path*, or the first free ``_take2``, ``_take3``... variant of it.

        Filenames carry a one-second timestamp. With cameras stopped and
        started one at a time, a restart inside that second would name the
        second take exactly as the first and open it for writing over the top.
        """
        taken = set(self._file_paths.values())
        candidate, take = path, 2
        while candidate.exists() or candidate in taken:
            candidate = path.with_name(f"{path.stem}_take{take}{path.suffix}")
            take += 1
        return candidate

    def _writer_thread_kwargs(self) -> dict:
        if self._buffer_size is None:
            return {}
        return {"max_queue_size": self._buffer_size}

    def _begin_run(self, record_annotated: bool) -> None:
        """Reset the per-run bookkeeping, for a run starting from idle."""
        self._record_annotated = record_annotated
        self._writer_error = None
        self._clocks.clear()
        self._start_time = datetime.now()
        self._file_paths.clear()
        self._frame_counts.clear()
        self._frames_dropped.clear()
        self._camera_started.clear()
        self._camera_stopped.clear()
        self._annotated_frame_count = 0
        # A run without an annotated video must not report the last run's.
        self._annotated_file_path = None

    def _open_camera(self, camera_id: str, camera, experiment_name: str) -> bool:
        """Open one camera's file and writer thread. Call with ``self._lock`` held.

        Returns False, having opened nothing, if the camera has no settings or
        its writer cannot be created.
        """
        settings = self._multi_cam.get_camera_settings(camera_id)
        if settings is None:
            return False

        # Determine FPS
        actual_fps = camera.current_fps
        recording_fps = actual_fps if actual_fps > 1.0 else settings.fps
        self._recording_fps[camera_id] = recording_fps

        file_path = self._unique_path(
            self._output_dir / self._generate_filename(experiment_name, camera_id)
        )

        # Create writer (falls back to a compatible codec if needed)
        writer, _ = open_video_writer(
            file_path,
            self._video_format.codec,
            recording_fps,
            settings.resolution,
        )

        if writer is None:
            logger.error(f"Failed to create video writer for {camera_id}: {file_path}")
            return False

        self._writers[camera_id] = writer
        self._file_paths[camera_id] = file_path
        self._frame_counts[camera_id] = 0
        self._frames_dropped[camera_id] = 0
        self._clocks[camera_id] = FrameClock()
        self._camera_started[camera_id] = datetime.now()
        self._camera_stopped.pop(camera_id, None)

        # Wrap in FrameWriterThread. Published last: _on_frame starts feeding
        # a camera as soon as its thread is in the dict, so the counters and
        # clock above must already exist.
        fwt = FrameWriterThread(
            writer,
            error_callback=self._on_writer_error,
            frame_size=settings.resolution,
            **self._writer_thread_kwargs(),
        )
        fwt.start()
        self._writer_threads[camera_id] = fwt

        # Register frame callback. Remove-then-add rather than a
        # "registered" flag: re-adding a camera after a preview
        # restart gives it a fresh, empty callback list, and a
        # stale flag left that run writing zero frames.
        self._multi_cam.remove_frame_callback(camera_id, self._on_frame)
        self._multi_cam.on_frame(camera_id, self._on_frame)

        logger.info(f"Recording {camera_id} to {file_path} at {recording_fps:.1f} fps")
        return True

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
        if self._state in ACTIVE_STATES:
            # Paused or errored: finalize it rather than overwrite live writers.
            await self.stop()

        self._begin_run(record_annotated)

        with self._lock:
            try:
                # Create writer for each camera
                for camera_id, camera in self._multi_cam.cameras.items():
                    self._open_camera(camera_id, camera, experiment_name)

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
                        self._annotated_writer, _ = open_video_writer(
                            self._annotated_file_path,
                            self._video_format.codec,
                            recording_fps,
                            primary_settings.resolution,
                        )

                        if self._annotated_writer is not None:
                            self._annotated_writer_thread = FrameWriterThread(
                                self._annotated_writer,
                                error_callback=self._on_writer_error,
                                frame_size=primary_settings.resolution,
                                **self._writer_thread_kwargs(),
                            )
                            self._annotated_writer_thread.start()
                            logger.info(f"Recording annotated video to {self._annotated_file_path}")
                        else:
                            logger.warning("Failed to create annotated video writer")
                            self._annotated_writer = None
            except Exception:
                # Partial failure: release any writers/threads we already allocated
                # so we don't leak handles or leave zombie threads.
                self._release_partial_writers()
                raise

        self._state = RecordingState.RECORDING
        logger.info(f"Started multi-camera recording ({len(self._writers)} cameras)")
        return self._file_paths

    async def start_camera(
        self, camera_id: str, experiment_name: str = "experiment"
    ) -> Path | None:
        """Start recording one camera, on its own or alongside a run already going.

        From idle this begins a run with just this camera in it; during a run
        it adds the camera to it. Either way the others are left as they are.

        Returns:
            The file being written, or None if the camera could not be started.
        """
        if camera_id in self._writer_threads:
            return self._file_paths.get(camera_id)
        if self._state in (RecordingState.FINALIZING, RecordingState.ERROR):
            logger.warning(f"Cannot start {camera_id} while the recorder is {self._state.name}")
            return None
        if camera_id in self._saving:
            logger.warning(f"Cannot start {camera_id}: its last file is still being saved")
            return None
        camera = self._multi_cam.cameras.get(camera_id)
        if camera is None:
            logger.warning(f"Cannot start {camera_id}: no such camera")
            return None

        new_run = self._state == RecordingState.IDLE
        if new_run:
            self._begin_run(record_annotated=False)

        with self._lock:
            try:
                opened = self._open_camera(camera_id, camera, experiment_name)
            except Exception:
                self._release_camera_writer(camera_id)
                raise
        if not opened:
            return None

        if new_run:
            self._state = RecordingState.RECORDING
            logger.info(f"Started multi-camera recording ({camera_id} only)")
        return self._file_paths[camera_id]

    async def stop_camera(self, camera_id: str) -> Path | None:
        """Stop one camera and finish its file, leaving the others recording.

        The file is finished on a worker thread: draining the writer and
        correcting the frame rate can take seconds, and blocking the event loop
        for that long would freeze every other camera's preview mid-run while
        their frames piled up behind it.

        Stopping the last camera ends the run, exactly as :meth:`stop` would.

        Returns:
            The finished file, or None if this camera was not recording.
        """
        if self._state == RecordingState.FINALIZING:
            return None  # stop() is already finishing every file
        take = self._detach(camera_id)
        if take is None:
            return None

        leftover = None
        if not self._writer_threads and self._state in ACTIVE_STATES:
            # That was the last one, so the run is over. An annotated writer
            # still open here outlived its primary; it is finished as is.
            leftover = self._detach_annotated()
            self._state = RecordingState.IDLE
            logger.info(f"Stopped multi-camera recording ({camera_id} was the last camera)")

        def _finish() -> None:
            self._finalize(take)
            if leftover is not None:
                self._finalize_annotated(leftover, 0.0, None, 0)

        saving = asyncio.ensure_future(asyncio.to_thread(_finish))
        self._saving[camera_id] = saving
        try:
            await saving
        finally:
            if self._saving.get(camera_id) is saving:
                del self._saving[camera_id]
        return take.path

    def _release_camera_writer(self, camera_id: str) -> None:
        """Undo a half-finished :meth:`_open_camera` for one camera."""
        fwt = self._writer_threads.pop(camera_id, None)
        if fwt is not None:
            try:
                fwt.stop()
            except Exception:
                logger.exception("Error stopping FrameWriterThread for %s", camera_id)
        writer = self._writers.pop(camera_id, None)
        if writer is not None:
            try:
                writer.release()
            except Exception:
                logger.exception("Error releasing VideoWriter for %s", camera_id)
        self._file_paths.pop(camera_id, None)

    def _release_partial_writers(self) -> None:
        """Release all writers and join threads allocated during a failed start()."""
        for cam_id, fwt in list(self._writer_threads.items()):
            try:
                fwt.stop()
            except Exception:
                logger.exception("Error stopping FrameWriterThread for %s", cam_id)
        self._writer_threads.clear()

        for cam_id, writer in list(self._writers.items()):
            try:
                writer.release()
            except Exception:
                logger.exception("Error releasing VideoWriter for %s", cam_id)
        self._writers.clear()

        if self._annotated_writer_thread is not None:
            try:
                self._annotated_writer_thread.stop()
            except Exception:
                logger.exception("Error stopping annotated FrameWriterThread")
            self._annotated_writer_thread = None
        if self._annotated_writer is not None:
            try:
                self._annotated_writer.release()
            except Exception:
                logger.exception("Error releasing annotated VideoWriter")
            self._annotated_writer = None
        self._annotated_file_path = None
        self._file_paths.clear()

    def _on_writer_error(self, exc: BaseException) -> None:
        """
        Handle an unrecoverable writer error from any FrameWriterThread.

        Called from the writer thread. Transitions the recorder into the
        ERROR state so subsequent frames are rejected and callers can
        observe the failure via ``state``/``writer_error``.
        """
        self._writer_error = exc
        if self._state in (RecordingState.RECORDING, RecordingState.PAUSED):
            self._state = RecordingState.ERROR
        logger.error(f"MultiVideoRecorder: writer aborted due to {exc!r}")

    @property
    def writer_error(self) -> BaseException | None:
        """The exception that caused a writer to abort, if any."""
        return self._writer_error

    @property
    def frames_dropped(self) -> dict[str, int]:
        """Frames dropped so far, per camera.

        A copy, so a caller polling this for a status display cannot reach in
        and disturb the counters the writer threads are updating.

        Readable during the run, not only from the summary logged at stop. On a
        sixteen-camera rig one starved writer silently shortens one animal's
        recording, and that is worth catching while the session can still be
        restarted.
        """
        return dict(self._frames_dropped)

    @property
    def total_frames_dropped(self) -> int:
        """Frames dropped across every camera."""
        return sum(self._frames_dropped.values())

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
                clock = self._clocks.get(camera_id)
                if clock is not None:
                    clock.tick(timestamp)
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

        # Read once: stopping the primary on its own detaches this writer.
        fwt = self._annotated_writer_thread
        if fwt is not None:
            if fwt.enqueue(frame.copy()):
                self._annotated_frame_count += 1
                return True
        return False

    def _detach(self, camera_id: str) -> _ClosingTake | None:
        """Take one camera out of the recording and hand back what finishes it.

        Once its writer thread is out of the dict, :meth:`_on_frame` stops
        feeding it. The annotated video goes with the primary: it is derived
        from the primary's frames, and its frame rate is corrected from them.
        """
        with self._lock:
            fwt = self._writer_threads.pop(camera_id, None)
            if fwt is None:
                return None
            writer = self._writers.pop(camera_id, None)
        self._camera_stopped[camera_id] = datetime.now()
        annotated = None
        if camera_id == self._multi_cam.primary_camera_id:
            annotated = self._detach_annotated()
        return _ClosingTake(
            camera_id=camera_id,
            writer_thread=fwt,
            writer=writer,
            path=self._file_paths.get(camera_id),
            recorded_fps=self._recording_fps.get(camera_id, 0.0),
            clock=self._clocks.get(camera_id),
            frames=self._frame_counts.get(camera_id, 0),
            dropped=self._frames_dropped.get(camera_id, 0),
            annotated=annotated,
        )

    def _detach_annotated(self) -> _ClosingAnnotated | None:
        with self._lock:
            fwt, writer = self._annotated_writer_thread, self._annotated_writer
            self._annotated_writer_thread = None
            self._annotated_writer = None
        if fwt is None and writer is None:
            return None
        return _ClosingAnnotated(
            fwt, writer, self._annotated_file_path, self._annotated_frame_count
        )

    def _finalize(self, take: _ClosingTake) -> None:
        """Drain, close and frame-rate-correct one camera's file.

        Touches only what the take carries, so it is safe on a worker thread.
        """
        take.writer_thread.stop(timeout=30.0)
        if take.dropped > 0:
            logger.warning(
                f"MultiVideoRecorder: {take.camera_id} total frames dropped: {take.dropped}"
            )
        if take.writer is not None:
            take.writer.release()
            logger.info(f"Saved {take.camera_id} video: {take.path} ({take.frames} frames)")

        # Each camera runs at its own real rate; correct each header from its
        # frames' capture timestamps (the single-camera recorder does the same).
        measured_fps = take.clock.fps if take.clock else None
        if take.path is not None:
            correct_video_fps(take.path, take.recorded_fps, measured_fps, self._video_format.codec)
        if take.annotated is not None:
            self._finalize_annotated(take.annotated, take.recorded_fps, measured_fps, take.frames)

    def _finalize_annotated(
        self,
        annotated: _ClosingAnnotated,
        recorded_fps: float,
        measured_fps: float | None,
        raw_frames: int,
    ) -> None:
        if annotated.writer_thread is not None:
            annotated.writer_thread.stop(timeout=30.0)
        if annotated.writer is not None:
            annotated.writer.release()
            logger.info(f"Saved annotated video: {annotated.path} ({annotated.frames} frames)")
        if annotated.path is not None and measured_fps and raw_frames and annotated.frames:
            correct_video_fps(
                annotated.path,
                recorded_fps,
                measured_fps * annotated.frames / raw_frames,
                self._video_format.codec,
            )

    async def _wait_for_saves(self) -> None:
        """Wait for cameras stopped on their own to finish writing."""
        pending = list(self._saving.values())
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def stop(self) -> dict[str, Path]:
        """
        Stop recording and finalize all video files.

        Finalizes from RECORDING, PAUSED and ERROR; a no-op when idle.

        Returns:
            Dictionary of camera_id -> saved file path, for every camera
            recorded this run -- including any already stopped on their own.
        """
        if self._state not in ACTIVE_STATES:
            return {}

        self._state = RecordingState.FINALIZING

        takes = []
        for camera_id in list(self._writer_threads):
            take = self._detach(camera_id)
            if take is not None:
                takes.append(take)
        # Still attached only if the primary itself was not recording.
        orphaned_annotated = self._detach_annotated()

        for take in takes:
            self._finalize(take)
        if orphaned_annotated is not None:
            self._finalize_annotated(orphaned_annotated, 0.0, None, 0)

        # A camera stopped on its own moments ago may still be writing; the
        # caller is about to treat every file as finished.
        await self._wait_for_saves()

        saved_paths = self._file_paths.copy()
        self._state = RecordingState.IDLE

        duration = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        logger.info(f"Stopped multi-camera recording ({len(saved_paths)} files, {duration:.1f}s)")

        return saved_paths

    async def pause(self) -> None:
        """Pause recording (frames will be skipped)."""
        if self._state == RecordingState.RECORDING:
            self._state = RecordingState.PAUSED
            for clock in self._clocks.values():
                clock.break_()  # paused time is not frame time
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

    def set_camera_labels(self, labels: dict[str, str]) -> None:
        """Name the cameras, for the filenames this recorder writes.

        A name is read only when a camera's file is opened, so a file already
        open keeps the name it was opened with. Accepted mid-run for that
        reason: with cameras started one at a time, refusing it would start a
        camera renamed while another was recording under its old name.
        """
        self._camera_labels = {str(k): str(v) for k, v in (labels or {}).items() if str(v).strip()}

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
