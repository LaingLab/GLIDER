"""
Audio Recorder - Records microphone audio during experiments.

Records audio from a selected input device to WAV files using sounddevice
and soundfile. Follows the same lifecycle pattern as VideoRecorder.
"""

import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional imports — audio recording is disabled if not installed.
# Catch OSError as well as ImportError: sounddevice/soundfile import fine as
# Python modules but raise OSError at import time when their native backend
# (PortAudio / libsndfile) is missing — common on headless CI runners and
# minimal Pi images. Without this, importing anything from `glider` would
# hard-fail on such machines.
try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None
    logger.info("sounddevice/PortAudio not available — audio recording disabled")

try:
    import soundfile as sf
except (ImportError, OSError):
    sf = None
    logger.info("soundfile/libsndfile not available — audio recording disabled")


class AudioRecorder:
    """
    Records microphone audio to WAV files.

    Uses sounddevice.InputStream for capture and soundfile for WAV writing.
    Thread-safe: the InputStream callback runs on a separate thread.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._output_dir = Path.cwd()

        self._recording = False
        self._paused = False
        self._file_path: Path | None = None
        self._stream: object | None = None  # sd.InputStream
        self._wav_file: object | None = None  # sf.SoundFile
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def file_path(self) -> Path | None:
        return self._file_path

    def set_output_directory(self, path: Path) -> None:
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, experiment_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in experiment_name)
        safe_name = safe_name.strip().replace(" ", "_")
        if not safe_name:
            safe_name = "experiment"
        return f"{safe_name}_{timestamp}.wav"

    @staticmethod
    def enumerate_devices() -> list[tuple[int, str]]:
        """List available audio input devices as (index, name) tuples."""
        if sd is None:
            return []
        try:
            devices = sd.query_devices()
            return [
                (int(d.get("index", i)), d["name"])
                for i, d in enumerate(devices)
                if d.get("max_input_channels", 0) > 0
            ]
        except Exception as e:
            logger.error(f"Failed to enumerate audio devices: {e}")
            return []

    @staticmethod
    def resolve_device_by_name(name: str) -> int | None:
        """Find device index by name. Returns None if not found."""
        for idx, dev_name in AudioRecorder.enumerate_devices():
            if dev_name == name:
                return idx
        return None

    @staticmethod
    def is_ffmpeg_available() -> bool:
        """Check whether ffmpeg is on PATH."""
        return shutil.which("ffmpeg") is not None

    async def start(
        self,
        experiment_name: str = "experiment",
        device_index: int | None = None,
    ) -> Path | None:
        """Start recording audio."""
        if sd is None or sf is None:
            logger.warning("Audio recording unavailable — missing sounddevice/soundfile")
            return None

        with self._lock:
            if self._recording:
                logger.warning("Audio recording already in progress")
                return self._file_path

            filename = self._generate_filename(experiment_name)
            self._file_path = self._output_dir / filename
            self._output_dir.mkdir(parents=True, exist_ok=True)

            try:
                self._wav_file = sf.SoundFile(
                    str(self._file_path),
                    mode="w",
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    subtype="PCM_16",
                )
                self._stream = sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype="int16",
                    device=device_index,
                    callback=self._audio_callback,
                )
                self._paused = False
                self._recording = True
                self._stream.start()
                logger.info(f"Started audio recording to {self._file_path}")
                return self._file_path
            except Exception as e:
                logger.error(f"Failed to start audio recording: {e}")
                self._cleanup()
                return None

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice from its audio thread."""
        if status:
            logger.warning(f"Audio status: {status}")
        with self._lock:
            if self._recording and not self._paused and self._wav_file is not None:
                self._wav_file.write(indata.copy())

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    async def stop(self) -> Path | None:
        """Stop recording and close the WAV file."""
        with self._lock:
            if not self._recording:
                return None
            self._recording = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
            self._stream = None

        if self._wav_file is not None:
            try:
                self._wav_file.close()
            except Exception as e:
                logger.error(f"Error closing WAV file: {e}")
            self._wav_file = None

        logger.info(f"Audio recording saved to {self._file_path}")
        return self._file_path

    def _cleanup(self) -> None:
        """Clean up on failed start."""
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._wav_file is not None:
            try:
                self._wav_file.close()
            except Exception:
                pass
            self._wav_file = None
        self._file_path = None


async def mux_audio_video(video_path: Path, audio_path: Path) -> bool:
    """Mux audio into a video file using FFmpeg.

    Tries -c:v copy first (fast, no re-encode). Falls back to
    -c:v libx264 if copy fails (codec incompatibility).

    Replaces the original video file with the muxed version.

    Args:
        video_path: Path to the video file (.mp4).
        audio_path: Path to the audio file (.wav).

    Returns:
        True if muxing succeeded, False otherwise.
    """
    import asyncio as _asyncio

    muxed_path = video_path.with_name(video_path.stem + "_muxed" + video_path.suffix)

    for attempt, video_codec in enumerate(
        [
            ["-c:v", "copy"],
            ["-c:v", "libx264", "-crf", "18"],
        ]
    ):
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                *video_codec,
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(muxed_path),
            ]
            proc = await _asyncio.create_subprocess_exec(
                *cmd,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                # Replace original with muxed file
                muxed_path.replace(video_path)
                logger.info(f"Muxed audio into {video_path}")
                return True
            else:
                if attempt == 0:
                    logger.warning(
                        f"FFmpeg copy failed for {video_path}, retrying with re-encode: "
                        f"{stderr.decode(errors='replace')[:200]}"
                    )
                    # Clean up failed temp file
                    if muxed_path.exists():
                        muxed_path.unlink()
                    continue
                else:
                    logger.error(
                        f"FFmpeg re-encode also failed for {video_path}: "
                        f"{stderr.decode(errors='replace')[:200]}"
                    )
                    if muxed_path.exists():
                        muxed_path.unlink()
                    return False

        except FileNotFoundError:
            logger.error("FFmpeg not found — cannot mux audio into video")
            return False
        except Exception as e:
            logger.error(f"Error muxing audio into {video_path}: {e}")
            if muxed_path.exists():
                muxed_path.unlink()
            return False

    return False
