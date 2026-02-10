"""
Audio Recorder - Record audio synced with experiment lifecycle.

Records audio to WAV files synchronized with experiment start/stop,
generating filenames that match the DataRecorder/VideoRecorder pattern.
"""

import logging
import threading
import wave
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import sounddevice as sd

    _sounddevice_available = True
except (ImportError, OSError):
    _sounddevice_available = False

logger = logging.getLogger(__name__)


class RecordingState(Enum):
    """State of the audio recording."""

    IDLE = auto()
    RECORDING = auto()
    PAUSED = auto()
    FINALIZING = auto()


@dataclass
class AudioSettings:
    """Audio recording settings."""

    enabled: bool = False
    device_index: Optional[int] = None  # None = system default
    device_name: str = ""
    sample_rate: int = 44100
    channels: int = 1
    gain: float = 1.0  # Software gain, 0.0-5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "gain": self.gain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioSettings":
        return cls(
            enabled=data.get("enabled", False),
            device_index=data.get("device_index"),
            device_name=data.get("device_name", ""),
            sample_rate=data.get("sample_rate", 44100),
            channels=data.get("channels", 1),
            gain=data.get("gain", 1.0),
        )


class AudioRecorder:
    """
    Records audio synchronized with experiment lifecycle.

    Integrates with GliderCore to automatically start/stop recording
    when experiments run. Saves WAV files alongside CSV data and video files
    with matching timestamps.
    """

    def __init__(self):
        """Initialize the audio recorder."""
        self._settings = AudioSettings()
        self._state = RecordingState.IDLE
        self._output_dir: Path = Path.cwd()
        self._file_path: Optional[Path] = None
        self._start_time: Optional[datetime] = None
        self._sample_count = 0
        self._lock = threading.Lock()
        self._stream: Optional[Any] = None  # sd.InputStream
        self._wav_file: Optional[wave.Wave_write] = None
        self._is_available = _sounddevice_available

    @property
    def is_available(self) -> bool:
        """Whether sounddevice is available."""
        return self._is_available

    @property
    def is_recording(self) -> bool:
        """Whether audio is currently being recorded."""
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
        """Path to the current/last audio file."""
        return self._file_path

    @property
    def settings(self) -> AudioSettings:
        """Current audio settings."""
        return self._settings

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def set_output_directory(self, path: Path) -> None:
        """Set the output directory for audio files."""
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Audio output directory: {self._output_dir}")

    def update_settings(self, settings: AudioSettings) -> None:
        """Update audio settings."""
        self._settings = settings

    @staticmethod
    def get_available_devices() -> list[dict[str, Any]]:
        """
        Get list of available audio input devices.

        Returns:
            List of dicts with device info (index, name, channels, sample_rate)
        """
        if not _sounddevice_available:
            return []

        devices = []
        try:
            device_list = sd.query_devices()
            for i, dev in enumerate(device_list):
                if dev["max_input_channels"] > 0:
                    devices.append(
                        {
                            "index": i,
                            "name": dev["name"],
                            "channels": dev["max_input_channels"],
                            "sample_rate": int(dev["default_samplerate"]),
                        }
                    )
        except Exception as e:
            logger.error(f"Failed to enumerate audio devices: {e}")

        return devices

    def _generate_filename(self, experiment_name: str) -> str:
        """Generate filename matching DataRecorder/VideoRecorder pattern."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in experiment_name)
        safe_name = safe_name.strip().replace(" ", "_") or "experiment"
        return f"{safe_name}_{timestamp}_audio.wav"

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio stream callback - write data to WAV file."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self._state != RecordingState.RECORDING:
            return

        with self._lock:
            if self._wav_file is None:
                return

            # Apply software gain
            if self._settings.gain != 1.0:
                data = indata * self._settings.gain
                data = np.clip(data, -1.0, 1.0)
            else:
                data = indata

            # Convert float32 to int16 for WAV
            int_data = (data * 32767).astype(np.int16)
            self._wav_file.writeframes(int_data.tobytes())
            self._sample_count += frames

    async def start(self, experiment_name: str = "experiment") -> Path:
        """
        Start recording audio.

        Args:
            experiment_name: Name for the audio file

        Returns:
            Path to the audio file being created
        """
        if not self._is_available:
            raise RuntimeError("sounddevice is not available")

        if self._state == RecordingState.RECORDING:
            logger.warning("Audio recording already in progress")
            return self._file_path

        filename = self._generate_filename(experiment_name)
        self._file_path = self._output_dir / filename
        self._start_time = datetime.now()
        self._sample_count = 0

        with self._lock:
            # Open WAV file for writing
            self._wav_file = wave.open(str(self._file_path), "wb")
            self._wav_file.setnchannels(self._settings.channels)
            self._wav_file.setsampwidth(2)  # 16-bit
            self._wav_file.setframerate(self._settings.sample_rate)

        # Start audio stream
        try:
            self._stream = sd.InputStream(
                device=self._settings.device_index,
                samplerate=self._settings.sample_rate,
                channels=self._settings.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            with self._lock:
                if self._wav_file is not None:
                    self._wav_file.close()
                    self._wav_file = None
            raise RuntimeError(f"Failed to start audio stream: {e}") from e

        self._state = RecordingState.RECORDING
        logger.info(
            f"Started audio recording to {self._file_path} "
            f"({self._settings.sample_rate} Hz, {self._settings.channels}ch)"
        )
        return self._file_path

    async def stop(self) -> Optional[Path]:
        """
        Stop recording and finalize WAV file.

        Returns:
            Path to the saved audio file, or None if not recording
        """
        if self._state not in (RecordingState.RECORDING, RecordingState.PAUSED):
            return None

        self._state = RecordingState.FINALIZING

        # Stop the audio stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
            self._stream = None

        # Close WAV file
        with self._lock:
            if self._wav_file is not None:
                try:
                    self._wav_file.close()
                except Exception as e:
                    logger.error(f"Error closing WAV file: {e}")
                self._wav_file = None

        saved_path = self._file_path
        duration = self.duration

        self._state = RecordingState.IDLE

        logger.info(
            f"Stopped audio recording. Saved to {saved_path} "
            f"({self._sample_count} samples, {duration:.1f}s)"
        )
        return saved_path

    async def pause(self) -> None:
        """Pause recording (audio data will be skipped)."""
        if self._state == RecordingState.RECORDING:
            self._state = RecordingState.PAUSED
            logger.info("Audio recording paused")

    async def resume(self) -> None:
        """Resume recording after pause."""
        if self._state == RecordingState.PAUSED:
            self._state = RecordingState.RECORDING
            logger.info("Audio recording resumed")

    def get_recording_info(self) -> dict:
        """Get information about the current recording."""
        return {
            "state": self._state.name,
            "file_path": str(self._file_path) if self._file_path else None,
            "sample_count": self._sample_count,
            "duration": self.duration,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "sample_rate": self._settings.sample_rate,
            "channels": self._settings.channels,
        }
