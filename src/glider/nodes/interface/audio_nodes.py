"""
Audio Nodes - Audio playback for experiments.

Provides an AudioPlayback node that plays WAV/MP3 files through
a user-selected sound output device. Playback is non-blocking:
the node starts audio in the background and immediately fires
the next exec output.

Requires: sounddevice, soundfile (WAV), pydub (MP3 support).
"""

import logging

from glider.nodes.base_node import (
    GliderNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

logger = logging.getLogger(__name__)


class AudioPlaybackNode(GliderNode):
    """Play a WAV or MP3 audio file through a selected output device."""

    definition = NodeDefinition(
        name="AudioPlayback",
        category=NodeCategory.INTERFACE,
        description="Play an audio file (WAV/MP3)",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
            PortDefinition("Volume", PortType.DATA, float, 1.0, "Playback volume (0.0-1.0)"),
        ],
        outputs=[
            PortDefinition("next", PortType.EXEC, description="Triggers after playback starts"),
        ],
        color="#5a4a2d",
    )

    def __init__(self):
        super().__init__()
        self._state.setdefault("file_path", "")
        self._state.setdefault("device_index", None)
        self._state.setdefault("device_name", "")
        self._state.setdefault("volume", 1.0)

    def update_event(self) -> None:
        """Called when inputs change."""
        pass

    async def execute(self) -> None:
        """Load and play the audio file, then fire the next exec output."""
        file_path = self._state.get("file_path", "")
        if not file_path:
            logger.warning("AudioPlayback: no file path set")
            await self._fire_exec_output("next")
            return

        # Read volume from connected input or fall back to state
        volume = self.get_input_by_name("Volume")
        if volume is None:
            volume = self._state.get("volume", 1.0)
        volume = max(0.0, min(1.0, float(volume)))

        device_index = self._state.get("device_index")

        try:
            import numpy as np

            data, samplerate = self._load_audio(file_path)

            # Apply volume scaling
            if volume != 1.0:
                data = (data * volume).astype(np.float32)

            import sounddevice as sd

            sd.play(data, samplerate, device=device_index)
            logger.info(
                f"AudioPlayback: playing '{file_path}' "
                f"(sr={samplerate}, vol={volume}, device={device_index})"
            )
        except ImportError as e:
            logger.error(f"AudioPlayback: missing dependency - {e}")
            self.set_error(f"Missing audio package: {e}")
        except Exception as e:
            logger.error(f"AudioPlayback: playback error - {e}")
            self.set_error(str(e))

        await self._fire_exec_output("next")

    @staticmethod
    def _load_audio(file_path: str):
        """
        Load an audio file and return (numpy_array, samplerate).

        WAV files are loaded via soundfile; MP3 files via pydub.
        """
        lower = file_path.lower()
        if lower.endswith(".mp3"):
            import numpy as np
            from pydub import AudioSegment

            seg = AudioSegment.from_mp3(file_path)
            samplerate = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            # Normalise int samples to -1..1
            samples = samples / (2 ** (seg.sample_width * 8 - 1))
            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels)
            return samples, samplerate
        else:
            import soundfile as sf

            data, samplerate = sf.read(file_path, dtype="float32")
            return data, samplerate

    async def stop(self) -> None:
        """Stop any active playback."""
        try:
            import sounddevice as sd

            sd.stop()
        except ImportError:
            pass

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        for callback in self._update_callbacks:
            callback("next", True)


def register_audio_nodes(flow_engine) -> None:
    """Register audio nodes with the flow engine."""
    flow_engine.register_node("AudioPlayback", AudioPlaybackNode)
    logger.info("Registered audio nodes")
