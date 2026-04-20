"""Tests for AudioPlaybackNode event emission."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from glider.nodes.interface.audio_nodes import AudioPlaybackNode


@pytest.fixture
def node():
    n = AudioPlaybackNode()
    n._state["file_path"] = "/tmp/fake.wav"
    return n


@pytest.fixture
def captured_outputs(node):
    """Capture all set_output calls so we can assert against them."""
    events: list[tuple[str, object]] = []

    def capture(output_name, value):
        events.append((output_name, value))

    node._update_callbacks.append(capture)
    return events


class TestAudioPlaybackEvents:
    """Option 2: success logged only after sd.play(); errors logged to 'error' output."""

    @pytest.mark.asyncio
    async def test_playing_event_fires_after_successful_play(self, node, captured_outputs):
        fake_sd = MagicMock()
        fake_sd.play = MagicMock()

        with (
            patch.object(
                AudioPlaybackNode,
                "_load_audio",
                return_value=(np.zeros(100, dtype=np.float32), 44100),
            ),
            patch.dict(sys.modules, {"sounddevice": fake_sd}),
        ):
            await node.execute()

        output_names = [e[0] for e in captured_outputs]
        assert "playing" in output_names, "expected 'playing' event on success"
        assert "error" not in output_names, "no 'error' event on success"
        playing_value = next(v for n, v in captured_outputs if n == "playing")
        assert playing_value == "/tmp/fake.wav"

    @pytest.mark.asyncio
    async def test_play_failure_emits_error_not_playing(self, node, captured_outputs):
        fake_sd = MagicMock()
        fake_sd.play.side_effect = RuntimeError("device busy")

        with (
            patch.object(
                AudioPlaybackNode,
                "_load_audio",
                return_value=(np.zeros(100, dtype=np.float32), 44100),
            ),
            patch.dict(sys.modules, {"sounddevice": fake_sd}),
        ):
            await node.execute()

        output_names = [e[0] for e in captured_outputs]
        assert "playing" not in output_names, "no 'playing' event when sd.play fails"
        assert "error" in output_names, "expected 'error' event"
        error_value = next(v for n, v in captured_outputs if n == "error")
        assert "device busy" in error_value
        assert "/tmp/fake.wav" in error_value

    @pytest.mark.asyncio
    async def test_load_failure_emits_error_not_playing(self, node, captured_outputs):
        with patch.object(
            AudioPlaybackNode, "_load_audio", side_effect=ValueError("unsupported codec")
        ):
            await node.execute()

        output_names = [e[0] for e in captured_outputs]
        assert "playing" not in output_names
        assert "error" in output_names
        error_value = next(v for n, v in captured_outputs if n == "error")
        assert "unsupported codec" in error_value
