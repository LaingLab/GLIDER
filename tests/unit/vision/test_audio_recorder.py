"""Tests for AudioRecorder."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "recordings"


@pytest.fixture
def mock_sounddevice():
    with patch("glider.vision.audio_recorder.sd") as mock_sd:
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Mic", "max_input_channels": 2, "max_output_channels": 0, "index": 0},
            {"name": "USB Mic", "max_input_channels": 1, "max_output_channels": 0, "index": 1},
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2, "index": 2},
        ]
        yield mock_sd


@pytest.fixture
def mock_soundfile():
    with patch("glider.vision.audio_recorder.sf") as mock_sf:
        yield mock_sf


class TestAudioRecorderLifecycle:
    def test_initial_state(self, output_dir):
        from glider.vision.audio_recorder import AudioRecorder

        recorder = AudioRecorder()
        assert not recorder.is_recording
        assert recorder.file_path is None

    async def test_start_creates_wav_file(self, output_dir, mock_sounddevice, mock_soundfile):
        from glider.vision.audio_recorder import AudioRecorder

        recorder = AudioRecorder()
        recorder.set_output_directory(output_dir)
        path = await recorder.start("test_experiment", device_index=0)

        assert path is not None
        assert path.suffix == ".wav"
        assert "test_experiment" in path.name
        assert recorder.is_recording
        mock_sounddevice.InputStream.assert_called_once()

    async def test_stop_returns_path(self, output_dir, mock_sounddevice, mock_soundfile):
        from glider.vision.audio_recorder import AudioRecorder

        recorder = AudioRecorder()
        recorder.set_output_directory(output_dir)
        await recorder.start("test_experiment", device_index=0)
        path = await recorder.stop()

        assert path is not None
        assert not recorder.is_recording

    async def test_stop_when_not_recording_returns_none(self, output_dir):
        from glider.vision.audio_recorder import AudioRecorder

        recorder = AudioRecorder()
        path = await recorder.stop()
        assert path is None

    async def test_start_while_recording_returns_existing_path(
        self, output_dir, mock_sounddevice, mock_soundfile
    ):
        from glider.vision.audio_recorder import AudioRecorder

        recorder = AudioRecorder()
        recorder.set_output_directory(output_dir)
        path1 = await recorder.start("test", device_index=0)
        path2 = await recorder.start("test2", device_index=0)
        assert path1 == path2

    async def test_pause_and_resume(self, output_dir, mock_sounddevice, mock_soundfile):
        from glider.vision.audio_recorder import AudioRecorder

        recorder = AudioRecorder()
        recorder.set_output_directory(output_dir)
        await recorder.start("test", device_index=0)

        recorder.pause()
        assert recorder.is_paused

        recorder.resume()
        assert not recorder.is_paused


class TestAudioRecorderDeviceEnumeration:
    def test_enumerate_devices_returns_input_only(self, mock_sounddevice):
        from glider.vision.audio_recorder import AudioRecorder

        devices = AudioRecorder.enumerate_devices()
        assert len(devices) == 2
        assert devices[0] == (0, "Built-in Mic")
        assert devices[1] == (1, "USB Mic")

    def test_enumerate_devices_handles_import_error(self):
        from glider.vision.audio_recorder import AudioRecorder

        with patch("glider.vision.audio_recorder.sd", None):
            devices = AudioRecorder.enumerate_devices()
            assert devices == []


class TestAudioRecorderFilename:
    def test_filename_sanitization(self, output_dir, mock_sounddevice, mock_soundfile):
        from glider.vision.audio_recorder import AudioRecorder

        recorder = AudioRecorder()
        recorder.set_output_directory(output_dir)
        name = recorder._generate_filename("my experiment/test")
        assert "/" not in name
        assert name.endswith(".wav")
        assert "my_experiment" in name
        assert "test" in name


class TestResolveDeviceByName:
    def test_resolves_existing_device(self, mock_sounddevice):
        from glider.vision.audio_recorder import AudioRecorder

        idx = AudioRecorder.resolve_device_by_name("USB Mic")
        assert idx == 1

    def test_returns_none_for_unknown_device(self, mock_sounddevice):
        from glider.vision.audio_recorder import AudioRecorder

        idx = AudioRecorder.resolve_device_by_name("Nonexistent Mic")
        assert idx is None


class TestIsFFmpegAvailable:
    def test_returns_true_when_found(self):
        from glider.vision.audio_recorder import AudioRecorder

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assert AudioRecorder.is_ffmpeg_available() is True

    def test_returns_false_when_missing(self):
        from glider.vision.audio_recorder import AudioRecorder

        with patch("shutil.which", return_value=None):
            assert AudioRecorder.is_ffmpeg_available() is False


class TestDeviceOpenFailure:
    async def test_start_returns_none_on_device_error(
        self, output_dir, mock_sounddevice, mock_soundfile
    ):
        from glider.vision.audio_recorder import AudioRecorder

        mock_sounddevice.InputStream.side_effect = OSError("Device not available")

        recorder = AudioRecorder()
        recorder.set_output_directory(output_dir)
        path = await recorder.start("test", device_index=99)

        assert path is None
        assert not recorder.is_recording


class TestMuxAudioVideo:
    async def test_mux_calls_ffmpeg_with_correct_args(self, tmp_path):
        from glider.vision.audio_recorder import mux_audio_video

        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"
        video_path.write_bytes(b"fake video")
        audio_path.write_bytes(b"fake audio")
        muxed_path = video_path.with_name("video_muxed.mp4")

        with patch("asyncio.create_subprocess_exec") as mock_exec:

            async def fake_exec(*args, **kwargs):
                # Simulate FFmpeg creating the output file
                muxed_path.write_bytes(b"muxed output")
                mock_proc = MagicMock()

                async def fake_communicate():
                    return (b"", b"")

                mock_proc.communicate = fake_communicate
                mock_proc.returncode = 0
                return mock_proc

            mock_exec.side_effect = fake_exec

            result = await mux_audio_video(video_path, audio_path)

            assert result is True
            args = mock_exec.call_args_list[0][0]
            assert args[0] == "ffmpeg"
            assert "-c:v" in args
            assert "copy" in args
            assert video_path.read_bytes() == b"muxed output"

    async def test_mux_returns_false_when_ffmpeg_missing(self, tmp_path):
        from glider.vision.audio_recorder import mux_audio_video

        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"
        video_path.write_bytes(b"fake video")
        audio_path.write_bytes(b"fake audio")

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await mux_audio_video(video_path, audio_path)
            assert result is False

    async def test_mux_fallback_to_reencode(self, tmp_path):
        from glider.vision.audio_recorder import mux_audio_video

        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"
        video_path.write_bytes(b"fake video")
        audio_path.write_bytes(b"fake audio")
        muxed_path = video_path.with_name("video_muxed.mp4")

        call_count = 0

        async def fake_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_proc = MagicMock()
            if call_count == 1:

                async def fake_communicate():
                    return (b"", b"error")

                mock_proc.communicate = fake_communicate
                mock_proc.returncode = 1
            else:
                muxed_path.write_bytes(b"re-encoded output")

                async def fake_communicate():
                    return (b"", b"")

                mock_proc.communicate = fake_communicate
                mock_proc.returncode = 0
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = await mux_audio_video(video_path, audio_path)
            assert result is True
            assert call_count == 2


class TestCameraConfigAudio:
    def test_audio_fields_default_to_none(self):
        from glider.core.experiment_session import CameraConfig

        config = CameraConfig()
        assert config.audio_device_name is None
        assert config.audio_device_index is None

    def test_audio_fields_round_trip(self):
        from glider.core.experiment_session import CameraConfig

        config = CameraConfig(audio_device_name="USB Mic", audio_device_index=1)
        data = config.to_dict()
        restored = CameraConfig.from_dict(data)
        assert restored.audio_device_name == "USB Mic"
        assert restored.audio_device_index == 1

    def test_audio_fields_absent_in_old_data(self):
        from glider.core.experiment_session import CameraConfig

        # Simulate loading a .glider file saved before audio was added
        data = {"camera_index": 0, "fps": 30}
        config = CameraConfig.from_dict(data)
        assert config.audio_device_name is None
        assert config.audio_device_index is None
