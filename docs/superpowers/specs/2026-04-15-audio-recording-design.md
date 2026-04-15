# Audio Recording & Video Muxing Design

## Overview

Add microphone recording during experiments and mux the captured audio into all output video files. The user selects a microphone from the Camera Settings dialog. Audio is recorded to a WAV file during the experiment, then combined with each video file via FFmpeg on experiment stop.

## Components

### 1. AudioRecorder (`src/glider/vision/audio_recorder.py`)

New class following the same lifecycle pattern as `VideoRecorder`.

- **Recording engine:** `sounddevice.InputStream` with a callback that writes chunks to a `soundfile.SoundFile` in WAV format (16-bit PCM / `int16` for size and compatibility).
- **Lifecycle:** `start(experiment_name)` -> records -> `stop()` -> returns file path.
- **Filename pattern:** `{experiment_name}_{timestamp}.wav` (matches video convention).
- **Configuration:** device index, sample rate (default 44100), channels (default 1/mono).
- **Device enumeration:** Static method `enumerate_devices()` wrapping `sounddevice.query_devices()` to list input devices. Returns list of `(index, name)` tuples for input-capable devices.
- **Output directory:** Shares the same directory as video/data recorders, set via `set_output_directory()` (matches `VideoRecorder` method name).
- **Thread safety:** Uses `threading.Lock` for state protection, since `sounddevice.InputStream` fires its callback from a separate thread while `start()`/`stop()` are called from the asyncio event loop.
- **Pause/resume:** `pause()` sets a flag that causes the InputStream callback to discard incoming samples (write nothing to the WAV file). `resume()` clears the flag. This matches `VideoRecorder` behavior, which skips frames during pause. Both audio and video files become shorter than wall-clock time by the pause duration, keeping them in sync with each other.
- **Optional dependency:** If `sounddevice` or `soundfile` is not installed, all audio recording features are disabled with a logged warning. Import errors are caught gracefully — the experiment still runs without audio.
- **Device open failures:** If the selected device cannot be opened at `start()` time (e.g., unplugged), log a warning and skip audio recording. The experiment still starts normally.

### 2. FFmpeg Muxing (`mux_audio_video` in `audio_recorder.py`)

Utility function to combine audio and video after recording stops.

- **Command:** `ffmpeg -y -i video.mp4 -i audio.wav -c:v copy -c:a aac -b:a 128k -shortest -movflags +faststart output.mp4`
  - Copies video stream as-is (no re-encode).
  - Encodes audio to AAC at 128 kbps.
  - `-shortest` handles slight length differences between streams.
  - `-movflags +faststart` for web-compatible MP4.
  - `-y` overwrites any stale temporary files from previous runs.
- **Fallback:** If `-c:v copy` fails (codec incompatibility), retry with `-c:v libx264 -crf 18` to re-encode video.
- **Execution:** `asyncio.create_subprocess_exec` to avoid blocking the event loop.
- **Ordering:** Muxing runs AFTER `VideoRecorder.stop()` completes (which includes any `_fix_video_fps` re-encoding). This avoids conflicts with the FPS-fix step that also renames files.
- **Workflow per video file:**
  1. Mux to a temporary output file (`_muxed.mp4` suffix).
  2. Replace the original video with the muxed file.
  3. Delete the `.wav` only after ALL video files are successfully muxed. If any mux fails, leave the `.wav` intact for manual retry.
- **Video file discovery:** After `VideoRecorder.stop()`, collect the returned raw video path AND check `VideoRecorder.annotated_file_path` for the annotated video (it is stored as a property but not included in the return value). For `MultiVideoRecorder.stop()`, iterate over the returned `dict[str, Path]` and similarly check for annotated paths. Mux audio into every collected path.
- **Applied to:** raw video, annotated video, and all multi-camera videos — each gets the same audio track.
- **Graceful degradation:** If FFmpeg is not installed, log a warning and leave the separate `.mp4` and `.wav` files intact.
- **Early detection:** Check for FFmpeg availability at experiment start (via `shutil.which("ffmpeg")`). If not found and audio recording is enabled, log a warning so the user knows audio won't be muxed — but still record the `.wav`.

### 3. Microphone Selection UI (Camera Settings Dialog)

- **Location:** New "Audio" tab in `CameraSettingsDialog`.
- **Dropdown:** Lists available input devices from `AudioRecorder.enumerate_devices()`. Refresh button to re-scan.
- **Default:** "None (no audio recording)" — audio recording is opt-in.
- **Storage:** Selected device stored in `ExperimentSession` metadata as `audio_device_name` (string) and `audio_device_index` (int). Device name is the primary identifier; index is used as a runtime cache. On load, look up the device by name and update the index. If the named device is not found, disable audio recording and warn the user. The `.glider` serialization schema should be updated to include these fields.
- **Test button:** Records 1 second of audio and plays it back via `sounddevice.play()`. If playback fails (e.g., headless Pi), show a message "Test recording captured successfully" instead of playing back.

### 4. Experiment Lifecycle Integration (GliderCore)

- `GliderCore.__init__`: Create `_audio_recorder` instance.
- `set_recording_directory()`: Call `_audio_recorder.set_output_directory(path)` (delegates to same method name as `VideoRecorder`).
- `start_experiment()`: If `audio_device_name` is set in session metadata, resolve device index by name, start audio recording alongside video recording. Also check for FFmpeg and warn if missing. If audio device fails to open, log warning and continue without audio.
- `pause_experiment()` / `resume_experiment()`: Call `_audio_recorder.pause()` / `_audio_recorder.resume()`.
- `_stop_recorders()`: Stop audio recording, stop video recording (including FPS fix), collect all video paths (raw + annotated from return values and properties), then mux audio into each, then delete WAV if all muxes succeeded.
- Audio recording is enabled when a device is configured — no separate toggle needed.

## Data Flow

```
Experiment Start
  |
  +-- Check FFmpeg availability (warn if missing)
  +-- VideoRecorder.start() --> raw.mp4 (+ annotated.mp4, cam1.mp4, ...)
  +-- AudioRecorder.start() --> audio.wav
  |
  ... experiment runs ...
  |
Experiment Stop
  |
  +-- AudioRecorder.stop() --> audio.wav path
  +-- VideoRecorder.stop() --> list of video paths (includes _fix_video_fps)
  +-- for each video path:
  |     mux_audio_video(video.mp4, audio.wav) --> video.mp4 (with audio)
  +-- if all muxes succeeded: delete audio.wav
```

## Testing

- **AudioRecorder unit tests:** Mock `sounddevice` and `soundfile` to test start/stop lifecycle, filename generation, device enumeration, pause/resume sample discarding, graceful import failure.
- **mux_audio_video unit tests:** Mock `asyncio.create_subprocess_exec` to verify correct FFmpeg arguments, test `-c:v copy` fallback to re-encode, and test graceful failure when FFmpeg is missing.
- **GliderCore integration:** Verify audio recorder starts/stops with experiment when device is configured, is skipped when no device is selected, and that `set_recording_directory` propagates to audio recorder.
- Actual audio capture and FFmpeg execution are not tested in CI (system-level dependencies).

## Notes

- **Disk usage:** WAV at 44100 Hz mono 16-bit is ~5.3 MB/minute (~317 MB/hour). Acceptable for typical experiment lengths.
- **Audio/video sync:** Audio records at a precise hardware clock (44100 Hz). Video frame timing is less precise. For typical experiment durations (minutes to low hours), drift is negligible. The `-shortest` flag trims any length mismatch.
