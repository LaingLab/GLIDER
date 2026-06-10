"""
Fixtures for analysis tests — generate synthetic GLIDER recordings.

The CSVs are written to match the real format produced by
``event_logger``, ``data_recorder``, and ``tracking_logger`` byte-for-
byte at the header level, so the analysis library's parsers are
exercised against the same surface real users will hit. A binary fixture
file would drift; a generator stays correct as long as the contract
above stays correct (and any divergence shows up immediately as a test
failure on either side).

The generator parameters are exposed so individual tests can build
recordings with specific shapes (e.g., flow markers in odd positions,
multi-object scenes, missing artifacts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Anchors: pick a fixed date so tests are deterministic across runs.
_BASE_DATETIME = datetime(2026, 5, 25, 14, 0, 30)
_BASE_UNIX = _BASE_DATETIME.timestamp()


@dataclass
class RecordingSpec:
    """Parameters for building a synthetic recording. Defaults produce
    a 5-second, 30 FPS, single-object recording with a 1-second pre-flow
    period and three behavioral states cycling resting→active→resting.
    """

    experiment_name: str = "test_experiment"
    n_pre_flow_frames: int = 30  # 1s of camera/setup before flow start
    n_post_flow_frames: int = 120  # 4s of actual experiment
    fps: float = 30.0
    state_schedule: tuple[tuple[float, str], ...] = (
        (0.0, "resting"),
        (2000.0, "active"),
        (3500.0, "resting"),
    )
    # Zone schedule mirrors state_schedule: (flow_ms_threshold, zone_ids_str).
    # Default keeps the subject in zone1 the whole flow.
    zone_schedule: tuple[tuple[float, str], ...] = ((0.0, "zone1"),)
    # Position is a linear walk across the frame for non-zero variance
    # (so occupancy / trajectory tests have meaningful data to chew on).
    position_start: tuple[float, float] = (100.0, 100.0)
    position_end: tuple[float, float] = (500.0, 300.0)
    # Per-state velocity in px/frame. Anything not listed defaults to 0.
    state_velocities: dict[str, float] = field(
        default_factory=lambda: {"resting": 0.0, "active": 5.0, "locomotion": 8.0}
    )
    # Extra event rows to append after the flow_marker[start] but before
    # flow_marker[end]. Each entry is (flow_ms, source, board_id, pin, value).
    # Example: ((1000.0, "output_write", "board0", "5", "1"),) writes an LED-on
    # event 1s after flow start.
    extra_events: tuple[tuple[float, str, str, str, str], ...] = ()
    write_tracking: bool = True
    write_data: bool = True
    write_events: bool = True


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def _state_at(flow_ms: float, schedule: tuple[tuple[float, str], ...]) -> str:
    current = schedule[0][1]
    for transition_ms, state in schedule:
        if flow_ms >= transition_ms:
            current = state
    return current


def _position_at(
    frame_idx: int, total_frames: int, start: tuple[float, float], end: tuple[float, float]
) -> tuple[float, float]:
    """Linear interpolation between start and end across all frames so
    occupancy/trajectory data has actual variance to test."""
    if total_frames <= 1:
        return start
    t = frame_idx / (total_frames - 1)
    return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)


def write_synthetic_recording(directory: Path, spec: RecordingSpec) -> Path:
    """Materialize a synthetic recording in ``directory``.

    Returns the directory for convenience. Files are named
    ``<experiment>_<timestamp>{,_tracking,_events}.csv`` to mimic the
    real ``DataRecorder.generate_filename`` convention.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    ts_suffix = _BASE_DATETIME.strftime("%Y%m%d_%H%M%S")
    base_name = f"{spec.experiment_name}_{ts_suffix}"
    flow_start_dt = _BASE_DATETIME + timedelta(seconds=spec.n_pre_flow_frames / spec.fps)
    flow_start_unix = flow_start_dt.timestamp()
    total_frames = spec.n_pre_flow_frames + spec.n_post_flow_frames
    flow_end_dt = _BASE_DATETIME + timedelta(seconds=total_frames / spec.fps)
    duration_s = total_frames / spec.fps

    if spec.write_tracking:
        _write_tracking_csv(
            directory / f"{base_name}_tracking.csv",
            spec,
            flow_start_unix,
            flow_end_dt,
            duration_s,
            total_frames,
        )
    if spec.write_data:
        _write_data_csv(
            directory / f"{base_name}.csv",
            spec,
            flow_start_unix,
            flow_end_dt,
            duration_s,
        )
    if spec.write_events:
        _write_events_csv(
            directory / f"{base_name}_events.csv",
            spec,
            flow_start_dt,
            flow_end_dt,
            duration_s,
        )

    return directory


def _write_tracking_csv(
    path: Path,
    spec: RecordingSpec,
    flow_start_unix: float,
    flow_end_dt: datetime,
    duration_s: float,
    total_frames: int,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GLIDER Tracking Data\n")
        f.write(f"# Experiment,{spec.experiment_name}\n")
        f.write(f"# Start Time,{_iso(_BASE_DATETIME)}\n")
        f.write("# Pixels/mm,4.0000\n")
        f.write("# Calibration Resolution,640x480\n")
        f.write("\n")
        f.write(
            "frame,timestamp,elapsed_ms,flow_elapsed_ms,object_id,class,x,y,w,h,"
            "confidence,center_x,center_y,distance_px,distance_mm,cumulative_mm,"
            "zone_ids,behavioral_state,velocity_px_frame\n"
        )

        for i in range(total_frames):
            t_unix = _BASE_UNIX + i / spec.fps
            t_dt = datetime.fromtimestamp(t_unix)
            elapsed_ms = (t_unix - _BASE_UNIX) * 1000
            in_flow = i >= spec.n_pre_flow_frames
            cx, cy = _position_at(i, total_frames, spec.position_start, spec.position_end)
            # Bbox tracks center so x/y/w/h stay self-consistent with center_x/y.
            bbox_w, bbox_h = 20.0, 20.0
            bx, by = cx - bbox_w / 2, cy - bbox_h / 2
            if in_flow:
                flow_ms = (t_unix - flow_start_unix) * 1000
                flow_cell = f"{flow_ms:.1f}"
                state = _state_at(flow_ms, spec.state_schedule)
                zone_ids = _state_at(flow_ms, spec.zone_schedule)
                velocity = spec.state_velocities.get(state, 0.0)
            else:
                flow_cell = ""
                state = "unknown"
                zone_ids = ""
                velocity = 0.0
            f.write(
                f"{i + 1},{_iso(t_dt)},{elapsed_ms:.1f},{flow_cell},0,mouse,"
                f"{bx:.1f},{by:.1f},{bbox_w:.1f},{bbox_h:.1f},0.900,"
                f"{cx:.1f},{cy:.1f},0.00,0.00,0.00,{zone_ids},{state},{velocity:.2f}\n"
            )

        f.write("\n")
        f.write(f"# End Time,{_iso(flow_end_dt)}\n")
        f.write(f"# Duration (s),{duration_s:.2f}\n")
        f.write(f"# Total Frames,{total_frames}\n")


def _write_data_csv(
    path: Path,
    spec: RecordingSpec,
    flow_start_unix: float,
    flow_end_dt: datetime,
    duration_s: float,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GLIDER Experiment Data\n")
        f.write(f"# Experiment Name,{spec.experiment_name}\n")
        f.write(f"# Start Time,{_iso(_BASE_DATETIME)}\n")
        f.write("# Sample Interval (s),0.1\n")
        f.write("\n")
        f.write("# Boards\n")
        f.write("\n")
        f.write("# Devices\n")
        f.write("\n")
        f.write("frame,timestamp,elapsed_ms,flow_elapsed_ms\n")

        # 100ms samples for full duration. No devices configured so the row
        # is just timing columns.
        n_samples = int(duration_s * 10)
        for i in range(n_samples):
            t_unix = _BASE_UNIX + i * 0.1
            t_dt = datetime.fromtimestamp(t_unix)
            elapsed_ms = i * 100.0
            flow_offset_ms = (t_unix - flow_start_unix) * 1000
            flow_cell = f"{flow_offset_ms:.1f}" if flow_offset_ms >= 0 else ""
            f.write(f",{_iso(t_dt)},{elapsed_ms:.1f},{flow_cell}\n")

        f.write("\n")
        f.write(f"# End Time,{_iso(flow_end_dt)}\n")
        f.write(f"# Duration (s),{duration_s:.2f}\n")


def _write_events_csv(
    path: Path,
    spec: RecordingSpec,
    flow_start_dt: datetime,
    flow_end_dt: datetime,
    duration_s: float,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GLIDER Device Event Log\n")
        f.write(f"# Experiment,{spec.experiment_name}\n")
        f.write(f"# Start Time,{_iso(_BASE_DATETIME)}\n")
        f.write("\n")
        f.write(
            "frame,timestamp,elapsed_ms,source,board_id,device_id,device_type,"
            "pin,pin_type,value\n"
        )

        # flow_marker rows — the actual analysis input we care about
        flow_start_elapsed = (flow_start_dt - _BASE_DATETIME).total_seconds() * 1000
        flow_end_elapsed = (flow_end_dt - _BASE_DATETIME).total_seconds() * 1000
        f.write(
            f"{spec.n_pre_flow_frames},{_iso(flow_start_dt)},{flow_start_elapsed:.1f},"
            "flow_marker,,,,,,start\n"
        )
        # Extra synthetic events (e.g., output_write at known flow times) so
        # event_triggered tests have something to bind to.
        for flow_ms, source, board_id, pin, value in spec.extra_events:
            event_dt = flow_start_dt + timedelta(milliseconds=flow_ms)
            event_elapsed = (event_dt - _BASE_DATETIME).total_seconds() * 1000
            frame = spec.n_pre_flow_frames + int(flow_ms / 1000.0 * spec.fps)
            f.write(
                f"{frame},{_iso(event_dt)},{event_elapsed:.1f},"
                f"{source},{board_id},,{pin},,{pin},{value}\n"
            )
        f.write(
            f"{spec.n_pre_flow_frames + spec.n_post_flow_frames},"
            f"{_iso(flow_end_dt)},{flow_end_elapsed:.1f},flow_marker,,,,,,end\n"
        )

        f.write("\n")
        f.write(f"# End Time,{_iso(flow_end_dt)}\n")
        f.write(f"# Duration (s),{duration_s:.2f}\n")


@pytest.fixture
def synthetic_recording(tmp_path: Path) -> Path:
    """Default 5-second recording with flow markers + state transitions."""
    return write_synthetic_recording(tmp_path / "recording", RecordingSpec())


@pytest.fixture
def recording_factory(tmp_path: Path):
    """Build customized recordings: ``recording_factory(spec)``."""
    counter = {"n": 0}

    def _make(spec: RecordingSpec | None = None) -> Path:
        counter["n"] += 1
        return write_synthetic_recording(
            tmp_path / f"recording_{counter['n']}", spec or RecordingSpec()
        )

    return _make
