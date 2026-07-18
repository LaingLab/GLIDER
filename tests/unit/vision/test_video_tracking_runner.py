"""Tests for VideoTrackingRunner — batch tracking over a recorded video."""

from pathlib import Path

import numpy as np
import pandas as pd

from glider.vision.cv_processor import CVSettings, MotionResult, TrackedObject
from glider.vision.video_tracking_runner import VideoTrackingConfig, VideoTrackingRunner
from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape


class FakeCV:
    """Stand-in CVProcessor: reports one object whose x = frame index * 5."""

    is_initialized = True

    def __init__(self):
        self._n = -1

    def initialize(self):  # pragma: no cover - trivial
        return True

    def reset(self):  # pragma: no cover - trivial
        pass

    def process_frame(self, frame, timestamp):
        self._n += 1
        obj = TrackedObject(
            track_id=1,
            class_name="subject",
            bbox=(self._n * 5, 20, 8, 8),
            confidence=0.9,
            centroid=(self._n * 5 + 4, 24),
        )
        return [], [obj], MotionResult(False, 0.0)


class FakeCVKeypoints:
    """CVProcessor stand-in that reports one object with a keypoint at (40, 30)."""

    is_initialized = True

    def initialize(self):  # pragma: no cover - trivial
        return True

    def reset(self):  # pragma: no cover - trivial
        pass

    def process_frame(self, frame, timestamp):
        obj = TrackedObject(
            track_id=1,
            class_name="mouse",
            bbox=(8, 8, 6, 6),
            confidence=0.9,
            centroid=(11, 11),
            keypoints=np.array([[40.0, 30.0, 0.9]], dtype=np.float32),
        )
        return [], [obj], MotionResult(False, 0.0)


def test_frame_cb_receives_annotated_frames(synthetic_clip: Path, tmp_path: Path):
    """A live-preview frame_cb is called per processed frame, even with no writer."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=None,
        write_tracking=False,
        write_zone_events=False,
        write_annotated=False,  # preview must work independently of the writer
    )
    seen: list[tuple[int, np.ndarray]] = []
    VideoTrackingRunner(cfg, cv_processor=FakeCV()).run(
        frame_cb=lambda frame, n: seen.append((n, frame.copy()))
    )

    # synthetic_clip is 12 frames at 10 fps; preview throttle at 10 fps -> stride 1.
    assert [n for n, _ in seen] == list(range(12))
    # Each is an annotated BGR frame the size of the source (64x48).
    assert seen[0][1].shape == (48, 64, 3)
    # The bbox overlay was drawn (frame is no longer all-black).
    assert seen[0][1].any()


def test_frame_cb_frame_includes_keypoints(synthetic_clip: Path, tmp_path: Path):
    """The preview/annotated frame carries pose keypoint dots."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=None,
        write_tracking=False,
        write_zone_events=False,
        write_annotated=False,
    )
    seen: list[np.ndarray] = []
    VideoTrackingRunner(cfg, cv_processor=FakeCVKeypoints()).run(
        frame_cb=lambda frame, n: seen.append(frame.copy())
    )

    # Default keypoint color is BGR red; keypoint placed at x=40, y=30.
    assert tuple(int(c) for c in seen[0][30, 40]) == (0, 0, 255)


def test_batch_run_writes_keypoints_csv_with_operator_names(synthetic_clip: Path, tmp_path: Path):
    """The batch path must honour configured bodypart names.

    It carried cv_settings all the way here but never called
    set_keypoint_names, so offline pose runs silently labelled every keypoint
    positionally (0, 1, 2, ...) and discarded the operator's names — while the
    live path (GliderCore) got them right. Nothing covered this path.
    """
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=None,
        cv_settings=CVSettings(keypoint_names=["snout"]),
        write_tracking=True,
        write_zone_events=False,
        write_annotated=False,
    )
    VideoTrackingRunner(cfg, cv_processor=FakeCVKeypoints()).run()

    keypoints = list(out.glob("*_keypoints.csv"))
    assert len(keypoints) == 1
    df = pd.read_csv(keypoints[0])
    assert df["keypoint"].unique().tolist() == ["snout"]
    assert df["x"].iloc[0] == 40.0
    assert df["y"].iloc[0] == 30.0
    assert len(df) == 12  # one keypoint, one object, 12 frames


def test_batch_run_without_names_falls_back_to_indices(synthetic_clip: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=None,
        write_tracking=True,
        write_zone_events=False,
        write_annotated=False,
    )
    VideoTrackingRunner(cfg, cv_processor=FakeCVKeypoints()).run()

    # dtype=str: an index-labelled keypoint column reads back as int otherwise.
    df = pd.read_csv(next(iter(out.glob("*_keypoints.csv"))), dtype={"keypoint": str})
    assert df["keypoint"].unique().tolist() == ["0"]


def test_batch_run_without_pose_writes_no_keypoints_csv(synthetic_clip: Path, tmp_path: Path):
    """FakeCV reports no keypoints — the file must not be created."""
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=None,
        write_tracking=True,
        write_zone_events=False,
        write_annotated=False,
    )
    VideoTrackingRunner(cfg, cv_processor=FakeCV()).run()

    assert list(out.glob("*_keypoints.csv")) == []


def test_run_writes_tracking_csv(synthetic_clip: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=None,
        write_tracking=True,
        write_zone_events=False,
        write_annotated=False,
    )
    runner = VideoTrackingRunner(cfg, cv_processor=FakeCV())
    runner.run()

    tracking = list(out.glob("*tracking*.csv"))
    assert len(tracking) == 1
    # Skip the GLIDER metadata header rows (commented with '#').
    df = pd.read_csv(tracking[0], comment="#")
    assert len(df) == 12  # one row per frame, single object
    # elapsed_ms derives from the video timeline: frame / fps * 1000, fps=10.
    assert df["elapsed_ms"].iloc[0] == 0.0
    assert abs(df["elapsed_ms"].iloc[1] - 100.0) < 1.0
    # flow_elapsed_ms is empty (no flow).
    assert df["flow_elapsed_ms"].isna().all()


def test_progress_callback_reports_total(synthetic_clip: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=None,
        write_zone_events=False,
        write_annotated=False,
    )
    seen = []
    VideoTrackingRunner(cfg, cv_processor=FakeCV()).run(
        progress_cb=lambda done, total: seen.append((done, total))
    )
    assert seen[-1] == (12, 12)


def test_tracking_csv_populates_zone_ids_when_zones_set(synthetic_clip: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    zones = ZoneConfiguration()
    # Whole-frame zone (normalized top-left .. bottom-right), so every centroid
    # falls inside it and zone_ids is populated for every data row.
    zones.add_zone(
        Zone(id="all", name="all", shape=ZoneShape.RECTANGLE, vertices=[(0.0, 0.0), (1.0, 1.0)])
    )
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=zones,
        write_zone_events=False,
        write_annotated=False,
    )
    VideoTrackingRunner(cfg, cv_processor=FakeCV()).run()

    tracking = list(out.glob("*tracking*.csv"))[0]
    df = pd.read_csv(tracking, comment="#")
    # zone_ids should contain the zone name for at least the rows where the object
    # is inside the frame. Non-empty for the first row at minimum.
    assert df["zone_ids"].notna().any()
    assert (df["zone_ids"].astype(str).str.contains("all")).any()


def _right_half_zone() -> ZoneConfiguration:
    cfg = ZoneConfiguration()
    cfg.add_zone(
        Zone(
            id="z1",
            name="right",
            shape=ZoneShape.RECTANGLE,
            # RECTANGLE vertices are [(x1,y1), (x2,y2)] = top-left, bottom-right
            # in normalized coords. Right half = x 0.5..1.0, full height.
            vertices=[(0.5, 0.0), (1.0, 1.0)],
        )
    )
    return cfg


def test_zone_events_enter_and_occupancy(synthetic_clip: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=_right_half_zone(),
        write_tracking=False,
        write_zone_events=True,
        write_annotated=False,
    )
    VideoTrackingRunner(cfg, cv_processor=FakeCV()).run()

    events = pd.read_csv(out / "zone_events.csv")
    # FakeCV centroid x = n*5+4 on a 64px-wide frame; crosses x=32 (norm 0.5)
    # at n=6 (34). Exactly one 'enter' for object 1, no exits.
    enters = events[events["event"] == "enter"]
    assert list(enters["object_id"]) == [1]
    assert int(enters["frame"].iloc[0]) == 7  # logger frames are 1-based

    occ = pd.read_csv(out / "zone_occupancy.csv")
    row = occ[occ["zone_id"] == "z1"].iloc[0]
    assert int(row["frames_in_zone"]) >= 1
    assert abs(row["seconds"] - row["frames_in_zone"] / 10.0) < 1e-6


def test_zone_files_absent_when_disabled(synthetic_clip: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=_right_half_zone(),
        write_tracking=False,
        write_zone_events=False,  # guard: no zone files even though zones exist
        write_annotated=False,
    )
    VideoTrackingRunner(cfg, cv_processor=FakeCV()).run()
    assert not (out / "zone_events.csv").exists()
    assert not (out / "zone_occupancy.csv").exists()


def test_annotated_video_written_with_discoverable_name(synthetic_clip: Path, tmp_path: Path):
    import pytest

    out = tmp_path / "out"
    out.mkdir()
    cfg = VideoTrackingConfig(
        source_path=synthetic_clip,
        output_dir=out,
        zone_config=_right_half_zone(),
        write_tracking=False,
        write_zone_events=False,
        write_annotated=True,
        annotated_codec="mp4v",  # widely available; open_video_writer falls back if not
    )
    VideoTrackingRunner(cfg, cv_processor=FakeCV()).run()

    # Stem must end in "_annotated" so analysis._io.find_artifacts discovers it.
    annotated = list(out.glob("*_annotated.mp4"))
    if not annotated:
        pytest.skip("this OpenCV build cannot open any mp4 writer")
    assert len(annotated) == 1
    assert annotated[0].stat().st_size > 0
