"""Tests for VideoTrackingRunner — batch tracking over a recorded video."""

from pathlib import Path

import pandas as pd

from glider.vision.cv_processor import MotionResult, TrackedObject
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
