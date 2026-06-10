"""Tests for VideoTrackingRunner — batch tracking over a recorded video."""

from pathlib import Path

import pandas as pd

from glider.vision.cv_processor import MotionResult, TrackedObject
from glider.vision.video_tracking_runner import VideoTrackingConfig, VideoTrackingRunner


class FakeCV:
    """Stand-in CVProcessor: reports one object whose x = frame index * 5."""

    is_initialized = True

    def initialize(self):  # pragma: no cover - trivial
        return True

    def reset(self):  # pragma: no cover - trivial
        pass

    def process_frame(self, frame, timestamp):
        # Centroid marches rightward so it can cross a zone later.
        self._n = getattr(self, "_n", -1) + 1
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
