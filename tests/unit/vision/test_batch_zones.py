"""run_batch scoring zones as it goes, so no second pass over the video."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from glider.vision.pose.batch import EventKind, run_batch
from glider.vision.pose.core import PoseData
from glider.vision.zone_scoring import zone_output_dir
from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape

NAMES = ["nose", "body_center", "tail_base"]
RESOLUTION = (640, 480)
INSIDE = (320.0, 240.0)
OUTSIDE = (60.0, 60.0)


def _zone_config() -> ZoneConfiguration:
    config = ZoneConfiguration()
    config.config_width, config.config_height = RESOLUTION
    config.add_zone(
        Zone(
            id="z1",
            name="Zone 1",
            shape=ZoneShape.POLYGON,
            vertices=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
        )
    )
    return config


def _fake_infer(track):
    """Stand in for the model: returns a track without touching a video."""

    def infer(**kwargs):
        n = len(track)
        xy = np.zeros((n, 3, 2), dtype=float)
        xy[:, 1, :] = np.asarray(track, dtype=float)
        return PoseData(
            xy=xy,
            confidence=np.ones((n, 3)),
            keypoint_names=NAMES,
            fps=30.0,
            source="test",
            metadata={"resolution": list(RESOLUTION)},
        )

    return infer


@pytest.fixture
def batch(tmp_path):
    video = tmp_path / "t1_d2.mp4"
    video.touch()
    model = tmp_path / "train-6.pt"
    model.touch()
    return video, model


def _run(batch, track, zones, **kwargs):
    video, model = batch
    return run_batch([video], model, NAMES, infer=_fake_infer(track), zones=zones, **kwargs)


class TestZoneOutput:
    def test_zone_csvs_are_written_beside_the_video(self, batch):
        video, _ = batch
        _run(batch, [INSIDE] * 10, {video.resolve(): _zone_config()})
        out = zone_output_dir(video)
        assert (out / "zone_events.csv").exists()
        assert (out / "zone_occupancy.csv").exists()

    def test_occupancy_matches_the_track(self, batch):
        video, _ = batch
        _run(batch, [OUTSIDE] * 4 + [INSIDE] * 6, {video.resolve(): _zone_config()})
        with open(zone_output_dir(video) / "zone_occupancy.csv") as f:
            row = list(csv.reader(f))[1]
        assert row[2] == "6"
        assert float(row[3]) == pytest.approx(0.2)

    def test_nothing_is_written_without_a_zone_for_that_video(self, batch):
        video, _ = batch
        _run(batch, [INSIDE] * 5, {})
        assert not zone_output_dir(video).exists()

    def test_zones_are_optional(self, batch):
        video, model = batch
        result = run_batch([video], model, NAMES, infer=_fake_infer([INSIDE] * 5))
        assert result.completed == [video.resolve()]
        assert not zone_output_dir(video).exists()

    def test_the_video_still_counts_as_completed(self, batch):
        video, _ = batch
        result = _run(batch, [INSIDE] * 5, {video.resolve(): _zone_config()})
        assert result.completed == [video.resolve()]


class TestFailureIsolation:
    def test_a_zone_failure_does_not_fail_the_video(self, batch):
        # The pose CSV is already written and valid by this point. Losing the
        # zone scoring must not throw that away or mark the video failed.
        video, _ = batch
        config = _zone_config()
        result = _run(
            batch, [INSIDE] * 5, {video.resolve(): config}, zone_keypoint="not_a_keypoint"
        )
        assert result.completed == [video.resolve()]
        assert result.failed == []

    def test_a_zone_failure_is_reported(self, batch):
        video, _ = batch
        events = []
        _run(
            batch,
            [INSIDE] * 5,
            {video.resolve(): _zone_config()},
            zone_keypoint="not_a_keypoint",
            on_event=events.append,
        )
        warnings = [e for e in events if e.kind is EventKind.WROTE and e.message]
        assert warnings and "not_a_keypoint" in warnings[0].message

    def test_the_pose_csv_survives_a_zone_failure(self, batch):
        from glider.vision.pose.batch import dlc_output_path

        video, model = batch
        _run(batch, [INSIDE] * 5, {video.resolve(): _zone_config()}, zone_keypoint="nope")
        assert dlc_output_path(video, model).exists()


class TestKeypointChoice:
    def test_the_scoring_keypoint_is_configurable(self, batch):
        video, _ = batch
        _run(
            batch,
            [OUTSIDE] * 5,
            {video.resolve(): _zone_config()},
            zone_keypoint="nose",
        )
        # nose sits at (0, 0) in this fixture, outside the centre zone.
        with open(zone_output_dir(video) / "zone_occupancy.csv") as f:
            assert list(csv.reader(f))[1][2] == "0"
