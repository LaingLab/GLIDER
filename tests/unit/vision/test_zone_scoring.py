"""Scoring zone occupancy straight off a pose CSV."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import to_dlc_csv
from glider.vision.zone_scoring import (
    KeypointMissingError,
    score_pose,
    write_zone_csvs,
)
from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape

NAMES = ["nose", "body_center", "tail_base"]
RESOLUTION = (640, 480)


def _centre_zone() -> ZoneConfiguration:
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


def _pose(body_xy, confidence=None) -> PoseData:
    """A pose whose body_center follows *body_xy*, in pixels."""
    n = len(body_xy)
    xy = np.zeros((n, 3, 2), dtype=float)
    xy[:, 1, :] = np.asarray(body_xy, dtype=float)
    conf = np.ones((n, 3), dtype=float)
    if confidence is not None:
        conf[:, 1] = confidence
    return PoseData(xy=xy, confidence=conf, keypoint_names=NAMES, fps=30.0, source="test")


# Pixel points, given the 0.4-0.6 zone on a 640x480 frame (256-384, 192-288).
INSIDE = (320.0, 240.0)
OUTSIDE = (60.0, 60.0)


class TestOccupancy:
    def test_counts_frames_inside_the_zone(self):
        result = score_pose(_pose([INSIDE] * 10), _centre_zone(), resolution=RESOLUTION)
        assert result.frames_in_zone["z1"] == 10

    def test_counts_nothing_when_always_outside(self):
        result = score_pose(_pose([OUTSIDE] * 10), _centre_zone(), resolution=RESOLUTION)
        assert result.frames_in_zone["z1"] == 0

    def test_seconds_use_the_pose_fps(self):
        result = score_pose(_pose([INSIDE] * 30), _centre_zone(), resolution=RESOLUTION)
        assert result.seconds_in_zone["z1"] == pytest.approx(1.0)

    def test_a_mixed_track_counts_only_the_inside_frames(self):
        track = [INSIDE] * 4 + [OUTSIDE] * 6
        result = score_pose(_pose(track), _centre_zone(), resolution=RESOLUTION)
        assert result.frames_in_zone["z1"] == 4


class TestEvents:
    def test_entering_and_leaving_emit_one_event_each(self):
        track = [OUTSIDE] * 2 + [INSIDE] * 3 + [OUTSIDE] * 2
        result = score_pose(_pose(track), _centre_zone(), resolution=RESOLUTION)
        assert [(e.frame, e.event) for e in result.events] == [(2, "enter"), (5, "exit")]

    def test_starting_inside_emits_an_enter(self):
        result = score_pose(_pose([INSIDE] * 3), _centre_zone(), resolution=RESOLUTION)
        assert [(e.frame, e.event) for e in result.events] == [(0, "enter")]

    def test_staying_put_emits_nothing_further(self):
        result = score_pose(_pose([INSIDE] * 50), _centre_zone(), resolution=RESOLUTION)
        assert len(result.events) == 1

    def test_events_carry_the_zone_identity(self):
        result = score_pose(_pose([INSIDE]), _centre_zone(), resolution=RESOLUTION)
        assert (result.events[0].zone_id, result.events[0].zone_name) == ("z1", "Zone 1")


class TestDropouts:
    # A dropout is missing information, not movement. Scoring one as "outside"
    # would invent an exit and an entry the animal never made.
    LOW = [0.9, 0.9, 0.05, 0.9, 0.9]

    def test_a_dropout_does_not_emit_an_exit(self):
        result = score_pose(
            _pose([INSIDE] * 5, confidence=self.LOW),
            _centre_zone(),
            resolution=RESOLUTION,
            min_confidence=0.5,
        )
        assert [e.event for e in result.events] == ["enter"]

    def test_a_dropout_is_not_counted_as_occupancy(self):
        result = score_pose(
            _pose([INSIDE] * 5, confidence=self.LOW),
            _centre_zone(),
            resolution=RESOLUTION,
            min_confidence=0.5,
        )
        assert result.frames_in_zone["z1"] == 4

    def test_coverage_is_reported(self):
        result = score_pose(
            _pose([INSIDE] * 5, confidence=self.LOW),
            _centre_zone(),
            resolution=RESOLUTION,
            min_confidence=0.5,
        )
        assert (result.frames_scored, result.frames_total) == (4, 5)

    def test_nan_coordinates_count_as_dropouts(self):
        track = [INSIDE, (float("nan"), float("nan")), INSIDE]
        result = score_pose(_pose(track), _centre_zone(), resolution=RESOLUTION)
        assert result.frames_scored == 2
        assert [e.event for e in result.events] == ["enter"]


class TestKeypointSelection:
    def test_a_missing_keypoint_raises_rather_than_scoring_zero(self):
        # The failure this guards against is silent: a vocabulary mismatch
        # (center_body for body_center) would otherwise produce a clean-looking
        # file of zeros.
        with pytest.raises(KeypointMissingError, match="centre_body"):
            score_pose(
                _pose([INSIDE]),
                _centre_zone(),
                resolution=RESOLUTION,
                keypoint="centre_body",
            )

    def test_the_error_lists_what_is_available(self):
        with pytest.raises(KeypointMissingError, match="body_center"):
            score_pose(_pose([INSIDE]), _centre_zone(), resolution=RESOLUTION, keypoint="nope")

    def test_a_named_keypoint_is_used(self):
        pose = _pose([OUTSIDE])
        pose.xy[:, 0, :] = INSIDE  # nose inside, body_center outside
        result = score_pose(pose, _centre_zone(), resolution=RESOLUTION, keypoint="nose")
        assert result.frames_in_zone["z1"] == 1


class TestCsvOutput:
    def test_files_match_the_tracking_runner_schema(self, tmp_path):
        result = score_pose(
            _pose([OUTSIDE] * 2 + [INSIDE] * 3), _centre_zone(), resolution=RESOLUTION
        )
        write_zone_csvs(result, tmp_path)

        with open(tmp_path / "zone_events.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["frame", "elapsed_ms", "zone_id", "zone_name", "object_id", "event"]

        with open(tmp_path / "zone_occupancy.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["zone_id", "zone_name", "frames_in_zone", "seconds"]
        assert rows[1][:3] == ["z1", "Zone 1", "3"]

    def test_elapsed_ms_follows_the_fps(self, tmp_path):
        result = score_pose(_pose([OUTSIDE] * 30 + [INSIDE]), _centre_zone(), resolution=RESOLUTION)
        write_zone_csvs(result, tmp_path)
        with open(tmp_path / "zone_events.csv") as f:
            row = list(csv.reader(f))[1]
        assert float(row[1]) == pytest.approx(1000.0)

    def test_it_creates_the_output_directory(self, tmp_path):
        result = score_pose(_pose([INSIDE]), _centre_zone(), resolution=RESOLUTION)
        write_zone_csvs(result, tmp_path / "deep" / "out")
        assert (tmp_path / "deep" / "out" / "zone_occupancy.csv").exists()


class TestFromCsv:
    def test_scores_a_csv_written_by_the_pose_batch(self, tmp_path):
        from glider.vision.zone_scoring import score_csv

        pose = _pose([OUTSIDE] * 2 + [INSIDE] * 8)
        csv_path = tmp_path / "Test 1DLC_x.csv"
        to_dlc_csv(pose, csv_path)

        result = score_csv(csv_path, _centre_zone(), resolution=RESOLUTION)
        assert result.frames_in_zone["z1"] == 8
        assert [e.event for e in result.events] == ["enter"]

    def test_resolution_comes_from_the_sidecar_when_not_given(self, tmp_path):
        from glider.vision.pose.dlc import backfill_resolution
        from glider.vision.zone_scoring import score_csv

        pose = _pose([INSIDE] * 4)
        csv_path = tmp_path / "Test 2DLC_x.csv"
        to_dlc_csv(pose, csv_path)
        backfill_resolution(csv_path, RESOLUTION)

        assert score_csv(csv_path, _centre_zone()).frames_in_zone["z1"] == 4

    def test_the_zone_config_resolution_is_the_last_fallback(self, tmp_path):
        from glider.vision.zone_scoring import score_csv

        csv_path = tmp_path / "Test 3DLC_x.csv"
        to_dlc_csv(_pose([INSIDE] * 3), csv_path, write_meta=False)
        # No sidecar and no explicit resolution, but the zones record the size
        # they were drawn at, which is the same thing.
        assert score_csv(csv_path, _centre_zone()).frames_in_zone["z1"] == 3

    def test_a_wholly_unknown_resolution_is_refused(self, tmp_path):
        # Nothing to place the pixel track against. Guessing here would
        # mis-score silently rather than fail.
        from glider.vision.zone_scoring import score_csv

        config = _centre_zone()
        config.config_width = config.config_height = 0
        csv_path = tmp_path / "Test 4DLC_x.csv"
        to_dlc_csv(_pose([INSIDE]), csv_path, write_meta=False)
        with pytest.raises(ValueError, match="resolution"):
            score_csv(csv_path, config)
