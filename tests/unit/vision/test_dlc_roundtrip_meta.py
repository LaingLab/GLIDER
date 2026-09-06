"""The sidecar has to survive a read/write round-trip.

from_dlc_csv dropped the sidecar's resolution, so any pipeline that read a
pose CSV, transformed it and wrote it back produced a sidecar with no
resolution. Downstream that is not an error - it is a blank speed axis, because
resolution is what converts px to cm, and without it freezing and darting are
never scored. A filtered re-score of 31 sessions produced 21,576 empty speed
cells before anyone noticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import (
    fps_for_csv,
    from_dlc_csv,
    resolution_for_csv,
    to_dlc_csv,
)


def _pose(**meta) -> PoseData:
    return PoseData(
        xy=np.zeros((5, 2, 2)),
        confidence=np.ones((5, 2)),
        keypoint_names=["nose", "tail_base"],
        fps=30.0009,
        source="yolo_exp-7",
        metadata=meta,
    )


class TestResolutionSurvives:
    def test_a_read_carries_the_resolution_into_metadata(self, tmp_path):
        to_dlc_csv(_pose(resolution=[640, 480]), tmp_path / "a.csv")
        assert from_dlc_csv(tmp_path / "a.csv").metadata.get("resolution") == [640, 480]

    def test_a_round_trip_keeps_the_sidecar_intact(self, tmp_path):
        to_dlc_csv(_pose(resolution=[640, 480]), tmp_path / "a.csv")
        to_dlc_csv(from_dlc_csv(tmp_path / "a.csv"), tmp_path / "b.csv")
        assert resolution_for_csv(tmp_path / "b.csv") == (640, 480)

    def test_the_fps_survives_too(self, tmp_path):
        to_dlc_csv(_pose(resolution=[640, 480]), tmp_path / "a.csv")
        to_dlc_csv(from_dlc_csv(tmp_path / "a.csv"), tmp_path / "b.csv")
        assert fps_for_csv(tmp_path / "b.csv") == pytest.approx(30.0009)

    def test_a_csv_with_no_sidecar_reads_without_inventing_one(self, tmp_path):
        to_dlc_csv(_pose(resolution=[640, 480]), tmp_path / "a.csv", write_meta=False)
        assert from_dlc_csv(tmp_path / "a.csv").metadata.get("resolution") is None

    def test_other_provenance_is_preserved(self, tmp_path):
        to_dlc_csv(_pose(resolution=[640, 480]), tmp_path / "a.csv")
        back = from_dlc_csv(tmp_path / "a.csv")
        assert back.source == "yolo_exp-7"
        assert back.keypoint_names == ["nose", "tail_base"]
