"""run_batch with more than one animal. The single-animal path must not move."""

import numpy as np
import pytest

from glider.vision.pose.batch import dlc_output_path, run_batch
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import header_depth, list_individuals
from glider.vision.pose.identity import identity_output_path
from glider.vision.pose.tracks import PoseTracks

NAMES = ["snout", "tail"]


def fake_tracks(n_animals=2, n_frames=20):
    tracks = {}
    for slot in range(n_animals):
        xy = np.full((n_frames, 2, 2), 100.0 * (slot + 1))
        tracks[slot] = PoseData(
            xy=xy,
            confidence=np.ones((n_frames, 2)),
            keypoint_names=list(NAMES),
            fps=30.0,
            source="yolo_test",
        )
    t = PoseTracks(tracks=tracks, fps=30.0)
    t.metadata["stitched"] = {s: [] for s in range(n_animals)}
    return t


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "session01.mp4"
    path.write_bytes(b"not really a video")
    return path


def test_two_animals_write_one_four_row_csv(video, tmp_path):
    result = run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
    )
    assert result.completed == [video]
    out = dlc_output_path(video, tmp_path / "m.pt")
    assert header_depth(out) == 4
    assert list_individuals(out) == ["animal0", "animal1"]


def test_the_identity_sidecar_is_written_beside_it(video, tmp_path):
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
    )
    assert identity_output_path(dlc_output_path(video, tmp_path / "m.pt")).exists()


def test_n_animals_one_still_takes_the_single_animal_path(video, tmp_path):
    # The guard against a regression in every existing project's output.
    called = {"single": 0, "multi": 0}

    def single(**kw):
        called["single"] += 1
        return PoseData(
            xy=np.zeros((20, 2, 2)),
            confidence=np.ones((20, 2)),
            keypoint_names=list(NAMES),
            fps=30.0,
        )

    def multi(**kw):
        called["multi"] += 1
        return fake_tracks(1)

    run_batch([video], tmp_path / "m.pt", NAMES, n_animals=1, infer=single, infer_tracks=multi)
    assert called == {"single": 1, "multi": 0}
    assert header_depth(dlc_output_path(video, tmp_path / "m.pt")) == 3


def test_the_consolidation_knobs_reach_the_tracker(video, tmp_path):
    seen = {}

    def capture(**kw):
        seen.update(kw)
        return fake_tracks(2)

    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        max_travel_px_per_frame=25.0,
        min_fragment_frames=9,
        infer_tracks=capture,
    )
    assert seen["n_animals"] == 2
    assert seen["max_travel_px_per_frame"] == 25.0
    assert seen["min_fragment_frames"] == 9


def test_a_failing_video_does_not_end_the_batch(video, tmp_path):
    second = tmp_path / "session02.mp4"
    second.write_bytes(b"x")

    def boom(**kw):
        if "session01" in kw["video_path"]:
            raise RuntimeError("bad video")
        return fake_tracks(2)

    result = run_batch([video, second], tmp_path / "m.pt", NAMES, n_animals=2, infer_tracks=boom)
    assert [v for v, _ in result.failed] == [video]
    assert result.completed == [second]
