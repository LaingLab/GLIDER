"""run_batch with more than one animal. The single-animal path must not move."""

import numpy as np
import pytest

from glider.vision.pose.batch import FilterSettings, dlc_output_path, raw_output_path, run_batch
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import from_dlc_csv, header_depth, list_individuals, read_pose_meta
from glider.vision.pose.identity import identity_output_path
from glider.vision.pose.tracks import PoseTracks

NAMES = ["snout", "tail"]

#: For the gating tests below -- same fronto-parallel square used in
#: test_batch.py's arena-gating tests, and the same four-keypoint arrangement
#: (three off the arena floor, one on it) that trips ``min_inside_fraction``.
GATE_NAMES = ["a", "b", "c", "d"]

SQUARE = [
    (120 / 640, 40 / 480),
    (520 / 640, 40 / 480),
    (520 / 640, 440 / 480),
    (120 / 640, 440 / 480),
]


def _arena():
    from glider.vision.arena import ArenaCalibration

    return ArenaCalibration(corners=SQUARE, width_cm=30.0, height_cm=30.0, frame_size=(640, 480))


def _gate_settings(**kw):
    from glider.vision.arena_gate import ArenaGateSettings

    return ArenaGateSettings(**kw)


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


# --------------------------------------------------------------------------
# Fix round 1: the riskiest new code -- per-slot gating, per-slot filtering,
# _raw provenance, and identity separation -- shipped with no coverage. These
# mirror the single-animal idiom in test_batch.py's arena-gating section
# (test_gating_runs_before_zone_scoring, test_raw_is_written_when_gating_
# without_filtering, test_the_primary_carries_the_gate_block,
# test_no_arena_means_no_gating) and test_filtering_writes_both_primary_and_raw.
# --------------------------------------------------------------------------


def _gate_tracks(n_frames=5):
    """Slot 0 relocated off the arena floor (blanked whole); slot 1 clean.

    Two different reports let a test tell whether the sidecar's arena_gate
    block came from slot 0 or slot 1 -- ``frames_blanked`` differs between
    them, so the assertion cannot pass by accident.
    """
    relocated = [[-900.0, -900.0]] * 3 + [[320.0, 240.0]]  # 1/4 inside < 0.5
    clean = [[320.0, 240.0]] * 4  # 4/4 inside

    def _pose(points):
        xy = np.array([points] * n_frames, dtype=float)
        return PoseData(
            xy=xy,
            confidence=np.full((n_frames, 4), 0.9),
            keypoint_names=GATE_NAMES,
            fps=30.0,
            source="yolo_test",
        )

    tracks = {0: _pose(relocated), 1: _pose(clean)}
    t = PoseTracks(tracks=tracks, fps=30.0)
    t.metadata["stitched"] = {0: [], 1: []}
    return t


def fake_tracks_with_spike(n_animals=2, n_frames=20, spike_frame=10, spike_delta=500.0):
    """Like ``fake_tracks``, but one isolated frame is way off-baseline.

    A constant track gives ``median_filter`` nothing to do, so filtering
    "changing something" needs an actual outlier -- one a 5-frame median
    filter smooths back to baseline, and that raw companion still shows.
    """
    tracks = {}
    for slot in range(n_animals):
        baseline = 100.0 * (slot + 1)
        xy = np.full((n_frames, 2, 2), baseline)
        xy[spike_frame] += spike_delta
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


def test_gating_lands_on_slot_zero_and_reaches_the_sidecar(video, tmp_path):
    """write_tracks_meta reads arena_gate off tracks[0] only -- see its
    docstring. Slot 0 is the relocated (blanked) track here and slot 1 is
    clean, so a sidecar reporting slot 1's report by mistake (frames_blanked
    == 0) is caught, not just an absent block."""
    run_batch(
        [video],
        tmp_path / "m.pt",
        GATE_NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: _gate_tracks(n_frames=5),
        arenas={video: _arena()},
        gate=_gate_settings(),
    )
    meta = read_pose_meta(dlc_output_path(video, tmp_path / "m.pt"))
    assert meta["arena_gate"]["gated"] is True
    assert meta["arena_gate"]["frames_blanked"] == 5


def test_raw_is_written_when_gating_without_filtering(video, tmp_path):
    """_raw is 'what the model actually said'. Gating without it would
    discard data with no companion -- and it must be the four-row format."""
    run_batch(
        [video],
        tmp_path / "m.pt",
        GATE_NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: _gate_tracks(),
        filtering=None,
        arenas={video: _arena()},
        gate=_gate_settings(),
    )
    raw = raw_output_path(video, tmp_path / "m.pt")
    assert raw.exists()
    assert header_depth(raw) == 4
    assert list_individuals(raw) == ["animal0", "animal1"]


def test_raw_is_written_when_filtering_without_gating(video, tmp_path):
    """The other half of the ``or``: filtering alone must also produce _raw."""
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
        filtering=FilterSettings(),
    )
    raw = raw_output_path(video, tmp_path / "m.pt")
    assert raw.exists()
    assert header_depth(raw) == 4
    assert list_individuals(raw) == ["animal0", "animal1"]


def test_filtering_rebuilds_every_slot(video, tmp_path):
    """Both slots are smoothed, not just one -- the primary must differ from
    _raw for animal0 *and* animal1, at the individual each name actually
    belongs to."""
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks_with_spike(2),
        filtering=FilterSettings(),
    )
    primary = dlc_output_path(video, tmp_path / "m.pt")
    raw = raw_output_path(video, tmp_path / "m.pt")
    assert list_individuals(primary) == ["animal0", "animal1"]

    for slot, baseline in ((0, 100.0), (1, 200.0)):
        smoothed = from_dlc_csv(primary, individual=slot)
        unsmoothed = from_dlc_csv(raw, individual=slot)
        assert not np.array_equal(smoothed.xy, unsmoothed.xy)
        # The spike is real in _raw, and the median filter erased it in the
        # primary -- so each animal is verified against its own baseline,
        # not just "some difference happened somewhere".
        assert np.allclose(unsmoothed.xy[10], baseline + 500.0)
        assert np.allclose(smoothed.xy[10], baseline)


def test_identity_min_separation_px_reaches_identity_flags(video, tmp_path):
    """A knob passed straight through must actually change the sidecar.

    fake_tracks(2) puts the two animals' centroids ~141px apart (100,100 vs
    200,200). The default separation (60px) is too small to flag them close;
    a separation past 141px is not.
    """

    def identity_text(min_separation_px):
        model = tmp_path / f"m-{min_separation_px}.pt"
        run_batch(
            [video],
            model,
            NAMES,
            n_animals=2,
            infer_tracks=lambda **kw: fake_tracks(2),
            identity_min_separation_px=min_separation_px,
        )
        return identity_output_path(dlc_output_path(video, model)).read_text()

    assert "close" not in identity_text(60.0)
    assert "close" in identity_text(200.0)
