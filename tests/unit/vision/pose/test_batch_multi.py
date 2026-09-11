"""run_batch with more than one animal. The single-animal path must not move."""

import csv

import numpy as np
import pytest

from glider.vision.pose.batch import (
    EventKind,
    FilterSettings,
    animals_dir,
    dlc_output_path,
    raw_output_path,
    run_batch,
)
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import from_dlc_csv, header_depth, list_individuals, read_pose_meta
from glider.vision.pose.identity import identity_output_path
from glider.vision.pose.tracks import PoseTracks
from glider.vision.zone_scoring import zone_output_dir
from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape

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


def test_two_animals_complete_and_write_per_animal_files(video, tmp_path):
    """Batch completion is still reported, and output is now one three-row
    file per slot under animals_dir -- the primary is never written."""
    result = run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
    )
    assert result.completed == [video]
    primary = dlc_output_path(video, tmp_path / "m.pt")
    assert not primary.exists()
    out = animals_dir(primary)
    assert sorted(p.name for p in out.glob("*.csv")) == ["animal0.csv", "animal1.csv"]
    for p in out.glob("*.csv"):
        assert header_depth(p) == 3


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


def test_gating_lands_on_each_animals_own_sidecar(video, tmp_path):
    """Each animal's own sidecar carries its own gate report.

    D1 wrote one aggregate sidecar via write_tracks_meta, which reads
    arena_gate off tracks[0] only -- see its docstring -- so this test used
    to check that slot 0's report, not slot 1's, landed there. D2 no longer
    writes that aggregate file at all: each animal's file gets its gate
    report straight from its own PoseData.metadata via write_pose_meta, so
    the thing worth pinning is that neither slot's report leaks into the
    other's file. Slot 0 is the relocated (blanked) track here and slot 1 is
    clean, so a swap (frames_blanked == 0 on animal0, or == 5 on animal1) is
    caught, not just an absent block."""
    run_batch(
        [video],
        tmp_path / "m.pt",
        GATE_NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: _gate_tracks(n_frames=5),
        arenas={video: _arena()},
        gate=_gate_settings(),
    )
    d = animals_dir(dlc_output_path(video, tmp_path / "m.pt"))
    meta0 = read_pose_meta(d / "animal0.csv")
    meta1 = read_pose_meta(d / "animal1.csv")
    assert meta0["arena_gate"]["gated"] is True
    assert meta0["arena_gate"]["frames_blanked"] == 5
    assert meta1["arena_gate"]["gated"] is True
    assert meta1["arena_gate"]["frames_blanked"] == 0


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
    """Both slots are smoothed, not just one -- each animal's own file must
    differ from its own _raw slot, at the individual each name actually
    belongs to."""
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks_with_spike(2),
        filtering=FilterSettings(),
    )
    out = animals_dir(dlc_output_path(video, tmp_path / "m.pt"))
    raw = raw_output_path(video, tmp_path / "m.pt")
    assert sorted(p.name for p in out.glob("*.csv")) == ["animal0.csv", "animal1.csv"]

    for slot, baseline in ((0, 100.0), (1, 200.0)):
        smoothed = from_dlc_csv(out / f"animal{slot}.csv")
        unsmoothed = from_dlc_csv(raw, individual=slot)
        assert not np.array_equal(smoothed.xy, unsmoothed.xy)
        # The spike is real in _raw, and the median filter erased it in this
        # animal's own file -- so each animal is verified against its own
        # baseline, not just "some difference happened somewhere".
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


# --------------------------------------------------------------------------
# Zone scoring, per animal. This is the branch's per-animal zone deliverable
# and, before this test, nothing exercised _score_zones_multi through
# run_batch. Mirrors test_batch_zones.py's single-animal idiom.
# --------------------------------------------------------------------------

ZONE_NAMES = ["nose", "body_center", "tail_base"]
ZONE_RESOLUTION = (640, 480)
ZONE_INSIDE = (320.0, 240.0)


def _zone_config() -> ZoneConfiguration:
    config = ZoneConfiguration()
    config.config_width, config.config_height = ZONE_RESOLUTION
    config.add_zone(
        Zone(
            id="z1",
            name="Zone 1",
            shape=ZoneShape.POLYGON,
            vertices=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
        )
    )
    return config


def _zone_tracks(n_frames=6):
    """Both animals sit inside the zone for every frame.

    Resolution lives only on slot 0's ``PoseData.metadata``, the way a real
    ``infer_video_tracks`` run sets it -- ``_score_zones_multi`` reads
    ``tracks[0].metadata``, not ``tracks.metadata``, so a test that put
    resolution at the container level wouldn't catch a regression back to
    the wrong one.
    """
    tracks = {}
    for slot in range(2):
        xy = np.zeros((n_frames, 3, 2), dtype=float)
        xy[:, 1, :] = ZONE_INSIDE  # body_center, the default zone keypoint
        meta = {"resolution": list(ZONE_RESOLUTION)} if slot == 0 else {}
        tracks[slot] = PoseData(
            xy=xy,
            confidence=np.ones((n_frames, 3)),
            keypoint_names=ZONE_NAMES,
            fps=30.0,
            source="yolo_test",
            metadata=meta,
        )
    t = PoseTracks(tracks=tracks, fps=30.0)
    t.metadata["stitched"] = {0: [], 1: []}
    return t


def test_zone_occupancy_carries_rows_for_every_animal(video, tmp_path):
    run_batch(
        [video],
        tmp_path / "m.pt",
        ZONE_NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: _zone_tracks(),
        zones={video.resolve(): _zone_config()},
    )
    with open(zone_output_dir(video) / "zone_occupancy.csv") as f:
        rows = list(csv.reader(f))[1:]
    assert {row[0] for row in rows} == {"animal0", "animal1"}


# --------------------------------------------------------------------------
# D2: one three-row CSV per animal, not one four-row primary. The primary is
# no longer written by run_batch at all -- it survives only as a naming
# anchor for animals_dir, _raw and the identity sidecar.
# --------------------------------------------------------------------------


def test_two_animals_write_one_three_row_csv_each(video, tmp_path):
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
    )
    d = animals_dir(dlc_output_path(video, tmp_path / "m.pt"))
    assert sorted(p.name for p in d.glob("*.csv")) == ["animal0.csv", "animal1.csv"]
    for p in d.glob("*.csv"):
        assert header_depth(p) == 3


def test_no_four_row_file_is_written(video, tmp_path):
    # The four-row CSV is an export now, never a batch product.
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
    )
    assert not dlc_output_path(video, tmp_path / "m.pt").exists()


def test_each_animal_carries_its_own_coordinates(video, tmp_path):
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
    )
    d = animals_dir(dlc_output_path(video, tmp_path / "m.pt"))
    assert np.allclose(from_dlc_csv(d / "animal0.csv").xy, 100.0)
    assert np.allclose(from_dlc_csv(d / "animal1.csv").xy, 200.0)


def test_one_animal_writes_no_animals_directory(video, tmp_path):
    # The byte-identity constraint. n_animals=1 is exactly today.
    def single(**kw):
        return PoseData(
            xy=np.zeros((20, 2, 2)),
            confidence=np.ones((20, 2)),
            keypoint_names=list(NAMES),
            fps=30.0,
        )

    run_batch([video], tmp_path / "m.pt", NAMES, n_animals=1, infer=single)
    primary = dlc_output_path(video, tmp_path / "m.pt")
    assert header_depth(primary) == 3
    assert not animals_dir(primary).exists()


# --------------------------------------------------------------------------
# Fix round: reconciling with what a *previous*, different-shaped run left
# behind. Neither direction used to clean up after the other, so
# find_pose_csv could hand back a superseded track as current, the
# skip/resume check tested an artifact _process_multi never writes, and
# shrinking n_animals orphaned a slot's file.
# --------------------------------------------------------------------------


def _single_pose():
    return PoseData(
        xy=np.zeros((20, 2, 2)),
        confidence=np.ones((20, 2)),
        keypoint_names=list(NAMES),
        fps=30.0,
    )


def test_switching_single_to_multi_replaces_the_stale_primary(video, tmp_path):
    """The dangerous direction: a video tracked single-animal, then rerun
    multi-animal with the default overwrite=False. The old primary must not
    survive next to the new animals_dir -- find_pose_csv would keep handing
    it back as the pose CSV for a video that has since been multi-tracked."""
    run_batch([video], tmp_path / "m.pt", NAMES, n_animals=1, infer=lambda **kw: _single_pose())
    primary = dlc_output_path(video, tmp_path / "m.pt")
    assert primary.exists()

    result = run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
    )
    assert result.completed == [video]
    assert not primary.exists()
    d = animals_dir(primary)
    assert sorted(p.name for p in d.glob("*.csv")) == ["animal0.csv", "animal1.csv"]


def test_multi_animal_rerun_skips_by_default(video, tmp_path):
    """Resume: primary.exists() is always False for n_animals > 1, so the old
    check never skipped a multi-animal rerun. The corrected check must."""
    calls = {"n": 0}

    def infer(**kw):
        calls["n"] += 1
        return fake_tracks(2)

    run_batch([video], tmp_path / "m.pt", NAMES, n_animals=2, infer_tracks=infer)
    result = run_batch([video], tmp_path / "m.pt", NAMES, n_animals=2, infer_tracks=infer)
    assert calls["n"] == 1
    assert result.skipped == [video]
    assert result.completed == []


def test_switching_multi_to_single_replaces_the_stale_animals_dir(video, tmp_path):
    """The reverse case: a fresh single-animal primary must not sit beside a
    stale animals_dir from an earlier multi-animal run -- any future
    per-animal loading keyed on animals_dir's presence must not mistake this
    video for still being multi-tracked."""
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
    )
    primary = dlc_output_path(video, tmp_path / "m.pt")
    d = animals_dir(primary)
    assert d.exists()

    result = run_batch(
        [video], tmp_path / "m.pt", NAMES, n_animals=1, infer=lambda **kw: _single_pose()
    )
    assert result.completed == [video]
    assert primary.exists()
    assert not d.exists()


def test_shrinking_animal_count_orphans_no_slot(video, tmp_path):
    """n_animals=3, then n_animals=2: animal2.csv (and its sidecar) must not
    survive -- the per-slot write loop only overwrites slots the new run
    produces, so a shrink used to leave the old slot behind."""
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=3,
        infer_tracks=lambda **kw: fake_tracks(3),
    )
    primary = dlc_output_path(video, tmp_path / "m.pt")
    d = animals_dir(primary)
    assert sorted(p.name for p in d.glob("*.csv")) == [
        "animal0.csv",
        "animal1.csv",
        "animal2.csv",
    ]

    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
        overwrite=True,
    )
    assert sorted(p.name for p in d.glob("*.csv")) == ["animal0.csv", "animal1.csv"]
    assert not (d / "animal2.csv").exists()
    assert not (d / "animal2.meta.json").exists()


def test_single_animal_only_history_is_unaffected_by_any_of_this(video, tmp_path):
    """No multi-animal history anywhere: skip/resume and reconciliation are
    both no-ops, exactly as before this fix round."""
    calls = {"n": 0}

    def single(**kw):
        calls["n"] += 1
        return _single_pose()

    run_batch([video], tmp_path / "m.pt", NAMES, n_animals=1, infer=single)
    result = run_batch([video], tmp_path / "m.pt", NAMES, n_animals=1, infer=single)
    primary = dlc_output_path(video, tmp_path / "m.pt")
    assert calls["n"] == 1
    assert result.skipped == [video]
    assert header_depth(primary) == 3
    assert not animals_dir(primary).exists()


def test_wrote_event_names_the_animals_dir_for_multi(video, tmp_path):
    """Fix 4: the WROTE event for a multi-animal run must not claim it wrote
    primary -- run_batch never writes that file on this branch."""
    events = []
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks(2),
        on_event=events.append,
    )
    wrote = next(e for e in events if e.kind is EventKind.WROTE)
    assert wrote.output == animals_dir(dlc_output_path(video, tmp_path / "m.pt"))


def test_wrote_event_still_names_primary_for_single_animal(video, tmp_path):
    events = []
    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=1,
        infer=lambda **kw: _single_pose(),
        on_event=events.append,
    )
    wrote = next(e for e in events if e.kind is EventKind.WROTE)
    assert wrote.output == dlc_output_path(video, tmp_path / "m.pt")


def test_gating_raw_survives_a_multi_animal_rerun(video, tmp_path):
    """The stale-primary cleanup after switching to multi must not delete the
    _raw companion the *current* multi-animal run just wrote -- raw_output_path
    is keyed by video and model alone, so a single-animal run's _raw and this
    run's own _raw are the same file."""
    run_batch([video], tmp_path / "m.pt", NAMES, n_animals=1, infer=lambda **kw: _single_pose())

    run_batch(
        [video],
        tmp_path / "m.pt",
        NAMES,
        n_animals=2,
        infer_tracks=lambda **kw: fake_tracks_with_spike(2),
        filtering=FilterSettings(),
    )
    raw = raw_output_path(video, tmp_path / "m.pt")
    assert raw.exists()
    assert header_depth(raw) == 4
    assert list_individuals(raw) == ["animal0", "animal1"]
