"""Batch pose inference: discovery, output naming, and run_batch.

Inference is injected via ``run_batch(infer=...)``, so nothing here needs
torch, a GPU, or a real video file.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pytest

from glider.vision.pose import batch
from glider.vision.pose.core import PoseCancelledError, PoseData

NAMES = ["a", "b"]
MODEL = Path("exp-6.pt")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _pose(n_frames: int = 3) -> PoseData:
    return PoseData(
        xy=np.zeros((n_frames, len(NAMES), 2)),
        confidence=np.ones((n_frames, len(NAMES))),
        keypoint_names=NAMES,
    )


def _fake_infer(**kwargs) -> PoseData:
    return _pose()


@pytest.fixture(autouse=True)
def stub_device(monkeypatch):
    """Keep run_batch's preflight from importing torch to resolve a device."""
    import glider.vision.pose.device as device_mod

    monkeypatch.setattr(device_mod, "resolve_device", lambda d, require_gpu=False: "cpu")


@pytest.fixture
def videos(tmp_path):
    return [_touch(tmp_path / f"v{i}.mp4") for i in range(3)]


# --------------------------------------------------------------------------
# discover_videos
# --------------------------------------------------------------------------


def test_discovers_videos_recursively(tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.avi")
    _touch(tmp_path / "notes.txt")
    assert [p.name for p in batch.discover_videos([tmp_path])] == ["a.mp4", "b.avi"]


def test_non_recursive_skips_subdirectories(tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.mp4")
    found = batch.discover_videos([tmp_path], recursive=False)
    assert [p.name for p in found] == ["a.mp4"]


def test_extension_match_is_case_insensitive(tmp_path):
    _touch(tmp_path / "a.MP4")
    assert len(batch.discover_videos([tmp_path])) == 1


def test_accepts_individual_files(tmp_path):
    video = _touch(tmp_path / "a.mp4")
    assert batch.discover_videos([video]) == [video.resolve()]


def test_dedupes_overlapping_selections(tmp_path):
    video = _touch(tmp_path / "a.mp4")
    assert len(batch.discover_videos([video, video, tmp_path])) == 1


def test_ignores_missing_paths(tmp_path):
    assert batch.discover_videos([tmp_path / "nope"]) == []


def test_ignores_non_video_files(tmp_path):
    _touch(tmp_path / "readme.md")
    assert batch.discover_videos([tmp_path]) == []


# --------------------------------------------------------------------------
# output naming
# --------------------------------------------------------------------------


def test_dlc_output_path_uses_dlc_naming(tmp_path):
    out = batch.dlc_output_path(tmp_path / "session01.mp4", MODEL)
    assert out == tmp_path / "session01DLC_exp-6.csv"


def test_raw_output_path_appends_raw(tmp_path):
    out = batch.raw_output_path(tmp_path / "session01.mp4", MODEL)
    assert out == tmp_path / "session01DLC_exp-6_raw.csv"


# --------------------------------------------------------------------------
# run_batch
# --------------------------------------------------------------------------


def test_writes_one_csv_per_video(videos):
    result = batch.run_batch(videos, MODEL, NAMES, infer=_fake_infer)
    assert len(result.completed) == 3
    assert not result.failed
    assert all(batch.dlc_output_path(v, MODEL).is_file() for v in videos)


def test_written_csv_is_readable_as_dlc(videos):
    from glider.vision.pose.dlc import from_dlc_csv

    batch.run_batch(videos[:1], MODEL, NAMES, infer=_fake_infer)
    pose = from_dlc_csv(batch.dlc_output_path(videos[0], MODEL))
    assert pose.keypoint_names == NAMES
    assert pose.n_frames == 3


def test_skips_existing_output(videos):
    batch.dlc_output_path(videos[0], MODEL).write_text("stale")
    result = batch.run_batch(videos, MODEL, NAMES, infer=_fake_infer)
    assert result.skipped == [videos[0].resolve()]
    assert len(result.completed) == 2
    assert batch.dlc_output_path(videos[0], MODEL).read_text() == "stale"


def test_overwrite_replaces_existing(videos):
    batch.dlc_output_path(videos[0], MODEL).write_text("stale")
    result = batch.run_batch(videos, MODEL, NAMES, overwrite=True, infer=_fake_infer)
    assert not result.skipped
    assert len(result.completed) == 3
    assert batch.dlc_output_path(videos[0], MODEL).read_text() != "stale"


def test_one_failure_does_not_stop_the_batch(videos):
    def flaky(**kwargs):
        if Path(kwargs["video_path"]).name == "v1.mp4":
            raise OSError("corrupt file")
        return _pose()

    result = batch.run_batch(videos, MODEL, NAMES, infer=flaky)
    assert len(result.completed) == 2
    assert len(result.failed) == 1
    assert "corrupt file" in result.failed[0][1]


def test_failed_video_leaves_no_csv(videos):
    def flaky(**kwargs):
        raise OSError("corrupt file")

    batch.run_batch(videos[:1], MODEL, NAMES, infer=flaky)
    assert not batch.dlc_output_path(videos[0], MODEL).exists()


def test_cancel_stops_batch_and_writes_nothing_further(videos):
    def cancelling(**kwargs):
        if Path(kwargs["video_path"]).name == "v1.mp4":
            raise PoseCancelledError("stopped")
        return _pose()

    result = batch.run_batch(videos, MODEL, NAMES, infer=cancelling)
    assert result.cancelled is True
    assert len(result.completed) == 1
    assert not batch.dlc_output_path(videos[1], MODEL).exists()
    assert not batch.dlc_output_path(videos[2], MODEL).exists()


def test_cancel_cb_checked_between_videos(videos):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1

    result = batch.run_batch(videos, MODEL, NAMES, cancel_cb=cancel, infer=_fake_infer)
    assert result.cancelled is True
    assert len(result.completed) == 1


def test_filtering_writes_both_primary_and_raw(videos):
    result = batch.run_batch(
        videos[:1], MODEL, NAMES, filtering=batch.FilterSettings(), infer=_fake_infer
    )
    assert result.completed
    assert batch.dlc_output_path(videos[0], MODEL).is_file()
    assert batch.raw_output_path(videos[0], MODEL).is_file()


def test_no_filtering_writes_only_primary(videos):
    batch.run_batch(videos[:1], MODEL, NAMES, infer=_fake_infer)
    assert not batch.raw_output_path(videos[0], MODEL).exists()


def test_stale_raw_file_does_not_suppress_a_run(videos):
    batch.raw_output_path(videos[0], MODEL).write_text("stale")
    result = batch.run_batch(videos[:1], MODEL, NAMES, infer=_fake_infer)
    assert len(result.completed) == 1


def test_preflight_rejects_duplicate_names(videos):
    with pytest.raises(ValueError, match="unique"):
        batch.run_batch(videos, MODEL, ["a", "a"], infer=_fake_infer)


def test_preflight_rejects_empty_names(videos):
    with pytest.raises(ValueError, match="keypoint_names"):
        batch.run_batch(videos, MODEL, [], infer=_fake_infer)


def test_preflight_runs_before_any_video(videos):
    """A bad device must abort the batch, not fail each video individually."""
    import glider.vision.pose.device as device_mod

    def boom(device, require_gpu=False):
        raise RuntimeError("no CUDA device")

    device_mod_resolve = device_mod.resolve_device
    device_mod.resolve_device = boom
    try:
        with pytest.raises(RuntimeError, match="no CUDA device"):
            batch.run_batch(videos, MODEL, NAMES, infer=_fake_infer)
    finally:
        device_mod.resolve_device = device_mod_resolve
    assert not batch.dlc_output_path(videos[0], MODEL).exists()


def test_emits_events_in_order(videos):
    kinds = []
    batch.run_batch(
        videos[:1], MODEL, NAMES, infer=_fake_infer, on_event=lambda e: kinds.append(e.kind)
    )
    assert kinds == [batch.EventKind.STARTED, batch.EventKind.WROTE]


def test_skip_emits_a_skipped_event(videos):
    batch.dlc_output_path(videos[0], MODEL).write_text("stale")
    kinds = []
    batch.run_batch(
        videos[:1], MODEL, NAMES, infer=_fake_infer, on_event=lambda e: kinds.append(e.kind)
    )
    assert kinds == [batch.EventKind.SKIPPED]


def test_result_summary_reports_all_counts(videos):
    batch.dlc_output_path(videos[0], MODEL).write_text("stale")

    def flaky(**kwargs):
        if Path(kwargs["video_path"]).name == "v1.mp4":
            raise OSError("bad")
        return _pose()

    result = batch.run_batch(videos, MODEL, NAMES, infer=flaky)
    assert "1 completed" in result.summary
    assert "1 skipped" in result.summary
    assert "1 failed" in result.summary


def test_raises_inside_a_running_event_loop(videos):
    async def go():
        batch.run_batch(videos, MODEL, NAMES, infer=_fake_infer)

    with pytest.raises(RuntimeError, match="event loop"):
        asyncio.run(go())


# --------------------------------------------------------------------------
# find_pose_csv
#
# The behavior tools used to look for "<stem>.csv" while run_batch wrote
# "<stem>DLC_<model>.csv", so pointing the annotator at a folder this tool
# had just filled reported every session as missing.
# --------------------------------------------------------------------------


def test_finds_the_csv_run_batch_actually_writes(tmp_path):
    video = tmp_path / "session01.mp4"
    video.touch()
    written = tmp_path / "session01DLC_exp-6.csv"
    written.touch()

    assert batch.find_pose_csv(video) == written


def test_finds_a_plain_stem_csv(tmp_path):
    """Hand-placed or DeepLabCut-exported files keep working."""
    video = tmp_path / "session01.mp4"
    video.touch()
    plain = tmp_path / "session01.csv"
    plain.touch()

    assert batch.find_pose_csv(video) == plain


def test_exact_stem_wins_over_a_batch_output(tmp_path):
    """A deliberately placed file is never shadowed by a batch run."""
    video = tmp_path / "session01.mp4"
    video.touch()
    plain = tmp_path / "session01.csv"
    plain.touch()
    (tmp_path / "session01DLC_exp-6.csv").touch()

    assert batch.find_pose_csv(video) == plain


def test_skips_the_raw_companion(tmp_path):
    """_raw is the unsmoothed inference — never the one to analyze."""
    video = tmp_path / "session01.mp4"
    video.touch()
    (tmp_path / "session01DLC_exp-6_raw.csv").touch()
    primary = tmp_path / "session01DLC_exp-6.csv"
    primary.touch()

    assert batch.find_pose_csv(video) == primary


def test_skips_the_annotations_companion(tmp_path):
    """The annotator writes "<pose stem>_annotations.csv" beside the pose CSV.

    For a batch-named folder that is "<stem>DLC_<model>_annotations.csv", which
    matches the same glob and — being written later — won the most-recent tie
    break. The second launch on any annotated batch folder then read a file of
    behavior zones as though it were pose data.
    """
    video = tmp_path / "session01.mp4"
    video.touch()
    primary = tmp_path / "session01DLC_exp-6.csv"
    primary.touch()
    annotations = tmp_path / "session01DLC_exp-6_annotations.csv"
    annotations.touch()
    # Make the decoy unambiguously newer, which is how it wins on a real disk.
    os.utime(annotations, (time.time() + 60, time.time() + 60))

    assert batch.find_pose_csv(video) == primary


def test_annotations_alone_is_not_a_match(tmp_path):
    """With the pose CSV gone, an annotations file must not stand in for it."""
    video = tmp_path / "session01.mp4"
    video.touch()
    (tmp_path / "session01DLC_exp-6_annotations.csv").touch()

    assert batch.find_pose_csv(video) is None


def test_raw_alone_is_not_a_match(tmp_path):
    video = tmp_path / "session01.mp4"
    video.touch()
    (tmp_path / "session01DLC_exp-6_raw.csv").touch()

    assert batch.find_pose_csv(video) is None


def test_returns_none_when_nothing_matches(tmp_path):
    video = tmp_path / "session01.mp4"
    video.touch()
    (tmp_path / "someone_else.csv").touch()

    assert batch.find_pose_csv(video) is None


def test_searches_a_separate_poses_dir(tmp_path):
    """The annotator lets the operator keep poses in their own folder."""
    videos = tmp_path / "videos"
    poses = tmp_path / "poses"
    videos.mkdir()
    poses.mkdir()
    video = videos / "session01.mp4"
    video.touch()
    written = poses / "session01DLC_exp-6.csv"
    written.touch()

    assert batch.find_pose_csv(video, poses) == written
    assert batch.find_pose_csv(video) is None


def test_missing_search_dir_is_not_an_error(tmp_path):
    video = tmp_path / "session01.mp4"
    video.touch()

    assert batch.find_pose_csv(video, tmp_path / "nope") is None


def test_multiple_models_resolve_deterministically(tmp_path):
    video = tmp_path / "session01.mp4"
    video.touch()
    (tmp_path / "session01DLC_zeta.csv").touch()
    (tmp_path / "session01DLC_alpha.csv").touch()

    assert batch.find_pose_csv(video) == tmp_path / "session01DLC_alpha.csv"


def test_the_two_pipelines_agree_on_what_counts_as_a_video():
    """Both containers used to be accepted by only one half of the pipeline."""
    assert {".wmv", ".webm"} <= batch.VIDEO_EXTS

    from glider.analysis.behavior import project
    from glider.gui.behavior import window

    assert project.VIDEO_EXTS == batch.VIDEO_EXTS
    assert window._VIDEO_EXTS == batch.VIDEO_EXTS


# --------------------------------------------------------------------------
# Several models over one video
#
# Alphabetical order silently prefers exp-5 over exp-7, scoring a cohort with
# a superseded pose model and reporting nothing.
# --------------------------------------------------------------------------


def _touch_at(path, when):
    import os

    path.touch()
    os.utime(path, (when, when))
    return path


def test_the_newest_model_output_wins_not_the_first_alphabetically(tmp_path):
    video = tmp_path / "session01.mp4"
    video.touch()
    old = _touch_at(tmp_path / "session01DLC_exp-5.csv", 1_000_000)
    new = _touch_at(tmp_path / "session01DLC_exp-7.csv", 2_000_000)

    assert batch.find_pose_csv(video) == new
    assert old.exists()  # nothing is removed, only deselected


def test_newest_wins_even_when_it_sorts_first(tmp_path):
    """The tie-break must be time, not a different alphabetical accident."""
    video = tmp_path / "session01.mp4"
    video.touch()
    _touch_at(tmp_path / "session01DLC_zzz.csv", 1_000_000)
    new = _touch_at(tmp_path / "session01DLC_aaa.csv", 2_000_000)

    assert batch.find_pose_csv(video) == new


def test_the_choice_is_logged(tmp_path, caplog):
    video = tmp_path / "session01.mp4"
    video.touch()
    _touch_at(tmp_path / "session01DLC_exp-5.csv", 1_000_000)
    _touch_at(tmp_path / "session01DLC_exp-7.csv", 2_000_000)

    with caplog.at_level("INFO", logger="glider.vision.pose.batch"):
        batch.find_pose_csv(video)
    assert "exp-7" in caplog.text
    assert "exp-5" in caplog.text  # both candidates named, so the pick is auditable


def test_a_single_candidate_is_not_logged_as_ambiguous(tmp_path, caplog):
    video = tmp_path / "session01.mp4"
    video.touch()
    only = _touch_at(tmp_path / "session01DLC_exp-7.csv", 1_000_000)

    with caplog.at_level("INFO", logger="glider.vision.pose.batch"):
        assert batch.find_pose_csv(video) == only
    assert caplog.text == ""


# --------------------------------------------------------------------------
# Arena gating
#
# The detector is confident when it finds something that is not the animal, so
# the confidence mask inside smooth() never sees it. Geometry does: keypoints
# that landed off the arena floor are blanked before anything downstream —
# zone scoring included — is allowed to read them.
# --------------------------------------------------------------------------

GATE_NAMES = ["a", "b", "c", "d"]

#: The fronto-parallel square from test_arena_gate.py: a real 400x400 px square
#: in a 640x480 frame, which is close to what these rigs produce.
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


def _gate_pose(points) -> PoseData:
    """One frame of four keypoints at *points*, all confidently detected."""
    xy = np.array([points], dtype=float)
    return PoseData(xy=xy, confidence=np.full(xy.shape[:2], 0.9), keypoint_names=GATE_NAMES)


def _relocated_pose() -> PoseData:
    """Three of four keypoints out on the bench floor, one still in the arena.

    Below ``min_inside_fraction=0.5``, so the gate blanks the frame whole —
    the case gating exists for, and the one a per-keypoint mask would keep.
    """
    return _gate_pose([[-900.0, -900.0]] * 3 + [[320.0, 240.0]])


def _clean_pose() -> PoseData:
    """Centred in the arena: nothing for the gate to blank."""
    return _gate_pose([[320.0, 240.0]] * 4)


def _run(tmp_path, video, *, pose, **kw):
    return batch.run_batch(
        [video],
        tmp_path / "exp-7.pt",
        GATE_NAMES,
        infer=lambda **_: pose.copy(),
        **kw,
    )


@pytest.fixture
def video(tmp_path):
    return _touch(tmp_path / "session.mp4")


def test_gating_runs_before_zone_scoring(tmp_path, video, monkeypatch):
    """Centre-time computed from bench-floor detections is meaningless."""
    seen = {}

    def fake_score(video, pose, zones, keypoint):
        seen["blanked"] = int(np.isnan(pose.xy[:, 0, 0]).sum())
        return ""

    monkeypatch.setattr(batch, "_score_zones", fake_score)
    _run(
        tmp_path,
        video,
        pose=_relocated_pose(),
        arenas={video: _arena()},
        gate=_gate_settings(),
    )
    assert seen["blanked"] == 1


def test_raw_is_written_when_gating_without_filtering(tmp_path, video):
    """_raw is 'what the model actually said'. Gating without it would discard
    data with no companion."""
    _run(
        tmp_path,
        video,
        pose=_relocated_pose(),
        filtering=None,
        arenas={video: _arena()},
        gate=_gate_settings(),
    )
    assert batch.raw_output_path(video, tmp_path / "exp-7.pt").exists()


def test_the_primary_carries_the_gate_block(tmp_path, video):
    from glider.vision.pose.dlc import read_pose_meta

    _run(
        tmp_path,
        video,
        pose=_relocated_pose(),
        arenas={video: _arena()},
        gate=_gate_settings(),
    )
    meta = read_pose_meta(batch.dlc_output_path(video, tmp_path / "exp-7.pt"))
    assert meta["arena_gate"]["gated"] is True
    assert meta["arena_gate"]["frames_blanked"] == 1


def test_no_arena_means_no_gating(tmp_path, video):
    """arenas={} must leave the pipeline byte-identical to today."""
    from glider.vision.pose.dlc import read_pose_meta

    _run(tmp_path, video, pose=_relocated_pose(), arenas={}, gate=_gate_settings())
    meta = read_pose_meta(batch.dlc_output_path(video, tmp_path / "exp-7.pt"))
    assert "arena_gate" not in meta
    assert not batch.raw_output_path(video, tmp_path / "exp-7.pt").exists()


def test_a_clean_video_reports_no_warning(tmp_path, video):
    """gate_warning must be initialized per video, not only assigned inside the
    over-threshold branch — otherwise the common case raises NameError."""
    result = _run(
        tmp_path,
        video,
        pose=_clean_pose(),
        arenas={video: _arena()},
        gate=_gate_settings(),
    )
    assert result.completed == [video.resolve()]


def test_a_heavily_blanked_video_is_called_out(tmp_path, video):
    messages = []
    _run(
        tmp_path,
        video,
        pose=_relocated_pose(),
        arenas={video: _arena()},
        gate=_gate_settings(),
        on_event=lambda e: messages.append(e.message),
    )
    assert any("gate blanked" in m for m in messages)


def test_a_degenerate_arena_does_not_fail_the_video(tmp_path, video):
    """Mirrors _score_zones: by here the inference is done and valid."""
    from glider.vision.arena import ArenaCalibration
    from glider.vision.pose.dlc import read_pose_meta

    bad = ArenaCalibration(corners=[(0.5, 0.5)] * 4, frame_size=(640, 480))
    result = _run(
        tmp_path, video, pose=_relocated_pose(), arenas={video: bad}, gate=_gate_settings()
    )
    assert result.completed == [video.resolve()]
    assert not result.failed
    assert "arena_gate" not in read_pose_meta(batch.dlc_output_path(video, tmp_path / "exp-7.pt"))


def test_the_gate_block_survives_filtering(tmp_path, video):
    """smooth() returns a new PoseData; the provenance must reach the sidecar."""
    from glider.vision.pose.dlc import read_pose_meta

    _run(
        tmp_path,
        video,
        pose=_clean_pose(),
        filtering=batch.FilterSettings(),
        arenas={video: _arena()},
        gate=_gate_settings(),
    )
    meta = read_pose_meta(batch.dlc_output_path(video, tmp_path / "exp-7.pt"))
    assert meta["arena_gate"]["gated"] is True


def test_each_video_reports_only_its_own_zone_warning(tmp_path, monkeypatch):
    """zone_warning used to be bound once, before the loop.

    Nothing observable came of it — it was reassigned before every emit — but
    it is one edited branch away from labelling a clean video with the
    previous one's problem, which is exactly the shape of bug gate_warning
    would have had. Both are per-video now, and this holds them there.
    """
    first = _touch(tmp_path / "a.mp4")
    second = _touch(tmp_path / "b.mp4")
    monkeypatch.setattr(
        batch,
        "_score_zones",
        lambda video, pose, zones, keypoint: (
            "zones not scored: boom" if video.name == "a.mp4" else ""
        ),
    )
    messages = {}
    batch.run_batch(
        [first, second],
        tmp_path / "exp-7.pt",
        GATE_NAMES,
        infer=lambda **_: _clean_pose(),
        on_event=lambda e: messages.setdefault(e.video.name, []).append(e.message),
    )
    assert any("boom" in m for m in messages["a.mp4"])
    assert not any("boom" in m for m in messages["b.mp4"])
