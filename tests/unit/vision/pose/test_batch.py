"""Batch pose inference: discovery, output naming, and run_batch.

Inference is injected via ``run_batch(infer=...)``, so nothing here needs
torch, a GPU, or a real video file.
"""

from __future__ import annotations

import asyncio
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
