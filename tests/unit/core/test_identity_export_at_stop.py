"""GliderCore hands the live identity export the video's own timing, off the loop."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from glider.core.glider_core import GliderCore


@pytest.mark.asyncio
async def test_export_gets_recorder_timing_and_runs_off_the_event_loop(tmp_path, monkeypatch):
    import glider.vision.identity_export as ie

    calls = []
    main = threading.get_ident()
    monkeypatch.setattr(
        ie,
        "export_live_identity",
        lambda *a: calls.append((a, threading.get_ident() != main)) or [],
    )
    core = GliderCore()
    core._cv_processor = SimpleNamespace(
        identity_active=True,
        settings=SimpleNamespace(
            model_path="m.pt", animals=[{"name": "A"}], keypoint_names=["nose"]
        ),
    )
    core._tracking_logger = SimpleNamespace(keypoints_file_path=tmp_path / "k.csv")
    core._video_recorder = SimpleNamespace(frame_timestamps=[12.5, 12.6], fps=29.5)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    await core._export_identity(tmp_path / "t.csv", video)
    ((args, off_loop),) = calls
    assert args[6:] == (29.5, [12.5, 12.6]) and off_loop
    assert args[3] == Path("m.pt")


@pytest.mark.asyncio
async def test_export_skipped_without_a_first_frame(tmp_path, monkeypatch):
    import glider.vision.identity_export as ie

    called = MagicMock(return_value=[])
    monkeypatch.setattr(ie, "export_live_identity", called)
    core = GliderCore()
    core._cv_processor = SimpleNamespace(identity_active=True, settings=MagicMock())
    core._tracking_logger = SimpleNamespace(keypoints_file_path=tmp_path / "k.csv")
    core._video_recorder = SimpleNamespace(frame_timestamps=[], fps=30.0)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    await core._export_identity(tmp_path / "t.csv", video)
    called.assert_not_called()
