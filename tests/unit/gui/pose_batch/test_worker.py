"""PoseBatchWorker: what it actually hands the Qt-free batch core.

The worker is the *only* ``run_batch`` caller in ``src/``, so anything it does
not forward is unreachable from the application no matter how well the core
implements it. These tests exist to keep that seam honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glider.gui.pose_batch.worker import PoseBatchWorker
from glider.vision.arena import ArenaCalibration
from glider.vision.arena_gate import ArenaGateSettings

TRAPEZOID = [(0.28, 0.1), (0.72, 0.1), (0.76, 0.9), (0.24, 0.9)]


class _Result:
    summary = "1 completed"


@pytest.fixture
def calls(monkeypatch):
    """Capture the kwargs ``run_batch`` is called with, without running it."""
    seen: list[dict] = []

    def fake_run_batch(videos, model_path, names, **kwargs):
        seen.append(kwargs)
        return _Result()

    monkeypatch.setattr("glider.vision.pose.batch.run_batch", fake_run_batch)
    return seen


def _worker(tmp_path, **kwargs) -> PoseBatchWorker:
    return PoseBatchWorker(
        [tmp_path / "a.mp4"], tmp_path / "exp-7.pt", ["nose", "tail_base"], **kwargs
    )


def test_arenas_reach_run_batch(tmp_path, calls):
    """Without this the gate cannot fire: run_batch skips a video with no arena."""
    arena = ArenaCalibration(corners=TRAPEZOID, frame_size=(640, 480))
    video = Path(tmp_path / "a.mp4").resolve()

    _worker(tmp_path, arenas={video: arena}).run()

    assert calls[0]["arenas"] == {video: arena}


def test_the_gate_settings_reach_run_batch(tmp_path, calls):
    """Passed through, not reconstructed: a hard-coded ``ArenaGateSettings()``
    at the call site would silently ignore whatever the caller chose."""
    settings = ArenaGateSettings(margin_cm=3.0, min_inside_fraction=0.75)

    _worker(tmp_path, gate=settings).run()

    assert calls[0]["gate"] == settings
    assert calls[0]["gate"] != ArenaGateSettings()


def test_no_arenas_leaves_the_core_ungated(tmp_path, calls):
    """A worker built without arenas must not invent one."""
    _worker(tmp_path).run()

    assert not calls[0]["arenas"]
