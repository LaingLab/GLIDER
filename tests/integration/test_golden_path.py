"""Golden-path integration coverage for a complete experiment lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from glider.core.experiment_session import SessionState
from glider.core.glider_core import GliderCore


@pytest.mark.asyncio
async def test_save_reopen_run_and_stop_produces_clean_artifacts(tmp_path: Path):
    """A configured experiment survives reopening and completes a safe run."""
    experiment_path = tmp_path / "golden-path.glider"
    recordings_dir = tmp_path / "recordings"

    authoring_core = GliderCore()
    await authoring_core.initialize()
    try:
        authoring_core.session.name = "Golden Path"
        authoring_core.flow_engine.create_node(
            node_id="start",
            node_type="StartExperiment",
        )
        authoring_core.flow_engine.create_node(
            node_id="delay",
            node_type="Delay",
            state={"duration_seconds": 30.0, "use_input": False},
        )
        authoring_core.flow_engine.create_connection(
            connection_id="start-to-delay",
            from_node_id="start",
            from_output=0,
            to_node_id="delay",
            to_input=0,
            connection_type="exec",
        )

        await authoring_core.save_experiment(experiment_path)
    finally:
        await authoring_core.shutdown()

    assert experiment_path.exists()

    run_core = GliderCore()
    await run_core.initialize()
    run_core.set_recording_directory(recordings_dir)
    run_core.video_recording_enabled = False
    run_core.cv_processing_enabled = False
    try:
        await run_core.load_experiment(experiment_path)

        assert run_core.session.name == "Golden Path"
        assert set(run_core.flow_engine.nodes) == {"start", "delay"}
        assert len(run_core.flow_engine.get_connections()) == 1

        await run_core.start_experiment()
        assert run_core.state is SessionState.RUNNING

        await run_core.pause_experiment()
        assert run_core.state is SessionState.PAUSED

        await run_core.resume_experiment()
        assert run_core.state is SessionState.RUNNING

        await run_core.stop_experiment()
        assert run_core.state is SessionState.READY
        assert not run_core.data_recorder.is_recording
        assert not run_core.event_logger.is_recording
    finally:
        await run_core.shutdown()

    csv_artifacts = list(recordings_dir.glob("*.csv"))
    assert len(csv_artifacts) >= 2
    assert any("_events" in path.stem for path in csv_artifacts)
    assert all(path.read_text(encoding="utf-8").strip() for path in csv_artifacts)
    assert not list(recordings_dir.glob("*.partial"))
