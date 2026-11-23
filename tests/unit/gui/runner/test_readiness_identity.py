"""Start waits for live identity: named animals, and a backend that can name them."""

from types import SimpleNamespace

import pytest

from glider.core.experiment_session import FlowConfig, NodeConfig
from glider.gui.runner.readiness import compute_readiness


def _core(mode="continuity", *, ready=False, reason=None, enabled=True):
    flow = FlowConfig(nodes=[NodeConfig(id="0", node_type="StartExperiment")])
    cv = SimpleNamespace(
        settings=SimpleNamespace(enabled=enabled, identity_mode=mode),
        identity_ready=ready,
        identity_unavailable_reason=reason,
    )
    return SimpleNamespace(
        session=SimpleNamespace(flow=flow, metadata=SimpleNamespace(name="")),
        hardware_manager=SimpleNamespace(
            is_any_board_connected=lambda: True, connected_board_description=lambda: "Uno"
        ),
        cv_processor=cv,
    )


def test_unnamed_animals_block_start():
    r = compute_readiness(_core())
    assert r.experiment_ready is False
    assert r.identity_message == "Name the animals on the camera view"


def test_named_animals_let_start_go():
    r = compute_readiness(_core(ready=True))
    assert r.experiment_ready is True
    assert r.identity_message is None


@pytest.mark.parametrize("mode", ["continuity", "appearance"])
def test_an_unavailable_backend_says_why(mode):
    reason = "Identity needs a YOLO model; the active backend is POSE_MODEL."
    r = compute_readiness(_core(mode, reason=reason))
    assert r.experiment_ready is False
    assert r.identity_message == reason


def test_appearance_mode_never_blocks_start():
    r = compute_readiness(_core("appearance", ready=False))
    assert r.experiment_ready is True
    assert r.identity_message is None


@pytest.mark.parametrize("core", [_core("off"), _core(enabled=False)])
def test_identity_off_or_cv_off_changes_nothing(core):
    r = compute_readiness(core)
    assert r.experiment_ready is True
    assert r.identity_message is None
