from types import SimpleNamespace

from glider.core.experiment_session import FlowConfig, NodeConfig
from glider.gui.runner.readiness import compute_readiness


def _core(*, board_connected, node_types, board_desc="", exp_name=""):
    flow = FlowConfig(nodes=[NodeConfig(id=str(i), node_type=t) for i, t in enumerate(node_types)])
    session = SimpleNamespace(flow=flow, metadata=SimpleNamespace(name=exp_name))
    hw = SimpleNamespace(
        is_any_board_connected=lambda: board_connected,
        connected_board_description=lambda: board_desc,
    )
    return SimpleNamespace(session=session, hardware_manager=hw)


def test_nothing_ready():
    r = compute_readiness(_core(board_connected=False, node_types=[]))
    assert r.board_ready is False
    assert r.experiment_ready is False
    assert r.all_ready is False


def test_board_only():
    r = compute_readiness(_core(board_connected=True, node_types=["DigitalWrite"]))
    assert r.board_ready is True
    assert r.experiment_ready is False


def test_experiment_only():
    r = compute_readiness(_core(board_connected=False, node_types=["StartExperiment"]))
    assert r.board_ready is False
    assert r.experiment_ready is True


def test_all_ready():
    r = compute_readiness(
        _core(board_connected=True, node_types=["StartExperiment", "Delay"], exp_name="My Exp")
    )
    assert r.all_ready is True
    assert r.experiment_label == "My Exp"


def test_no_session_is_not_ready():
    core = SimpleNamespace(
        session=None,
        hardware_manager=SimpleNamespace(
            is_any_board_connected=lambda: True, connected_board_description=lambda: "Uno"
        ),
    )
    r = compute_readiness(core)
    assert r.experiment_ready is False
