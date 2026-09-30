"""Pure computation of Runner readiness gates (board + experiment).

Used by the Dashboard readiness strip and START enablement. UI-free and
independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass

_START_EXPERIMENT = "StartExperiment"


@dataclass(frozen=True)
class Readiness:
    board_ready: bool
    experiment_ready: bool
    board_label: str
    experiment_label: str

    @property
    def all_ready(self) -> bool:
        return self.board_ready and self.experiment_ready


def _has_runnable_flow(session) -> bool:
    """A runnable flow is loaded iff the flow graph has a StartExperiment node.

    Mirrors FlowEngine.validate, which requires a StartExperiment node to run.
    """
    if session is None:
        return False
    return any(n.node_type == _START_EXPERIMENT for n in session.flow.nodes)


def _needs_no_board(session) -> bool:
    """True when the experiment configures no boards at all (e.g. video only).

    Such a session has nothing to connect, so waiting for a board would keep
    START disabled forever. GliderCore.start_experiment already handles it:
    connecting zero boards succeeds and the run proceeds.
    """
    hardware = getattr(session, "hardware", None)
    return hardware is not None and not hardware.boards


def compute_readiness(core) -> Readiness:
    hw = core.hardware_manager
    session = getattr(core, "session", None)
    board_ready = bool(hw.is_any_board_connected())
    board_label = ""
    if board_ready:
        desc = getattr(hw, "connected_board_description", lambda: "")()
        board_label = desc or "Board connected"
    elif _needs_no_board(session):
        board_ready = True
        board_label = "None needed"

    experiment_ready = _has_runnable_flow(session)
    experiment_label = ""
    if session is not None and getattr(session, "metadata", None) is not None:
        experiment_label = session.metadata.name or ""

    return Readiness(
        board_ready=board_ready,
        experiment_ready=experiment_ready,
        board_label=board_label,
        experiment_label=experiment_label,
    )
