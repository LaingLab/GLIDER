"""Pure computation of Runner readiness gates (board + experiment).

Used by the Dashboard readiness strip and START enablement. UI-free and
independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from glider.core.presets import missing_required

_START_EXPERIMENT = "StartExperiment"


@dataclass(frozen=True)
class Readiness:
    board_ready: bool
    experiment_ready: bool
    board_label: str
    experiment_label: str
    missing_field: str | None = None
    #: Why live identity keeps Start disabled (folded into experiment_ready).
    identity_message: str | None = None

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


def _identity_message(core) -> str | None:
    """Why live identity is not ready to run, or None."""
    cv = getattr(core, "cv_processor", None)
    settings = getattr(cv, "settings", None)
    mode = getattr(settings, "identity_mode", "off")
    if mode not in ("appearance", "continuity"):
        return None
    if getattr(settings, "enabled", False) is not True:
        # Naming happens on the tracked camera view, so it needs CV running.
        return "Turn on Computer Vision to name the animals" if mode == "continuity" else None
    reason = cv.identity_unavailable_reason
    if reason:
        return reason
    if mode == "continuity" and not cv.identity_ready:
        return "Name the animals on the camera view"
    return None


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

    identity_message = _identity_message(core)
    experiment_ready = _has_runnable_flow(session) and identity_message is None
    experiment_label = ""
    if session is not None and getattr(session, "metadata", None) is not None:
        experiment_label = session.metadata.name or ""

    missing = None
    md = getattr(session, "metadata", None) if session is not None else None
    fields = getattr(md, "run_fields", None)
    if isinstance(fields, list) and fields:  # tolerate mock sessions
        missing = missing_required(fields, md.run_values)

    return Readiness(
        board_ready=board_ready,
        experiment_ready=experiment_ready,
        board_label=board_label,
        experiment_label=experiment_label,
        missing_field=missing,
        identity_message=identity_message,
    )
