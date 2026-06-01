"""Async coordinator that runs a graph StartFunction chain on demand.

Used by the Runner-mode manual-control page. Keeps run semantics (lazy graph
load, one-run-at-a-time, error mapping) out of the Qt widget so they are
unit-testable without an event loop wired to widgets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RunOutcome(Enum):
    SUCCESS = "success"
    NOT_FOUND = "function-not-found"
    NO_HARDWARE = "hardware-not-ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass(frozen=True)
class RunResult:
    outcome: RunOutcome
    error: str | None = None


def _default_function_runner_factory(start_node_id: str, flow_engine: Any):
    from glider.nodes.flow_function_nodes import FlowFunctionRunner

    return FlowFunctionRunner(start_node_id, flow_engine)


class ManualControlRunner:
    """Runs one manual graph function at a time against the live flow engine."""

    def __init__(
        self,
        core: Any,
        function_runner_factory: Callable[[str, Any], Any] = _default_function_runner_factory,
    ):
        self._core = core
        self._make_runner = function_runner_factory
        self._busy = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    async def run(self, start_node_id: str) -> RunResult:
        if self._busy:
            return RunResult(RunOutcome.BUSY)
        if not self._core.hardware_manager.is_any_board_connected():
            return RunResult(RunOutcome.NO_HARDWARE)

        self._busy = True
        try:
            engine = self._core.flow_engine
            if not engine.nodes:
                self._core.setup_flow()

            node = engine.get_node(start_node_id)
            if node is None:
                return RunResult(RunOutcome.NOT_FOUND)

            runner = self._make_runner(start_node_id, engine)
            await runner.execute()
            return RunResult(RunOutcome.SUCCESS)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            logger.exception("Manual control run failed for %s", start_node_id)
            return RunResult(RunOutcome.ERROR, str(exc))
        finally:
            self._busy = False
