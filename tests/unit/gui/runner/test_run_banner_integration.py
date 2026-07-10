"""Integration: the real RunnerPanel -> RunnerContainer banner chain.

Wires the same connections main_window makes (panel.elapsed_updated ->
container.set_banner_time) and exercises the core behavior of Task 9: the
header timer hides during a live run while the banner owns the visible timer,
elapsed updates reach the banner, and the header reappears once idle.
"""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.usefixtures("qtbot")


def _wire(qtbot, mock_core):
    from PyQt6.QtWidgets import QWidget

    from glider.gui.panels.runner_panel import RunnerPanel
    from glider.gui.runner.runner_container import RunnerContainer

    # Leaving this a MagicMock would make _snap_timer_to_flow_duration paint a
    # mock into the timer on IDLE; None takes its documented early-return path.
    mock_core.last_flow_duration_s = None
    panel = RunnerPanel(mock_core, MagicMock())
    container = RunnerContainer(mock_core, panel, QWidget())
    panel.elapsed_updated.connect(container.set_banner_time)
    qtbot.addWidget(container)
    container.show()
    return panel, container


def test_header_timer_hidden_during_run_and_reshown_idle(qtbot, mock_core):
    panel, container = _wire(qtbot, mock_core)

    panel.update_state("RUNNING")
    assert panel._runner_timer.isVisibleTo(panel) is False

    # A live tick flows panel -> banner via elapsed_updated.
    panel._set_timer_display(5.0)
    assert container._banner._time.text() == "00:05.00"

    panel.update_state("IDLE")
    assert panel._runner_timer.isVisibleTo(panel) is True
