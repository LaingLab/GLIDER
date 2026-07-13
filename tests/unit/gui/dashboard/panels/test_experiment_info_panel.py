import pytest

from glider.gui.dashboard.panels.experiment_info_panel import ExperimentInfoPanel

pytestmark = pytest.mark.usefixtures("qtbot")


@pytest.fixture
def panel(qtbot, mock_core):
    from PyQt6.QtWidgets import QWidget

    p = ExperimentInfoPanel(mock_core, hardware_widget=QWidget())
    qtbot.addWidget(p)
    return p


def test_reexposes_open_requested(panel, qtbot):
    with qtbot.waitSignal(panel.open_requested, timeout=1000):
        panel._page.open_requested.emit()


def test_reexposes_save_requested(panel, qtbot):
    with qtbot.waitSignal(panel.save_requested, timeout=1000):
        panel._page.save_requested.emit()


def test_forwards_update_state(panel):
    # RunnerSetupPage has no update_state; the wrapper must no-op safely, not raise.
    panel.update_state("RUNNING")


def test_forwards_refresh(panel):
    panel.refresh()  # delegates to the inner page's refresh(); must not raise
