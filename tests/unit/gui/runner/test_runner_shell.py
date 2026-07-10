import pytest
from PyQt6.QtWidgets import QWidget

pytestmark = pytest.mark.usefixtures("qtbot")


def _shell(qtbot, mock_core):
    from glider.gui.runner.runner_shell import RunnerShell

    s = RunnerShell(mock_core, QWidget(), QWidget(), QWidget(), QWidget())  # setup, run, manual, camera
    qtbot.addWidget(s)
    s.show()
    return s


def test_boots_on_setup(qtbot, mock_core):
    s = _shell(qtbot, mock_core)
    assert s._stack.currentIndex() == 0


def test_banner_hidden_when_idle(qtbot, mock_core):
    s = _shell(qtbot, mock_core)
    s.update_state("IDLE")
    assert not s._banner.isVisibleTo(s)


def test_banner_shown_off_run_tab_while_running(qtbot, mock_core):
    s = _shell(qtbot, mock_core)
    s.update_state("RUNNING")
    s.select_tab(0)
    assert s._banner.isVisibleTo(s)


def test_banner_hidden_on_run_tab_while_running(qtbot, mock_core):
    s = _shell(qtbot, mock_core)
    s.update_state("RUNNING")
    s.select_tab(1)
    assert not s._banner.isVisibleTo(s)


def test_banner_reappears_when_leaving_run_tab(qtbot, mock_core):
    s = _shell(qtbot, mock_core)
    s.update_state("RUNNING")
    s.select_tab(1)
    assert not s._banner.isVisibleTo(s)
    s.select_tab(2)
    assert s._banner.isVisibleTo(s)


def test_paused_treated_as_live(qtbot, mock_core):
    s = _shell(qtbot, mock_core)
    s.update_state("PAUSED")
    s.select_tab(3)
    assert s._banner.isVisibleTo(s)


def test_stop_reemits(qtbot, mock_core):
    s = _shell(qtbot, mock_core)
    with qtbot.waitSignal(s.stop_requested):
        s._banner.stop_requested.emit()


def test_set_banner_time_forwards(qtbot, mock_core):
    s = _shell(qtbot, mock_core)
    s.set_banner_time("00:07.00")
    assert s._banner._time.text() == "00:07.00"
