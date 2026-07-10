import pytest

pytestmark = pytest.mark.usefixtures("qtbot")


def test_banner_hidden_when_idle(qtbot, mock_core):
    from PyQt6.QtWidgets import QWidget

    from glider.gui.runner.runner_container import RunnerContainer

    c = RunnerContainer(mock_core, QWidget(), QWidget())
    qtbot.addWidget(c)
    assert c._banner.isVisibleTo(c) is False


def test_banner_shown_while_running(qtbot, mock_core):
    from PyQt6.QtWidgets import QWidget

    from glider.gui.runner.runner_container import RunnerContainer

    c = RunnerContainer(mock_core, QWidget(), QWidget())
    qtbot.addWidget(c)
    c.show()
    c.update_state("RUNNING")
    assert c._banner.isVisibleTo(c) is True
    c.update_state("IDLE")
    assert c._banner.isVisibleTo(c) is False


def test_banner_shown_while_paused(qtbot, mock_core):
    from PyQt6.QtWidgets import QWidget

    from glider.gui.runner.runner_container import RunnerContainer

    c = RunnerContainer(mock_core, QWidget(), QWidget())
    qtbot.addWidget(c)
    c.show()
    c.update_state("PAUSED")
    assert c._banner.isVisibleTo(c) is True
    assert c._banner._state.text() == "PAUSED"
    c.update_state("IDLE")
    assert c._banner.isVisibleTo(c) is False


def test_container_reemits_stop(qtbot, mock_core):
    from PyQt6.QtWidgets import QWidget

    from glider.gui.runner.runner_container import RunnerContainer

    c = RunnerContainer(mock_core, QWidget(), QWidget())
    qtbot.addWidget(c)
    with qtbot.waitSignal(c.stop_requested):
        c._banner.stop_requested.emit()


def test_set_banner_time_forwards(qtbot, mock_core):
    from PyQt6.QtWidgets import QWidget

    from glider.gui.runner.runner_container import RunnerContainer

    c = RunnerContainer(mock_core, QWidget(), QWidget())
    qtbot.addWidget(c)
    c.set_banner_time("00:05.00")
    assert c._banner._time.text() == "00:05.00"
