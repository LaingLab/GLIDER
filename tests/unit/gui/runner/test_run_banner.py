import pytest

pytestmark = pytest.mark.usefixtures("qtbot")


def test_set_time_updates_label(qtbot):
    from glider.gui.runner.run_banner import RunBanner

    b = RunBanner()
    qtbot.addWidget(b)
    b.set_time("01:23.45")
    assert b._time.text() == "01:23.45"


def test_stop_button_emits(qtbot):
    from glider.gui.runner.run_banner import RunBanner

    b = RunBanner()
    qtbot.addWidget(b)
    with qtbot.waitSignal(b.stop_requested):
        b._stop.click()


def test_set_state_updates_pill(qtbot):
    from glider.gui.runner.run_banner import RunBanner

    b = RunBanner()
    qtbot.addWidget(b)
    b.set_state("RUNNING", recording=False)
    assert b._state.text() == "RUNNING"


def test_recording_toggles_rec_visibility(qtbot):
    from glider.gui.runner.run_banner import RunBanner

    b = RunBanner()
    qtbot.addWidget(b)
    b.show()
    b.set_state("RUNNING", recording=True)
    assert b._rec.isVisibleTo(b)
    b.set_state("RUNNING", recording=False)
    assert not b._rec.isVisibleTo(b)
