import pytest
from PyQt6.QtCore import Qt

from glider.gui.runner.notice_bar import NoticeBar

pytestmark = pytest.mark.usefixtures("qtbot")


def test_hidden_until_a_notice_arrives(qtbot):
    bar = NoticeBar()
    qtbot.addWidget(bar)
    assert bar.objectName() == "piNoticeBar"
    assert bar.isHidden()
    bar.show_notice("Saved", "box_pi_side.glider", "info")
    assert not bar.isHidden()
    assert bar.text() == "Saved: box_pi_side.glider"
    assert bar.property("level") == "info"


def test_the_message_wraps(qtbot):
    bar = NoticeBar()
    qtbot.addWidget(bar)
    bar.show_notice("Saved", "x", "info")
    assert bar._label.wordWrap()


def test_critical_is_not_replaced_by_info(qtbot):
    bar = NoticeBar()
    qtbot.addWidget(bar)
    bar.show_notice("UNSAFE STATE — Hardware may still be active", "Check outputs.", "critical")
    bar.show_notice("Saved", "box_pi_side.glider", "info")
    assert bar.text().startswith("UNSAFE STATE")
    assert bar.property("level") == "critical"
    bar.dismiss()
    bar.show_notice("Saved", "x", "info")
    assert bar.text() == "Saved: x" and not bar.isHidden()


def test_only_a_critical_notice_is_sticky(qtbot):
    bar = NoticeBar()
    qtbot.addWidget(bar)
    bar.show_notice("Saved", "a", "info")
    bar.show_notice("Heads up", "b", "warning")
    assert bar.text() == "Heads up: b" and bar.property("level") == "warning"
    bar.show_notice("UNSAFE", "c", "critical")
    bar.show_notice("Heads up", "d", "warning")
    assert bar.text() == "UNSAFE: c"
    bar.show_notice("UNSAFE", "e", "critical")  # the newer critical wins
    assert bar.text() == "UNSAFE: e"


def test_a_left_tap_dismisses_and_a_right_tap_does_not(qtbot):
    bar = NoticeBar()
    qtbot.addWidget(bar)
    bar.show_notice("Saved", "x", "info")
    qtbot.mouseClick(bar, Qt.MouseButton.RightButton)
    assert not bar.isHidden()
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton)
    assert bar.isHidden()
