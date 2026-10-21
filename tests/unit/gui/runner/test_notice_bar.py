import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

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


def _bar_at(qtbot, width=480):
    """A bar in a column ``width`` wide, the way the window holds it."""
    host = QWidget()
    qtbot.addWidget(host)
    bar = NoticeBar()
    QVBoxLayout(host).addWidget(bar)
    host.resize(width, 200)
    host.show()
    return host, bar


def _lines(bar):
    return bar._label.text().split("\n")


def test_info_and_warning_are_one_elided_line_with_the_text_in_the_tooltip(qtbot):
    host, bar = _bar_at(qtbot)
    long = "the camera stopped. " * 20
    for level in ("info", "warning"):
        bar.show_notice("Heads up", long, level)
        qtbot.wait(10)
        assert len(_lines(bar)) == 1 and _lines(bar)[0].endswith("…")
        assert bar._label.sizeHint().height() < 2 * bar._label.fontMetrics().lineSpacing()
        assert bar._label.toolTip() == bar.text() == f"Heads up: {long}"


def test_critical_wraps_to_two_lines_at_most(qtbot):
    host, bar = _bar_at(qtbot)
    unbreakable = "/dev/" + "x" * 120
    for message in ("Hardware may still be active. " * 10, unbreakable):
        bar.show_notice("UNSAFE STATE", message, "critical")
        qtbot.wait(10)
        metrics = bar._label.fontMetrics()
        assert len(_lines(bar)) == 2 and _lines(bar)[1].endswith("…")
        assert all(metrics.horizontalAdvance(line) <= bar._label.width() for line in _lines(bar))
        assert bar._label.sizeHint().height() < 3 * metrics.lineSpacing()
        assert bar.text() == f"UNSAFE STATE: {message}"
    assert host.minimumSizeHint().width() <= 480 and bar.width() <= 480


def test_a_short_critical_notice_stays_on_one_line(qtbot):
    host, bar = _bar_at(qtbot)
    bar.show_notice("UNSAFE", "Check outputs.", "critical")
    qtbot.wait(10)
    assert _lines(bar) == ["⚠  UNSAFE: Check outputs."]


def test_each_level_has_its_glyph(qtbot):
    host, bar = _bar_at(qtbot)
    for level, glyph in (("info", "ⓘ"), ("warning", "⚠"), ("critical", "⚠")):
        bar.show_notice("T", "m", level)
        bar.dismiss()
        assert bar._label.text().startswith(glyph)


def test_the_close_button_dismisses(qtbot):
    host, bar = _bar_at(qtbot)
    bar.show_notice("UNSAFE", "c", "critical")
    assert bar._close.text() == "✕" and bar._close.height() >= 44
    bar._close.click()
    assert bar.isHidden()


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
