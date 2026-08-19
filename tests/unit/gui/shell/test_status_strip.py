"""What the one piece of chrome that cannot be summoned has to guarantee.

Every other surface in the Builder shell is on demand. This one is not, because
a board dropping 40 minutes into an unattended overnight run has to be visible
without anyone asking for it. The case these tests are written against is
mockup state 3: a device down *while recording*. If that reads as ambiguous or
easy to miss, the widget has failed however clean it looks.

**Every test here runs against a shown strip**, mirroring
``test_side_panel.py``. Qt settles geometry, visibility and style resolution on
the *show* event; a height assertion against an unshown widget measures a
sizeHint, not the thing that ships. On an earlier branch 36 dialog tests passed
against never-shown widgets, two while asserting the opposite of reality.

Three tests go further and mount the strip under the **real** ``desktop.qss``,
then read the colour Qt actually resolved. That is the only way to tell the
three failure modes apart: property set but never re-polished, re-polished but
no matching rule in the stylesheet, and a rule that fires. A widget whose
``state`` property is correct and whose colour never changed is exactly the bug
this file exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QVBoxLayout, QWidget

import glider.gui.styles
from glider.gui.shell.status_strip import STRIP_HEIGHT, StatusStrip
from glider.gui.styles.colors import STATE_ERR

#: The mockup's rig: two healthy boards and a camera that wants a look at.
DEVICES = [("arduino", "ok"), ("harp1", "ok"), ("camera", "warn")]

#: Deliberately taller than the strip: the height assertions must prove the
#: strip refuses the space, not that nobody offered it any.
_HOST_HEIGHT = 200


def _text_colour(widget: QWidget) -> str:
    """The colour Qt resolved for *widget*'s text, after the stylesheet ran."""
    return widget.palette().color(QPalette.ColorRole.WindowText).name()


def _shown(qtbot, *, themed: bool = False, width: int = 900) -> tuple[QWidget, StatusStrip]:
    """A populated strip inside a host, shown and exposed.

    The host exists so ``themed`` can carry the application stylesheet without
    the strip itself owning one -- which is the very thing
    :func:`test_nothing_in_the_strip_sets_its_own_stylesheet` forbids, and is
    also how ``MainWindow`` applies the theme in the running app.

    Both are returned because ``qtbot.addWidget`` keeps only a weak reference:
    drop the host and Qt destroys it and, with it, the strip under test.
    """
    host = QWidget()
    qtbot.addWidget(host)
    if themed:
        qss = Path(glider.gui.styles.__file__).with_name("desktop.qss")
        host.setStyleSheet(qss.read_text(encoding="utf-8"))

    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    strip = StatusStrip(parent=host)
    layout.addWidget(strip)
    layout.addStretch(1)

    strip.set_experiment("open-field-cohort-3", dirty=True)
    strip.set_run_state("idle")
    strip.set_devices(DEVICES)

    host.resize(width, _HOST_HEIGHT)
    with qtbot.waitExposed(host):
        host.show()
    qtbot.waitUntil(strip.isVisible)
    return host, strip


@pytest.fixture
def strip(qtbot):
    # Both fixtures hold the host for the whole test on purpose; see _shown.
    host, strip = _shown(qtbot)
    yield strip
    del host


@pytest.fixture
def themed_strip(qtbot):
    host, strip = _shown(qtbot, themed=True)
    yield strip
    del host


# --------------------------------------------------------------- the experiment


def test_the_experiment_name_and_its_dirty_marker_both_render(strip):
    assert strip.name_label().text() == "open-field-cohort-3"
    assert strip.name_label().isVisible()
    assert strip.dirty_label().isVisible()
    assert strip.dirty_label().text().strip() != ""


def test_the_dirty_marker_is_kept_out_of_the_name(strip):
    """Two labels, not one string: the stylesheet greys the marker without
    greying the name, and a caller reading the name back gets the name."""
    assert strip.dirty_label().text() not in strip.name_label().text()


def test_saving_takes_the_dirty_marker_away_and_editing_brings_it_back(strip, qtbot):
    strip.set_experiment("open-field-cohort-3", dirty=False)
    qtbot.waitUntil(lambda: not strip.dirty_label().isVisible())
    assert strip.name_label().text() == "open-field-cohort-3"

    strip.set_experiment("open-field-cohort-3", dirty=True)
    qtbot.waitUntil(strip.dirty_label().isVisible)


def test_an_unnamed_experiment_still_leaves_a_readable_strip(strip):
    """A brand new session has no file yet. The strip must not go blank there."""
    strip.set_experiment("", dirty=False)

    assert strip.name_label().text().strip() != ""
    assert strip.height() == STRIP_HEIGHT


# ------------------------------------------------------------------- run state


@pytest.mark.parametrize("state", ["idle", "running", "recording", "error"])
def test_every_run_state_tags_the_pill_and_names_itself(strip, state):
    strip.set_run_state(state)

    assert strip.pill().property("state") == state
    assert strip.pill().text().strip() != ""


def test_the_four_run_states_do_not_share_a_label(strip):
    """A pill that says the same thing in two states is decoration."""
    labels = []
    for state in ("idle", "running", "recording", "error"):
        strip.set_run_state(state)
        labels.append(strip.pill().text())

    assert len(set(labels)) == 4


def test_an_unknown_run_state_is_refused_rather_than_shown_wrong(strip):
    """Run state comes from our own code, not from hardware. A pill quietly
    stuck on 'Idle' through a recording is the failure this widget exists to
    prevent, so a typo is loud."""
    with pytest.raises(ValueError):
        strip.set_run_state("paused")

    assert strip.pill().property("state") == "idle"


def test_changing_the_run_state_repolishes_so_the_new_colour_lands(themed_strip):
    """Qt resolves ``[state="..."]`` at polish time, not when the property is
    set. Without the re-polish the pill keeps the colour it opened with, and
    the property assertions above would all still pass."""
    idle = _text_colour(themed_strip.pill())

    themed_strip.set_run_state("recording")

    assert _text_colour(themed_strip.pill()) != idle
    assert _text_colour(themed_strip.pill()) == STATE_ERR


# --------------------------------------------------------------------- devices


def test_one_dot_per_device_in_the_order_given(strip):
    assert strip.device_names() == ["arduino", "harp1", "camera"]
    assert len(strip.device_dots()) == len(DEVICES)
    assert all(dot.isVisible() for dot in strip.device_dots())


def test_each_dot_carries_its_state_for_the_stylesheet(strip):
    assert [dot.property("state") for dot in strip.device_dots()] == ["ok", "ok", "warn"]


def test_a_device_says_its_name_and_its_state_on_hover(strip):
    """The dot is 8 px. Anyone who has to act on a red one needs the state
    spelled out, not inferred from a colour."""
    tips = [chip.toolTip() for chip in strip.device_chips()]

    assert "harp1" in tips[1]
    assert "warn" in tips[2].lower()


def test_changing_one_device_touches_only_that_device(strip):
    """The row is updated in place rather than rebuilt, so a strip refreshed on
    a timer does not churn every widget on it -- and so 'only that dot changed'
    is a claim that can be checked at all."""
    before = strip.device_dots()

    strip.set_devices([("arduino", "ok"), ("harp1", "error"), ("camera", "warn")])

    assert strip.device_dots() == before, "same names must reuse the same widgets"
    assert [dot.property("state") for dot in strip.device_dots()] == ["ok", "error", "warn"]


def test_a_device_going_down_repolishes_its_dot(themed_strip):
    healthy = _text_colour(themed_strip.device_dots()[1])

    themed_strip.set_devices([("arduino", "ok"), ("harp1", "error"), ("camera", "warn")])

    assert _text_colour(themed_strip.device_dots()[1]) != healthy
    assert _text_colour(themed_strip.device_dots()[0]) == healthy


def test_a_different_rig_replaces_the_row(strip):
    strip.set_devices([("pi-cam", "ok")])

    assert strip.device_names() == ["pi-cam"]
    assert len(strip.device_dots()) == 1


def test_a_rig_with_no_devices_renders_rather_than_raising(strip):
    """The Builder opens before anything is connected. An empty list is the
    normal first thing this widget is told."""
    strip.set_devices([])

    assert strip.device_names() == []
    assert strip.device_dots() == []
    assert strip.height() == STRIP_HEIGHT
    assert strip.isVisible()


def test_an_unfamiliar_device_state_never_renders_as_healthy(strip):
    """Device states arrive from drivers, so the vocabulary can widen without
    this file changing. What must never happen is a state nobody recognises
    being painted the same green as a working board."""
    strip.set_devices([("harp1", "reconnecting")])

    assert strip.device_dots()[0].property("state") == "unknown"
    assert "reconnecting" in strip.device_chips()[0].toolTip()


# ------------------------------------------------------ mockup state 3, in full


def test_a_device_down_mid_recording_stands_out_from_its_neighbours(themed_strip):
    """The case the whole widget is judged on.

    Being a different colour than it was a moment ago is not enough -- nobody
    was watching a moment ago. The down device has to differ from the healthy
    devices *beside it*, in the same glance, while the pill is red for a
    reason of its own.
    """
    themed_strip.set_run_state("recording")
    themed_strip.set_devices([("arduino", "ok"), ("harp1", "error"), ("camera", "ok")])

    down, healthy = themed_strip.device_chips()[1], themed_strip.device_chips()[0]

    assert themed_strip.pill().property("state") == "recording"
    assert down.property("state") == "error"
    assert _text_colour(down) != _text_colour(healthy), "the name must carry the alarm too"
    assert _text_colour(themed_strip.device_dots()[1]) != _text_colour(
        themed_strip.device_dots()[0]
    )


# --------------------------------------------------------------------- signals


def test_the_left_toggle_announces_itself_and_nothing_else(strip, qtbot):
    left: list[object] = []
    right: list[object] = []
    strip.left_toggled.connect(lambda: left.append(1))
    strip.right_toggled.connect(lambda: right.append(1))

    qtbot.mouseClick(strip.left_toggle(), Qt.MouseButton.LeftButton)

    assert len(left) == 1
    assert right == []


def test_the_right_toggle_announces_itself_and_nothing_else(strip, qtbot):
    left: list[object] = []
    right: list[object] = []
    strip.left_toggled.connect(lambda: left.append(1))
    strip.right_toggled.connect(lambda: right.append(1))

    qtbot.mouseClick(strip.right_toggle(), Qt.MouseButton.LeftButton)

    assert len(right) == 1
    assert left == []


def test_the_toggles_show_which_side_is_open(strip):
    """Mockup state 3 has the left panel open and the right one collapsed. The
    strip is told; it does not guess."""
    strip.set_left_expanded(True)
    strip.set_right_expanded(False)

    assert strip.left_toggle().isChecked() is True
    assert strip.right_toggle().isChecked() is False


def test_being_told_the_panel_state_does_not_look_like_a_click(strip):
    """The owner drives both directions. If echoing the panel's state re-emitted
    the toggle, the two would ping-pong."""
    seen: list[object] = []
    strip.left_toggled.connect(lambda: seen.append(1))

    strip.set_left_expanded(False)
    strip.set_left_expanded(True)

    assert seen == []


def test_the_palette_hint_is_a_button_and_not_a_decoration(strip, qtbot):
    """A hint nobody can press teaches the shortcut and then refuses to do the
    thing. Every user of GLIDER is a first-time user for now."""
    seen: list[object] = []
    strip.palette_requested.connect(lambda: seen.append(1))

    qtbot.mouseClick(strip.palette_hint(), Qt.MouseButton.LeftButton)

    assert len(seen) == 1


def test_the_palette_hint_names_the_shortcut(strip):
    text = strip.palette_hint().text()

    assert "Ctrl" in text
    assert "K" in text


# -------------------------------------------------------- geometry and styling


def test_the_strip_is_forty_pixels_tall_and_refuses_more(strip):
    """40 px is the budget the whole design is measured against: one strip in
    place of a menu bar and two toolbars."""
    assert STRIP_HEIGHT == 40
    assert strip.height() == STRIP_HEIGHT
    assert strip.maximumHeight() == STRIP_HEIGHT


def test_nothing_in_the_strip_sets_its_own_stylesheet(strip):
    """A widget-level stylesheet out-prioritises the application stylesheet and
    silently blocks all future theming. That trap has cost this project once."""
    offenders = [
        w.objectName() or type(w).__name__ for w in strip.findChildren(QWidget) if w.styleSheet()
    ]

    assert strip.styleSheet() == ""
    assert offenders == []
