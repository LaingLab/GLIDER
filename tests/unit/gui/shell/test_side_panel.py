"""What a side panel guarantees, collapsed and expanded.

**Every test here runs against a shown panel.** That is not stylistic: Qt
installs focus, default-button behaviour and final geometry on the *show*
event, and the width assertions below are meaningless without it. On an earlier
branch 36 dialog tests passed against never-shown widgets, two of them while
asserting the opposite of reality.

The assertions are about *what a user can reach* -- which tabs exist, which one
is current, whether the rail is clickable while collapsed, and whether the
width they dragged comes back -- not about pixels of padding.
"""

from __future__ import annotations

import pytest
from PyQt6 import sip
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QWidget

from glider.gui.shell.side_panel import DEFAULT_WIDTH, RAIL_WIDTH, SidePanel

#: (key, label, icon) for the three tabs the Builder's left side will carry.
TABS = [("nodes", "Nodes", "N"), ("hardware", "Hardware", "H"), ("files", "Files", "F")]


def _shown(qtbot, width: int = DEFAULT_WIDTH, side: str = "left") -> SidePanel:
    """A panel with three tabs, sized, shown and exposed."""
    panel = SidePanel(side=side)
    qtbot.addWidget(panel)
    for key, label, icon in TABS:
        content = QLabel(label)
        content.setObjectName(f"content-{key}")
        panel.add_tab(key, label, content, icon)
    panel.resize(width, 420)
    with qtbot.waitExposed(panel):
        panel.show()
    qtbot.waitUntil(lambda: panel.width() == width)
    return panel


@pytest.fixture
def panel(qtbot) -> SidePanel:
    return _shown(qtbot)


# ----------------------------------------------------------------- tabs


def test_tabs_come_back_in_the_order_they_were_added(panel):
    assert panel.keys() == ["nodes", "hardware", "files"]
    assert panel.current_key() == "nodes"


def test_selecting_a_tab_announces_it_exactly_once(panel):
    seen: list[str] = []
    panel.current_changed.connect(seen.append)

    panel.set_current("files")

    assert panel.current_key() == "files"
    assert seen == ["files"]

    panel.set_current("files")  # already there
    assert seen == ["files"]


def test_an_unknown_tab_key_is_refused_rather_than_ignored(panel):
    """A caller restoring a stale saved tab needs to hear about it, so the
    fallback can happen where the default is known."""
    with pytest.raises(KeyError):
        panel.set_current("analysis")


def test_the_current_tab_is_the_visible_one(panel, qtbot):
    panel.set_current("hardware")
    qtbot.waitUntil(panel.widget_for("hardware").isVisible)

    assert panel.widget_for("nodes").isVisible() is False


# ------------------------------------------------------------ collapsing


def test_toggle_collapses_to_the_rail_and_announces_it(panel, qtbot):
    seen: list[bool] = []
    panel.expanded_changed.connect(seen.append)

    panel.toggle()

    assert panel.expanded is False
    assert seen == [False]
    qtbot.waitUntil(lambda: panel.width() == RAIL_WIDTH)


def test_the_collapsed_panel_shows_one_rail_button_per_tab(panel):
    """Collapsed is a rail, not nothing. Every GLIDER user is a first-time
    user for the foreseeable future; the areas stay discoverable."""
    panel.toggle()

    buttons = panel.rail_buttons()
    assert len(buttons) == len(TABS)
    assert [b.toolTip() for b in buttons] == ["Nodes", "Hardware", "Files"]
    assert all(b.isVisible() for b in buttons)


def test_the_rail_is_out_of_the_way_while_expanded(panel):
    assert panel.expanded is True
    assert [b.isVisible() for b in panel.rail_buttons()] == [False, False, False]


def test_clicking_a_rail_button_expands_to_that_tab(panel, qtbot):
    panel.toggle()
    seen: list[str] = []
    panel.current_changed.connect(seen.append)

    qtbot.mouseClick(panel.rail_buttons()[2], Qt.MouseButton.LeftButton)

    assert panel.expanded is True
    assert panel.current_key() == "files"
    assert seen == ["files"]
    qtbot.waitUntil(panel.widget_for("files").isVisible)


def test_expanding_restores_the_width_the_user_chose(qtbot):
    """A panel dragged to 320 px and collapsed comes back at 320 px."""
    panel = _shown(qtbot, width=320)
    assert DEFAULT_WIDTH != 320, "the width under test must not be the default"

    panel.toggle()
    qtbot.waitUntil(lambda: panel.width() == RAIL_WIDTH)

    panel.toggle()
    qtbot.waitUntil(lambda: panel.width() == 320)
    assert panel.expanded_width() == 320


def test_expanding_an_already_expanded_panel_announces_nothing(panel):
    seen: list[bool] = []
    panel.expanded_changed.connect(seen.append)

    panel.set_expanded(True)

    assert seen == []
    assert panel.expanded is True


def test_a_tab_widget_survives_being_collapsed(panel, qtbot):
    """The right panel will host a live camera feed. Collapsing must park it,
    not destroy it."""
    hosted = panel.widget_for("hardware")
    panel.set_current("hardware")

    panel.set_expanded(False)
    panel.set_expanded(True)

    assert not sip.isdeleted(hosted)
    assert panel.widget_for("hardware") is hosted
    assert hosted.parent() is not None
    qtbot.waitUntil(hosted.isVisible)


# -------------------------------------------------------------- styling


def test_the_collapsed_state_is_published_for_the_stylesheet(panel):
    """desktop.qss selects on this property; Python sets no colour."""
    assert panel.property("state") == "expanded"

    panel.toggle()

    assert panel.property("state") == "collapsed"


def test_nothing_in_the_panel_sets_its_own_stylesheet(panel):
    """A widget-level stylesheet out-prioritises the application stylesheet and
    silently blocks all future theming. That trap has cost this project once."""
    offenders = [
        w.objectName() or type(w).__name__ for w in panel.findChildren(QWidget) if w.styleSheet()
    ]

    assert panel.styleSheet() == ""
    assert offenders == []
