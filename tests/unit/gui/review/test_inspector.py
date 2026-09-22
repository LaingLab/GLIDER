"""The inspector lays out what the window computes; it computes nothing."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QColor  # noqa: E402

from glider.gui.review.inspector import Inspector, MarkerRow  # noqa: E402
from glider.gui.styles import colors  # noqa: E402


@pytest.fixture
def inspector(qtbot):
    widget = Inspector()
    qtbot.addWidget(widget)
    return widget


def test_it_has_range_session_and_markers_tabs(inspector):
    assert [inspector.tabText(i) for i in range(inspector.count())] == [
        "Range",
        "Session",
        "Markers",
    ]


def test_the_range_card_shows_what_it_is_given(inspector):
    inspector.set_range("IN 00:02:00:00   OUT 00:02:45:00", "45.00 s · 1,350 frames")
    assert "OUT 00:02:45:00" in inspector.range_title.text()
    assert inspector.range_detail.text() == "45.00 s · 1,350 frames"


def test_kpis_and_clearing(inspector):
    inspector.set_kpis("183.4", "4.08", "27.9")
    assert (
        inspector.distance.text(),
        inspector.mean_speed.text(),
        inspector.peak_speed.text(),
    ) == (
        "183.4",
        "4.08",
        "27.9",
    )
    inspector.clear_range()
    assert inspector.distance.text() == "—"
    assert inspector.range_title.text() == "No range selected"


def test_clearing_empties_the_bout_and_zone_tables(inspector):
    inspector.bouts.setRowCount(1)
    inspector.zones.setRowCount(1)
    inspector.set_zone_count(1)
    inspector.clear_range()
    assert inspector.bouts.rowCount() == 0
    assert inspector.zones.rowCount() == 0
    assert inspector.zones_title.text() == "ZONES"


def test_hardware_rows(inspector):
    inspector.set_hardware([("LED 470 nm", "20.0 s on · 200 pulses", QColor(colors.LANE_OUTPUT))])
    assert inspector.hardware.rowCount() == 1
    assert inspector.hardware.item(0, 1).text() == "20.0 s on · 200 pulses"


def test_zone_count_in_the_title(inspector):
    inspector.set_zone_count(3)
    assert inspector.zones_title.text() == "ZONES (3)"
    inspector.set_zone_count(0)
    assert inspector.zones_title.text() == "ZONES"


def test_tables_are_as_tall_as_their_rows(qtbot):
    inspector = Inspector()
    qtbot.addWidget(inspector)
    inspector.resize(330, 2000)
    inspector.show()
    qtbot.waitExposed(inspector)
    inspector.set_hardware([("a", "1.0 s on", QColor(colors.LANE_OUTPUT))])
    qtbot.wait(20)
    one_row = inspector.hardware.height()
    assert one_row <= inspector.hardware.sizeHint().height() + 2
    inspector.set_hardware([("a", "1.0 s on", QColor(colors.LANE_OUTPUT))] * 4)
    qtbot.wait(20)
    assert inspector.hardware.height() > one_row


def test_session_review_qss_uses_only_token_colours():
    import re

    from glider.gui.styles import load_stylesheet

    qss = load_stylesheet("tools")
    review = qss[qss.index("Session Review: timeline panel") :]
    tokens = {
        v.lower()
        for k, v in vars(colors).items()
        if k.isupper() and isinstance(v, str) and v.startswith("#")
    }
    used = {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b", review)}
    assert used <= tokens, sorted(used - tokens)


def _rows():
    colour = QColor(colors.MARKER_CYAN)
    return [
        MarkerRow("r1", "range", "Stim", colour, "cohort", "1:00.00 – 4:20.00", "200.0 s"),
        MarkerRow("p1", "point", "door stuck", colour, "session", "2:11.50", "", "lever jammed"),
    ]


def test_markers_are_listed_and_counted(inspector):
    inspector.set_markers(_rows())
    assert inspector.marker_list.count() == 2
    assert inspector.tabText(2) == "Markers (2)"
    assert "Stim" in inspector.marker_list.item(0).text()
    assert "lever jammed" in inspector.marker_list.item(1).text()
    inspector.set_markers([])
    assert inspector.tabText(2) == "Markers"


def test_the_filters_show_one_kind(inspector):
    inspector.set_markers(_rows())
    inspector.marker_filters["point"].click()
    hidden = [inspector.marker_list.item(i).isHidden() for i in range(2)]
    assert hidden == [True, False]
    inspector.marker_filters["all"].click()
    assert not any(inspector.marker_list.item(i).isHidden() for i in range(2))


def test_clicking_a_marker_picks_it(inspector):
    inspector.set_markers(_rows())
    seen = []
    inspector.marker_picked.connect(seen.append)
    inspector.marker_list.itemClicked.emit(inspector.marker_list.item(1))
    assert seen == ["p1"]


def test_the_range_card_names_the_markers_it_is_inside(inspector):
    inspector.set_inside(["Baseline", "Stim"])
    assert inspector.inside.text() == "inside Baseline, Stim"
    assert not inspector.inside.isHidden()
    inspector.clear_range()
    assert inspector.inside.isHidden()
    assert not inspector.save_marker_btn.isEnabled()


def test_a_marker_note_shows_only_when_there_is_one(inspector):
    inspector.set_marker_note("review_markers.json could not be read")
    assert not inspector.marker_note.isHidden()
    inspector.set_marker_note("")
    assert inspector.marker_note.isHidden()
