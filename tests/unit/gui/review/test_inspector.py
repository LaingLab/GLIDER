"""The inspector lays out what the window computes; it computes nothing."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QColor  # noqa: E402

from glider.gui.review.inspector import Inspector  # noqa: E402
from glider.gui.styles import colors  # noqa: E402


@pytest.fixture
def inspector(qtbot):
    widget = Inspector()
    qtbot.addWidget(widget)
    return widget


def test_it_has_a_range_and_a_session_tab(inspector):
    assert [inspector.tabText(i) for i in range(inspector.count())] == ["Range", "Session"]


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


def test_hardware_rows(inspector):
    inspector.set_hardware([("LED 470 nm", "20.0 s on · 200 pulses", QColor(colors.LANE_OUTPUT))])
    assert inspector.hardware.rowCount() == 1
    assert inspector.hardware.item(0, 1).text() == "20.0 s on · 200 pulses"


def test_zone_count_in_the_title(inspector):
    inspector.set_zone_count(3)
    assert inspector.zones_title.text() == "ZONES (3)"
    inspector.set_zone_count(0)
    assert inspector.zones_title.text() == "ZONES"
