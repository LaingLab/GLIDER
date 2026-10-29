"""The time-bin export's options."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from glider.gui.review.time_bins_dialog import SHEETS, TimeBinsDialog  # noqa: E402


@pytest.fixture
def dialog(qtbot):
    widget = TimeBinsDialog(3.0, has_selection=False)
    qtbot.addWidget(widget)
    return widget


def test_the_tick_makes_a_sheet_per_measure(dialog):
    assert dialog.layout_shape() == "tidy"
    dialog.per_sheet.setChecked(True)
    assert dialog.layout_shape() == SHEETS
    # Tidy or wide means nothing for a workbook.
    assert not dialog.shape.isEnabled()
    dialog.per_sheet.setChecked(False)
    assert dialog.shape.isEnabled() and dialog.layout_shape() == "tidy"


def test_the_tick_can_start_on(qtbot):
    widget = TimeBinsDialog(1.0, has_selection=False, per_sheet=True)
    qtbot.addWidget(widget)
    assert widget.layout_shape() == SHEETS and not widget.shape.isEnabled()


def test_the_width_is_in_seconds(dialog):
    assert dialog.bin_seconds() == pytest.approx(180.0)
