"""The sessions panel: one row per animal, grouped by treatment."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from glider.gui.review.pool import PoolEntry, SessionPool, ethogram_strip  # noqa: E402
from glider.gui.review.timeline import behavior_qcolor  # noqa: E402


def _entries():
    return [
        PoolEntry("M11", "ChR2", 300.0, True, True, True),
        PoolEntry("M12", "ChR2", 300.0, True, False, True),
        PoolEntry("M21", "eYFP", 300.0, False, True, False),
    ]


@pytest.fixture
def pool(qtbot):
    widget = SessionPool()
    qtbot.addWidget(widget)
    widget.set_entries(_entries())
    return widget


def test_sessions_are_grouped(pool):
    assert pool.count() == 3
    groups = [pool.tree.topLevelItem(i).text(0) for i in range(pool.tree.topLevelItemCount())]
    assert groups == ["CHR2  ·  2", "EYFP  ·  1"]


def test_an_ungrouped_cohort_is_a_flat_list(qtbot):
    pool = SessionPool()
    qtbot.addWidget(pool)
    pool.set_entries(
        [PoolEntry("a", "", 1.0, False, False, False), PoolEntry("b", "", 1.0, False, False, False)]
    )
    assert pool.tree.topLevelItemCount() == 2


def test_selecting_a_row_reports_its_index(pool):
    seen = []
    pool.session_picked.connect(seen.append)
    pool.select(2)
    assert seen == [2]
    assert pool.current() == 2


def test_the_filter_hides_non_matching_rows(pool):
    pool.filter.setText("m2")
    assert [item.isHidden() for item in pool._items] == [True, True, False]
    assert pool.tree.topLevelItem(0).isHidden() is True  # nothing left in ChR2


def test_the_strip_is_the_ethogram_in_miniature(qtbot):
    image = ethogram_strip(["a"] * 50 + ["b"] * 50, ["a", "b"], width=10)
    assert image.width() == 10
    assert image.pixelColor(2, 0) == behavior_qcolor("a", ["a", "b"])
    assert image.pixelColor(8, 0) == behavior_qcolor("b", ["a", "b"])
