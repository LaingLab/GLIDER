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


def test_reloading_releases_the_old_rows(qtbot):
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent

    pool = SessionPool()
    qtbot.addWidget(pool)
    pool.set_entries(_entries())
    old = [pool.tree.itemWidget(item, 0) for item in pool._items]
    pool.set_entries(_entries())
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert all(sip.isdeleted(widget) for widget in old)


def test_a_narrow_pool_shows_every_badge_whole(qtbot):
    """At the default width the badges were clipped to 'VI', 'PO', 'H'."""
    from PyQt6.QtWidgets import QLabel

    pool = SessionPool()
    qtbot.addWidget(pool)
    strip = ethogram_strip(["a"] * 10, ["a"])
    pool.set_entries(
        [PoolEntry("a_rather_long_session_name_2026-05-25", "", 300.0, True, True, True, strip)]
    )
    pool.resize(200, 400)
    pool.show()
    qtbot.waitExposed(pool)
    badges = [b for b in pool.findChildren(QLabel) if b.objectName() == "PoolBadge"]
    assert [b.text() for b in badges] == ["VID", "POSE", "HW"]
    for badge in badges:
        assert badge.isVisible()
        assert badge.sizeHint().width() <= badge.width(), badge.text()


class TestGrouping:
    def _actions(self, menu):
        return {a.text(): a for a in menu.actions() if a.text()}

    def test_several_rows_can_be_selected(self, pool):
        pool._items[0].setSelected(True)
        pool._items[2].setSelected(True)
        assert pool.selected_indices() == [0, 2]

    def test_group_headers_are_not_sessions(self, pool):
        pool.tree.topLevelItem(0).setSelected(True)
        assert pool.selected_indices() == []

    def test_the_menu_offers_every_group_and_says_how_many(self, pool):
        actions = self._actions(pool.group_menu([0, 2]))
        assert "2 sessions" in actions
        assert {"Move to ChR2", "Move to eYFP", "New group…", "Remove from group"} <= set(actions)

    def test_moving_to_the_group_they_are_in_is_disabled(self, pool):
        actions = self._actions(pool.group_menu([0, 1]))
        assert not actions["Move to ChR2"].isEnabled()
        assert actions["Move to eYFP"].isEnabled()

    def test_choosing_a_group_reports_the_sessions_and_the_group(self, pool):
        seen = []
        pool.group_assigned.connect(lambda i, g: seen.append((i, g)))
        self._actions(pool.group_menu([0, 2]))["Move to eYFP"].trigger()
        self._actions(pool.group_menu([2]))["Remove from group"].trigger()
        assert seen == [([0, 2], "eYFP"), ([2], "")]

    def test_a_new_group_asks_for_its_name(self, pool, monkeypatch):
        seen = []
        pool.group_assigned.connect(lambda i, g: seen.append((i, g)))
        monkeypatch.setattr(pool, "_ask_group_name", lambda: "Saline")
        self._actions(pool.group_menu([1]))["New group…"].trigger()
        monkeypatch.setattr(pool, "_ask_group_name", lambda: "")  # cancelled
        self._actions(pool.group_menu([1]))["New group…"].trigger()
        assert seen == [([1], "Saline")]

    def test_an_ungrouped_cohort_can_remove_nothing(self, qtbot):
        pool = SessionPool()
        qtbot.addWidget(pool)
        pool.set_entries([PoolEntry("a", "", 1.0, False, False, False)])
        actions = self._actions(pool.group_menu([0]))
        assert not actions["Remove from group"].isEnabled()
        assert not any(t.startswith("Move to") for t in actions)
