"""The epoch table: sessions down, epochs across, a metric per column."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from glider.analysis.epochs import CURRENT, Epoch, metric_catalog  # noqa: E402
from glider.gui.review.epoch_table import BAR_ROLE, EpochSession, EpochTable  # noqa: E402

EPOCHS = [Epoch("m1", "Baseline", 0.0, 60.0, "slate"), Epoch(CURRENT, "Current range", 10.0, 20.0)]
SESSIONS = [
    EpochSession(0, "m01", "ctrl"),
    EpochSession(1, "m02", "ctrl"),
    EpochSession(2, "m03", "stim"),
]


def _row(session, distance, t0="flow_start"):
    return {
        "session": session,
        "t0": t0,
        "duration_s": 60.0,
        "distance_cm": distance,
        "mean_cm_s": 1.0,
        "peak_cm_s": 2.0,
        "freezing_s": 6.0,
        "darting_s": 0.0,
        "_states": ("freezing",),
    }


def _rows(t0_m03="flow_start"):
    block = [_row("m01", 100.0), _row("m02", 200.0), _row("m03", None, t0_m03)]
    return {"m1": block, CURRENT: [dict(r) for r in block]}


def _fill(widget, rows=None):
    rows = rows or _rows()
    widget.set_data(EPOCHS, SESSIONS, rows, metric_catalog(rows["m1"]))


@pytest.fixture
def table(qtbot):
    widget = EpochTable()
    qtbot.addWidget(widget)
    _fill(widget)
    return widget


def _text(table, r, c):
    item = table.table.item(r, c)
    return "" if item is None else item.text()


def _find(table, text):
    return next(r for r in range(table.table.rowCount()) if _text(table, r, 0) == text)


def _col(table, epoch, label):
    labels = [m.label for m in table.metrics()]
    return 1 + epoch * len(labels) + labels.index(label)


def test_columns_are_epochs_times_metrics(table):
    assert table.table.columnCount() == 1 + 2 * 3
    headers = [table.table.horizontalHeaderItem(c).text() for c in range(1, 4)]
    assert headers == ["Distance (cm)", "Speed (cm/s)", "Freezing %"]


def test_the_first_row_names_each_epoch(table):
    assert _text(table, 0, 1).startswith("Baseline")
    assert table.table.columnSpan(0, 1) == 3
    assert _text(table, 0, 4).startswith("Current range")


def test_group_titles_head_their_sessions(table):
    assert _find(table, "ctrl") < table.row_of(0) < table.row_of(1) < _find(table, "ctrl mean")
    assert _find(table, "stim") < table.row_of(2)


def test_each_group_gets_a_mean_and_sem_row(table):
    r = _find(table, "ctrl mean")
    c = _col(table, 0, "Distance (cm)")
    assert _text(table, r, c) == "150.0"
    assert _text(table, r + 1, 0) == "SEM"
    assert _text(table, r + 1, c) == "± 50.0"


def test_a_missing_value_is_a_dash_with_a_reason(table):
    item = table.table.item(table.row_of(2), _col(table, 0, "Distance (cm)"))
    assert item.text() == "—"
    assert item.toolTip() == "No calibration for this session"


def test_the_data_bar_is_the_share_of_the_column_max(table):
    c = _col(table, 0, "Distance (cm)")
    assert table.table.item(table.row_of(1), c).data(BAR_ROLE) == pytest.approx(1.0)
    assert table.table.item(table.row_of(0), c).data(BAR_ROLE) == pytest.approx(0.5)


def test_clicking_a_session_picks_it_and_a_summary_row_picks_nothing(table):
    seen = []
    table.session_picked.connect(seen.append)
    table.table.cellClicked.emit(table.row_of(1), 0)
    table.table.cellClicked.emit(_find(table, "ctrl mean"), 0)
    assert seen == [1]


def test_double_clicking_a_session_opens_it(table):
    seen = []
    table.session_opened.connect(seen.append)
    table.table.cellDoubleClicked.emit(table.row_of(2), 0)
    assert seen == [2]


def test_switching_an_epoch_off_drops_its_columns_and_survives_a_refill(table):
    table.chips[CURRENT].click()
    assert [e.key for e in table.epochs()] == ["m1"]
    assert table.table.columnCount() == 1 + 3
    _fill(table)
    assert [e.key for e in table.epochs()] == ["m1"]


def test_choosing_metrics(table):
    table.set_metrics(["distance_cm"])
    assert table.table.columnCount() == 1 + 2
    _fill(table)
    assert [m.key for m in table.metrics()] == ["distance_cm"]


def test_the_metrics_menu_offers_the_catalog_with_the_chosen_checked(table):
    actions = {a.text(): a for a in table.metrics_btn.menu().actions()}
    assert "Peak speed (cm/s)" in actions
    assert actions["Distance (cm)"].isChecked() and not actions["Peak speed (cm/s)"].isChecked()
    actions["Peak speed (cm/s)"].trigger()
    assert "peak_cm_s" in [m.key for m in table.metrics()]


def test_a_session_on_another_zero_is_flagged(qtbot):
    widget = EpochTable()
    qtbot.addWidget(widget)
    _fill(widget, _rows(t0_m03="video_start"))
    item = widget.table.item(widget.row_of(2), 0)
    assert item.text().startswith("⚠")
    assert "first video frame" in item.toolTip()


def test_it_is_empty_until_there_is_an_epoch(table):
    table.set_data([], SESSIONS, {}, [])
    assert table.table.isHidden() and not table.empty.isHidden()
    assert table.session_count() == 0


def test_the_shown_session_is_highlighted(table):
    table.set_shown(1)
    assert {i.row() for i in table.table.selectedIndexes()} == {table.row_of(1)}


def test_showing_a_session_does_not_report_it_as_picked(table):
    seen = []
    table.session_picked.connect(seen.append)
    table.set_shown(1)
    assert {i.row() for i in table.table.selectedIndexes()} == {table.row_of(1)}
    assert seen == []


def test_metrics_fall_back_to_the_defaults_when_the_catalog_no_longer_offers_them(table):
    zoned = _rows()
    for block in zoned.values():
        for r in block:
            r["_zones"] = ("open",)
            r["zone_open_s"] = 5.0
    table.set_data(EPOCHS, SESSIONS, zoned, metric_catalog(zoned["m1"]))
    table.set_metrics(["zone_open_s"])
    assert [m.key for m in table.metrics()] == ["zone_open_s"]

    _fill(table)  # refill from a catalog with no zones at all
    assert table.session_count() == 3
    headers = [table.table.horizontalHeaderItem(c).text() for c in range(1, 4)]
    assert headers == ["Distance (cm)", "Speed (cm/s)", "Freezing %"]


def test_export_asks_for_a_shape(table):
    seen = []
    table.export_requested.connect(seen.append)
    for action in table.export_btn.menu().actions():
        action.trigger()
    assert seen == ["tidy", "wide"]
