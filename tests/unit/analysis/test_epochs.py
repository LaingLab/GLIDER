"""The epoch table's arithmetic. Qt-free."""

from __future__ import annotations

import math

import pytest

from glider.analysis.epochs import (
    CURRENT,
    Epoch,
    Metric,
    default_metrics,
    metric_catalog,
    metric_text,
    metric_value,
    missing_reason,
    summarize,
    threshold_text,
    tidy_frame,
    wide_frame,
)


def _row(session, *, distance=100.0, freezing=6.0, group="ctrl", **extra) -> dict:
    return {
        "session": session,
        "group": group,
        "t0": "flow_start",
        "start_s": 0.0,
        "end_s": 60.0,
        "duration_s": 60.0,
        "distance_cm": distance,
        "mean_cm_s": 1.5,
        "peak_cm_s": 9.0,
        "freezing_s": freezing,
        "darting_s": 0.0,
        "groom_s": 12.0,
        "_states": ("freezing", "groom"),
        **extra,
    }


def test_the_catalog_reads_movement_behaviour_zones_devices_thresholds():
    rows = [
        _row(
            "m01",
            zone_open_s=10.0,
            zone_open_entries=3,
            zone_open_latency_s=2.0,
            _zones=("open",),
            hw_LED_on_s=4.0,
            _devices=("LED",),
        )
    ]
    keys = [m.key for m in metric_catalog(rows)]
    assert keys[:3] == ["distance_cm", "mean_cm_s", "peak_cm_s"]
    assert keys.index("darting_s") < keys.index("groom_pct") < keys.index("zone_open_s")
    assert keys.index("zone_open_latency_s") < keys.index("hw_LED_on_s")
    assert keys[-2:] == ["freeze_threshold", "dart_threshold"]


def test_default_metrics_take_the_first_zone_when_there_is_one():
    plain = metric_catalog([_row("m01")])
    zoned = metric_catalog([_row("m01", zone_open_s=1.0, _zones=("open",))])
    assert default_metrics(plain) == ["distance_cm", "mean_cm_s", "freezing_pct"]
    assert default_metrics(zoned) == ["distance_cm", "mean_cm_s", "freezing_pct", "zone_open_s"]


def test_a_percentage_is_seconds_over_the_range():
    assert metric_value(_row("m01"), "freezing_pct") == pytest.approx(10.0)


def test_an_outside_row_answers_nothing():
    row = {"session": "m01", "outside": True}
    assert metric_value(row, "distance_cm") is None
    assert missing_reason(row, "distance_cm") == "The range is outside this session"


def test_a_missing_or_nan_value_is_none_with_a_reason():
    row = _row("m01", distance=None, zone_open_latency_s=math.nan)
    assert metric_value(row, "distance_cm") is None
    assert missing_reason(row, "distance_cm") == "No calibration for this session"
    assert metric_value(row, "zone_open_latency_s") is None
    assert missing_reason(row, "zone_open_latency_s") == "Never entered in this range"
    assert missing_reason(row, "hw_LED_on_s") == "This session has no such device"


def test_metric_text_uses_the_metric_s_decimals():
    assert metric_text(_row("m01"), Metric("distance_cm", "Distance", 1)) == "100.0"
    assert metric_text(_row("m01", distance=None), Metric("distance_cm", "Distance", 1)) == "—"


def test_thresholds_fall_back_to_pixels_rather_than_nothing():
    row = _row("m01", freeze_threshold_cm_s=None, freeze_threshold_px_frame=0.25)
    assert threshold_text(row, "freeze") == "0.250 px/f"
    assert metric_text(row, Metric("freeze_threshold", "Freeze <")) == "0.250 px/f"
    assert threshold_text(_row("m01", freeze_threshold_cm_s=0.5), "freeze") == "0.50"
    assert threshold_text(_row("m01"), "dart") == "—"


def test_summarize_is_mean_and_sem_over_what_is_present():
    mean, sem = summarize([1.0, 2.0, 3.0, None])
    assert mean == pytest.approx(2.0)
    assert sem == pytest.approx(1.0 / math.sqrt(3))
    assert summarize([5.0]) == (5.0, None)
    assert summarize([None, math.nan]) == (None, None)


EPOCHS = [Epoch("m1", "Baseline", 0.0, 60.0, "slate"), Epoch(CURRENT, "Current range", 10.0, 20.0)]
ROWS = {
    "m1": [_row("m01"), _row("m02", distance=200.0), _row("m03", group="stim")],
    CURRENT: [_row("m01"), _row("m02"), {"session": "m03", "group": "stim", "outside": True}],
}


def test_tidy_is_one_row_per_session_and_epoch():
    tidy = tidy_frame(EPOCHS, ROWS, ["distance_cm", "freezing_pct"])
    assert len(tidy) == 6
    assert list(tidy.columns) == [
        "session",
        "group",
        "range",
        "t0",
        "start_s",
        "end_s",
        "distance_cm",
        "freezing_pct",
    ]
    assert tidy.loc[tidy["session"] == "m03", "distance_cm"].isna().tolist() == [False, True]


def test_wide_is_one_row_per_session():
    wide = wide_frame(tidy_frame(EPOCHS, ROWS, ["distance_cm"]), ["distance_cm"])
    assert list(wide["session"]) == ["m01", "m02", "m03"]
    assert list(wide.columns) == [
        "session",
        "group",
        "Baseline__distance_cm",
        "Current range__distance_cm",
    ]
    assert wide.loc[wide["session"] == "m02", "Baseline__distance_cm"].item() == 200.0


def test_duplicate_epoch_names_get_unique_range_labels():
    # Two epochs sharing a name: tidy should make their range labels unique
    epochs = [
        Epoch("m1", "Baseline", 0.0, 60.0, "slate"),
        Epoch("m2", "Baseline", 60.0, 120.0, "slate"),
    ]
    rows_dict = {
        "m1": [_row("m01"), _row("m02")],
        "m2": [_row("m01"), _row("m02")],
    }
    tidy = tidy_frame(epochs, rows_dict, ["distance_cm"])
    # Range labels should be unique: "Baseline" and "Baseline (2)"
    ranges = tidy["range"].unique().tolist()
    assert len(ranges) == 2
    assert "Baseline" in ranges
    assert "Baseline (2)" in ranges
    # wide_frame should not crash with duplicate labels
    wide = wide_frame(tidy, ["distance_cm"])
    assert (
        len(wide.columns) == 4
    )  # session, group, Baseline__distance_cm, Baseline (2)__distance_cm


def test_nan_in_state_duration_returns_none_not_nan():
    row = _row("m01", distance=None, freezing_s=math.nan)
    assert metric_value(row, "freezing_pct") is None
    assert metric_text(row, Metric("freezing_pct", "Freezing %", 1)) == "—"
