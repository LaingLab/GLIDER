"""The epoch table's arithmetic: metrics, group mean and SEM, and the CSVs. Qt-free.

An epoch is a named range applied to every session on its own zero (see
:mod:`glider.analysis.markers`): the cohort's Baseline / Stim / Post markers,
and the current selection. The window computes one row per session per epoch
(``AnalysisWindow.range_rows``); this module decides what those rows can
answer and how each answer reads.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "CURRENT",
    "Epoch",
    "Metric",
    "default_metrics",
    "metric_catalog",
    "metric_sheets",
    "metric_text",
    "metric_value",
    "missing_reason",
    "summarize",
    "threshold_text",
    "tidy_frame",
    "time_bins",
    "wide_frame",
]

#: The key of the Current range epoch, which has no marker id.
CURRENT = "current"


@dataclass(frozen=True)
class Epoch:
    """One column group: a cohort range marker, or the current selection."""

    key: str  # marker id, or CURRENT
    name: str
    start_s: float
    end_s: float
    color: str = ""  # swatch name; "" for the current range


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    decimals: int = 2


_MOVEMENT = (
    Metric("distance_cm", "Distance (cm)", 1),
    Metric("mean_cm_s", "Speed (cm/s)", 2),
    Metric("peak_cm_s", "Peak speed (cm/s)", 2),
)
# Pup retrieval's per-range numbers (AnalysisWindow._retrieval_columns). No
# `_pct` key: metric_value would read it as seconds over the range.
_RETRIEVAL = (
    Metric("pups_retrieved", "Pups retrieved", 0),
    Metric("dam_in_nest_frac", "Dam in nest (fraction)", 2),
    Metric("nose_nest_mean_cm", "Nose to nest (cm)", 1),
)
_THRESHOLDS = (
    Metric("freeze_threshold", "Freeze < (cm/s)", 2),
    Metric("dart_threshold", "Dart > (cm/s)", 2),
)


def _title(name: str) -> str:
    return name.replace("_", " ").capitalize()


def metric_catalog(rows: Iterable[dict]) -> list[Metric]:
    """Every metric the rows can answer, in reading order.

    Movement; each behaviour as seconds and as a share of the range; each
    zone; each device; pup retrieval where a session was scored for it;
    then the thresholds each session was scored with. Freezing and darting
    are always offered -- every row carries them.
    """
    rows = list(rows)
    states = sorted({"freezing", "darting"} | {s for r in rows for s in r.get("_states", ())})
    zones = sorted({z for r in rows for z in r.get("_zones", ())})
    devices = sorted({d for r in rows for d in r.get("_devices", ())})
    out = list(_MOVEMENT)
    for state in states:
        out += [
            Metric(f"{state}_s", f"{_title(state)} (s)", 2),
            Metric(f"{state}_pct", f"{_title(state)} %", 1),
        ]
    for zone in zones:
        out += [
            Metric(f"zone_{zone}_s", f"{zone} time (s)", 2),
            Metric(f"zone_{zone}_entries", f"{zone} entries", 0),
            Metric(f"zone_{zone}_latency_s", f"{zone} latency (s)", 2),
        ]
    out += [Metric(f"hw_{device}_on_s", f"{device} on (s)", 2) for device in devices]
    if any("pups_retrieved" in row for row in rows):
        out += list(_RETRIEVAL)
    return out + list(_THRESHOLDS)


def default_metrics(catalog: list[Metric]) -> list[str]:
    """Distance, Speed, Freezing % and the first zone's time -- whichever exist."""
    keys = [m.key for m in catalog]
    wanted = ["distance_cm", "mean_cm_s", "freezing_pct"]
    first_zone = next(
        (k for k in keys if k.startswith("zone_") and k.endswith("_s") and "_latency" not in k),
        None,
    )
    if first_zone is not None:
        wanted.append(first_zone)
    return [k for k in wanted if k in keys]


def metric_value(row: dict, key: str) -> float | None:
    """The number a cell holds, or None when this session cannot answer it."""
    if row.get("outside"):
        return None
    if key.endswith("_pct"):
        seconds, duration = row.get(key[: -len("_pct")] + "_s"), row.get("duration_s")
        if seconds is None or not duration:
            return None
        seconds = float(seconds)
        duration = float(duration)
        if math.isnan(seconds) or math.isnan(duration):
            return None
        return 100.0 * seconds / duration
    if key in ("freeze_threshold", "dart_threshold"):
        key = f"{key}_cm_s"
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def threshold_text(row: dict, side: str) -> str:
    """A cut-off in cm/s, falling back to px/frame, or an em dash.

    An uncalibrated run has real thresholds in pixels; showing nothing would
    claim it had none, and showing a converted number would invent the scale
    it never had.
    """
    real = row.get(f"{side}_threshold_cm_s")
    if real is not None:
        return f"{real:.2f}"
    pixels = row.get(f"{side}_threshold_px_frame")
    return "—" if pixels is None else f"{pixels:.3f} px/f"


def metric_text(row: dict, metric: Metric) -> str:
    """What a cell reads."""
    if metric.key in ("freeze_threshold", "dart_threshold") and not row.get("outside"):
        return threshold_text(row, metric.key.split("_")[0])
    value = metric_value(row, metric.key)
    return "—" if value is None else f"{value:.{metric.decimals}f}"


def missing_reason(row: dict, key: str) -> str:
    """Why a cell is blank, for its tooltip."""
    if row.get("outside"):
        return "The range is outside this session"
    if key in ("distance_cm", "mean_cm_s", "peak_cm_s"):
        return "No calibration for this session"
    if key.startswith("zone_"):
        if key.endswith("_latency_s") and key in row:
            return "Never entered in this range"
        return "No zone data for this session"
    if key.startswith("hw_"):
        return "This session has no such device"
    if key in ("freeze_threshold", "dart_threshold"):
        if row.get(f"{key}_px_frame") is not None:
            return "Uncalibrated: shown in px/frame"
        return "No thresholds recorded for this session (run.json)"
    return "Not available for this session"


def summarize(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    """Mean and SEM (sample SD / sqrt n) over the values present."""
    present = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not present:
        return None, None
    mean = statistics.fmean(present)
    if len(present) < 2:
        return mean, None
    return mean, statistics.stdev(present) / math.sqrt(len(present))


def tidy_frame(
    epochs: list[Epoch], rows_by_epoch: dict[str, list[dict]], metrics: list[str]
) -> pd.DataFrame:
    """One row per session x epoch: ``session, group, range, t0, start_s, end_s, <metric>…``.

    ``start_s`` / ``end_s`` are what that session was actually measured over
    -- a range past its end is clipped -- so a row can be checked on its own.
    Range labels are unique per epoch: the first occurrence of a name keeps it,
    later ones get a suffix ``(2)``, ``(3)``, etc. This ensures ``wide_frame``'s
    pivot operation never encounters duplicate column names.
    """
    # Build unique range labels: count occurrences of each name and suffix duplicates
    name_counts = Counter()
    epoch_labels = {}
    for epoch in epochs:
        name_counts[epoch.name] += 1
        count = name_counts[epoch.name]
        label = epoch.name if count == 1 else f"{epoch.name} ({count})"
        epoch_labels[epoch.key] = label

    records = []
    for epoch in epochs:
        range_label = epoch_labels[epoch.key]
        for row in rows_by_epoch[epoch.key]:
            records.append(
                {
                    "session": row["session"],
                    "group": row.get("group", ""),
                    "range": range_label,
                    "t0": row.get("t0", ""),
                    "start_s": row.get("start_s"),
                    "end_s": row.get("end_s"),
                    **{key: metric_value(row, key) for key in metrics},
                }
            )
    columns = ["session", "group", "range", "t0", "start_s", "end_s", *metrics]
    return pd.DataFrame.from_records(records, columns=columns)


def _bin_edge(seconds: float, unit: float) -> str:
    """``90`` s as ``1.5`` in minutes: whole numbers without a trailing ``.0``."""
    return f"{seconds / unit:g}"


def time_bins(start_s: float, end_s: float, bin_s: float) -> list[Epoch]:
    """``[start_s, end_s)`` cut into consecutive bins of ``bin_s``, as epochs.

    Bin edges sit on whole multiples of ``bin_s`` from zero (flow start, or the
    first video frame), so minute 3 is the same stretch in every export
    whatever span was asked for. The last bin is clipped to ``end_s`` and named
    for what it covers. Labelled in minutes when the bin is a whole number of
    them, in seconds otherwise.
    """
    if not bin_s > 0 or not end_s > start_s:
        return []
    unit, suffix = (60.0, "min") if float(bin_s) % 60 == 0 else (1.0, "s")
    first = math.floor(start_s / bin_s + 1e-9)
    bins = []
    k = first
    while k * bin_s < end_s - 1e-9:
        lo, hi = max(k * bin_s, start_s), min((k + 1) * bin_s, end_s)
        # A plain hyphen: an en dash reads as "â€“" in a CSV opened by Excel.
        name = f"{_bin_edge(lo, unit)}-{_bin_edge(hi, unit)} {suffix}"
        bins.append(Epoch(f"bin_{k - first + 1}", name, lo, hi))
        k += 1
    return bins


def wide_frame(tidy: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """One row per session: ``session, group, <range>__<metric>…``.

    Keyed on the session id, which must be unique: the review window
    suffixes a repeated one (two folders both named ``m01``).
    """
    out = tidy[["session", "group"]].drop_duplicates("session").set_index("session")
    for name, block in tidy.groupby("range", sort=False):
        block = block.set_index("session")
        for key in metrics:
            out[f"{name}__{key}"] = block[key]
    return out.reset_index()


#: Characters Excel refuses in a sheet name, and its length limit.
_SHEET_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")
_SHEET_MAX = 31


def _sheet_name(label: str, taken: set[str]) -> str:
    """``label`` as a legal, unique Excel sheet name."""
    base = _SHEET_FORBIDDEN.sub("-", label).strip("' ") or "Measure"
    name, n = base[:_SHEET_MAX], 1
    while name.lower() in taken:  # Excel compares sheet names case-blind
        n += 1
        suffix = f" ({n})"
        name = base[: _SHEET_MAX - len(suffix)] + suffix
    taken.add(name.lower())
    return name


def metric_sheets(
    tidy: pd.DataFrame, metrics: list[Metric], *, reserved: Iterable[str] = ()
) -> list[tuple[str, Metric, pd.DataFrame]]:
    """``(sheet name, metric, table)`` per metric: a row per session, a column per range.

    For a workbook with a sheet per measure, the layout a stats package or a
    plot wants. Sessions and ranges keep the order ``tidy`` has them in. A
    metric no session has a value for gets no sheet: an empty one is clutter.
    Sheet names are legal in Excel and unique, avoiding ``reserved``.
    """
    ranges = list(dict.fromkeys(tidy["range"]))
    sessions = tidy.drop_duplicates("session")[["session", "group"]]
    taken = {name.lower() for name in reserved}
    sheets: list[tuple[str, Metric, pd.DataFrame]] = []
    for metric in metrics:
        if metric.key not in tidy or tidy[metric.key].isna().all():
            continue
        table = tidy.pivot(index="session", columns="range", values=metric.key)
        table = table.reindex(index=sessions["session"], columns=ranges)
        table.columns.name = None
        table = table.reset_index()
        table.insert(1, "group", sessions["group"].to_numpy())
        sheets.append((_sheet_name(metric.label, taken), metric, table))
    return sheets
