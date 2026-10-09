"""Analysis tools for the GLIDER MCP server.

Thin wrappers over glider.analysis and glider.core.doctor that return compact
JSON. Large tables go to a CSV the agent names, so a long recording does not
flood its context; the response keeps a preview and the path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from glider.analysis import Session
from glider.analysis.cohort import discover_sessions
from glider.core.doctor import doctor
from glider.core.project import Project
from glider.mcp.paths import writable_path

MAX_ROWS = 200
PREVIEW_ROWS = 20


def _load(path: str) -> Session:
    p = Path(path).expanduser()
    if not p.is_dir():
        raise ValueError(
            f"{p} is not a folder; pass a recording folder (find_recordings lists them)"
        )
    return Session.load(p)


def _require_tracking(session: Session) -> None:
    if session.tracking.empty:
        raise ValueError(
            f"{session.directory} has no tracking CSV, so there is no ethogram, "
            "trajectory or kinematics; session_summary shows what it does have"
        )


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _table(df: pd.DataFrame, out_csv: str | None) -> dict[str, Any]:
    df = df.reset_index(drop=True)
    result: dict[str, Any] = {"columns": [str(c) for c in df.columns], "total_rows": len(df)}
    limit = MAX_ROWS
    if out_csv:
        target = writable_path(out_csv, ".csv")
        df.to_csv(target, index=False)
        result["csv"] = str(target)
        limit = PREVIEW_ROWS
    result["rows"] = _records(df.head(limit))
    result["truncated"] = len(df) > limit
    return result


def _unique(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df:
        return []
    return sorted(df[column].dropna().astype(str).unique())


def _number(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 3)


def find_recordings(root: str) -> dict[str, Any]:
    """Every GLIDER recording beneath a folder, with its group from glider_project.json."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise ValueError(f"{root_path} is not a folder")
    sources, warning = discover_sessions(root_path)
    sessions = [
        {
            "session_id": s.session_id,
            "path": str(s.path),
            "recording": str(s.recording) if s.recording else None,
            "group": s.group,
        }
        for s in sources
    ]
    return {"summary": f"{len(sessions)} session(s)", "sessions": sessions, "warning": warning}


def session_summary(path: str) -> dict[str, Any]:
    """What a recording holds: duration, frame rate, states, zones, event sources.

    Call this first; it names the values the other analysis tools accept.
    """
    s = _load(path)
    t = s.tracking
    zones = {z.strip() for v in _unique(t, "zone_ids") for z in v.split(",") if z.strip()}
    object_ids = (
        sorted(int(x) for x in t["object_id"].dropna().unique()) if "object_id" in t else []
    )
    has = {
        "tracking": not t.empty,
        "data": not s.data.empty,
        "events": not s.events.empty,
        "video": s.video_path is not None,
    }
    duration = _number(s.flow_duration_s)
    return {
        "summary": f"{s.directory.name}: {duration} s, "
        + ", ".join(k for k, v in has.items() if v),
        "directory": str(s.directory),
        "duration_s": duration,
        "frame_rate": _number(s.frame_rate),
        "has": has,
        "object_ids": object_ids,
        "states": _unique(t, "behavioral_state"),
        "zones": sorted(zones),
        "event_sources": _unique(s.events, "source"),
        "metadata": s.metadata,
    }


def _state_totals(intervals: pd.DataFrame) -> list[dict[str, Any]]:
    if intervals.empty:
        return []
    total = intervals["duration_ms"].sum()
    out = (
        intervals.groupby("state")["duration_ms"]
        .agg(total_ms="sum", n_bouts="count", mean_bout_ms="mean", median_bout_ms="median")
        .reset_index()
    )
    out["percent"] = 100 * out["total_ms"] / total if total else 0.0
    return _records(out.round(2))


def ethogram(
    path: str, object_id: int = 0, kind: str = "intervals", out_csv: str | None = None
) -> dict[str, Any]:
    """Behavioural states over the flow, with per-state time, %, bout count and length.

    Args:
        kind: "intervals" (one row per bout) or "transitions" (state -> state counts).
        out_csv: Optional absolute .csv path for the full table.
    """
    s = _load(path)
    _require_tracking(s)
    intervals = s.ethogram(object_id=object_id)
    if kind == "intervals":
        table = intervals
    elif kind == "transitions":
        table = s.state_transitions(object_id=object_id)
    else:
        raise ValueError(f"unknown kind '{kind}'; use 'intervals' or 'transitions'")
    totals = _state_totals(intervals)
    return {
        "summary": f"{len(intervals)} bouts across {len(totals)} states",
        "state_totals": totals,
        **_table(table, out_csv),
    }


def trajectory(
    path: str, object_id: int = 0, kind: str = "zone_dwell", out_csv: str | None = None
) -> dict[str, Any]:
    """Where the subject went.

    Args:
        kind: "zone_dwell" (time, entries, mean bout per zone), "zone_transitions",
            or "positions" (the centroid trace; use out_csv, it is long).
            For an occupancy heatmap use plot(kind="occupancy").
    """
    s = _load(path)
    _require_tracking(s)
    tables = {
        "zone_dwell": s.zone_dwell,
        "zone_transitions": s.zone_transitions,
        "positions": s.trajectory,
    }
    if kind not in tables:
        raise ValueError(f"unknown kind '{kind}'; use one of {', '.join(tables)}")
    table = tables[kind](object_id=object_id)
    return {"summary": f"{kind}: {len(table)} rows", **_table(table, out_csv)}


def kinematics(path: str, object_id: int = 0, out_csv: str | None = None) -> dict[str, Any]:
    """Speed statistics, total distance, and the speed distribution."""
    s = _load(path)
    _require_tracking(s)
    speed = pd.to_numeric(s.velocity(object_id=object_id)["velocity"], errors="coerce").dropna()
    cumulative = s.cumulative_distance(object_id=object_id)
    total = cumulative["cumulative_mm"].iloc[-1] if not cumulative.empty else None
    units = "px/s" if s.frame_rate else "px/frame"
    stats = {
        "speed_units": units,
        "mean_speed": _number(speed.mean()) if not speed.empty else None,
        "median_speed": _number(speed.median()) if not speed.empty else None,
        "max_speed": _number(speed.max()) if not speed.empty else None,
        "total_distance": _number(total),
        "distance_units": "mm if the camera was calibrated, otherwise px",
    }
    table = s.speed_distribution(object_id=object_id)
    return {
        "summary": f"mean speed {stats['mean_speed']} {units}",
        **stats,
        **_table(table, out_csv),
    }


def events(
    path: str,
    source: str | None = None,
    value: str | None = None,
    window_s: list[float] | None = None,
    out_csv: str | None = None,
) -> dict[str, Any]:
    """Hardware/flow events, or tracking around them (event-triggered analysis).

    Args:
        source: Event source to match (session_summary lists them).
        value: Optional event value to match.
        window_s: [seconds_before, seconds_after]. When given, returns the
            tracked velocity around each matching event instead of the events.
    """
    s = _load(path)
    if s.events.empty:
        raise ValueError(f"{s.directory} has no events CSV")
    if window_s is None:
        table = s.find_events(source=source, value=value)
        return {"summary": f"{len(table)} event(s)", **_table(table, out_csv)}
    if not source:
        raise ValueError(
            "event-triggered analysis needs a source; session_summary lists event_sources"
        )
    if len(window_s) != 2:
        raise ValueError("window_s must be [seconds_before, seconds_after]")
    _require_tracking(s)
    before, after = window_s
    table = s.event_triggered(
        source=source, value=value, window_ms=(-1000.0 * before, 1000.0 * after)
    )
    trials = table["trial_id"].nunique() if not table.empty else 0
    return {"summary": f"{trials} trial(s) around {source}", **_table(table, out_csv)}


def project_doctor(root: str) -> dict[str, Any]:
    """Problems in a project folder that would cost a result, and what each costs."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise ValueError(f"{root_path} is not a folder")
    findings = [
        {"check": f.check, "severity": f.severity, "session_id": f.session_id, "message": f.message}
        for f in doctor(Project.load(root_path))
    ]
    return {"summary": f"{len(findings)} finding(s)", "findings": findings}
