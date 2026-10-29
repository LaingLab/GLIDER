"""From the epoch table's tidy frame to the cells a test compares.

Two shapes: *between* (one epoch, the animals split by group) and *within*
(the same animals across epochs, as an animals x epochs matrix). Every animal
that cannot contribute is named with the reason, never silently dropped: a
reviewer reading "n = 7" when the cohort has 8 needs to know which one and why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["UNGROUPED", "Dropped", "between_cells", "within_matrix"]

#: The group of an animal with none in ``glider_project.json``.
UNGROUPED = "Ungrouped"


@dataclass(frozen=True)
class Dropped:
    """An animal left out of a test, and why."""

    session: str
    reason: str


def _missing(value) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _group(value) -> str:
    return UNGROUPED if value is None or str(value) in ("", "nan") else str(value)


def between_cells(
    tidy: pd.DataFrame, metric: str, epoch: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], list[Dropped]]:
    """The metric per group within one epoch.

    Returns ``(values, sessions, dropped)``: each group's values and the
    sessions they came from, in frame order, groups sorted by name with
    Ungrouped last. ``epoch`` None takes the frame as a single epoch.
    """
    rows = tidy if epoch is None else tidy[tidy["range"] == epoch]
    values: dict[str, list[float]] = {}
    sessions: dict[str, list[str]] = {}
    dropped = []
    for _, row in rows.iterrows():
        value = row.get(metric)
        if _missing(value):
            where = f" in {epoch}" if epoch is not None else ""
            dropped.append(Dropped(str(row["session"]), f"no {metric}{where}"))
            continue
        group = _group(row.get("group"))
        values.setdefault(group, []).append(float(value))
        sessions.setdefault(group, []).append(str(row["session"]))
    order = sorted(values, key=lambda g: (g == UNGROUPED, g))
    return (
        {g: np.asarray(values[g], dtype=float) for g in order},
        {g: sessions[g] for g in order},
        dropped,
    )


def within_matrix(
    tidy: pd.DataFrame, metric: str, epochs: list[str], group: str | None = None
) -> tuple[np.ndarray, list[str], list[Dropped]]:
    """The metric as an animals x epochs matrix, complete animals only.

    A within-subject test needs every animal in every epoch, so an animal
    missing any one is left out entirely (listwise), and named. ``group``
    restricts to one group's animals.
    """
    rows = tidy[tidy["range"].isin(epochs)]
    if group is not None:
        rows = rows[rows["group"].map(_group) == group]
    table: dict[str, dict[str, float]] = {}
    for _, row in rows.iterrows():
        cells = table.setdefault(str(row["session"]), {})
        value = row.get(metric)
        cells[str(row["range"])] = float("nan") if _missing(value) else float(value)
    kept, matrix, dropped = [], [], []
    for session, cells in table.items():
        missing = [e for e in epochs if _missing(cells.get(e))]
        if missing:
            dropped.append(Dropped(session, f"no {metric} in {', '.join(missing)}"))
            continue
        kept.append(session)
        matrix.append([cells[e] for e in epochs])
    out = np.asarray(matrix, dtype=float).reshape(len(kept), len(epochs))
    return out, kept, dropped
