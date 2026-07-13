"""Pure load/save of the dashboard layout to ~/.glider/dashboard_layout.json.

Any problem (missing file, corrupt JSON, unknown/duplicate panel keys, missing
quadrants) resolves to the default layout rather than raising — a bad layout
file must never block the dashboard from opening.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from glider.core.config import get_config
from glider.gui.dashboard.layout import (
    QUADRANTS,
    DashboardLayout,
    default_layout,
)
from glider.gui.dashboard.panel_registry import PANEL_KEYS

logger = logging.getLogger(__name__)


def _default_path() -> Path:
    return get_config().paths.user_config_dir / "dashboard_layout.json"


def to_dict(layout: DashboardLayout) -> dict[str, Any]:
    return {
        "assignment": dict(layout.assignment),
        "outer_sizes": list(layout.outer_sizes),
        "left_sizes": list(layout.left_sizes),
        "right_sizes": list(layout.right_sizes),
    }


def _valid_assignment(assignment: Any) -> bool:
    if not isinstance(assignment, dict):
        return False
    if set(assignment.keys()) != set(QUADRANTS):
        return False
    values = list(assignment.values())
    if any(v not in PANEL_KEYS for v in values):
        return False
    if len(set(values)) != len(QUADRANTS):  # no duplicates
        return False
    return True


def from_dict(data: Any) -> DashboardLayout:
    if not isinstance(data, dict) or not _valid_assignment(data.get("assignment")):
        return default_layout()

    def _sizes(key: str) -> tuple[int, ...]:
        raw = data.get(key, [])
        if isinstance(raw, list) and all(isinstance(n, int) for n in raw):
            return tuple(raw)
        return ()

    return DashboardLayout(
        assignment=dict(data["assignment"]),
        outer_sizes=_sizes("outer_sizes"),
        left_sizes=_sizes("left_sizes"),
        right_sizes=_sizes("right_sizes"),
    )


def load_layout(path: Path | None = None) -> DashboardLayout:
    path = path or _default_path()
    if not path.exists():
        return default_layout()
    try:
        with open(path) as f:
            data = json.load(f)
        return from_dict(data)
    except Exception as e:  # noqa: BLE001 - any read/parse/shape error falls back
        logger.warning("Failed to load dashboard layout from %s: %s", path, e)
        return default_layout()


def save_layout(layout: DashboardLayout, path: Path | None = None) -> None:
    path = path or _default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_dict(layout), f, indent=2)
