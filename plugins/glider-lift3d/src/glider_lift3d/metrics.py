"""Back-arch and posture measures per range, for Session Review's epoch table and exports.

Registered under ``glider.range_metrics``: every range GLIDER computes (an epoch, the
current selection, each time bin of *Export -> Time bins*) gets these columns for a
session whose 3-D lift has been computed (``<pose>.lift3d/frames.csv``). A session
without one gets none, and the columns are offered only when some session has them.

Means are over the frames in the range that have a value. The arch eccentricities are
blank while rearing and without a neck keypoint, so ``lift3d_arch_frames`` says how
many frames a mean stands on: a minute spent mostly rearing averages very few.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .curvature import MAX_REAR_WEIGHT
from .session import SidecarPaths, migrate_legacy

#: (key, label, decimals, frames.csv column or None for a derived measure)
_SPEC = (
    ("lift3d_arch_e_surface", "Arch e, surface (3-D)", 3, "ecc_surface"),
    ("lift3d_arch_e_depth", "Arch e, depth only (3-D)", 3, "ecc_depth"),
    ("lift3d_arch_frames", "Arch frames (3-D)", 0, None),
    ("lift3d_body_length_cm", "Body length (cm, 3-D)", 2, "body_length_cm"),
    ("lift3d_spine_length_cm", "Spine length (cm, 3-D)", 2, "spine_length_cm"),
    ("lift3d_rear_fraction", "Rearing fraction (3-D)", 3, None),
)


class Lift3DMetrics:
    name = "lift3d"

    def __init__(self):
        # pose path -> (frames.csv mtime, columns as arrays indexed by frame)
        self._tables: dict[Path, tuple[float, dict[str, np.ndarray]]] = {}

    def metrics(self):
        from glider.analysis.epochs import Metric

        return [Metric(key, label, decimals) for key, label, decimals, _ in _SPEC]

    def _table(self, pose_path) -> dict[str, np.ndarray] | None:
        if pose_path is None:
            return None
        pose_path = Path(pose_path)
        migrate_legacy(pose_path)
        table = SidecarPaths.for_pose(pose_path).table
        try:
            mtime = table.stat().st_mtime
        except OSError:
            self._tables.pop(pose_path, None)
            return None
        cached = self._tables.get(pose_path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        import pandas as pd

        df = pd.read_csv(table).set_index("frame").sort_index()
        n = int(df.index.max()) + 1 if len(df) else 0
        cols = {}
        for name in (
            "ecc_surface",
            "ecc_depth",
            "body_length_cm",
            "spine_length_cm",
            "rear_weight",
        ):
            arr = np.full(n, np.nan)
            if name in df:
                arr[df.index.to_numpy(int)] = pd.to_numeric(df[name], errors="coerce").to_numpy()
            cols[name] = arr
        self._tables[pose_path] = (mtime, cols)
        return cols

    def values(self, view, start_frame: int, end_frame: int) -> dict[str, float]:
        cols = self._table(getattr(view, "pose_path", None))
        if cols is None:
            return {}
        lo, hi = max(int(start_frame), 0), int(end_frame) + 1
        sl = {k: v[lo:hi] for k, v in cols.items()}

        def mean(a):
            a = a[np.isfinite(a)]
            return float(a.mean()) if a.size else math.nan

        rear = sl["rear_weight"]
        seen = rear[np.isfinite(rear)]
        out = {}
        for key, _label, _d, column in _SPEC:
            if column is not None:
                out[key] = mean(sl[column])
        out["lift3d_arch_frames"] = float(np.isfinite(sl["ecc_surface"]).sum())
        out["lift3d_rear_fraction"] = (
            float((seen >= MAX_REAR_WEIGHT).mean()) if seen.size else math.nan
        )
        return out


def provider() -> Lift3DMetrics:
    return Lift3DMetrics()
