"""Per-session 3-D lift, computed once and saved beside the pose CSV for review.

Scrubbing a session cannot re-run three models per frame, so the lift is run over the
whole session once (GPU, roughly real time) and saved as sidecars of the pose CSV:

* ``<pose stem>.lift3d.csv``         per frame: rear weight, body length, back-arch
  eccentricity (surface and depth-only), spine length, arch peak
* ``<pose stem>.lift3d_surface.npy`` ``(n_frames, GRID, GRID)`` float16 height map of
  the 20 cm crop around the mouse (NaN off the mouse), for the 3-D view
* ``<pose stem>.lift3d_spine.npy``   ``(n_frames, 5, 2)`` float32 neck .. tail base in
  grid coordinates (NaN without a neck keypoint), for the arch overlay
* ``<pose stem>.lift3d.json``        what produced them (scale, models, keypoints)

Files are written under temporary names and renamed at the end, so an interrupted run
never looks finished.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .curvature import back_arch
from .pipeline import Lift3D, Lift3DConfig
from .surface import CROP_CM, CROP_PX

GRID = 48  # surface cells per side: 4.2 mm each over the 20 cm crop
FORMAT_VERSION = 1
COLUMNS = [
    "frame",
    "t_s",
    "rear_weight",
    "body_ratio",
    "body_length_cm",
    "spine_length_cm",
    "ecc_surface",
    "ecc_depth",
    "arch_peak",
    "used_depth",
]


@dataclass(frozen=True)
class SidecarPaths:
    table: Path
    surface: Path
    spine: Path
    meta: Path

    @classmethod
    def for_pose(cls, pose_csv: str | Path) -> SidecarPaths:
        p = Path(pose_csv)
        base = p.with_suffix("")
        return cls(
            table=Path(f"{base}.lift3d.csv"),
            surface=Path(f"{base}.lift3d_surface.npy"),
            spine=Path(f"{base}.lift3d_spine.npy"),
            meta=Path(f"{base}.lift3d.json"),
        )

    def all(self) -> tuple[Path, ...]:
        return self.table, self.surface, self.spine, self.meta

    def exist(self) -> bool:
        return all(p.exists() for p in self.all())


def downsample(height: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Block-mean a CROP_PX height map to ``grid`` cells (NaN where a cell is off the mouse)."""
    k = height.shape[0] // grid
    h = height[: k * grid, : k * grid].reshape(grid, k, grid, k)
    valid = np.isfinite(h)
    total = np.where(valid, h, 0.0).sum(axis=(1, 3))
    count = valid.sum(axis=(1, 3))
    out = np.full((grid, grid), np.nan, np.float32)
    keep = count >= (k * k) // 3  # a cell mostly off the mouse is floor
    out[keep] = total[keep] / count[keep]
    return out


def _fmt(v, nd=4):
    return "" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def compute_session(
    video: str | Path,
    pose_csv: str | Path,
    config: Lift3DConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    lift: Lift3D | None = None,
    individual: str | int | None = None,
) -> SidecarPaths | None:
    """Lift every frame of a session and write its sidecars. None if stopped early."""
    import cv2

    from glider.vision.pose.dlc import from_dlc_csv

    pose = from_dlc_csv(pose_csv, individual=individual)
    paths = SidecarPaths.for_pose(pose_csv)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"could not open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or pose.fps or config.fps
    config.fps = fps
    n = pose.n_frames
    lift = lift or Lift3D(pose.keypoint_names, config)
    crop_ppc = CROP_PX / CROP_CM
    to_grid = GRID / CROP_PX

    tmp = {p: p.with_name(p.name + ".partial") for p in paths.all()}
    surface = np.lib.format.open_memmap(
        tmp[paths.surface], mode="w+", dtype=np.float16, shape=(n, GRID, GRID)
    )
    spine = np.lib.format.open_memmap(
        tmp[paths.spine], mode="w+", dtype=np.float32, shape=(n, 5, 2)
    )
    surface[:] = np.nan
    spine[:] = np.nan
    finished = False
    has_neck = lift.roles.neck is not None
    try:
        with open(tmp[paths.table], "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            for i in range(n):
                if should_stop is not None and should_stop():
                    return None
                ok, frame = cap.read()
                if not ok:
                    break
                r = lift.process(frame, pose.xy[i], frame_idx=i)
                if r is None:
                    w.writerow([i, f"{i / fps:.4f}"] + [""] * (len(COLUMNS) - 2))
                else:
                    arch = back_arch(r, crop_ppc)
                    surface[i] = downsample(r.height).astype(np.float16)
                    if r.spine_c is not None and len(r.spine_c) == 5:
                        spine[i] = r.spine_c * to_grid
                    w.writerow(
                        [
                            i,
                            f"{i / fps:.4f}",
                            _fmt(r.rear_weight),
                            _fmt(r.body_ratio),
                            _fmt(r.body_length_cm, 3),
                            _fmt(arch["spine_length_cm"], 3),
                            _fmt(arch["ecc_surface"]),
                            _fmt(arch["ecc_depth"]),
                            _fmt(arch["arch_peak"], 3),
                            int(r.used_depth),
                        ]
                    )
                if progress is not None:
                    progress(i + 1, n)
        surface.flush()
        spine.flush()
        meta = {
            "format": FORMAT_VERSION,
            "video": str(video),
            "pose_csv": str(pose_csv),
            "fps": fps,
            "n_frames": n,
            "ppc": config.ppc,
            "grid": GRID,
            "crop_cm": CROP_CM,
            "keypoint_names": list(pose.keypoint_names),
            "has_neck": has_neck,
            "used_mask_model": lift.segment is not None,
            "used_depth": lift.depth is not None,
        }
        tmp[paths.meta].write_text(json.dumps(meta, indent=2), encoding="utf-8")
        finished = True
    finally:
        cap.release()
        lift.close()
        del surface, spine  # Windows cannot rename a file that is still mapped
        for final, partial in tmp.items():
            if finished:
                os.replace(partial, final)
            else:
                partial.unlink(missing_ok=True)
    return paths


@dataclass
class LiftSession:
    """A session's saved lift, read for review."""

    table: object  # pandas.DataFrame indexed by frame
    surface: np.ndarray  # memory-mapped (n, GRID, GRID)
    spine: np.ndarray  # memory-mapped (n, 5, 2)
    meta: dict

    @property
    def n_frames(self) -> int:
        return len(self.surface)

    @property
    def cell_cm(self) -> float:
        return float(self.meta.get("crop_cm", CROP_CM)) / int(self.meta.get("grid", GRID))


def load_session(pose_csv: str | Path) -> LiftSession | None:
    """The saved lift for this pose CSV, or None if it has not been computed."""
    import pandas as pd

    paths = SidecarPaths.for_pose(pose_csv)
    if not paths.exist():
        return None
    table = pd.read_csv(paths.table).set_index("frame")
    return LiftSession(
        table=table,
        surface=np.load(paths.surface, mmap_mode="r"),
        spine=np.load(paths.spine, mmap_mode="r"),
        meta=json.loads(paths.meta.read_text(encoding="utf-8")),
    )
