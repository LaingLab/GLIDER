"""Offline 3-D lift of a recorded video from its pose CSV, written as a per-frame CSV.

python -m glider_lift3d VIDEO --pose-csv POSES.csv --ppc 12.7 [--arena ...]
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .keypoints import ANCHOR_NAMES
from .pipeline import Lift3D, Lift3DConfig, Lift3DFrame

CSV_COLUMNS = [
    "frame",
    "t_s",
    "rear_weight",
    "rear_score",
    "body_ratio",
    "body_length_cm",
    *(f"{a}_height" for a in ANCHOR_NAMES),
    "used_mask_model",
    "used_depth",
    "floor_resid_cm",
]


def lift_csv_path(video: str | Path) -> Path:
    v = Path(video)
    return v.with_name(f"{v.stem}_lift3d.csv")


def frame_row(frame_idx: int, fps: float, r: Lift3DFrame | None) -> list:
    if r is None:
        return [frame_idx, f"{frame_idx / fps:.4f}"] + [""] * (len(CSV_COLUMNS) - 2)

    def f(v, nd=4):
        return "" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"

    return [
        frame_idx,
        f"{frame_idx / fps:.4f}",
        f(r.rear_weight),
        f(r.rear_score),
        f(r.body_ratio),
        f(r.body_length_cm, 3),
        *(f(z, 3) for z in r.anchors_cm),
        int(r.used_mask_model),
        int(r.used_depth),
        f(r.floor_resid_cm, 3),
    ]


def run_video(
    video: str | Path,
    pose_csv: str | Path,
    config: Lift3DConfig,
    out_csv: str | Path | None = None,
    start: int = 0,
    n_frames: int | None = None,
    individual: str | int | None = None,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    lift: Lift3D | None = None,
) -> Path:
    """Lift every frame of ``video`` using the poses in ``pose_csv``; return the CSV path.

    Pose CSV rows are positional (row N = frame N). Frames without a usable pose get an
    empty row so the output stays frame-aligned with the video and the pose CSV.
    """
    import cv2

    from glider.vision.pose.dlc import from_dlc_csv

    pose = from_dlc_csv(pose_csv, individual=individual)
    out = Path(out_csv) if out_csv else lift_csv_path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"could not open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.fps
    config.fps = fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or pose.n_frames
    stop = total if n_frames is None else min(total, start + n_frames)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    lift = lift or Lift3D(pose.keypoint_names, config)
    try:
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_COLUMNS)
            for i in range(start, stop):
                if should_stop is not None and should_stop():
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                r = None
                if i < pose.n_frames:
                    r = lift.process(frame, pose.xy[i], frame_idx=i)
                w.writerow(frame_row(i, fps, r))
                if progress is not None:
                    progress(i - start + 1, stop - start)
    finally:
        cap.release()
        lift.close()
    return out


def parse_arena(text: str | None) -> np.ndarray | None:
    """``"x,y;x,y;x,y;x,y"`` -> (N, 2) array."""
    if not text:
        return None
    pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";") if p.strip()]
    if len(pts) < 3 or any(len(p) != 2 for p in pts):
        raise ValueError("arena needs at least 3 'x,y' points separated by ';'")
    return np.array(pts, float)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m glider_lift3d", description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--pose-csv", required=True, help="DLC-format pose CSV for the video")
    ap.add_argument("--ppc", type=float, required=True, help="pixels per cm at the floor")
    ap.add_argument("--arena", help="floor corners 'x,y;x,y;x,y;x,y' in image px")
    ap.add_argument("--out", help="output CSV (default: VIDEO_lift3d.csv)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=None, help="frames to process")
    ap.add_argument("--no-depth", action="store_true", help="skip the depth model")
    ap.add_argument("--no-mask", action="store_true", help="skip the mask model")
    ap.add_argument("--mask-model")
    ap.add_argument("--depth-model")
    ap.add_argument("--device")
    ap.add_argument("--individual", help="animal name/slot in a multi-animal CSV")
    a = ap.parse_args(argv)
    cfg = Lift3DConfig(
        ppc=a.ppc,
        arena_corners=parse_arena(a.arena),
        mask_model=a.mask_model,
        depth_model=a.depth_model,
        use_mask=not a.no_mask,
        use_depth=not a.no_depth,
        device=a.device,
    )

    def progress(done, total):
        if done % 300 == 0 or done == total:
            print(f"  {done}/{total} frames", flush=True)

    out = run_video(
        a.video,
        a.pose_csv,
        cfg,
        out_csv=a.out,
        start=a.start,
        n_frames=a.n,
        individual=a.individual,
        progress=progress,
    )
    print(f"wrote {out}")
    return 0
