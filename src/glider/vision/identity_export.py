"""Re-lay a live identity run's CSVs in the Batch Pose layout at Stop.

The offline per-animal tools (behaviour analysis, Session Review) read DLC
CSVs found beside the video. This writes the same files a Batch Pose run would,
so a recording made with identity on needs no special handling downstream.

Rows are placed on video frames by capture timestamp, not by the logger's
frame counter: each row goes to the video frame captured nearest its
timestamp, and is dropped when no frame lies within half a frame interval
(rows logged while recording was paused, or before/after the video).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from glider.analysis._io import parse_csv
from glider.vision.pose.batch import animals_dir, dlc_output_path
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import to_dlc_csv
from glider.vision.pose.identity import identity_output_path, write_identity_csv

logger = logging.getLogger(__name__)


def export_live_identity(
    tracking_csv: Path,
    keypoints_csv: Path,
    video: Path,
    model: Path,
    names: list[str],
    keypoint_names: list[str],
    fps: float,
    frame_ts,
) -> list[Path]:
    """Write one DLC CSV per named animal (plus identity sidecar); never raises."""
    try:
        return _export(
            Path(tracking_csv),
            Path(keypoints_csv),
            Path(video),
            Path(model),
            names,
            keypoint_names,
            fps,
            np.asarray(frame_ts, dtype=float),
        )
    except Exception:
        logger.exception("Could not export live identity run for %s", video)
        return []


def _video_frames(timestamps, frame_ts: np.ndarray, fps: float) -> np.ndarray:
    """0-based video frame of each logger ISO timestamp (naive local time), -1 if none.

    ``frame_ts`` is the sorted capture time of every video frame.
    """
    import pandas as pd

    ref = frame_ts[0]
    since = pd.to_datetime(timestamps) - pd.Timestamp(datetime.fromtimestamp(ref))
    t = since.dt.total_seconds().to_numpy() + ref
    step = np.median(np.diff(frame_ts)) if len(frame_ts) > 1 else 1.0 / fps
    hi = np.clip(np.searchsorted(frame_ts, t), 0, len(frame_ts) - 1)
    lo = np.maximum(hi - 1, 0)
    nearest = np.where(np.abs(frame_ts[hi] - t) < np.abs(t - frame_ts[lo]), hi, lo)
    return np.where(np.abs(frame_ts[nearest] - t) <= step / 2, nearest, -1)


def _export(tracking_csv, keypoints_csv, video, model, names, keypoint_names, fps, frame_ts):
    import pandas as pd

    n_frames = len(frame_ts)
    n_animals, n_kp = len(names), len(keypoint_names)
    kp_index = {k: i for i, k in enumerate(keypoint_names)}
    xy = np.full((n_animals, n_frames, n_kp, 2), np.nan)
    conf = np.full((n_animals, n_frames, n_kp), np.nan)

    kp = pd.read_csv(keypoints_csv)
    kp["vf"] = _video_frames(kp["timestamp"], frame_ts, fps)
    kp = kp[(kp["object_id"] < n_animals) & (kp["vf"] >= 0) & (kp["vf"] < n_frames)]
    for frame, oid, name, x, y, c in zip(
        kp["vf"],
        kp["object_id"],
        kp["keypoint"],
        kp["x"],
        kp["y"],
        kp["confidence"],
        strict=True,
    ):
        k = kp_index.get(name)
        if k is not None:
            xy[int(oid), int(frame), k] = (x, y)
            conf[int(oid), int(frame), k] = c

    _, tracking = parse_csv(tracking_csv)
    slot_of = {n: i for i, n in enumerate(names)}
    seen = [set() for _ in names]  # 0-based frames with a row
    flags: list[tuple[int, int, str]] = []
    if "animal" in tracking.columns:
        frames = _video_frames(tracking["timestamp"], frame_ts, fps)
        for frame, animal, flag in zip(
            frames, tracking["animal"], tracking["identity_flag"], strict=True
        ):
            slot = slot_of.get(animal)
            if slot is None or not 0 <= frame < n_frames:
                continue
            seen[slot].add(int(frame))
            if isinstance(flag, str) and flag:
                flags.append((int(frame), slot, flag))
    for slot in range(n_animals):
        flags += [(f, slot, "gap") for f in range(n_frames) if f not in seen[slot]]
    flags.sort()

    primary = dlc_output_path(video, model)
    poses = [
        PoseData(xy=xy[i], confidence=conf[i], keypoint_names=keypoint_names, fps=fps)
        for i in range(n_animals)
    ]
    if n_animals == 1:
        return [to_dlc_csv(poses[0], primary)]
    out_dir = animals_dir(primary)
    written = [to_dlc_csv(p, out_dir / f"animal{i}.csv") for i, p in enumerate(poses)]
    written.append(write_identity_csv(identity_output_path(primary), flags))
    return written
