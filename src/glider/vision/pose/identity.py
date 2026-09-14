"""Saying when a slot id is a guess.

Consolidation does not merely observe identity, it *infers* it: fragments are
joined across gaps on the strength of a plausible speed. That is a real
improvement over dropping them, and it is also exactly where a swap gets baked
in. So the frames where it happened are recorded.

The file is sparse -- a row exists only where there is doubt. Absent means
measured and unambiguous, so a left join from the pose CSV fills NaN and should
be read as ``""``. Writing every frame for every animal would be a hundred
thousand rows of mostly nothing for a half-hour recording of two mice.

The column is called ``identity_flag`` after the live tracking CSV column in the
shelved top-down work, so that if that is ever revived the two files join on one
vocabulary. Nothing else writes that column today -- the name is forward-looking,
not a consistency that already exists.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from glider.vision.pose.tracks import PoseTracks

__all__ = ["identity_flags", "identity_output_path", "write_identity_csv"]

#: Ordered so a multi-cause flag is deterministic and therefore testable.
_FLAG_ORDER = ("close", "stitched")


def _centroids(pose) -> np.ndarray:
    """``(F, 2)`` mean keypoint position per frame, NaN where nothing localized.

    Written out rather than ``np.nanmean`` because an all-NaN frame is normal
    here -- a dropout -- and nanmean emits a RuntimeWarning for every one of
    them. Keypoints are NaN'd in both coordinates together, so counting the
    finite x values is enough.
    """
    xy = pose.xy
    finite = np.isfinite(xy[:, :, 0])
    counts = finite.sum(axis=1)
    sums = np.nansum(xy, axis=1)
    safe = np.maximum(counts, 1)[:, None]
    return np.where(counts[:, None] > 0, sums / safe, np.nan)


def identity_flags(
    tracks: PoseTracks,
    *,
    stitched: dict[int, set[int]],
    min_separation_px: float = 60.0,
) -> list[tuple[int, int, str]]:
    """``(frame, slot, flag)`` for every frame where identity is in doubt."""
    per_slot = np.stack([_centroids(tracks[slot]) for slot in tracks])  # (N, F, 2)
    rows: list[tuple[int, int, str]] = []

    for frame in range(tracks.n_frames):
        points = per_slot[:, frame, :]
        for slot in range(tracks.n_animals):
            here = points[slot]
            if not np.all(np.isfinite(here)):
                # No position, so it cannot be near anything and the id is not
                # standing on anything either. One cause, not three.
                rows.append((frame, slot, "gap"))
                continue

            causes = set()
            others = np.delete(points, slot, axis=0)
            if others.size:
                distances = np.linalg.norm(others - here, axis=1)
                if np.any(distances[np.isfinite(distances)] < min_separation_px):
                    causes.add("close")
            if frame in stitched.get(slot, ()):
                causes.add("stitched")

            if causes:
                flag = "+".join(c for c in _FLAG_ORDER if c in causes)
                rows.append((frame, slot, flag))

    return rows


def identity_output_path(pose_csv: Path | str) -> Path:
    """``<stem>_identity.csv`` beside the pose CSV it describes."""
    pose_csv = Path(pose_csv)
    return pose_csv.with_name(f"{pose_csv.stem}_identity.csv")


def write_identity_csv(path: Path | str, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "individual", "identity_flag"])
        for frame, slot, flag in rows:
            writer.writerow([frame, f"animal{slot}", flag])
    return path
