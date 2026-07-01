"""Egocentric residual-motion features from the source video.

Dig and grooming are "obvious on video, invisible to the 7-keypoint
skeleton": the discriminating motion is in the forepaws / displaced
substrate, which aren't tracked. This module recovers that signal WITHOUT
paw keypoints by measuring how much the *image* changes frame-to-frame
after the mouse's own body motion is removed.

Per frame we warp the grayscale video frame into a fixed egocentric patch
(body center -> patch center, body axis -> +x, scaled by body length) with
the SAME translate/rotate/scale convention as
:mod:`glider.analysis.behavior.sequence`, then difference consecutive patches.
Because the body is registered, what's left is limb + substrate motion
*relative to the body*. Emitted per frame:

* ``motion_total``     - mean abs patch diff (residual motion energy)
* ``motion_anterior``  - same over the head-half of the patch (groom/dig)
* ``motion_posterior`` - same over the tail-half
* ``motion_spread``    - fraction of patch pixels whose diff exceeds a
  threshold (digging displaces a *wide* substrate area; grooming is local)

Limitation: rotation-compensation makes static bedding appear to move when
the body rotates. During dig/groom the body is near-stationary (that's why
they're confusable), so the diff is clean there; the artifact only shows on
high-rotation frames. Dropping rotation (translation-only) is a one-line
change if it adds noise.

Caching: computing these means decoding the whole video, so results are
cached to a per-session sidecar CSV keyed by the feature params; downstream
CV re-runs read the cache instead of re-decoding.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MOTION_COLUMNS: list[str] = [
    "motion_total",
    "motion_anterior",
    "motion_posterior",
    "motion_spread",
]

# Defaults. out_size px patch spanning `cover` body-lengths; thresh is the
# per-pixel abs-diff (0-255 intensity) above which a pixel counts as "moved".
OUT_SIZE = 160
COVER = 2.0
THRESH = 12.0
CACHE_DIR = Path("motion_cache")

_EPS = 1e-6


def video_path_for(pose_csv: Path) -> Path:
    """Resolve the source video for a pose CSV by project convention:
    ``.../poses/<stem>.csv`` -> ``.../videos/<stem>.mp4`` (a sibling
    ``videos/`` dir). Works for both the local repo layout and the Z: share.
    """
    pose_csv = Path(pose_csv)
    return pose_csv.parent.parent / "videos" / f"{pose_csv.stem}.mp4"


def egocentric_patch(
    gray: np.ndarray,
    center: np.ndarray,
    angle: float,
    body_len: float,
    out_size: int = OUT_SIZE,
    cover: float = COVER,
) -> np.ndarray:
    """Warp ``gray`` into an ``out_size`` x ``out_size`` egocentric patch.

    The body center lands at the patch center, the body axis aligns to +x,
    and ``cover`` body-lengths span the patch width. Same similarity-frame
    convention as :func:`glider.analysis.behavior.sequence.egocentric_batch`.
    """
    k = out_size / (cover * body_len)
    c, s = np.cos(angle), np.sin(angle)
    # Rotate by -angle so the body axis (head->tail) lands on +x, then scale.
    rot_mat = k * np.array([[c, s], [-s, c]], dtype=np.float64)
    b = np.array([out_size / 2.0, out_size / 2.0]) - rot_mat @ np.asarray(center, float)
    warp_mat = np.array(
        [[rot_mat[0, 0], rot_mat[0, 1], b[0]], [rot_mat[1, 0], rot_mat[1, 1], b[1]]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        gray.astype(np.float32),
        warp_mat,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def diff_features(prev: np.ndarray, cur: np.ndarray, thresh: float = THRESH) -> np.ndarray:
    """``(prev, cur)`` egocentric patches -> :data:`MOTION_COLUMNS` values.

    Head is the -x (left) half of the patch by the egocentric convention
    (axis = tail - head lands on +x), so ``motion_anterior`` is the left half.
    """
    d = np.abs(cur - prev)
    half = d.shape[1] // 2
    return np.array(
        [
            float(d.mean()),  # motion_total
            float(d[:, :half].mean()),  # motion_anterior (head side = left)
            float(d[:, half:].mean()),  # motion_posterior
            float((d > thresh).mean()),  # motion_spread
        ]
    )


def compute_motion_for_video(
    video_path: Path,
    xy: np.ndarray,
    body_axis: tuple[int, int],
    *,
    out_size: int = OUT_SIZE,
    cover: float = COVER,
    thresh: float = THRESH,
    progress=None,
) -> pd.DataFrame:
    """Stream ``video_path`` and compute per-frame motion features.

    ``xy`` is the session pose ``(n_frames, K, 2)``; video frame ``i`` is
    assumed to align with pose frame ``i`` (verified for this project). A
    frame whose body-axis keypoints are NaN (or zero-length) resets the diff
    chain — we never difference across a gap. The first valid frame of each
    run is NaN (no predecessor). Returns a DataFrame of length ``len(xy)``.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    head, tail = int(body_axis[0]), int(body_axis[1])
    out = np.full((n, len(MOTION_COLUMNS)), np.nan)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    prev_patch = None
    try:
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break  # video shorter than pose; rest stays NaN
            h, t = xy[i, head], xy[i, tail]
            if not (np.isfinite(h).all() and np.isfinite(t).all()):
                prev_patch = None
                continue
            axis = t - h
            blen = float(np.hypot(axis[0], axis[1]))
            if blen < _EPS:
                prev_patch = None
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            patch = egocentric_patch(
                gray, 0.5 * (h + t), float(np.arctan2(axis[1], axis[0])), blen, out_size, cover
            )
            if prev_patch is not None:
                out[i] = diff_features(prev_patch, patch, thresh)
            prev_patch = patch
            if progress and i and i % 5000 == 0:
                progress(i, n)
    finally:
        cap.release()
    return pd.DataFrame(out, columns=MOTION_COLUMNS)


def load_or_compute_motion(
    pose_csv: Path,
    video_path: Path,
    xy: np.ndarray,
    body_axis: tuple[int, int],
    *,
    out_size: int = OUT_SIZE,
    cover: float = COVER,
    thresh: float = THRESH,
    cache_dir: Path = CACHE_DIR,
    progress=None,
) -> pd.DataFrame:
    """Return cached motion features for a session, computing + caching on miss.

    The cache filename encodes the feature params, so changing any of them
    forces a recompute instead of silently reusing a stale sidecar.
    """
    pose_csv = Path(pose_csv)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{pose_csv.stem}_o{out_size}_c{cover:g}_t{thresh:g}.csv"
    cache_file = cache_dir / key

    n = len(np.asarray(xy))
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0)
        if len(df) == n and all(c in df.columns for c in MOTION_COLUMNS):
            return df[MOTION_COLUMNS].reset_index(drop=True)
        # stale (length/columns changed) -> fall through and recompute

    if not Path(video_path).exists():
        raise FileNotFoundError(
            f"motion features need the source video, but {video_path} is "
            f"missing (pose {pose_csv.name})."
        )
    df = compute_motion_for_video(
        video_path,
        xy,
        body_axis,
        out_size=out_size,
        cover=cover,
        thresh=thresh,
        progress=progress,
    )
    df.to_csv(cache_file)
    return df
