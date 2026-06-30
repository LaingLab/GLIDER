"""DeepLabCut format writer.

Produces the canonical 3-row-header CSV that B-SOID, VAME, Keypoint-MoSeq,
SimBA, and DLC's own analysis tools expect:

    scorer,my_yolo,my_yolo,my_yolo,...
    bodyparts,snout,snout,snout,left_ear,...
    coords,x,y,likelihood,x,y,likelihood,...
    0,412.3,288.1,0.97,...

The companion ``to_dlc_h5`` writer emits the same data as a pandas-flavor
HDF5 file, which is what DLC saves natively (``*_DLC_resnet50_*.h5``) — some
downstream tools (notably DLC's own ``analyze_videos`` post-processors) prefer
this form.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from glider.vision.pose.core import PoseData


def _build_dataframe(pose: PoseData) -> pd.DataFrame:
    """Build the DLC multi-index DataFrame (no I/O)."""
    n_frames = pose.n_frames
    n_kpts = pose.n_keypoints

    # Interleave x, y, likelihood per body part.
    flat = np.empty((n_frames, n_kpts * 3), dtype=float)
    flat[:, 0::3] = pose.xy[:, :, 0]
    flat[:, 1::3] = pose.xy[:, :, 1]
    flat[:, 2::3] = pose.confidence

    columns = pd.MultiIndex.from_product(
        [[pose.source], pose.keypoint_names, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(flat, columns=columns)
    # DLC convention: index has no name (so pd.to_csv emits exactly 3 header rows).
    return df


def to_dlc_csv(pose: PoseData, path: str | Path) -> Path:
    """Write a DeepLabCut-format CSV.

    Returns the resolved path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _build_dataframe(pose)
    df.to_csv(path)
    return path


def to_dlc_h5(pose: PoseData, path: str | Path, *, key: str = "df_with_missing") -> Path:
    """Write a DeepLabCut-style HDF5 (pandas pytables format).

    Some downstream tools require this rather than CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _build_dataframe(pose)
    # DLC stores the table under a fixed key with format='table' so it can be
    # appended to / queried; we mirror that.
    df.to_hdf(path, key=key, mode="w", format="table")
    return path


def from_dlc_csv(path: str | Path, *, fps: float = 30.0) -> PoseData:
    """Read a DLC-format CSV back into a PoseData.

    Useful for round-trip tests and for chaining with downstream filters.
    """
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    # Header shape: (scorer, bodyparts, coords)
    scorer_levels = df.columns.get_level_values("scorer").unique().tolist()
    if len(scorer_levels) != 1:
        raise ValueError(f"expected a single scorer column, got: {scorer_levels}")
    scorer = scorer_levels[0]

    bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
    n_frames = len(df)
    n_kpts = len(bodyparts)

    xy = np.empty((n_frames, n_kpts, 2), dtype=float)
    cf = np.empty((n_frames, n_kpts), dtype=float)
    for i, bp in enumerate(bodyparts):
        xy[:, i, 0] = df[(scorer, bp, "x")].to_numpy()
        xy[:, i, 1] = df[(scorer, bp, "y")].to_numpy()
        cf[:, i] = df[(scorer, bp, "likelihood")].to_numpy()

    return PoseData(
        xy=xy,
        confidence=cf,
        keypoint_names=bodyparts,
        fps=fps,
        source=scorer,
    )
