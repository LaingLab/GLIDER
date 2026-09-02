"""Sanity-check overlay: draw keypoints + names on a few frames.

Why this exists:

The single most common bug when bridging YOLO -> DLC is body-part order
mismatch. Your model emits the keypoints in the order they were trained,
but if your ``keypoint_names`` list to ``infer_video`` is in a different
order, every downstream cluster will be silently wrong.

Run ``overlay_frames(pose, video_path, "out.png")`` on the first few frames
of one video and eyeball it before processing 100 sessions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from glider.vision.pose.core import PoseData
from glider.vision.video_source import ExactFrameReader

# Reasonably distinct colors (BGR for OpenCV).
_PALETTE = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (128, 0, 255),
    (0, 128, 255),
    (255, 128, 0),
    (128, 255, 0),
    (0, 255, 128),
]


def _color_for(i: int) -> tuple[int, int, int]:
    return _PALETTE[i % len(_PALETTE)]


def overlay_frames(
    pose: PoseData,
    video_path: str | Path,
    output_path: str | Path,
    *,
    frame_indices: Sequence[int] | None = None,
    n_frames: int = 6,
    radius: int = 4,
    label_kpts: bool = True,
) -> Path:
    """Render selected frames with keypoints overlaid into a single PNG grid.

    Parameters
    ----------
    pose
        Pose data to overlay. Must come from the same video.
    video_path
        Source video.
    output_path
        Where to write the PNG grid.
    frame_indices
        Frames to render. If ``None``, evenly samples ``n_frames`` across the
        video.
    n_frames
        Number of frames to sample when ``frame_indices`` is ``None``.
    radius
        Keypoint dot radius in pixels.
    label_kpts
        If True, write the body-part name next to each keypoint.
    """
    import cv2

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or pose.n_frames

    if frame_indices is None:
        n = min(n_frames, pose.n_frames, total)
        frame_indices = np.linspace(0, min(pose.n_frames, total) - 1, n, dtype=int).tolist()
    frame_indices = [int(i) for i in frame_indices]

    # Exact frame access: this pairs pose with the image by index, and a
    # long-GOP seek lands several frames off, which draws the skeleton on a
    # frame the animal has already left.
    reader = ExactFrameReader(cap)
    rendered: list[np.ndarray] = []
    for idx in frame_indices:
        frame = reader.read(idx)
        if frame is None:
            continue
        if idx >= pose.n_frames:
            continue
        for k, name in enumerate(pose.keypoint_names):
            x, y = pose.xy[idx, k]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            xi, yi = int(round(x)), int(round(y))
            color = _color_for(k)
            cv2.circle(frame, (xi, yi), radius, color, -1)
            cv2.circle(frame, (xi, yi), radius + 1, (0, 0, 0), 1)
            if label_kpts:
                cv2.putText(
                    frame,
                    name,
                    (xi + radius + 2, yi - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(
            frame,
            f"frame {idx}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered.append(frame)

    cap.release()
    if not rendered:
        raise RuntimeError(
            f"no frames rendered (requested {frame_indices}); " f"check pose/video alignment"
        )

    grid = _arrange_grid(rendered)
    cv2.imwrite(str(output_path), grid)
    return output_path


def _arrange_grid(frames: list[np.ndarray]) -> np.ndarray:
    """Stack frames into a roughly square grid, padding the last row if needed."""
    import cv2

    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = frames[0].shape[:2]
    target_h, target_w = h, w

    cells = []
    for f in frames:
        if f.shape[:2] != (target_h, target_w):
            f = cv2.resize(f, (target_w, target_h))
        cells.append(f)
    blank = np.zeros_like(cells[0])
    while len(cells) < rows * cols:
        cells.append(blank)

    rows_imgs = []
    for r in range(rows):
        row = np.hstack(cells[r * cols : (r + 1) * cols])
        rows_imgs.append(row)
    return np.vstack(rows_imgs)
