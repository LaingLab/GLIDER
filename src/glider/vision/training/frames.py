"""Choose frames worth labelling and write them into a project.

Evenly spaced frames are the simple choice, but a session is mostly a mouse
sitting still, so most of them look alike. ``"diverse"`` reads a larger evenly
spaced pool and keeps the frames that differ most from those already chosen
(farthest-point sampling on small greyscale thumbnails), which is what a
labeller would pick by hand: rearing, grooming, turning, near each wall.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from glider.core.fileio import atomic_write_bytes
from glider.vision.training.project import FrameRecord, StudioProject
from glider.vision.video_source import ExactFrameReader

logger = logging.getLogger(__name__)

__all__ = [
    "METHODS",
    "ExtractionCancelledError",
    "diverse_indices",
    "extract_frames",
    "uniform_indices",
]

METHODS = ("diverse", "uniform")

#: How many candidates "diverse" reads for each frame it keeps.
_POOL_FACTOR = 6
#: Thumbnail edge for comparing frames: enough to see posture, small enough
#: that camera noise does not dominate.
_THUMB = 24
_JPEG_QUALITY = 95


class ExtractionCancelledError(Exception):
    """The caller asked extraction to stop."""


def uniform_indices(frame_count: int, n: int) -> list[int]:
    """``n`` evenly spaced frame indices, avoiding the very first and last frame."""
    if frame_count <= 0 or n <= 0:
        return []
    n = min(n, frame_count)
    step = frame_count / n
    return sorted({min(frame_count - 1, int(step * (i + 0.5))) for i in range(n)})


def diverse_indices(thumbs: np.ndarray, indices: list[int], n: int, seed: int = 0) -> list[int]:
    """Pick ``n`` of ``indices`` whose thumbnails differ most from each other.

    Args:
        thumbs: One flattened thumbnail per index, shape ``(len(indices), D)``.
        indices: The frame index each thumbnail came from.
        n: How many to keep.
        seed: Chooses the first pick; the rest follow deterministically.
    """
    if n >= len(indices):
        return sorted(indices)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(indices)))
    picks = [first]
    d2 = np.sum((thumbs - thumbs[first]) ** 2, axis=1)
    for _ in range(n - 1):
        d2[picks] = -1.0
        nxt = int(np.argmax(d2))
        picks.append(nxt)
        d2 = np.minimum(d2, np.sum((thumbs - thumbs[nxt]) ** 2, axis=1))
    return sorted(indices[i] for i in picks)


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(grey, (_THUMB, _THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Normalise brightness so a light flicker does not count as a new posture.
    small -= small.mean()
    norm = float(np.linalg.norm(small))
    return (small / norm if norm > 0 else small).ravel()


def _write_jpeg(path: Path, frame: np.ndarray) -> None:
    # imencode + write rather than imwrite: imwrite fails on non-ASCII paths on
    # Windows, and lab shares are full of them.
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise OSError(f"could not encode {path.name}")
    atomic_write_bytes(path, buf.tobytes())


def extract_frames(
    project: StudioProject,
    video: Path | str,
    n: int,
    method: str = "diverse",
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> list[FrameRecord]:
    """Add up to ``n`` new frames from ``video`` to the project, and save it.

    Frames the project already has from this video are not added twice, so
    running it again on the same video tops the set up rather than duplicating
    it. Returns the frames added.

    Raises:
        OSError: The video cannot be opened.
        ExtractionCancelledError: ``cancel_cb`` returned True. Frames written before
            the cancel are kept and saved.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video}")
    added: list[FrameRecord] = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reader = ExactFrameReader(cap)
        key = project.video_key(video)
        project.images_dir.mkdir(parents=True, exist_ok=True)

        def fresh(indices: list[int]) -> list[int]:
            return [i for i in indices if not project.has_frame(key, i)]

        if method == "uniform":
            chosen = fresh(uniform_indices(count, n))
            pool_steps = 0
        else:
            pool = fresh(uniform_indices(count, n * _POOL_FACTOR))
            pool_steps = len(pool)
        total = pool_steps + n

        if method == "diverse":
            thumbs, kept = [], []
            for step, index in enumerate(pool):
                if cancel_cb and cancel_cb():
                    raise ExtractionCancelledError()
                frame = reader.read(index)
                if frame is not None:
                    thumbs.append(_thumbnail(frame))
                    kept.append(index)
                if progress_cb:
                    progress_cb(step + 1, total)
            chosen = diverse_indices(np.stack(thumbs), kept, n, project.seed) if thumbs else []

        for step, index in enumerate(chosen):
            if cancel_cb and cancel_cb():
                raise ExtractionCancelledError()
            frame = reader.read(index)
            if frame is None:
                logger.warning("frame %d of %s could not be read; skipped", index, video)
                continue
            record = FrameRecord(project.frame_name(key, index), key, index)
            _write_jpeg(project.image_path(record), frame)
            project.frames.append(record)
            added.append(record)
            if progress_cb:
                progress_cb(pool_steps + step + 1, total)
    finally:
        cap.release()
        project.save()
    logger.info("added %d frames from %s (%s)", len(added), video.name, method)
    return added
