"""Pups from a semantic mask: N lifelong animals from "pup / not pup" pixels.

A semantic segmentation model answers one question per pixel -- is this pup?
-- and nothing about *which* pup. Two pups touching are one blob; a pup under
the dam is no blob at all. Nothing like ByteTrack can run on that, because
there are no instances to track.

So instances are rebuilt here, in three steps:

1. **Blobs.** Connected components of the mask, dropping specks.
2. **Splitting.** A blob much bigger than one pup is several pups touching. It
   is cut into ``round(area / single_pup_area)`` parts by k-means on its
   pixels -- pups are compact, so the parts land on the bodies.
3. **Identity.** Parts are linked frame to frame by optimal assignment on
   centroid distance, which yields short fragments, and the fragments go to
   :func:`~glider.vision.pose.consolidate.consolidate` with the litter size,
   exactly as multi-mouse pose tracking does. Each pup ends up as a one-
   keypoint (``centroid``) :class:`PoseData` slot, so zone scoring, smoothing
   and the review viewer accept it unchanged.

Tuned on a 640x480, 15 fps retrieval clip; the defaults are pixels at that
resolution.
"""

from __future__ import annotations

import io
import json
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from glider.core.fileio import atomic_write_bytes
from glider.vision.pose.consolidate import ConsolidationResult, Fragment, consolidate

__all__ = [
    "CENTROID",
    "PUPS_SUFFIX",
    "PupBlob",
    "PupTracks",
    "contours_from_mask",
    "link",
    "load_pups",
    "pups_path",
    "save_pups",
    "segment_video",
    "split_blobs",
    "track_pups",
    "track_video_pups",
    "tracks_from_result",
    "unit_area",
]

CENTROID = "centroid"
PUPS_SUFFIX = "_pups.npz"


@dataclass(frozen=True)
class PupBlob:
    """One pup's worth of mask in one frame."""

    x: float
    y: float
    area: float
    contour: np.ndarray  # (N, 2) int pixel outline
    merged: int  # how many pups the source blob held; 1 = alone


def contours_from_mask(mask: np.ndarray, *, min_area: float = 150) -> list[np.ndarray]:
    """Outer contours (``(N, 2)`` int) of every blob of at least *min_area* px.

    Contours, not masks, are what gets cached per frame: a few hundred bytes a
    pup instead of a full frame, and enough to redo splitting and nest
    distance without decoding the video again.
    """
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0, :].astype(np.int32) for c in found if cv2.contourArea(c) >= min_area]


def _area(contour: np.ndarray) -> float:
    return float(cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.int32)))


def unit_area(frames: list[list[np.ndarray]]) -> float:
    """One pup's area: the median blob area over every frame.

    Most blobs are single pups most of the time, so the median is one pup even
    when a few frames are full of huddles.
    """
    areas = [_area(c) for contours in frames for c in contours]
    if not areas:
        raise ValueError("no pup blobs in any frame")
    return float(np.median(areas))


def split_blobs(contours: list[np.ndarray], unit: float) -> list[PupBlob]:
    """One :class:`PupBlob` per pup, cutting blobs that hold several."""
    out: list[PupBlob] = []
    for contour in contours:
        contour = np.asarray(contour, dtype=np.int32)  # cv2 rejects int16 caches
        area = _area(contour)
        k = max(1, round(area / unit))
        if k == 1:
            m = cv2.moments(contour.reshape(-1, 1, 2))
            if m["m00"] == 0:
                continue
            out.append(PupBlob(m["m10"] / m["m00"], m["m01"] / m["m00"], area, contour, 1))
            continue
        x0, y0 = contour.min(axis=0)
        x1, y1 = contour.max(axis=0)
        canvas = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(canvas, [contour - [x0, y0]], 1)
        ys, xs = np.nonzero(canvas)
        if len(xs) < k:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        cv2.setRNGSeed(0)  # the same video must split the same way every run
        _, labels, centres = cv2.kmeans(pts, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        for part in range(k):
            sub = np.zeros_like(canvas)
            sub[ys[labels == part], xs[labels == part]] = 1
            found = contours_from_mask(sub, min_area=0)
            if not found:
                continue
            outline = max(found, key=_area) + [x0, y0]
            cx, cy = centres[part]
            out.append(
                PupBlob(float(cx + x0), float(cy + y0), float((labels == part).sum()), outline, k)
            )
    return out


def link(frames: list[list[PupBlob]], *, gate_px: float = 30.0, max_age: int = 3) -> list[Fragment]:
    """Frame-to-frame fragments by optimal assignment on centroid distance.

    A fragment survives up to *max_age* frames unseen (a pup briefly under the
    dam's tail), and may move ``gate_px`` per frame it was unseen. Anything
    that cannot be matched starts a new fragment; consolidation decides later
    which pup it was.
    """
    tracks: list[dict] = []
    live: list[dict] = []
    for f, blobs in enumerate(frames):
        live = [t for t in live if f - t["frames"][-1] <= max_age]
        used: set[int] = set()
        if live and blobs:
            last = np.array([t["xy"][-1] for t in live])
            here = np.array([(b.x, b.y) for b in blobs])
            cost = np.linalg.norm(last[:, None, :] - here[None, :, :], axis=2)
            for i, j in zip(*linear_sum_assignment(cost), strict=True):
                if cost[i, j] <= gate_px * (f - live[i]["frames"][-1]):
                    live[i]["frames"].append(f)
                    live[i]["xy"].append(here[j])
                    used.add(int(j))
        for j, b in enumerate(blobs):
            if j not in used:
                t = {"frames": [f], "xy": [(b.x, b.y)]}
                tracks.append(t)
                live.append(t)
    return [
        Fragment(
            track_id=k,
            frames=np.array(t["frames"]),
            xy=np.array(t["xy"], dtype=float)[:, None, :],
            confidence=np.ones((len(t["frames"]), 1)),
        )
        for k, t in enumerate(tracks)
    ]


def track_pups(
    frames: list[list[np.ndarray]],
    *,
    n_pups: int,
    fps: float,
    gate_px: float = 30.0,
    max_age: int = 3,
    max_travel_px_per_frame: float = 20.0,
    min_fragment_s: float = 0.5,
) -> tuple[ConsolidationResult, list[list[PupBlob]]]:
    """*n_pups* lifelong pup tracks from per-frame blob contours.

    Returns the consolidation (tracks, stitched frames, dropped fragments) and
    the split blobs per frame, which the caller keeps for nest distance and
    for drawing masks.

    ``min_fragment_s`` drops glimpses. A pup seen for a few frames under the
    dam's head mid-carry is otherwise handed to whichever pup has been hidden
    longest -- a long gap makes any jump look slow -- and that pup gains a
    carry it never had.
    """
    unit = unit_area(frames)
    blobs = [split_blobs(contours, unit) for contours in frames]
    fragments = link(blobs, gate_px=gate_px, max_age=max_age)
    result = consolidate(
        fragments,
        n_animals=n_pups,
        n_frames=len(frames),
        keypoint_names=[CENTROID],
        fps=fps,
        source="pups",
        max_travel_px_per_frame=max_travel_px_per_frame,
        min_fragment_frames=max(1, round(min_fragment_s * fps)),
    )
    return result, blobs


def pups_path(video: Path | str) -> Path:
    """Where the batch pup pass keeps a video's pups: ``<stem>_pups.npz`` beside it."""
    video = Path(video)
    return video.with_name(f"{video.stem}{PUPS_SUFFIX}")


@dataclass(frozen=True)
class PupTracks:
    """Every pup through one video, as the batch pup pass saved it."""

    xy: np.ndarray  # (N, F, 2) float, NaN where the pup was not seen
    outlines: list[list[np.ndarray]]  # per frame, each (K, 2) int32 pixel outline
    stitched: np.ndarray  # (N, F) bool: frames whose identity is a join
    meta: dict

    @property
    def n_pups(self) -> int:
        return int(self.xy.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.xy.shape[1])

    @property
    def names(self) -> list[str]:
        return [f"pup_{k}" for k in range(self.n_pups)]

    def outline_for(self, pup: int, frame: int) -> np.ndarray | None:
        """The frame's outline nearest the pup's tracked position, or None if unseen."""
        if not 0 <= frame < self.n_frames or not self.outlines[frame]:
            return None
        here = self.xy[pup, frame]
        if not np.isfinite(here).all():
            return None
        # ponytail: vertex mean, not the true centroid; pups are compact blobs,
        # so the nearest outline is the same either way.
        return min(self.outlines[frame], key=lambda o: float(np.hypot(*(o.mean(axis=0) - here))))


def tracks_from_result(
    result: ConsolidationResult, blobs: list[list[PupBlob]], *, n_pups: int, meta: dict
) -> PupTracks:
    """A :class:`PupTracks` from :func:`track_pups`' output."""
    n_frames = len(blobs)
    xy = np.full((n_pups, n_frames, 2), np.nan)
    stitched = np.zeros((n_pups, n_frames), dtype=bool)
    for slot in range(n_pups):
        pose = result.tracks.tracks.get(slot)
        if pose is not None:
            xy[slot] = pose.xy[:, 0, :]
        for f in result.stitched.get(slot, ()):
            stitched[slot, f] = True
    outlines = [[np.asarray(b.contour, dtype=np.int32) for b in frame] for frame in blobs]
    return PupTracks(xy, outlines, stitched, dict(meta))


def save_pups(path: Path | str, tracks: PupTracks) -> None:
    """Write *tracks* to *path* so a crash never leaves a half-written file."""
    path = Path(path)
    flat = [o for frame in tracks.outlines for o in frame]
    points = np.concatenate(flat).astype(np.int16) if flat else np.zeros((0, 2), np.int16)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        points=points,
        contour_ends=np.cumsum([len(o) for o in flat], dtype=np.int64),
        frame_ends=np.cumsum([len(f) for f in tracks.outlines], dtype=np.int64),
        xy=tracks.xy.astype(np.float32),
        stitched=tracks.stitched,
        meta=np.array(json.dumps(tracks.meta)),
    )
    atomic_write_bytes(path, buffer.getvalue())


def load_pups(path: Path | str) -> PupTracks:
    """Read a ``_pups.npz``; ``ValueError`` names the file if it is not one."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            points = data["points"].astype(np.int32)
            contour_ends = data["contour_ends"]
            frame_ends = data["frame_ends"]
            xy = data["xy"].astype(float)
            stitched = data["stitched"].astype(bool)
            meta = json.loads(str(data["meta"]))
    # A damaged member raises zlib.error or NotImplementedError from inside zipfile,
    # and an empty file raises EOFError; all of them mean "not a pups file".
    except (
        OSError,
        KeyError,
        ValueError,
        EOFError,
        NotImplementedError,
        zipfile.BadZipFile,
        zlib.error,
    ) as e:
        raise ValueError(f"{path.name} is not a pups file: {e}") from e
    contours = np.split(points, contour_ends[:-1]) if len(contour_ends) else []
    outlines, start = [], 0
    for end in frame_ends:
        outlines.append([np.asarray(c, dtype=np.int32) for c in contours[start:end]])
        start = int(end)
    return PupTracks(xy, outlines, stitched, meta)


def _numpy(data) -> np.ndarray:
    """A mask's data as numpy, whether Ultralytics gave a tensor or tests gave an array."""
    return data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)


def segment_video(
    model,
    video: Path,
    *,
    device: str,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> tuple[list[list[np.ndarray]], tuple[int, int]]:
    """Each frame's pup outlines in video pixels, and the frame size (w, h).

    The semantic model's mask can come back at its own input size; it is
    resized to the video's (nearest neighbour, so it stays binary) before any
    outline is taken, so every outline is in the video's pixels.
    """
    from glider.vision.pose.core import PoseCancelledError

    frames: list[list[np.ndarray]] = []
    size = (0, 0)
    for result in model.predict(source=str(video), stream=True, device=device, verbose=False):
        if cancel_cb is not None and cancel_cb():
            raise PoseCancelledError(f"cancelled during {Path(video).name}")
        h, w = result.orig_shape[:2]
        size = (int(w), int(h))
        mask = _numpy(result.semantic_mask.data).squeeze().astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        frames.append(contours_from_mask(mask))
        if progress_cb is not None:
            progress_cb(len(frames), 0)
    return frames, size


def _load_yolo(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path))


def track_video_pups(
    model_path: Path,
    video: Path,
    *,
    litter_size: int,
    device: str | None = None,
    require_gpu: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    load_model: Callable[[Path], object] | None = None,
) -> Path:
    """Segment, track and save one video's pups; returns ``<stem>_pups.npz``."""
    from glider import __version__
    from glider.vision.pose.core import _video_fps
    from glider.vision.pose.device import resolve_device

    video = Path(video)
    resolved = resolve_device(device, require_gpu=require_gpu)
    model = (load_model or _load_yolo)(Path(model_path))
    frames, (width, height) = segment_video(
        model, video, device=resolved, progress_cb=progress_cb, cancel_cb=cancel_cb
    )
    fps = _video_fps(video)
    result, blobs = track_pups(frames, n_pups=litter_size, fps=fps)
    meta = {
        "model": Path(model_path).name,
        "litter_size": int(litter_size),
        "fps": float(fps),
        "width": width,
        "height": height,
        "glider_version": __version__,
    }
    out = pups_path(video)
    save_pups(out, tracks_from_result(result, blobs, n_pups=litter_size, meta=meta))
    return out
