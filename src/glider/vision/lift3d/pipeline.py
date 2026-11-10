"""Per-frame 3-D lift: keypoints + frame -> mouse height surface and spine anchor heights.

Ported from the 3D-latent prototype's ``Pipeline2``. The prototype ran its own pose
model; here the caller passes keypoints from whatever GLIDER already ran (live tracker,
pose CSV), so the lift adds only the mask and depth models.

    keypoints -> fixed 20 cm crop -> (mask || depth) -> floor plane outside the mask
              -> surface (75% outline dome + 25% depth, rear profile from keypoints)
              -> five spine anchors (tail base .. nose) -> One-Euro smoothing

Every model is optional. Without the depth model the surface comes from the outline and
keypoints alone; without the mask model the outline is a dilated keypoint hull (weak).
Everything is causal (past frames only), so it runs live.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as _models
from .keypoints import CENTER, KeypointRoles, anchor_points
from .posture import KeypointHeight, anchor_values, body_surface, center_width
from .surface import (
    CROP_CM,
    CROP_PX,
    OneEuro,
    apply_affine,
    crop,
    fill_walls,
    floor_relative,
    outline_mask,
)

BACK_CM = 3.0  # assumed median back height of an adult C57; sets the per-session depth scale
SCALE_SAMPLES = 90  # frames of median back height before the depth scale is fixed
ANCHOR_RADIUS_CM = 0.6
MAX_HEIGHT_CM = 15.0

SegmentFn = Callable[[np.ndarray, tuple[int, int]], np.ndarray | None]
DepthFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Lift3DConfig:
    """Knobs for one lift run. ``ppc`` (pixels per cm) is the only required one."""

    ppc: float
    fps: float = 30.0
    # Arena floor polygon in image px (any order of 4 corners, or a longer polygon).
    arena_corners: np.ndarray | None = None
    # Model paths; None = look in models_dir(). use_* = False skips a model entirely.
    mask_model: str | Path | None = None
    depth_model: str | Path | None = None
    use_mask: bool = True
    use_depth: bool = True
    # Live: depth runs behind in its own thread every `depth_every` frames, and each
    # frame uses the newest result shifted to its crop (surface median 1.2 mm off).
    async_depth: bool = False
    depth_every: int = 6
    depth_imgsz: int = 768  # at 320 the depth model misses the mouse
    fill_walls: bool = True
    device: str | None = None
    # Parallax correction of anchor positions; both needed (from a phantom/block calibration).
    camera_height_cm: float | None = None
    principal_point_px: tuple[float, float] | None = None
    back_cm: float = BACK_CM


@dataclass
class Lift3DFrame:
    """One frame's lift. Heights are estimates in cm-like units (see ``posture``)."""

    anchors_cm: np.ndarray  # (5,) smoothed, tail base .. nose
    anchors_raw_cm: np.ndarray  # (5,) before smoothing
    rear_weight: float
    rear_score: float | None
    body_ratio: float
    body_length_cm: float
    height: np.ndarray  # (CROP_PX, CROP_PX) cm, NaN off the mouse
    mask: np.ndarray  # (CROP_PX, CROP_PX) bool
    M: np.ndarray  # 2x3 image -> crop affine
    depth_scale: float | None
    anchors_floor_px: np.ndarray  # (5, 2) image px, parallax-corrected when calibrated
    floor_resid_cm: float | None
    used_mask_model: bool
    used_depth: bool
    timings: dict = field(default_factory=dict)


def _ultralytics_segment(model, device, half) -> SegmentFn:
    def seg(crop_bgr, body_center):
        r = model(
            crop_bgr,
            imgsz=CROP_PX,
            half=half,
            device=device,
            verbose=False,
            retina_masks=True,
        )[0]
        if r.masks is None or len(r.masks) == 0:
            return None
        ms = r.masks.data.cpu().numpy().astype(bool)
        x, y = body_center
        hit = [k for k in range(len(ms)) if ms[k][y, x]]  # instance holding the body centre
        m = ms[hit[0]] if hit else ms[int(np.argmax(ms.sum((1, 2))))]
        if m.shape != (CROP_PX, CROP_PX):
            import cv2

            m = cv2.resize(
                m.astype(np.uint8), (CROP_PX, CROP_PX), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        return m

    return seg


def _ultralytics_depth(model, imgsz, device, half) -> DepthFn:
    def depth(crop_bgr):
        r = model(crop_bgr, imgsz=imgsz, half=half, device=device, verbose=False)[0]
        return np.squeeze(r.depth.data.float().cpu().numpy())

    return depth


class Lift3D:
    """Stateful per-session lift. Call :meth:`process` once per frame, in order."""

    def __init__(
        self,
        keypoint_names: list[str],
        config: Lift3DConfig,
        segment: SegmentFn | None = None,
        depth: DepthFn | None = None,
    ):
        """``segment`` / ``depth`` inject model callables (tests, or a caller that already
        loaded them); otherwise they are loaded from ``config`` paths / ``models_dir()``."""
        if not config.ppc or config.ppc <= 0:
            raise ValueError("the 3-D lift needs the camera's pixels per cm (ppc > 0)")
        self.roles = KeypointRoles.from_names(list(keypoint_names))
        self.config = config
        self.segment = segment
        self.depth = depth
        if segment is None or depth is None:
            self._load_models()
        self.arena = (
            None if config.arena_corners is None else np.asarray(config.arena_corners, float)
        )
        self.kh = KeypointHeight(fps=config.fps)
        self.filt = OneEuro(config.fps)
        self.scale: float | None = None
        self._scale_samples: list[float] = []
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="lift3d-depth")
        self._job: Future | None = None
        self._rel = None
        self._since = 0
        self._last_idx: int | None = None
        gv, gu = np.mgrid[0:CROP_PX, 0:CROP_PX]
        self._gu, self._gv = gu.astype(np.float64), gv.astype(np.float64)

    # -- setup ----------------------------------------------------------------------------
    def _load_models(self) -> None:
        cfg = self.config
        try:
            import torch

            half = bool(torch.cuda.is_available()) and cfg.device != "cpu"
        except ImportError:
            half = False
        if self.segment is None and cfg.use_mask:
            path = _models.find_model(_models.MASK_MODEL, cfg.mask_model)
            if path is not None:
                self.segment = _ultralytics_segment(_models.load_yolo(path), cfg.device, half)
        if self.depth is None and cfg.use_depth:
            path = _models.find_model(_models.DEPTH_MODEL, cfg.depth_model)
            if path is not None:
                _models.check_depth_support()
                self.depth = _ultralytics_depth(
                    _models.load_yolo(path), cfg.depth_imgsz, cfg.device, half
                )

    def reset(self) -> None:
        """Forget motion history, for a seek or a gap in the video. Keeps session scale."""
        self.kh.reset()
        self.filt.reset()
        self._rel = None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- depth ----------------------------------------------------------------------------
    def _depth_job(self, crop_bgr, mask, arena_c, center):
        img = crop_bgr
        if self.config.fill_walls and arena_c is not None:
            img = fill_walls(crop_bgr, mask, arena_c)
        rel, floor_m = floor_relative(self.depth(img), mask, arena_c)
        return rel.astype(np.float32), floor_m.astype(np.uint8), center

    def _latest_depth(self, crop_bgr, mask, arena_c, center):
        """Newest finished depth shifted to this crop; starts the next one when due."""
        import cv2

        if self._job is not None and self._job.done():
            self._rel, self._job = self._job.result(), None
        self._since += 1
        if self._job is None and (self._rel is None or self._since >= self.config.depth_every):
            self._since = 0
            self._job = self._pool.submit(self._depth_job, crop_bgr, mask, arena_c, center)
        if self._rel is None:
            self._rel, self._job = self._job.result(), None
        rel, floor_m, c0 = self._rel
        s = CROP_PX / (CROP_CM * self.config.ppc)
        shift = np.float32([[1, 0, s * (c0[0] - center[0])], [0, 1, s * (c0[1] - center[1])]])
        size = (CROP_PX, CROP_PX)
        return (
            cv2.warpAffine(rel, shift, size, borderMode=cv2.BORDER_REPLICATE),
            cv2.warpAffine(floor_m, shift, size, flags=cv2.INTER_NEAREST).astype(bool),
        )

    # -- main -----------------------------------------------------------------------------
    def process(
        self, frame_bgr: np.ndarray, keypoints: np.ndarray, frame_idx: int | None = None
    ) -> Lift3DFrame | None:
        """Lift one frame. ``keypoints`` is ``(K, 2+)`` in the model's own order.

        Returns None when a needed keypoint is missing (NaN) this frame. A seek
        (``frame_idx`` going backwards, or forward by more than a second) resets the
        motion history; the odd dropped frame under live back-pressure does not.
        """
        if frame_idx is not None:
            last = self._last_idx
            if last is not None and not (last < frame_idx <= last + max(int(self.config.fps), 1)):
                self.reset()
            self._last_idx = frame_idx
        xy = self.roles.canonical(keypoints)
        if not np.isfinite(xy).all():
            return None
        cfg = self.config
        t: dict[str, float] = {}

        lo, hi = xy.min(0), xy.max(0)
        center = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
        crop_bgr, affine = crop(frame_bgr, center, cfg.ppc)
        kp_c = apply_affine(affine, xy)
        bc = tuple(int(v) for v in np.clip(np.round(kp_c[CENTER]), 0, CROP_PX - 1))
        arena_c = None if self.arena is None else apply_affine(affine, self.arena)

        t0 = time.perf_counter()
        mask = self.segment(crop_bgr, bc) if self.segment is not None else None
        used_mask = mask is not None and bool(mask.any())
        if not used_mask:
            mask = outline_mask(kp_c)
        t["mask"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        rel = floor_m = None
        if self.depth is not None:
            if cfg.async_depth:
                rel, floor_m = self._latest_depth(crop_bgr, mask, arena_c, center)
            else:
                rel, fm, _ = self._depth_job(crop_bgr, mask, arena_c, center)
                floor_m = fm.astype(bool)
        t["depth"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        pts = anchor_points(kp_c)
        rad = ANCHOR_RADIUS_CM * CROP_PX / CROP_CM
        k = self.kh(xy, cfg.ppc, center_width(mask, kp_c[CENTER]) if used_mask else None)

        scale = None
        if rel is not None:
            # Per-session depth scale: median back height outside rears -> back_cm. Skip
            # frames where depth saw a flat mouse (blur, failures).
            near = mask & ((self._gu - pts[2][0]) ** 2 + (self._gv - pts[2][1]) ** 2 < rad**2)
            back = float(np.percentile(rel[near], 75)) if near.sum() > 5 else 0.0
            if self.scale is None:
                if back > 0.004 and k["weight"] < 0.1:
                    self._scale_samples.append(back)
                if len(self._scale_samples) >= SCALE_SAMPLES:
                    self.scale = cfg.back_cm / max(float(np.median(self._scale_samples)), 1e-6)
            scale = self.scale or cfg.back_cm / max(back, 0.004)

        height = body_surface(mask, rel, scale or 0.0, kp_c, k)
        z_raw = np.clip(anchor_values(height, pts, rad), 0.0, MAX_HEIGHT_CM)
        z = self.filt(z_raw)

        anchors_img = anchor_points(xy)
        if cfg.camera_height_cm and cfg.principal_point_px is not None:
            # A point z above the floor appears pushed away from the image centre by H/(H - z).
            c = np.asarray(cfg.principal_point_px, float)
            anchors_floor = c + (anchors_img - c) * (1 - z / cfg.camera_height_cm)[:, None]
        else:
            anchors_floor = anchors_img
        resid = None
        if rel is not None and floor_m is not None and floor_m.any() and scale:
            resid = float(np.std(rel[floor_m]) * scale)
        t["post"] = time.perf_counter() - t0

        return Lift3DFrame(
            anchors_cm=np.asarray(z, float).copy(),
            anchors_raw_cm=z_raw,
            rear_weight=float(k["weight"]),
            rear_score=k["score"],
            body_ratio=float(k["ratio"]),
            body_length_cm=float(k["length_cm"]),
            height=height,
            mask=mask,
            M=affine,
            depth_scale=scale,
            anchors_floor_px=anchors_floor,
            floor_resid_cm=resid,
            used_mask_model=used_mask,
            used_depth=rel is not None,
            timings=t,
        )
