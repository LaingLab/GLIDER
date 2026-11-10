"""Crop, floor-plane and smoothing helpers for the 3-D lift (Qt-free, model-free).

Ported from the 3D-latent prototype (``lift3d/pipeline2.py``).
"""

from __future__ import annotations

import numpy as np

CROP_CM, CROP_PX = 20.0, 320


class OneEuro:
    """One-Euro filter (Casiez et al. 2012): smooth when still, responsive when moving."""

    def __init__(
        self, fps: float, min_cutoff: float = 1.5, beta: float = 0.3, d_cutoff: float = 1.0
    ):
        self.fps, self.mc, self.beta, self.dc = fps, min_cutoff, beta, d_cutoff
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None

    @staticmethod
    def _a(cutoff, fps):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)

    def reset(self) -> None:
        self.x = self.dx = None

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx = x, np.zeros_like(x)
            return x
        dx = (x - self.x) * self.fps
        self.dx = self.dx + self._a(self.dc, self.fps) * (dx - self.dx)
        cutoff = self.mc + self.beta * np.abs(self.dx)
        self.x = self.x + self._a(cutoff, self.fps) * (x - self.x)
        return self.x


def crop_transform(center, ppc: float) -> np.ndarray:
    """2x3 affine taking image px to a CROP_CM-wide crop of CROP_PX px around ``center``.

    Fixed physical scale, so the depth model sees the mouse at the same size every frame.
    """
    s = CROP_PX / (CROP_CM * ppc)
    return np.array(
        [[s, 0, CROP_PX / 2 - s * center[0]], [0, s, CROP_PX / 2 - s * center[1]]], np.float32
    )


def crop(frame: np.ndarray, center, ppc: float) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    affine = crop_transform(center, ppc)
    out = cv2.warpAffine(
        frame, affine, (CROP_PX, CROP_PX), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return out, affine


def apply_affine(affine: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, float) @ affine[:, :2].T + affine[:, 2]


_GRID: tuple[np.ndarray, np.ndarray] | None = None


def _grid() -> tuple[np.ndarray, np.ndarray]:
    global _GRID
    if _GRID is None:
        gv, gu = np.mgrid[0:CROP_PX, 0:CROP_PX]
        _GRID = (gu.ravel().astype(np.float64), gv.ravel().astype(np.float64))
    return _GRID


def floor_relative(dep: np.ndarray, mask: np.ndarray, arena_c: np.ndarray | None = None):
    """Fit a plane to depth on floor pixels; return height above it as a fraction of the
    camera distance, plus the floor-pixel mask.

    Off-the-shelf depth models assume a forward camera and paint an overhead floor as
    sloping toward the image bottom. A plane fitted under each crop removes that.
    """
    import cv2

    gu, gv = _grid()
    floor_m = ~cv2.dilate(mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    if arena_c is not None:
        inside = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(inside, [np.round(arena_c).astype(np.int32)], 1)
        inside = cv2.erode(inside, np.ones((9, 9), np.uint8)).astype(bool)
        if (floor_m & inside).sum() > 500:
            floor_m &= inside
    sel = floor_m.ravel()
    design = np.c_[gu[sel], gv[sel], np.ones(sel.sum())]
    y = dep.ravel()[sel].astype(np.float64)
    keep = np.ones(len(y), bool)
    coef = np.zeros(3)
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
        res = y - design @ coef
        keep = np.abs(res) < 2.5 * np.median(np.abs(res[keep])) + 1e-9
    plane = (coef[0] * gu + coef[1] * gv + coef[2]).reshape(CROP_PX, CROP_PX)
    return (plane - dep) / np.median(plane), floor_m


def fill_walls(crop_bgr: np.ndarray, mask: np.ndarray, arena_c: np.ndarray) -> np.ndarray:
    """Paint everything outside the arena floor with floor pixels, keeping the animal.

    Walls in a crop change the depth model's scale: the same model read 3x higher beside
    a wall than mid-floor.
    """
    import cv2

    inside = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(inside, [np.round(arena_c).astype(np.int32)], 1)
    inside = inside.astype(bool)
    bad = ~inside & ~mask
    floor = inside & ~cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    if not bad.any() or floor.sum() < 50:
        return crop_bgr
    out = crop_bgr.copy()
    ys, xs = np.nonzero(bad)
    fy, fx = np.nonzero(floor)
    # Same seed every frame, so the fill does not flicker.
    i = np.random.default_rng(0).integers(0, len(fy), len(ys))
    out[ys, xs] = crop_bgr[fy[i], fx[i]]
    return out


def outline_mask(xy_c: np.ndarray, size: int = CROP_PX) -> np.ndarray:
    """A rough body mask from the keypoints' hull, for runs without a mask model.

    Only a fallback: the hull of seven points is far thinner than the real body, so
    width-based rear scoring and the dome surface are both weaker without the mask model.
    """
    import cv2

    pts = np.asarray(xy_c, float)
    pts = pts[np.isfinite(pts).all(1)]
    m = np.zeros((size, size), np.uint8)
    if len(pts) < 3:
        return m.astype(bool)
    hull = cv2.convexHull(np.round(pts).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    return m.astype(bool)
