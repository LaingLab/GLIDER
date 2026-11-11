"""Rear weight and body height surface from 2-D keypoints plus the mouse outline.

Ported from the 3D-latent prototype (``lift3d/keypoint_height.py``). A rearing mouse looks
short from above; the depth model does not see rears (it inflates the outline, so a
raised head at the edge reads low), so the keypoints supply what depth misses.

Seen nose-to-tail length ``d`` against the session's full length ``L`` gives a rear
weight, and the rise ``sqrt(L^2 - d^2)`` the height of the head above the tail base.
Once the session has a width baseline, a fitted score decides instead of length alone:
the torso looks narrower from above in a rear, and body length keeps changing, where a
curled mouse holds still.

Status (Nov 2026, 180 hand-labelled clips): held-out-session rear F1 0.66. Rears along a
wall are mostly missed: the head goes over the wall and the body looks full length.
All heights are model estimates, not measurements; treat them as body-length-scaled
posture, not centimetres of real height.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .keypoints import L_EAR, L_HIP, NOSE, R_EAR, R_HIP, TAIL

REAR_FULL, REAR_NONE = 0.55, 0.63  # body ratio at which the rear weight is 1 and 0
MAX_REAR_S = 4.0  # length-only rule: sitting and curling look as short but last longer
# Log-odds weights: 1 s minimum ratio, width share, 1 s ratio SD. Fitted on 180 clips.
SCORE_BIAS, SCORE_MIN, SCORE_WIDTH, SCORE_CHANGE = 3.597, -0.759, -5.789, 19.965
SCORE_CUT = 0.40
SCORE_SOFT = 0.1  # probability either side of the cut over which the weight goes 0 -> 1
SCORE_WINDOW = 31  # about 1 s
WIDTH_WINDOW = 13  # median over about 0.4 s; the outline flickers
WIDTH_MIN_BASELINE = 20  # stretched-out frames needed before the width means anything
STRETCHED = 0.85
MIN_GAP = 0.5  # ear or hip gap under this share of normal: keypoints collapsed, not a rear
REFRESH = 300
RISE_FACTOR = 0.9  # straight-line rise overshoots: rearing phantom 7.1 cm true, 7.9 straight
EASE = 0.2  # share of the way to the new rear weight per frame (~0.3 s rise)
HOLD_FRAMES = 6  # ~0.2 s below threshold before a rear can start; glitches are shorter
DEFAULT_LENGTH_CM = 9.0  # nose to tail base of an adult, until the session has its own
DEPTH_SHARE = 0.25  # share of the low-pose surface that comes from the depth model
LOW_POSE_MAX_CM = 4.5  # depth alone cannot go above this (mice by a wall read 7-8 cm)
TAIL_CM = 0.5
TAIL_KERNEL_PX = 17
# Rearing phantom: share of head height along tail base -> nose, from 9 photos.
PROFILE_T = np.linspace(0.05, 0.95, 10)
PROFILE = np.array([0.2, 0.48, 0.6, 0.72, 0.98, 0.96, 0.96, 0.97, 0.99, 1.0])


def _tail_kernel():
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TAIL_KERNEL_PX, TAIL_KERNEL_PX))


def body_length(xy: np.ndarray, ppc: float) -> float:
    """Seen nose to tail base, in cm (canonical keypoints)."""
    return float(np.linalg.norm(xy[NOSE] - xy[TAIL])) / ppc


def rise_height(full_length: float, seen: float) -> float:
    return RISE_FACTOR * float(np.sqrt(max(full_length**2 - seen**2, 0.0)))


def rear_weight(ratio: float) -> float:
    t = np.clip((REAR_NONE - ratio) / (REAR_NONE - REAR_FULL), 0, 1)
    return float(t * t * (3 - 2 * t))


def back_height(ratio: float) -> float:
    """Top of a low pose: normal phantom 2.9 cm at full length, hunched 3.5 cm at 0.8."""
    return 2.9 + 0.6 * float(np.clip((1 - ratio) / 0.2, 0, 1.5))


class KeypointHeight:
    """Per-frame rear weight and head rise. Keeps a running full body length.

    Causal: every value uses only frames already seen, so it runs live.
    """

    def __init__(self, fps: float = 30.0, min_history: int = 300, ref_length: float | None = None):
        self.fixed = ref_length
        self.history: list[float] = []
        self.gaps: list[np.ndarray] = []
        self.min_history = min_history
        self._L: float | None = None
        self._gap_ref: np.ndarray | None = None
        self.max_low = int(MAX_REAR_S * fps)
        self.still = False  # single photos have no motion: only the length rule applies
        self.forgive = 2  # a brief dip below the cut takes back this many hold frames
        self.baseline: list[float] = []
        self.reset()

    def reset(self) -> None:
        """Forget recent frames, for when the video jumps."""
        self.low, self.w = 0, 0.0
        self.recent: deque[float] = deque(maxlen=SCORE_WINDOW)
        self.widths: deque[float] = deque(maxlen=WIDTH_WINDOW)

    def score(self, ratio: float, width: float | None) -> float | None:
        """Probability of a rear, or None until the session has a width baseline."""
        self.recent.append(ratio)
        if width is None:
            return None
        self.widths.append(width)
        if ratio > STRETCHED:
            self.baseline.append(width)
        if self.still or len(self.baseline) < WIDTH_MIN_BASELINE or len(self.recent) < 2:
            return None
        share = float(np.median(self.widths)) / float(np.median(self.baseline[-300:]))
        change = float(np.std(self.recent, ddof=1))
        z = SCORE_BIAS + SCORE_MIN * min(self.recent) + SCORE_WIDTH * share + SCORE_CHANGE * change
        return float(1 / (1 + np.exp(-z)))

    @property
    def length(self) -> float:
        if self.fixed:
            return self.fixed
        if len(self.history) < self.min_history:
            return DEFAULT_LENGTH_CM
        if self._L is None or len(self.history) % REFRESH == 0:
            self._L = float(np.percentile(self.history[-9000:], 90))
            self._gap_ref = np.percentile(self.gaps[-9000:], 90, axis=0)
        return self._L

    def __call__(self, xy: np.ndarray, ppc: float, width: float | None = None) -> dict:
        d = body_length(xy, ppc)
        gaps = (
            np.array([np.linalg.norm(xy[L_EAR] - xy[R_EAR]), np.linalg.norm(xy[L_HIP] - xy[R_HIP])])
            / ppc
        )
        self.history.append(d)
        self.gaps.append(gaps)
        full_len = self.length
        ratio = d / full_len
        collapsed = self._gap_ref is not None and bool((gaps < MIN_GAP * self._gap_ref).any())
        p = self.score(ratio, width)
        if p is None:
            self.low = self.low + 1 if ratio < REAR_NONE else 0
            target = rear_weight(ratio) if HOLD_FRAMES <= self.low <= self.max_low else 0.0
        else:
            if p >= SCORE_CUT:
                self.low += 1
            else:
                self.low = max(self.low - self.forgive, 0)
            soft = (p - SCORE_CUT + SCORE_SOFT) / (2 * SCORE_SOFT)
            target = float(np.clip(soft, 0, 1)) if self.low >= HOLD_FRAMES else 0.0
        if collapsed:
            target = 0.0
        self.w += EASE * (target - self.w)
        return {
            "ratio": ratio,
            "weight": self.w,
            "rise_cm": rise_height(full_len, d),
            "length_cm": full_len,
            "score": p,
        }


def center_width(mask: np.ndarray, uv) -> float | None:
    """Crop-px distance from the body-centre keypoint to the outline, tail removed."""
    import cv2

    body = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _tail_kernel())
    if body.sum() < 500:
        return None
    y = int(np.clip(int(uv[1]), 0, mask.shape[0] - 1))
    x = int(np.clip(int(uv[0]), 0, mask.shape[1] - 1))
    return float(cv2.distanceTransform(body, cv2.DIST_L2, 5)[y, x])


def body_surface(
    mask: np.ndarray,
    rel: np.ndarray | None,
    depth_scale: float,
    xy_c: np.ndarray,
    k: dict,
    depth_share: float = DEPTH_SHARE,
) -> np.ndarray:
    """Height map (crop px, NaN outside the mask) from outline + keypoints + a little depth.

    On the phantom photos a dome from the outline beats the depth model (mean error 9.5
    vs 15 mm), and a quarter depth gives the best match within each photo. In a rear each
    pixel follows the measured profile along the tail-base-to-nose line. ``rel=None``
    (no depth model) gives the keypoint-and-outline surface alone.
    """
    import cv2

    body = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _tail_kernel()).astype(bool)
    if body.sum() < 500:  # the tail is too thin to survive the opening; a tiny mask isn't
        body = mask.astype(bool)
    dt = cv2.distanceTransform(body.astype(np.uint8), cv2.DIST_L2, 5)
    dome = np.sqrt(np.clip(1 - (1 - dt / max(float(dt.max()), 1.0)) ** 2, 0, 1))
    low = back_height(k["ratio"]) * dome * body + TAIL_CM * (mask & ~body)
    if rel is not None and depth_share > 0:
        d = np.clip(rel, 0, None) * depth_scale
        top = np.percentile(d[mask], 95) if mask.any() else 0.0
        low = (1 - depth_share) * low + depth_share * d * min(
            1.0, LOW_POSE_MAX_CM / max(float(top), 1e-6)
        )
    h = low
    if k["weight"] > 0.01:
        a, b = xy_c[TAIL], xy_c[NOSE]
        ab = b - a
        gv, gu = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
        t = np.clip(((gu - a[0]) * ab[0] + (gv - a[1]) * ab[1]) / max(float(ab @ ab), 1.0), 0, 1)
        rear = k["rise_cm"] * np.interp(t, PROFILE_T, PROFILE)
        h = (1 - k["weight"]) * low + k["weight"] * rear
    return np.where(mask, h, np.nan)


def anchor_values(height: np.ndarray, pts: np.ndarray, radius: float) -> np.ndarray:
    """Height at each anchor: 75th percentile within ``radius`` px, inside the mask."""
    gv, gu = np.mgrid[0 : height.shape[0], 0 : height.shape[1]]
    out = np.zeros(len(pts))
    ok = np.isfinite(height)
    for i, (u, v) in enumerate(pts):
        near = ok & ((gu - u) ** 2 + (gv - v) ** 2 < radius**2)
        if near.sum() > 5:
            out[i] = np.percentile(height[near], 75)
        else:
            y = int(np.clip(int(v), 0, height.shape[0] - 1))
            x = int(np.clip(int(u), 0, height.shape[1] - 1))
            out[i] = height[y, x] if ok[y, x] else 0.0
    return out
