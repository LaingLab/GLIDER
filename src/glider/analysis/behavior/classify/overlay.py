"""On-frame visualization: pose skeleton + behavior label + FPS.

Pure OpenCV drawing — no Qt, no matplotlib. The drawing functions take
an existing BGR numpy array and mutate it in place, returning the
same array for chaining convenience.

Default palette
---------------

* Keypoints: green dots (cv2 BGR ``(40, 200, 80)``), opacity-modulated
  by confidence.
* Skeleton lines: thin cyan ``(200, 180, 30)`` connecting adjacent
  keypoints along the configured edges.
* Behavior label badge: top-left, white-on-coloured background. The
  color is picked per behavior from the same categorical palette the
  annotator uses, so labels stay consistent across the two tools.
* FPS overlay: bottom-right, tiny gray text.

Edges
-----

If no explicit ``edges`` list is provided, we infer a sensible chain
``[0,1], [1,2], [2,3], ...`` covering consecutive keypoints. Pass
``edges=[(snout_idx, neck_idx), ...]`` for a real anatomical skeleton.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# Categorical palette (BGR). Matches the annotator's colours but
# converted from #RRGGBB hex to BGR ints for OpenCV.
_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (216, 78, 29),  # #1d4ed8 blue
    (87, 120, 4),  # #047857 emerald
    (9, 83, 180),  # #b45309 amber
    (237, 60, 124),  # #7c3aed violet
    (93, 24, 190),  # #be185d rose
    (110, 118, 15),  # #0f766e teal
    (7, 98, 161),  # #a16207 yellow
    (12, 65, 194),  # #c2410c orange
    (202, 56, 67),  # #4338ca indigo
    (61, 128, 21),  # #15803d green
    (77, 23, 157),  # #9d174d pink
    (28, 28, 185),  # #b91c1c red
)


def color_for_behavior(name: str, vocab_order: list[str] | None = None) -> tuple[int, int, int]:
    """Return a stable BGR triple for a behavior name.

    If ``vocab_order`` is supplied, the color comes from the behavior's
    position in that list (matches what the annotator showed). Otherwise
    a deterministic hash of the name picks one from the palette.
    """
    if vocab_order and name in vocab_order:
        return _PALETTE_BGR[vocab_order.index(name) % len(_PALETTE_BGR)]
    # Deterministic fallback for unknown labels — hash the name into
    # a palette slot so the same behavior always renders the same color
    # across runs.
    h = sum(ord(c) for c in name) % len(_PALETTE_BGR)
    return _PALETTE_BGR[h]


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    confidences: np.ndarray | None = None,
    edges: Iterable[tuple[int, int]] | None = None,
    keypoint_radius: int = 4,
    edge_thickness: int = 2,
    min_confidence: float = 0.3,
) -> np.ndarray:
    """Draw keypoints + edges on ``frame``.

    Parameters
    ----------
    frame
        BGR ndarray of shape ``(H, W, 3)``. Modified in place.
    keypoints
        Shape ``(K, 2)`` of xy pixel coordinates.
    confidences
        Optional ``(K,)`` confidences in ``[0, 1]``. Keypoints below
        ``min_confidence`` are skipped; edges that touch a low-conf
        keypoint are skipped too.
    edges
        Pairs of keypoint indices to connect with a line. Defaults to
        the consecutive chain.
    """
    import cv2

    if keypoints is None or keypoints.size == 0:
        return frame
    k = keypoints.shape[0]
    if confidences is None:
        confidences = np.ones(k, dtype=np.float64)

    # Default edge chain: consecutive keypoints. Crude but better than
    # nothing for a sanity check; the CLI lets you pass real edges.
    if edges is None:
        edges = [(i, i + 1) for i in range(k - 1)]

    # Edges first so the keypoint dots sit on top.
    cyan = (220, 180, 30)
    for a, b in edges:
        if a < 0 or b < 0 or a >= k or b >= k:
            continue
        if confidences[a] < min_confidence or confidences[b] < min_confidence:
            continue
        pa = (int(keypoints[a, 0]), int(keypoints[a, 1]))
        pb = (int(keypoints[b, 0]), int(keypoints[b, 1]))
        if not _finite_point(pa) or not _finite_point(pb):
            continue
        cv2.line(frame, pa, pb, cyan, edge_thickness, cv2.LINE_AA)

    green = (40, 200, 80)
    for i in range(k):
        if confidences[i] < min_confidence:
            continue
        p = (int(keypoints[i, 0]), int(keypoints[i, 1]))
        if not _finite_point(p):
            continue
        cv2.circle(frame, p, keypoint_radius, green, -1, cv2.LINE_AA)
        cv2.circle(frame, p, keypoint_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_label_badge(
    frame: np.ndarray,
    text: str,
    color_bgr: tuple[int, int, int] = (40, 40, 40),
    x: int = 16,
    y: int = 16,
) -> np.ndarray:
    """Draw a rounded badge at ``(x, y)`` with white ``text`` on ``color_bgr``."""
    import cv2

    if not text:
        return frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 12, 8
    x1, y1 = x, y
    x2, y2 = x + tw + 2 * pad_x, y + th + 2 * pad_y + baseline
    # Solid background.
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, -1, cv2.LINE_AA)
    # Thin white border for contrast against arbitrary video.
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        text,
        (x1 + pad_x, y1 + pad_y + th),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Tiny FPS readout in the bottom-right corner."""
    import cv2

    text = f"{fps:5.1f} fps"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    h, w = frame.shape[:2]
    x = w - tw - 12
    y = h - 12
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)
    return frame


def _finite_point(p: tuple[int, int]) -> bool:
    return all(np.isfinite(v) and -1e6 < v < 1e6 for v in p)
