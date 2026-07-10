"""Pure, non-threaded YOLO pose → keypoints decode.

This module holds the keypoint-decoding logic that used to live *inside*
:class:`~glider.analysis.behavior.classify.threads.PoseTracker`. Extracting it
here — with no threading and no Qt — lets a future live classifier decode pose
from a YOLO result *identically* to the offline path.

Single piece:

* :func:`extract_keypoints` — pull ``(K, 2)`` xy + ``(K,)`` confidence from one
  ultralytics ``Results`` object (or ``None``), NaN-masking keypoints below the
  confidence threshold so downstream feature math never uses them.
"""

from __future__ import annotations

import numpy as np


def extract_keypoints(
    result, conf_threshold: float, n_keypoints: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``(K, 2)`` xy + ``(K,)`` confidence from one YOLO Results object.

    ``result`` is a single ultralytics ``Results`` object OR ``None`` (callers
    pass ``results[0] if results else None``). Returns NaN-filled arrays if no
    detection landed. We always take the first detection (Ultralytics returns
    one Results per image — the first instance is the main subject in a single-
    animal setup).

    Both output arrays are sized by ``n_keypoints`` so ``K`` is always correct,
    even on the None / empty-detection branches.
    """
    keypoints = np.full((n_keypoints, 2), np.nan, dtype=np.float64)
    confidences = np.zeros(n_keypoints, dtype=np.float64)
    if result is None:
        return keypoints, confidences
    kp = getattr(result, "keypoints", None)
    if kp is None or kp.xy is None or kp.xy.shape[0] == 0:
        return keypoints, confidences
    xy = kp.xy[0].cpu().numpy()  # (K, 2)
    conf = kp.conf[0].cpu().numpy() if kp.conf is not None else np.ones(xy.shape[0])
    n = min(n_keypoints, xy.shape[0])
    keypoints[:n] = xy[:n]
    confidences[:n] = conf[:n]
    # Drop low-confidence keypoints to NaN so feature math doesn't use them.
    # The conf array is preserved for the overlay's per-dot fade logic.
    keypoints[confidences < conf_threshold] = np.nan
    return keypoints, confidences
