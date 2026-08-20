"""Pure-numpy decode of pose network outputs into keypoints.

Both decoders take ``(K, H, W)`` — the backend normalises tensor layout before
calling — and return ``(K, 2)`` xy in **model-input pixels** plus ``(K,)``
confidence. They know nothing about files, resizing, or padding; the backend
undoes those afterwards. That split is what makes this file testable against
synthetic heatmaps with no model, no onnxruntime, and no GPU.

The two families use *different* coordinate conventions, which is easy to get
wrong and expensive to get wrong quietly — a systematic half-cell offset still
produces a plausible-looking skeleton:

* **DeepLabCut** treats a grid cell as covering ``stride`` pixels and reports
  its *centre*, hence the ``+ stride / 2``. Sub-pixel accuracy comes from a
  separate location-refinement head.
* **SLEAP** samples on a grid of ``np.arange(0, size, output_stride)``, so cell
  ``i`` sits at exactly ``i * output_stride`` with no half-cell shift. Sub-pixel
  accuracy comes from refining the peak itself.

Only the parity test against real DeepLabCut/SLEAP output can confirm these hold
for a given exported model. The unit tests here prove self-consistency, not
agreement with the source tools.
"""

from __future__ import annotations

import numpy as np


def _peaks(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel argmax: returns (rows, cols, peak_values)."""
    k = maps.shape[0]
    flat = maps.reshape(k, -1)
    idx = flat.argmax(axis=1)
    rows, cols = np.unravel_index(idx, maps.shape[1:])
    return rows, cols, flat[np.arange(k), idx]


def decode_dlc_locref(
    heatmaps: np.ndarray,
    locref: np.ndarray | None = None,
    *,
    stride: float,
    locref_stdev: float,
    apply_sigmoid: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode DeepLabCut heatmaps (+ optional location refinement) to keypoints.

    Parameters
    ----------
    heatmaps
        ``(K, H, W)`` scoremaps.
    locref
        ``(2K, H, W)`` refinement field, x then y per keypoint (interleaved).
        ``None`` skips sub-pixel refinement and returns grid-cell centres.
    stride
        Pixels per grid cell (DLC's ``stride``, typically 8).
    locref_stdev
        Scale the refinement field was trained against (DLC's ``locref_stdev``,
        typically 7.2831).
    apply_sigmoid
        True when the export ended before the final sigmoid.

    Returns
    -------
    (xy, confidence)
        ``(K, 2)`` in model-input pixels and ``(K,)`` peak scores.
    """
    heatmaps = np.asarray(heatmaps, dtype=float)
    if heatmaps.ndim != 3:
        raise ValueError(f"heatmaps must be (K, H, W); got {heatmaps.shape}")
    if apply_sigmoid:
        heatmaps = 1.0 / (1.0 + np.exp(-heatmaps))

    k = heatmaps.shape[0]
    rows, cols, conf = _peaks(heatmaps)

    xy = np.empty((k, 2), dtype=float)
    xy[:, 0] = cols * stride + stride / 2.0
    xy[:, 1] = rows * stride + stride / 2.0

    if locref is not None:
        locref = np.asarray(locref, dtype=float)
        expected = (2 * k, *heatmaps.shape[1:])
        if locref.shape != expected:
            raise ValueError(f"locref must be (2K, H, W) == {expected}; got {locref.shape}")
        ar = np.arange(k)
        xy[:, 0] += locref[2 * ar, rows, cols] * locref_stdev
        xy[:, 1] += locref[2 * ar + 1, rows, cols] * locref_stdev

    return xy, conf


def decode_sleap_confmaps(
    confmaps: np.ndarray,
    *,
    stride: float,
    window: int = 5,
    apply_sigmoid: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode SLEAP single-instance confidence maps to keypoints.

    Takes the global peak per channel, then refines it to sub-pixel accuracy by
    an intensity-weighted centroid over a ``window``-square neighbourhood
    (SLEAP's integral refinement). Values are clipped at zero first so a
    negative logit cannot drag the centroid, and the window is clipped at the
    array edge so a peak on the border is safe.

    Note the coordinate convention: cell ``i`` maps to ``i * stride`` with **no**
    half-cell offset, unlike :func:`decode_dlc_locref`.

    Returns
    -------
    (xy, confidence)
        ``(K, 2)`` in model-input pixels and ``(K,)`` peak scores.
    """
    confmaps = np.asarray(confmaps, dtype=float)
    if confmaps.ndim != 3:
        raise ValueError(f"confmaps must be (K, H, W); got {confmaps.shape}")
    if apply_sigmoid:
        confmaps = 1.0 / (1.0 + np.exp(-confmaps))

    k, h, w = confmaps.shape
    rows, cols, conf = _peaks(confmaps)
    radius = max(int(window) // 2, 0)

    xy = np.empty((k, 2), dtype=float)
    for i in range(k):
        r0, r1 = max(0, rows[i] - radius), min(h, rows[i] + radius + 1)
        c0, c1 = max(0, cols[i] - radius), min(w, cols[i] + radius + 1)
        patch = np.clip(confmaps[i, r0:r1, c0:c1], 0.0, None)
        total = patch.sum()
        if total <= 0.0:
            # Nothing to weight by (flat or all-negative): keep the argmax cell.
            xy[i] = (cols[i], rows[i])
            continue
        rr = np.arange(r0, r1)[:, None]
        cc = np.arange(c0, c1)[None, :]
        xy[i, 0] = float((patch * cc).sum() / total)
        xy[i, 1] = float((patch * rr).sum() / total)

    xy *= stride
    return xy, conf
