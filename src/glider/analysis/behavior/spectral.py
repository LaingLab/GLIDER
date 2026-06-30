"""Windowed spectral features capturing rhythmicity.

Grooming and digging are *rhythmic but stationary*; locomotion is
*translational*. Mean/std/max over a window can't tell a rhythmic paw
oscillation from steady motion, but the frequency content can. These
two stats summarize that content per window:

* **dominant frequency** — the FFT bin (cycles per window) with the most
  power. Reported as a bin index, not Hz, so it's sample-rate-free and
  the training and live sides agree without plumbing fps around.
* **spectral flatness** — geometric-mean / arithmetic-mean of the power
  spectrum. Near 0 for a clean oscillation (energy in one bin), near 1
  for white noise (energy spread evenly). This is the "how rhythmic is
  it" knob.

The SAME :func:`window_spectral` runs on both sides of the pipeline —
the training rolling apply (:mod:`glider.analysis.behavior.windowing`) and the
live ring buffer (:mod:`glider.analysis.behavior.classify.buffer`) — so they can never
silently drift. Applied to kinematic features only (speeds /
accelerations / body angular velocity).
"""

from __future__ import annotations

import numpy as np

# Below this many real (finite) samples a spectrum is meaningless.
_MIN_FINITE = 4

# The two spectral stats and the kinematic features they apply to. Kept
# here so the training and live sides agree on names + ordering.
SPECTRAL_STATS: tuple[str, ...] = ("domfreq", "specflat")
# ``motion_*`` are the image residual-motion-energy features (analysis/behavior/motion.py).
# Their rhythm — a grooming face-wash oscillation vs an irregular sniff — is
# exactly what spectral is for, and unlike the trunk keypoints the anterior
# pixel motion actually carries the paw rhythm, so they belong in this set.
# (No-op unless those columns are present, i.e. unless --motion-features ran.)
_KINEMATIC_PREFIXES = ("speed_", "accel_", "motion_")
_KINEMATIC_EXACT = ("body_angular_velocity",)


def kinematic_feature_names(feature_names) -> list[str]:
    """Filter ``feature_names`` to the kinematic ones (speed / accel /
    body angular velocity / motion-energy) — the motion signals where
    rhythm lives."""
    return [n for n in feature_names if n.startswith(_KINEMATIC_PREFIXES) or n in _KINEMATIC_EXACT]


def spectral_column_names(kinematic_names) -> list[str]:
    """Windowed spectral column names, stat-major to match
    :func:`glider.analysis.behavior.windowing.apply_rolling`'s ``__mean``/``__std``
    layout: every ``__domfreq`` then every ``__specflat``."""
    return [f"{n}__{stat}" for stat in SPECTRAL_STATS for n in kinematic_names]


def window_spectral_batch(windows) -> np.ndarray:
    """Vectorized :func:`window_spectral` over many windows at once.

    Parameters
    ----------
    windows
        ``(m, w)`` array — ``m`` windows of length ``w``. Each row is one
        feature's values across one window.

    Returns
    -------
    ndarray
        ``(m, 2)`` of ``[dominant_frequency, spectral_flatness]`` per row,
        with exactly the same semantics (and NaN/constant handling) as the
        scalar :func:`window_spectral`. One batched FFT replaces the
        per-window Python calls — this is what makes ``--freq-features``
        affordable over millions of frames.
    """
    w_arr = np.asarray(windows, dtype=np.float64)
    if w_arr.ndim != 2:
        raise ValueError(f"windows must be 2-D (m, w); got shape {w_arr.shape}")
    m = w_arr.shape[0]
    out = np.full((m, 2), np.nan)

    finite = np.isfinite(w_arr)
    counts = finite.sum(axis=1)
    valid = counts >= _MIN_FINITE
    if not valid.any():
        return out

    w_valid = w_arr[valid]
    fv = finite[valid]
    cv = counts[valid]
    # Fill NaN with the row's finite mean, then remove DC (per row).
    row_mean = np.where(fv, w_valid, 0.0).sum(axis=1) / cv
    filled = np.where(fv, w_valid, row_mean[:, None])
    filled = filled - filled.mean(axis=1, keepdims=True)

    power = np.abs(np.fft.rfft(filled, axis=1)) ** 2
    power = power[:, 1:]  # drop the DC bin
    total = power.sum(axis=1)

    res = np.full((w_valid.shape[0], 2), np.nan)
    res[total <= 0.0, 0] = 0.0  # constant rows → (0.0, NaN)
    osc = total > 0.0
    if osc.any():
        p_osc = power[osc]
        res[osc, 0] = np.argmax(p_osc, axis=1) + 1  # +1 restores dropped DC
        pe = p_osc + 1e-12
        res[osc, 1] = np.exp(np.mean(np.log(pe), axis=1)) / np.mean(pe, axis=1)

    out[valid] = res
    return out


def window_spectral(values) -> tuple[float, float]:
    """Return ``(dominant_frequency, spectral_flatness)`` for one window.

    Parameters
    ----------
    values
        1-D sequence of a single feature's values across the window.
        NaNs (low-confidence frames) are filled with the window mean
        before the FFT.

    Returns
    -------
    (dominant_frequency, spectral_flatness)
        ``dominant_frequency`` is the cycles-per-window bin index of the
        peak (DC excluded); ``spectral_flatness`` is in ``(0, 1]``.
        Both are ``NaN`` when there are fewer than four finite samples.
        A constant (zero-variance) window returns ``(0.0, NaN)`` — no
        rhythm, flatness undefined.

    This is the single-window form the live buffer calls; it delegates to
    :func:`window_spectral_batch` so the live and training paths share one
    implementation and can't drift.
    """
    a = np.asarray(values, dtype=np.float64).reshape(1, -1)
    dom, flat = window_spectral_batch(a)[0]
    return float(dom), float(flat)
