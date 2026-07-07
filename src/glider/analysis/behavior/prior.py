"""Freeze/dart kinematic prior for the hybrid behavior model.

Pure functions over the windowed feature frame the base model consumes. Turns
graded freeze/dart activations into a per-class log-prior via semantic class
tags, so the same rules transfer across vocabularies (rules key off tags, not
class names). See docs/superpowers/specs/2026-07-07-hybrid-behavior-prior-fusion-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Single source of truth for the percentiles (shared with the live detector).
from glider.analysis.behavior.classify.speed_state import (
    DART_PCT_DEFAULT as DART_PCT,
)
from glider.analysis.behavior.classify.speed_state import (
    FREEZE_PCT_DEFAULT as FREEZE_PCT,
)


def prior_speed(windowed: pd.DataFrame) -> np.ndarray:
    """Scalar speed per row = mean across keypoints of the `speed_*__mean` columns."""
    cols = [c for c in windowed.columns if c.startswith("speed_") and c.endswith("__mean")]
    if not cols:
        raise ValueError("no speed_*__mean columns in the windowed frame")
    return windowed[cols].mean(axis=1).to_numpy(dtype=np.float64)


def calibrate_thresholds(
    speed: np.ndarray, *, freeze_pct: float = FREEZE_PCT, dart_pct: float = DART_PCT
) -> tuple[float, float]:
    """(freeze, dart) = the freeze_pct / dart_pct percentiles of `speed` (NaNs dropped)."""
    arr = np.asarray(speed, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("no valid speed samples to calibrate from")
    return float(np.percentile(arr, freeze_pct)), float(np.percentile(arr, dart_pct))
