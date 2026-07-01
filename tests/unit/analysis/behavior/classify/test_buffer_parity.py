"""Train/serve parity: the live SlidingFeatureBuffer must produce the same
windowed row as the offline apply_rolling for identical per-frame features.

Any divergence here means live predictions drift from the cross-validated
numbers. Covers the rolling aggregation (mean/std/max); the per-frame
gradient parity is exercised end-to-end by _diag_causal.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer
from glider.analysis.behavior.windowing import apply_rolling


def test_buffer_rolling_matches_apply_rolling():
    rng = np.random.default_rng(0)
    names = ["a", "b", "c"]
    window = 10
    stats = ("mean", "std", "max")
    data = rng.normal(size=(window, len(names)))

    offline = apply_rolling(pd.DataFrame(data, columns=names), window=window, stats=stats)
    off_row = offline.iloc[-1]

    buf = SlidingFeatureBuffer(names, window=window, stats=stats)
    for row in data:
        buf.push_features(row)
    cols, vals = buf.rolling_features()
    live = dict(zip(cols, vals, strict=False))

    mismatches = {
        c: (live[c], float(off_row[c]))
        for c in offline.columns
        if not np.isclose(live[c], off_row[c], rtol=1e-9, atol=1e-9)
    }
    assert not mismatches, f"live != offline for: {mismatches}"
