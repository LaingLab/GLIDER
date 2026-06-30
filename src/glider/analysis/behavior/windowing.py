"""Pandas-based temporal windowing.

Per the project plan: apply ``.rolling()`` to the per-frame feature
DataFrame to get 30-frame moving means, standard deviations, and maxes.
Each feature column is replaced with N new columns, one per stat,
suffixed ``__mean`` / ``__std`` / ``__max`` / ``__min``.

Why pandas and not the old custom sliding-window helper? Because pandas
``rolling`` handles edges (``min_periods``), centred-vs-trailing
windows, NaN, and per-group rolling out of the box — none of which we
want to reimplement.

Per-session independence
------------------------

When training on multiple recordings, the per-frame feature DataFrames
should each be rolled **independently**, then concatenated. Otherwise
the rolling window at the start of session N would borrow frames from
the tail of session N-1, which is meaningless. The :func:`apply_rolling`
function takes one DataFrame at a time; the orchestration in
:mod:`glider.analysis.behavior.pipeline` calls it per session and concatenates the results.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from glider.analysis.behavior.spectral import (
    kinematic_feature_names,
    spectral_column_names,
    window_spectral_batch,
)

DEFAULT_STATS: tuple[str, ...] = ("mean", "std", "max")


def apply_rolling(
    df: pd.DataFrame,
    window: int = 30,
    stats: Iterable[str] = DEFAULT_STATS,
    min_periods: int | None = None,
    center: bool = False,
) -> pd.DataFrame:
    """Apply ``window``-frame rolling stats to every column of ``df``.

    Returns a new DataFrame with ``len(df.columns) * len(stats)``
    columns. Output column names are ``<original>__<stat>``.

    Parameters
    ----------
    df
        Per-frame features. One row per frame, one column per feature.
    window
        Rolling window length in frames. At 30 fps the project default
        of 30 covers 1 second.
    stats
        Iterable of pandas-supported rolling-aggregation names. Defaults
        to the three the project plan calls for: mean / std / max.
        Supported: any name available on ``DataFrame.rolling`` (mean,
        std, max, min, sum, median, count, ...).
    min_periods
        Forwarded to ``DataFrame.rolling``. ``None`` defaults to
        ``window`` — the strict choice, so the first ``window-1`` rows
        come out NaN and get dropped downstream rather than being
        partial-window estimates. Pass ``1`` to fill the edges with
        whatever data exists.
    center
        Forwarded to ``DataFrame.rolling``. ``False`` = trailing window
        (the standard for live streaming).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    stats = tuple(stats)
    if not stats:
        raise ValueError("at least one stat is required")
    roller = df.rolling(window=window, min_periods=min_periods, center=center)
    parts: list[pd.DataFrame] = []
    for stat in stats:
        method = getattr(roller, stat, None)
        if method is None or not callable(method):
            raise ValueError(
                f"stat {stat!r} is not a pandas Rolling method; "
                f"try one of: mean, std, max, min, sum, median, count"
            )
        rolled = method().add_suffix(f"__{stat}")
        parts.append(rolled)
    out = pd.concat(parts, axis=1)
    out.index = df.index
    return out


def apply_spectral_rolling(
    df: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """Rolling spectral features (dominant frequency + spectral flatness)
    over the kinematic columns of ``df``.

    Returns a DataFrame with ``2 * n_kinematic`` columns named
    ``<feature>__domfreq`` / ``<feature>__specflat`` (all domfreq, then
    all specflat — matching :func:`apply_rolling`'s stat-major layout).
    Non-kinematic columns are ignored; if there are none the result is an
    empty-column frame aligned to ``df.index``.

    Each output row uses the trailing ``window`` frames; the first
    ``window-1`` rows are NaN and get dropped downstream, exactly like
    :func:`apply_rolling` with the strict ``min_periods=window``. The
    per-window math is :func:`~glider.analysis.behavior.spectral.window_spectral_batch`
    — the vectorized form of the same function the live buffer calls, so
    train and inference can't drift (a parity test pins this). One batched
    FFT over all windows replaces a Python call per window, which is what
    keeps this affordable over millions of frames.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    kin = kinematic_feature_names(df.columns)
    if not kin:
        return pd.DataFrame(index=df.index)

    n = len(df)
    cols: dict[str, np.ndarray] = {}
    for name in kin:
        dom = np.full(n, np.nan)
        flat = np.full(n, np.nan)
        if n >= window:
            col = df[name].to_numpy(dtype=np.float64)
            # (n-window+1, window): trailing window j covers frames
            # [j .. j+window-1], landing on output row j+window-1.
            views = np.lib.stride_tricks.sliding_window_view(col, window)
            res = window_spectral_batch(views)
            dom[window - 1 :] = res[:, 0]
            flat[window - 1 :] = res[:, 1]
        cols[f"{name}__domfreq"] = dom
        cols[f"{name}__specflat"] = flat

    out = pd.DataFrame(cols, index=df.index)
    # Pin column order to the shared canonical layout.
    return out[spectral_column_names(kin)]
