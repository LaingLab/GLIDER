"""Sliding feature buffer for live inference.

Wraps :class:`collections.deque` (the "buffer" the project plan calls
for) plus the per-tick rolling-stats computation that produces the
1-D feature row the trained model consumes.

This module deliberately avoids pandas in the hot path. The per-frame
features are stored as ``np.ndarray`` rows and the rolling stats are
computed via :func:`numpy.nanmean` / :func:`numpy.nanstd` /
:func:`numpy.nanmax` directly. For a 30-frame window over ~40 base
features that's well under a millisecond per tick, leaving the YOLO
forward pass as the real bottleneck.

The buffer is single-writer / single-reader by convention — owned by
the FeatureEngine thread. It is **not** internally thread-safe; the
class assumes the caller serializes access.
"""

from __future__ import annotations

import functools
from collections import deque
from collections.abc import Iterable

import numpy as np
import pandas as pd


class SlidingFeatureBuffer:
    """A bounded ring of per-frame feature rows + rolling-stat compute.

    Parameters
    ----------
    feature_names
        Names of the per-frame base features (i.e. the columns
        :func:`glider.analysis.behavior.features.compute_features` produces, NOT
        the windowed names with ``__mean`` suffix). Order is
        preserved internally.
    window
        Buffer capacity in frames. Defaults to 30 to match the project
        plan's "30-frame moving average" target.
    stats
        Which rolling aggregations to compute on each
        :meth:`rolling_features` call. Each is mapped to a NaN-safe
        numpy reduction.
    """

    # ddof=1 on std to match pandas ``.rolling().std()`` used in training
    # (windowing.apply_rolling); ddof=0 would scale every __std feature by
    # sqrt(N/(N-1)) and silently drift live predictions off the trained model.
    _REDUCTIONS: dict[str, callable] = {
        "mean": np.nanmean,
        "std": functools.partial(np.nanstd, ddof=1),
        "max": np.nanmax,
        "min": np.nanmin,
        "sum": np.nansum,
        "median": np.nanmedian,
    }

    def __init__(
        self,
        feature_names: list[str],
        window: int = 30,
        stats: Iterable[str] = ("mean", "std", "max"),
        spectral_features: list[str] | None = None,
    ):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.feature_names = list(feature_names)
        self.window = int(window)
        self.stats = tuple(stats)
        for s in self.stats:
            if s not in self._REDUCTIONS:
                raise ValueError(
                    f"stat {s!r} not supported; try one of " f"{list(self._REDUCTIONS)}"
                )
        # Kinematic base names that also get rolling spectral features
        # (dominant frequency + spectral flatness). Their column indices
        # are cached so the hot path doesn't re-scan feature_names.
        self.spectral_features = list(spectral_features) if spectral_features else []
        self._spectral_idx = [
            self.feature_names.index(n) for n in self.spectral_features if n in self.feature_names
        ]
        self._buffer: deque[np.ndarray] = deque(maxlen=self.window)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def push_features(self, features: dict[str, float] | np.ndarray | pd.Series) -> None:
        """Append one frame's features to the buffer.

        Accepts a dict / Series (keyed by feature name) or a raw 1-D
        array already in :attr:`feature_names` order. Missing or unknown
        keys become NaN.
        """
        if isinstance(features, np.ndarray):
            if features.shape != (len(self.feature_names),):
                raise ValueError(
                    f"array features must be 1-D of length "
                    f"{len(self.feature_names)}; got {features.shape}"
                )
            row = features.astype(np.float64, copy=True)
        elif isinstance(features, pd.Series):
            row = features.reindex(self.feature_names).to_numpy(dtype=np.float64)
        else:
            row = np.full(len(self.feature_names), np.nan, dtype=np.float64)
            for i, name in enumerate(self.feature_names):
                if name in features:
                    val = features[name]
                    row[i] = float(val) if val is not None else np.nan
        self._buffer.append(row)

    def clear(self) -> None:
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buffer)

    def is_full(self) -> bool:
        """True once the buffer has reached its capacity."""
        return len(self._buffer) >= self.window

    # ------------------------------------------------------------------
    # Rolling stats
    # ------------------------------------------------------------------
    def rolling_features(self) -> tuple[list[str], np.ndarray]:
        """Compute rolling stats over the current buffer contents.

        Returns ``(column_names, row_values)`` where ``column_names`` is
        a list of ``"{feature}__{stat}"`` strings (mean before std
        before max, in the configured order), and ``row_values`` is a
        1-D float64 array of the same length.

        When the buffer is empty, returns ``(names, all-NaN array)``.

        NaN columns (e.g. body_length when a keypoint is missing) are
        handled by ``nan*`` reductions: if every value in the window is
        NaN, the reduction returns NaN, otherwise it skips them. ``std``
        of a single non-NaN value is 0 (numpy convention).
        """
        from glider.analysis.behavior.spectral import spectral_column_names, window_spectral

        n_feat = len(self.feature_names)
        n_buf = len(self._buffer)
        column_names = [f"{name}__{stat}" for stat in self.stats for name in self.feature_names]
        spectral_names = spectral_column_names(self.spectral_features)

        if n_buf == 0:
            empty = np.full(n_feat * len(self.stats) + len(spectral_names), np.nan)
            return column_names + spectral_names, empty

        # Stack the buffered rows into a (n_buf, n_feat) array.
        arr = np.vstack(self._buffer)  # shape (n_buf, n_feat)

        # Compute each stat in turn. NaN-safe reductions emit a
        # RuntimeWarning when *all* values in a column are NaN; we
        # suppress that — the resulting NaN is the right answer for the
        # downstream classifier ("we don't know, emit blank").
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN slice encountered", RuntimeWarning)
            warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
            warnings.filterwarnings("ignore", r"Degrees of freedom <= 0", RuntimeWarning)
            parts: list[np.ndarray] = []
            for stat in self.stats:
                reduce_fn = self._REDUCTIONS[stat]
                parts.append(reduce_fn(arr, axis=0))
        row = np.concatenate(parts)

        if self._spectral_idx:
            # Compute (domfreq, specflat) per kinematic column via the SAME
            # function the training rolling apply uses. Stat-major order:
            # all domfreq, then all specflat — matching spectral_column_names.
            pairs = [window_spectral(arr[:, j]) for j in self._spectral_idx]
            dom = np.array([p[0] for p in pairs], dtype=np.float64)
            flat = np.array([p[1] for p in pairs], dtype=np.float64)
            row = np.concatenate([row, dom, flat])
            column_names = column_names + spectral_names
        return column_names, row

    def rolling_dict(self) -> dict[str, float]:
        """Convenience wrapper returning the rolling row as a dict."""
        names, row = self.rolling_features()
        return {name: float(val) for name, val in zip(names, row, strict=False)}
