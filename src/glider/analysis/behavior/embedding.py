"""Fit a 3D embedding of the training feature space.

The live pipeline streams classified windows through a 3D "galaxy" of
training points (birdsong-demo style). Because a fresh UMAP/PCA can't be
fit online, we fit the reducer **once** here on the training feature
vectors and store the fitted ``scaler`` + ``reducer`` in the model
bundle. At inference time :meth:`EmbeddingArtifact.transform` projects
each new windowed feature row to a 3D point using that same fit.

UMAP gives the tight, well-separated clusters the demo is famous for and
supports out-of-sample ``.transform``. When ``umap-learn`` isn't
importable (or ``method="pca"`` is requested) we fall back to PCA, which
is always available and trivially supports transforming new points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EmbeddingArtifact:
    """A fitted feature-space embedding, picklable into the model bundle.

    Attributes
    ----------
    method
        ``"umap"`` or ``"pca"`` — the reducer that actually ran (records
        the fallback, so the requested method and the stored one may
        differ).
    scaler
        Fitted ``StandardScaler`` applied before the reducer.
    reducer
        Fitted UMAP or ``PCA(n_components=3)``.
    coords
        ``(M, 3)`` embedded training points (``M`` <= ``max_points``).
    labels
        ``(M,)`` behavior names aligned to ``coords`` (the training
        "galaxy", colored per behavior at render time).
    feature_names
        Column order the scaler/reducer were fit on. Used to wrap bare
        ndarray rows (the live hot path) into a named DataFrame so
        sklearn doesn't warn about missing feature names.
    """

    method: str
    scaler: object
    reducer: object
    coords: np.ndarray
    labels: np.ndarray
    feature_names: list[str]

    def transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Project feature rows to ``(n, 3)`` via the stored scaler + reducer.

        Accepts a DataFrame or a bare ndarray; ndarrays are wrapped in
        :attr:`feature_names` so the scaler sees the columns it was fit
        on.
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(np.atleast_2d(x), columns=self.feature_names)
        scaled = self.scaler.transform(x)
        return np.asarray(self.reducer.transform(scaled), dtype=np.float64)


def fit_embedding(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    method: str = "umap",
    n_components: int = 3,
    max_points: int = 20_000,
    random_state: int = 42,
    supervised: bool = True,
) -> EmbeddingArtifact:
    """Fit a 3D embedding on training features.

    Parameters
    ----------
    x
        Training feature rows (no NaN). The same columns the model was
        fit on.
    y
        Labels aligned to ``x`` — kept alongside the embedded coords so
        the galaxy can be colored per behavior.
    method
        ``"umap"`` (default) or ``"pca"``. ``"umap"`` falls back to PCA
        if ``umap-learn`` can't be imported.
    n_components
        Embedding dimensionality (3 for the 3D view).
    max_points
        Subsample (label-stratified) to at most this many rows before
        fitting — caps UMAP fit cost and render weight.
    random_state
        Seed for the subsample and the reducer.
    supervised
        UMAP only: feed the labels into the fit so same-behavior points
        pull together (target_metric="categorical"). This produces the
        tight, well-separated "galaxy" clusters; out-of-sample
        ``.transform`` still projects new points without labels, so the
        live view is unaffected. PCA ignores this. Note it's a
        visualization aid — the geometry is label-guided, not proof the
        classes are separable in feature space.

    Returns
    -------
    EmbeddingArtifact
    """
    from sklearn.preprocessing import StandardScaler

    x = x.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    idx = _subsample_indices(y, max_points=max_points, random_state=random_state)
    x_sub = x.iloc[idx]
    y_sub = y.iloc[idx].to_numpy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_sub)

    reducer, resolved_method = _build_reducer(
        method=method, n_components=n_components, random_state=random_state
    )
    # Supervised UMAP consumes integer label codes; PCA has no y arg.
    if resolved_method == "umap" and supervised:
        codes = pd.Categorical(y_sub).codes
        coords = np.asarray(reducer.fit_transform(scaled, y=codes), dtype=np.float64)
    else:
        coords = np.asarray(reducer.fit_transform(scaled), dtype=np.float64)

    return EmbeddingArtifact(
        method=resolved_method,
        scaler=scaler,
        reducer=reducer,
        coords=coords,
        labels=y_sub,
        feature_names=list(x.columns),
    )


def _subsample_indices(y: pd.Series, *, max_points: int, random_state: int) -> np.ndarray:
    """Return sorted row indices subsampled to ``max_points``.

    Stratified by label so rare behaviors survive: each class keeps a
    share proportional to its size, with at least one row per class.
    """
    n = len(y)
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    keep: list[np.ndarray] = []
    for _label, grp in y.groupby(y):
        members = grp.index.to_numpy()
        share = max(1, int(round(max_points * len(members) / n)))
        share = min(share, len(members))
        keep.append(rng.choice(members, size=share, replace=False))
    idx = np.concatenate(keep)
    # Stratified rounding can overshoot max_points slightly — trim.
    if len(idx) > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    idx.sort()
    return idx


def _build_reducer(*, method: str, n_components: int, random_state: int):
    """Construct the reducer; UMAP with a PCA fallback when unavailable."""
    method = (method or "umap").lower()
    if method == "umap":
        try:
            import umap  # type: ignore

            return (
                umap.UMAP(n_components=n_components, random_state=random_state),
                "umap",
            )
        except ImportError:
            method = "pca"
    if method != "pca":
        raise ValueError(f"unknown embedding method {method!r}; expected 'umap' or 'pca'")
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, random_state=random_state), "pca"
