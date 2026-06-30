"""Tests for the train-time embedding fit (train/embedding.py).

The embedding projects high-dimensional windowed feature vectors into a
3D space so the live pipeline can stream classified points through a
"galaxy" of training points. We fit it once here on the training data
and reuse the fitted reducer at inference time via ``.transform``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")


def _make_feature_frame(n: int = 300, d: int = 8, seed: int = 0):
    """Three labeled blobs in d-dim feature space + their labels."""
    rng = np.random.default_rng(seed)
    per = n // 3
    chunks = []
    labels = []
    for c, center in enumerate((0.0, 5.0, -5.0)):
        chunks.append(rng.normal(center, 0.5, size=(per, d)))
        labels.extend([f"beh{c}"] * per)
    x = pd.DataFrame(np.vstack(chunks), columns=[f"f{i}" for i in range(d)])
    y = pd.Series(labels, name="label")
    return x, y


def test_fit_embedding_pca_returns_3d_coords_and_labels():
    from glider.analysis.behavior.embedding import fit_embedding

    x, y = _make_feature_frame()
    art = fit_embedding(x, y, method="pca")

    assert art is not None
    assert art.method == "pca"
    assert art.coords.shape == (len(x), 3)
    assert len(art.labels) == len(x)
    # Labels preserved (possibly reordered by subsampling, but same set).
    assert set(art.labels.tolist()) == {"beh0", "beh1", "beh2"}


def test_fit_embedding_subsamples_to_max_points():
    from glider.analysis.behavior.embedding import fit_embedding

    x, y = _make_feature_frame(n=300)
    art = fit_embedding(x, y, method="pca", max_points=60)

    assert art.coords.shape == (60, 3)
    assert len(art.labels) == 60


def test_fit_embedding_transform_maps_new_row_to_3d():
    from glider.analysis.behavior.embedding import fit_embedding

    x, y = _make_feature_frame()
    art = fit_embedding(x, y, method="pca")

    # A single new feature row -> one 3D point.
    point = art.transform(x.iloc[[0]])
    assert point.shape == (1, 3)
    # Deterministic: transforming the same row twice gives the same point.
    np.testing.assert_allclose(point, art.transform(x.iloc[[0]]))


def test_fit_embedding_transform_accepts_bare_ndarray_without_warning():
    """The live projector passes a raw 1-row ndarray; transform must wrap
    it in the stored feature columns so sklearn doesn't warn or differ."""
    import warnings

    from glider.analysis.behavior.embedding import fit_embedding

    x, y = _make_feature_frame()
    art = fit_embedding(x, y, method="pca")

    row_df = x.iloc[[3]]
    row_arr = x.iloc[3].to_numpy().reshape(1, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        from_arr = art.transform(row_arr)
    np.testing.assert_allclose(from_arr, art.transform(row_df))


def test_fit_embedding_artifact_round_trips_through_joblib(tmp_path):
    import joblib

    from glider.analysis.behavior.embedding import fit_embedding

    x, y = _make_feature_frame()
    art = fit_embedding(x, y, method="pca")

    path = tmp_path / "art.joblib"
    joblib.dump(art, path)
    loaded = joblib.load(path)

    # The reloaded reducer still transforms identically.
    np.testing.assert_allclose(loaded.transform(x.iloc[[5]]), art.transform(x.iloc[[5]]))


def _tiny_model(embedding=None):
    """A minimal fitted BehaviorModel for bundle round-trip tests."""
    from sklearn.ensemble import RandomForestClassifier

    from glider.analysis.behavior import BehaviorModel, FeatureSpec

    x = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["x", "y"])
    clf = RandomForestClassifier(n_estimators=4, random_state=0).fit(x, ["a", "b"])
    return BehaviorModel(
        classifier=clf,
        feature_names=["x", "y"],
        spec=FeatureSpec(),
        window=10,
        stats=("mean",),
        fps=30.0,
        classes=["a", "b"],
        embedding=embedding,
    )


def test_model_bundle_round_trips_with_embedding(tmp_path):
    from glider.analysis.behavior import BehaviorModel
    from glider.analysis.behavior.embedding import fit_embedding

    x, y = _make_feature_frame()
    art = fit_embedding(x, y, method="pca")
    model = _tiny_model(embedding=art)

    path = tmp_path / "model.pkl"
    model.save(path)
    reloaded = BehaviorModel.load(path)

    assert reloaded.embedding is not None
    assert reloaded.embedding.method == "pca"
    np.testing.assert_allclose(reloaded.embedding.coords, art.coords)
    # The reloaded reducer still projects a new row identically.
    np.testing.assert_allclose(
        reloaded.embedding.transform(x.iloc[[0]]), art.transform(x.iloc[[0]])
    )


def test_model_bundle_round_trips_without_embedding(tmp_path):
    from glider.analysis.behavior import BehaviorModel

    model = _tiny_model(embedding=None)
    path = tmp_path / "model.pkl"
    model.save(path)
    reloaded = BehaviorModel.load(path)

    assert reloaded.embedding is None


def test_legacy_bundle_without_embedding_key_loads_as_none(tmp_path):
    """A format_version-1 bundle (no 'embedding' key) loads with embedding=None."""
    import joblib

    from glider.analysis.behavior import BehaviorModel

    model = _tiny_model(embedding=None)
    path = tmp_path / "legacy.pkl"
    model.save(path)
    # Strip the key to simulate a bundle written before embeddings existed.
    payload = joblib.load(path)
    payload.pop("embedding", None)
    payload["format_version"] = 1
    joblib.dump(payload, path)

    reloaded = BehaviorModel.load(path)
    assert reloaded.embedding is None


def _one_nn_label_purity(coords, labels):
    """Fraction of points whose nearest other point shares their label."""
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    hits = 0
    for i in range(len(coords)):
        d = np.linalg.norm(coords - coords[i], axis=1)
        d[i] = np.inf
        j = int(np.argmin(d))
        hits += int(labels[j] == labels[i])
    return hits / len(coords)


def test_supervised_umap_uses_labels_to_separate_overlapping_classes():
    """Supervised UMAP pulls same-label points together even when the
    feature blobs overlap — proving it actually consumes ``y``. Two classes
    drawn from the SAME distribution can't be separated unsupervised, but
    can be when the labels guide the fit."""
    pytest.importorskip("umap")
    from glider.analysis.behavior.embedding import fit_embedding

    rng = np.random.default_rng(0)
    # Two classes, identical feature distribution → fully overlapping.
    d = 8
    x = pd.DataFrame(rng.normal(0.0, 1.0, size=(160, d)), columns=[f"f{i}" for i in range(d)])
    y = pd.Series(["a"] * 80 + ["b"] * 80, name="label")

    sup = fit_embedding(x, y, method="umap", supervised=True)
    uns = fit_embedding(x, y, method="umap", supervised=False)

    sup_purity = _one_nn_label_purity(sup.coords, sup.labels)
    uns_purity = _one_nn_label_purity(uns.coords, uns.labels)

    # Supervised should cleanly group by label; unsupervised can't (chance ~0.5).
    assert sup_purity > 0.9
    assert sup_purity > uns_purity + 0.2


def test_supervised_umap_transform_needs_no_labels():
    """Out-of-sample .transform projects new points without labels (the
    live galaxy never knows the true class), so it stays usable."""
    pytest.importorskip("umap")
    from glider.analysis.behavior.embedding import fit_embedding

    x, y = _make_feature_frame()
    art = fit_embedding(x, y, method="umap", supervised=True)
    point = art.transform(x.iloc[[0]])
    assert point.shape == (1, 3)


def test_fit_embedding_umap_falls_back_to_pca_when_unavailable(monkeypatch):
    """Requesting umap when umap-learn can't be imported records pca."""
    import builtins

    import glider.analysis.behavior.embedding as emb

    real_import = builtins.__import__

    def _no_umap(name, *args, **kwargs):
        if name == "umap" or name.startswith("umap."):
            raise ImportError("simulated missing umap-learn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_umap)

    x, y = _make_feature_frame()
    art = emb.fit_embedding(x, y, method="umap")
    assert art.method == "pca"
    assert art.coords.shape == (len(x), 3)
