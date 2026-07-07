import pytest

from glider.analysis.behavior.pipeline import _build_classifier


def test_default_classifier_is_lightgbm():
    clf = _build_classifier(
        classifier_type="lightgbm",
        n_estimators=50,
        random_state=0,
        class_weight=None,
        lgbm_reg=None,
    )
    assert type(clf).__name__ == "LGBMClassifier"


def test_hybrid_hard_requires_lightgbm(monkeypatch):
    # When lightgbm import fails, the hybrid path must ERROR, not silently
    # fall back to RandomForest.
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "lightgbm":
            raise ImportError("no lightgbm")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="lightgbm"):
        _build_classifier(
            classifier_type="lightgbm",
            n_estimators=50,
            random_state=0,
            class_weight=None,
            lgbm_reg=None,
            require=True,
        )
