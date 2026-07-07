import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from glider.analysis.behavior.features import FeatureSpec
from glider.analysis.behavior.model import BehaviorModel


def _fitted_model():
    x = pd.DataFrame({"f0": [0, 0, 1, 1, 0, 1], "f1": [0, 1, 0, 1, 0, 1]})
    y = ["a", "a", "b", "b", "a", "b"]
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(x, y)
    return BehaviorModel(clf, ["f0", "f1"], FeatureSpec(), 1, ("mean",), 30.0, ["a", "b"])


def test_posteriors_aligned_and_masks_nan_rows():
    m = _fitted_model()
    x = pd.DataFrame({"f0": [0.0, 1.0, np.nan], "f1": [0.0, 1.0, 0.0]})
    probs, valid = m.posteriors(x)
    assert list(m.classifier.classes_) == ["a", "b"]
    assert probs.shape == (2, 2)  # only the 2 non-NaN rows
    assert valid.tolist() == [True, True, False]
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_posteriors_all_nan_returns_empty():
    m = _fitted_model()
    x = pd.DataFrame({"f0": [np.nan, np.nan], "f1": [0.0, 1.0]})
    probs, valid = m.posteriors(x)
    assert probs.shape == (0, 2)
    assert valid.tolist() == [False, False]


def test_posteriors_raises_when_unfitted():
    m = BehaviorModel(object(), ["f0"], FeatureSpec(), 1, ("mean",), 30.0, ["a"])
    with pytest.raises(RuntimeError):
        m.posteriors(pd.DataFrame({"f0": [0.0]}))
