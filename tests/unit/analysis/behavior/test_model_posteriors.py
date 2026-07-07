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


def test_posteriors_column_order_follows_classifier_classes_():
    # classifier.classes_ is sklearn-sorted -> ["a", "b"], but the model is
    # constructed with self.classes reversed -> ["b", "a"]. The returned
    # probability columns must follow classifier.classes_, not self.classes.
    x = pd.DataFrame({"f0": [0, 0, 1, 1, 0, 1], "f1": [0, 1, 0, 1, 0, 1]})
    y = ["a", "a", "b", "b", "a", "b"]
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(x, y)
    assert list(clf.classes_) == ["a", "b"]
    m = BehaviorModel(clf, ["f0", "f1"], FeatureSpec(), 1, ("mean",), 30.0, ["b", "a"])
    assert m.classes == ["b", "a"]
    # A [0, 0] row is an unambiguous "a" example in the training data.
    probs, valid = m.posteriors(pd.DataFrame({"f0": [0.0], "f1": [0.0]}))
    assert valid.tolist() == [True]
    assert clf.classes_[probs[0].argmax()] == "a"


def test_posteriors_raises_when_unfitted():
    m = BehaviorModel(object(), ["f0"], FeatureSpec(), 1, ("mean",), 30.0, ["a"])
    with pytest.raises(RuntimeError):
        m.posteriors(pd.DataFrame({"f0": [0.0]}))
