import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from glider.analysis.behavior.features import FeatureSpec
from glider.analysis.behavior.hybrid import HybridModel
from glider.analysis.behavior.model import BehaviorModel
from glider.analysis.behavior.prior import KinematicPrior


def _base():
    x = pd.DataFrame({"speed_a__mean": [0, 0, 9, 9, 0, 9], "speed_b__mean": [0, 0, 9, 9, 0, 9]})
    y = ["rest", "rest", "locomote", "locomote", "rest", "locomote"]
    clf = RandomForestClassifier(n_estimators=8, random_state=0).fit(x, y)
    return BehaviorModel(
        clf,
        ["speed_a__mean", "speed_b__mean"],
        FeatureSpec(),
        1,
        ("mean",),
        30.0,
        ["locomote", "rest"],
    )


TAG_MAP = {"rest": frozenset({"stationary"}), "locomote": frozenset({"locomotory"})}


def _prior():
    p = KinematicPrior(tag_map=TAG_MAP)
    p.calibrate(
        pd.DataFrame(
            {"speed_a__mean": np.linspace(0, 9, 20), "speed_b__mean": np.linspace(0, 9, 20)}
        )
    )
    return p


def _frame(v):
    return pd.DataFrame({"speed_a__mean": v, "speed_b__mean": v})


def test_lambda_zero_is_exactly_supervised():
    base, prior = _base(), _prior()
    hyb = HybridModel(base, prior, lam=0.0, tag_map=TAG_MAP)
    x = _frame([0.0, 9.0, 4.5])
    np.testing.assert_array_equal(hyb.predict(x), base.predict(x))


def test_nan_row_passes_through_as_unknown():
    base, prior = _base(), _prior()
    hyb = HybridModel(base, prior, lam=0.5, tag_map=TAG_MAP)
    x = _frame([0.0, np.nan])
    assert hyb.predict(x)[1] == ""


def test_prior_shifts_argmax_toward_stationary_when_slow():
    base, prior = _base(), _prior()
    hi = HybridModel(base, prior, lam=0.9, tag_map=TAG_MAP)
    x = _frame([0.0])  # unambiguously slow -> freeze prior favors 'rest'
    assert hi.predict(x)[0] == "rest"


def test_classes_property():
    base, prior = _base(), _prior()
    hyb = HybridModel(base, prior, lam=0.3, tag_map=TAG_MAP)
    assert set(hyb.classes) == {"rest", "locomote"}


def test_save_load_round_trip(tmp_path):
    base, prior = _base(), _prior()
    hyb = HybridModel(base, prior, lam=0.3, tag_map=TAG_MAP)
    p = tmp_path / "hybrid.pkl"
    hyb.save(p)
    loaded = HybridModel.load(p)
    x = _frame([0.0, 9.0])
    np.testing.assert_array_equal(loaded.predict(x), hyb.predict(x))
    # Would catch a dropped prior field: thresholds, rules, and tag_map must
    # all survive the round-trip, not just the predictions on one frame.
    assert loaded.prior._freeze_thr == prior._freeze_thr
    assert loaded.prior._dart_thr == prior._dart_thr
    assert loaded.prior._scale == prior._scale
    assert loaded.prior.freeze_pct == prior.freeze_pct
    assert loaded.prior.dart_pct == prior.dart_pct
    assert loaded.prior.tag_map == prior.tag_map
    assert [(r.name, r.tag_weights) for r in loaded.prior.rules] == [
        (r.name, r.tag_weights) for r in prior.rules
    ]


def test_lam_out_of_range_raises():
    base, prior = _base(), _prior()
    with pytest.raises(ValueError):
        HybridModel(base, prior, lam=1.5, tag_map=TAG_MAP)
    with pytest.raises(ValueError):
        HybridModel(base, prior, lam=-0.1, tag_map=TAG_MAP)


def test_load_rejects_non_hybrid_bundle(tmp_path):
    import joblib

    p = tmp_path / "bad.pkl"
    joblib.dump({"kind": "something_else"}, p)
    with pytest.raises(ValueError):
        HybridModel.load(p)
