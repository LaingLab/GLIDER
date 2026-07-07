import numpy as np
import pandas as pd

from glider.analysis.behavior.prior import KinematicPrior


def _frame(mean_speed):
    return pd.DataFrame({"speed_a__mean": mean_speed, "speed_b__mean": mean_speed})


TAG_MAP = {
    "rest": frozenset({"stationary"}),
    "locomote": frozenset({"locomotory"}),
    "rear": frozenset(),
}  # untagged -> neutral
CLASSES = ["locomote", "rear", "rest"]


def _prior():
    p = KinematicPrior(tag_map=TAG_MAP)
    p.calibrate(_frame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    return p


def test_freeze_row_boosts_stationary_over_locomotory():
    p = _prior()
    lp = p.log_prior(_frame([0.0]), classes=CLASSES)  # very low speed => freeze
    row = lp[0]
    d = dict(zip(CLASSES, row, strict=True))
    assert d["rest"] > d["locomote"]  # stationary boosted, locomotory suppressed
    assert abs(d["rear"]) < 1e-9  # untagged neutral (0)


def test_dart_row_boosts_locomotory_over_stationary():
    p = _prior()
    lp = p.log_prior(_frame([100.0]), classes=CLASSES)  # very high speed => dart
    d = dict(zip(CLASSES, lp[0], strict=True))
    assert d["locomote"] > d["rest"]


def test_midspeed_row_is_neutral():
    p = _prior()
    lp = p.log_prior(_frame([5.0]), classes=CLASSES)  # between thresholds
    assert np.allclose(lp[0], 0.0, atol=1e-6)


def test_tag_transfer_same_rules_new_names():
    # Rename classes but keep the tags: same effect on the same frame.
    tag_map = {"still": frozenset({"stationary"}), "run": frozenset({"locomotory"})}
    p = KinematicPrior(tag_map=tag_map)
    p.calibrate(_frame([0, 2, 4, 6, 8, 10]))
    lp = p.log_prior(_frame([0.0]), classes=["run", "still"])
    d = dict(zip(["run", "still"], lp[0], strict=True))
    assert d["still"] > d["run"]


def test_log_prior_requires_calibration():
    import pytest

    p = KinematicPrior(tag_map=TAG_MAP)
    with pytest.raises(RuntimeError):
        p.log_prior(_frame([0.0]), classes=CLASSES)


def test_unknown_rule_name_rejected_eagerly():
    import pytest

    from glider.analysis.behavior.prior import KinematicPrior, Rule

    with pytest.raises(ValueError, match="activation"):
        KinematicPrior(
            tag_map={"x": frozenset({"stationary"})},
            rules=(Rule("rhythmic", {"rhythmic": 1.0}),),  # no matching activation
        )
