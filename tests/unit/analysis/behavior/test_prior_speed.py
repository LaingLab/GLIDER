import numpy as np
import pandas as pd

from glider.analysis.behavior.prior import calibrate_thresholds, prior_speed


def _frame(speeds_by_kp):
    # two keypoints, windowed mean columns + an unrelated column
    return pd.DataFrame(
        {
            "speed_nose__mean": speeds_by_kp[0],
            "speed_tail__mean": speeds_by_kp[1],
            "dist_a_b__mean": [0.1] * len(speeds_by_kp[0]),
        }
    )


def test_prior_speed_is_mean_over_keypoint_speed_means():
    df = _frame([[0.0, 2.0], [2.0, 4.0]])
    np.testing.assert_allclose(prior_speed(df), [1.0, 3.0])


def test_prior_speed_raises_without_speed_columns():
    import pytest

    with pytest.raises(ValueError):
        prior_speed(pd.DataFrame({"dist_a_b__mean": [0.1, 0.2]}))


def test_calibrate_returns_freeze_below_dart():
    s = np.linspace(0, 10, 101)
    freeze, dart = calibrate_thresholds(s, freeze_pct=10.0, dart_pct=99.5)
    assert freeze < dart
    assert 0.5 < freeze < 1.5  # ~10th percentile of 0..10
    assert dart > 9.0  # ~99.5th percentile


def test_calibrate_ignores_nan():
    s = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
    freeze, dart = calibrate_thresholds(s, freeze_pct=0.0, dart_pct=100.0)
    assert freeze == 1.0 and dart == 3.0
