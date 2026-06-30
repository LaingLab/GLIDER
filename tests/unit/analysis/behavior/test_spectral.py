"""Tests for the shared windowed spectral features (analysis/behavior/spectral.py).

These two stats — dominant frequency (cycles per window) and spectral
flatness — capture rhythmicity, which is what tells rhythmic-but-
stationary behaviors (grooming, digging) apart from translational ones
(locomotion). The SAME function runs on the training side (pandas
rolling apply) and the live side (numpy buffer), so its behavior is
pinned here once.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_dominant_frequency_of_pure_sine_is_its_cycle_count():
    from glider.analysis.behavior.spectral import window_spectral

    n, k = 30, 5
    t = np.arange(n)
    sine = np.sin(2 * np.pi * k * t / n)

    dom_freq, _flat = window_spectral(sine)
    assert dom_freq == pytest.approx(float(k))


def test_rhythmic_signal_is_more_tonal_than_white_noise():
    """A clean oscillation has far lower spectral flatness than noise."""
    from glider.analysis.behavior.spectral import window_spectral

    n = 30
    t = np.arange(n)
    sine = np.sin(2 * np.pi * 4 * t / n)
    noise = np.random.default_rng(0).normal(size=n)

    _f1, sine_flat = window_spectral(sine)
    _f2, noise_flat = window_spectral(noise)

    assert sine_flat < 0.1
    assert noise_flat > sine_flat + 0.1


def test_too_few_finite_points_returns_nan():
    from glider.analysis.behavior.spectral import window_spectral

    arr = np.array([1.0, np.nan, np.nan, np.nan, 2.0])  # only 2 finite
    dom_freq, flat = window_spectral(arr)
    assert np.isnan(dom_freq)
    assert np.isnan(flat)


def test_constant_signal_has_no_dominant_frequency():
    from glider.analysis.behavior.spectral import window_spectral

    dom_freq, flat = window_spectral(np.full(30, 3.7))
    assert dom_freq == 0.0
    assert np.isnan(flat)


def test_apply_spectral_rolling_emits_only_kinematic_columns():
    import pandas as pd

    from glider.analysis.behavior.windowing import apply_spectral_rolling

    n, window, k = 60, 30, 5
    t = np.arange(n)
    df = pd.DataFrame(
        {
            "dist_nose_tail": np.linspace(0, 1, n),  # not kinematic → ignored
            "speed_nose": np.sin(2 * np.pi * k * t / window),
            "accel_tail": np.zeros(n),
        }
    )
    out = apply_spectral_rolling(df, window=window)

    # Only kinematic features get spectral columns.
    assert set(out.columns) == {
        "speed_nose__domfreq",
        "accel_tail__domfreq",
        "speed_nose__specflat",
        "accel_tail__specflat",
    }
    # Partial windows (first window-1 rows) are NaN; full windows resolve.
    assert out["speed_nose__domfreq"].iloc[: window - 1].isna().all()
    assert out["speed_nose__domfreq"].iloc[-1] == pytest.approx(float(k))
    assert len(out) == n


def test_window_spectral_batch_matches_scalar_row_by_row():
    """The vectorized batch must give identical results to the scalar
    window_spectral for every row — that equality is what keeps the fast
    training path and the live per-window path in lockstep."""
    from glider.analysis.behavior.spectral import window_spectral, window_spectral_batch

    rng = np.random.default_rng(3)
    _n, w = 50, 30
    rows = []
    rows.append(np.sin(2 * np.pi * 5 * np.arange(w) / w))  # tonal
    rows.append(rng.normal(size=w))  # noise
    rows.append(np.full(w, 2.0))  # constant
    holed = np.sin(2 * np.pi * 6 * np.arange(w) / w)
    holed[[1, 9, 17]] = np.nan
    rows.append(holed)  # with NaNs
    rows.append(np.array([1.0] + [np.nan] * (w - 1)))  # too few finite
    stack = np.vstack(rows)

    batch = window_spectral_batch(stack)
    for i in range(len(stack)):
        dom, flat = window_spectral(stack[i])
        np.testing.assert_allclose(batch[i, 0], dom, equal_nan=True)
        np.testing.assert_allclose(batch[i, 1], flat, equal_nan=True)


def test_kinematic_feature_names_selects_speed_accel_angvel_only():
    from glider.analysis.behavior.spectral import kinematic_feature_names

    cols = [
        "body_length",
        "dist_nose_tail",
        "angle_a_b_c",
        "speed_nose",
        "accel_tail",
        "body_angular_velocity",
    ]
    assert kinematic_feature_names(cols) == [
        "speed_nose",
        "accel_tail",
        "body_angular_velocity",
    ]


def test_kinematic_feature_names_includes_motion_energy_columns():
    """Image motion-energy columns (analysis/behavior/motion.py) count as kinematic, so
    `--motion-features --freq-features` puts spectral (rhythmicity) on them —
    the grooming face-wash signal the trunk keypoints can't carry. Non-motion
    columns are still excluded."""
    from glider.analysis.behavior.spectral import kinematic_feature_names

    cols = [
        "body_length",
        "dist_nose_tail",
        "speed_nose",
        "motion_total",
        "motion_anterior",
        "motion_spread",
    ]
    assert kinematic_feature_names(cols) == [
        "speed_nose",
        "motion_total",
        "motion_anterior",
        "motion_spread",
    ]


def test_spectral_column_names_groups_domfreq_then_specflat():
    from glider.analysis.behavior.spectral import spectral_column_names

    names = spectral_column_names(["speed_nose", "accel_tail"])
    assert names == [
        "speed_nose__domfreq",
        "accel_tail__domfreq",
        "speed_nose__specflat",
        "accel_tail__specflat",
    ]


def test_nans_are_filled_before_fft_so_rhythm_still_detected():
    """A few missing frames (NaN'd low-confidence keypoints) shouldn't hide
    a clear oscillation — they're filled with the window mean first."""
    from glider.analysis.behavior.spectral import window_spectral

    n, k = 30, 6
    t = np.arange(n)
    sine = np.sin(2 * np.pi * k * t / n)
    holed = sine.copy()
    holed[[3, 11, 20]] = np.nan

    dom_freq, _flat = window_spectral(holed)
    assert dom_freq == pytest.approx(float(k))
