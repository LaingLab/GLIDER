"""Trends across time bins: slopes, polynomial contrasts, first vs last.

No outside library computes repeated-measures polynomial contrasts the way
these do, so they are pinned by identities that must hold exactly: on
equally spaced bins the linear contrast *is* the slope test, and with three
bins the two contrasts together *are* the repeated-measures ANOVA.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from glider.analysis.stats import (
    animal_slopes,
    compare_trend,
    polynomial_contrasts,
    run_one_sample,
    run_paired,
    run_repeated,
    run_trend,
    trend_contrasts,
)

RNG = np.random.default_rng(11)
# Twelve animals over six one-minute bins, declining about 1.5 a minute.
DECLINE = (50 + RNG.normal(0, 4, (12, 1)) - 1.5 * np.arange(6) + RNG.normal(0, 1.5, (12, 6))).round(
    2
)
MINUTES = np.arange(6) + 0.5


class TestOneSample:
    VALUES = np.array([0.4, -0.2, 1.1, 0.8, 0.5, 0.9, 0.3, 1.4, -0.1, 0.6])

    def test_t_matches_scipy(self):
        r = run_one_sample(self.VALUES, 0.0, "x", "t")
        ref = stats.ttest_1samp(self.VALUES, 0.0)
        assert r.statistic == pytest.approx(ref.statistic)
        assert r.p == pytest.approx(ref.pvalue)
        assert r.effect == pytest.approx(self.VALUES.mean() / self.VALUES.std(ddof=1))
        assert r.test == "One-sample t-test" and r.effect_name == "Cohen's d"

    def test_mu_shifts_the_comparison(self):
        r = run_one_sample(self.VALUES, 1.0, "ratio", "t")
        assert r.statistic == pytest.approx(stats.ttest_1samp(self.VALUES, 1.0).statistic)
        assert r.difference == pytest.approx(self.VALUES.mean() - 1.0)

    def test_wilcoxon_matches_scipy(self):
        r = run_one_sample(self.VALUES, 0.0, "x", "wilcoxon")
        assert r.p == pytest.approx(stats.wilcoxon(self.VALUES).pvalue)

    def test_its_mean_ci_brackets_the_mean(self):
        low, high = run_one_sample(self.VALUES).difference_ci
        assert low < self.VALUES.mean() < high

    def test_paired_is_one_sample_on_the_differences(self):
        a, b = self.VALUES + 3, self.VALUES[::-1]
        paired = run_paired(a, b, test="paired_t")
        single = run_one_sample(a - b, test="t")
        assert paired.statistic == pytest.approx(single.statistic)
        assert paired.p == pytest.approx(single.p)


class TestContrastWeights:
    def test_orthonormal_zero_sum_and_signed_to_rise(self):
        w = polynomial_contrasts(MINUTES)
        assert w.shape == (6, 2)
        assert w.T @ w == pytest.approx(np.eye(2), abs=1e-12)
        assert w.sum(axis=0) == pytest.approx([0, 0], abs=1e-12)
        assert w[-1, 0] > w[0, 0]  # linear rises with time
        assert w[0, 1] > w[2, 1]  # quadratic is U-shaped

    def test_unequal_spacing_is_respected(self):
        times = np.array([0.5, 1.5, 2.5, 3.25])  # a short last bin
        linear = polynomial_contrasts(times)[:, 0]
        centred = times - times.mean()
        assert linear == pytest.approx(centred / np.linalg.norm(centred))

    def test_two_bins_have_only_a_linear_contrast(self):
        assert polynomial_contrasts([0.5, 1.5]).shape == (2, 1)


class TestIdentities:
    def test_slopes_match_linregress(self):
        slopes = animal_slopes(DECLINE, MINUTES)
        for row, values in zip(slopes.itertuples(), DECLINE, strict=True):
            fit = stats.linregress(MINUTES, values)
            assert row.slope == pytest.approx(fit.slope)
            assert row.r2 == pytest.approx(fit.rvalue**2)

    def test_equal_spacing_makes_the_linear_contrast_the_slope_test(self):
        linear = trend_contrasts(DECLINE, MINUTES).iloc[0]
        slopes = animal_slopes(DECLINE, MINUTES)["slope"]
        slope_t = stats.ttest_1samp(slopes, 0.0)
        assert linear["t"] == pytest.approx(slope_t.statistic)
        assert linear["p"] == pytest.approx(slope_t.pvalue)

    def test_with_three_bins_the_contrasts_sum_to_the_anova(self):
        three = DECLINE[:, :3]
        contrasts = trend_contrasts(three, MINUTES[:3])
        n = three.shape[0]
        ss_contrasts = n * (contrasts["estimate"] ** 2).sum()
        grand = three.mean()
        ss_bins = n * ((three.mean(axis=0) - grand) ** 2).sum()
        assert ss_contrasts == pytest.approx(ss_bins)
        # And partial eta squared of the linear trend sits under 1.
        assert 0 < contrasts.iloc[0]["partial_eta2"] < 1

    def test_contrasts_are_holm_adjusted_together(self):
        c = trend_contrasts(DECLINE, MINUTES)
        assert (c["p_adj"] >= c["p"]).all()

    def test_a_missing_bin_still_gets_a_slope(self):
        holes = DECLINE.copy()
        holes[0, 2] = np.nan
        slopes = animal_slopes(holes, MINUTES)
        fit = stats.linregress(np.delete(MINUTES, 2), np.delete(holes[0], 2))
        assert slopes.iloc[0]["slope"] == pytest.approx(fit.slope)
        assert slopes.iloc[0]["n_bins"] == 5

    def test_under_three_bins_there_is_no_slope(self):
        sparse = np.full((1, 6), np.nan)
        sparse[0, :2] = [1.0, 2.0]
        assert math.isnan(animal_slopes(sparse, MINUTES).iloc[0]["slope"])


def _tidy(groups: dict[str, np.ndarray], *, last_end: float | None = None) -> pd.DataFrame:
    """As the time-bin export writes it: one row per animal per one-minute bin."""
    rows = []
    for group, matrix in groups.items():
        for i, animal in enumerate(matrix):
            for b, value in enumerate(animal):
                start = 60.0 * b
                end = 60.0 * (b + 1)
                if b == len(animal) - 1 and last_end is not None:
                    end = last_end
                rows.append(
                    {
                        "session": f"{group}{i:02d}",
                        "group": group,
                        "bin": f"{b}–{b + 1} min",
                        "bin_start_s": start,
                        "bin_end_s": end,
                        "dist": value,
                    }
                )
    return pd.DataFrame(rows)


class TestCompareTrend:
    def test_one_group(self):
        r = compare_trend(_tidy({"all": DECLINE}), "dist")
        trend = r.groups["all"]
        assert r.time_unit == "min"
        assert r.times == pytest.approx(MINUTES.tolist())
        assert trend.slope_test.difference == pytest.approx(
            animal_slopes(DECLINE, MINUTES)["slope"].mean()
        )
        assert trend.slope_test.p < 0.001
        assert trend.anova.test == "Repeated-measures ANOVA"
        assert r.slope_comparison is None and r.mixed is None

    def test_first_vs_last_is_a_paired_test_of_the_end_bins(self):
        r = compare_trend(_tidy({"all": DECLINE}), "dist")
        expected = run_paired(DECLINE[:, 0], DECLINE[:, -1], test="auto")
        assert r.groups["all"].habituation.statistic == pytest.approx(expected.statistic)
        animal = r.per_animal.iloc[0]
        assert animal["change"] == pytest.approx(DECLINE[0, -1] - DECLINE[0, 0])
        assert animal["ratio"] == pytest.approx(DECLINE[0, -1] / DECLINE[0, 0])

    def test_groups_are_compared_and_crossed_with_bins(self):
        flat = (50 + RNG.normal(0, 4, (10, 1)) + RNG.normal(0, 1.5, (10, 6))).round(2)
        r = compare_trend(_tidy({"chr2": DECLINE, "ctl": flat}), "dist")
        assert set(r.groups) == {"chr2", "ctl"}
        assert r.slope_comparison.p < 0.001
        assert r.mixed is not None
        assert list(r.mixed.effects) == ["group", "bin", "group × bin"]
        assert r.mixed.effects["group × bin"].p < 0.001
        assert "per min" in r.methods()

    def test_a_partial_last_bin_is_left_out_by_default(self):
        """A total over half a minute halves: it must not read as habituation."""
        steady = (50 + RNG.normal(0, 4, (12, 1)) + RNG.normal(0, 1.0, (12, 6))).round(2)
        partial = steady.copy()
        partial[:, -1] *= 0.5  # the last bin is 30 s of a steady animal
        tidy = _tidy({"all": partial}, last_end=330.0)

        kept = compare_trend(tidy, "dist", include_partial=True)
        assert kept.groups["all"].slope_test.p < 0.05  # the fake decline

        r = compare_trend(tidy, "dist")
        assert r.excluded_bins == ["5–6 min"]
        assert len(r.bins) == 5
        assert "partial bin" in r.warnings[0]
        assert r.groups["all"].slope_test.p > 0.05  # steady, as it was

    def test_an_animal_with_too_few_bins_is_named(self):
        tidy = _tidy({"all": DECLINE})
        short = tidy["session"].eq("all03") & tidy["bin_start_s"].ge(120)
        tidy.loc[short, "dist"] = np.nan
        r = compare_trend(tidy, "dist")
        assert any(d.session == "all03" and "slope" in d.reason for d in r.dropped)
        # The contrasts and ANOVA use complete animals only.
        assert r.groups["all"].anova.n["0–1 min"] == 11

    def test_without_bin_times_labels_are_taken_in_order(self):
        tidy = _tidy({"all": DECLINE}).drop(columns=["bin_start_s", "bin_end_s"])
        tidy = tidy.rename(columns={"bin": "range"})
        r = compare_trend(tidy, "dist")
        assert r.time_unit == "bin"
        assert r.times == [0, 1, 2, 3, 4, 5]
        assert r.excluded_bins == []

    def test_one_bin_is_refused(self):
        with pytest.raises(ValueError):
            run_trend({"all": (DECLINE[:, :1], ["a"] * 12)}, ["0–1 min"], [0.5])

    def test_the_anova_matches_running_it_directly(self):
        r = compare_trend(_tidy({"all": DECLINE}), "dist")
        bins = r.bins
        direct = run_repeated(DECLINE, bins, "dist")
        assert r.groups["all"].anova.statistic == pytest.approx(direct.statistic)
