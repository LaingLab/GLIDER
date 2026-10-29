"""The statistics engine, pinned to independent reference values.

The reference numbers below were produced once by pingouin 0.7.0 and
scikit-posthocs 0.17.1 (scipy 1.18.1) on the datasets beside them, in a
throwaway environment. GLIDER does not depend on either library; these
constants are what keeps its own arithmetic honest.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from glider.analysis.stats import (
    UNGROUPED,
    adjust,
    between_cells,
    compare_epochs,
    compare_groups,
    dunn,
    games_howell,
    run_between,
    run_paired,
    tukey,
    within_matrix,
)
from glider.analysis.stats.result import format_p

# Unequal n, unequal spread, rounded so ranks tie.
A = np.array([9.8, 9.4, 9.8, 10.2, 9.0, 11.1, 10.7, 10.1, 10.6])
B = np.array([9.5, 10.2, 12.8, 9.4, 13.3, 13.6, 13.1])
C = np.array([14.5, 21.6, 13.4, 12.7, 13.3, 13.2, 15.0, 10.5, 7.1, 10.3, 16.0])
# The same ten animals, before and during stimulation.
BASE = np.array([18.0, 22.0, 24.2, 15.9, 23.7, 21.3, 21.1, 19.9, 24.3, 17.2])
STIM = np.array([21.1, 22.9, 23.6, 17.2, 26.5, 25.8, 26.6, 22.1, 24.5, 17.8])

TOL = {"rel": 1e-9, "abs": 1e-9}


def _pair(frame: pd.DataFrame, a: str, b: str) -> pd.Series:
    return frame[(frame["a"] == a) & (frame["b"] == b)].iloc[0]


class TestTwoGroups:
    def test_welch(self):
        r = run_between({"A": A, "B": B}, "x", "welch")
        assert r.statistic == pytest.approx(-2.1568514551531504, **TOL)
        assert r.df[0] == pytest.approx(7.132295867688487, **TOL)
        assert r.p == pytest.approx(0.06720500593969006, **TOL)

    def test_student(self):
        r = run_between({"A": A, "B": B}, "x", "student")
        assert r.statistic == pytest.approx(-2.3985588414152024, **TOL)
        assert r.df == (14.0,)

    def test_hedges_g_and_its_interval(self):
        r = run_between({"A": A, "B": B}, "x", "welch")
        assert r.effect_name == "Hedges' g"
        assert r.effect == pytest.approx(-1.142827672262066, **TOL)
        assert r.effect_ci == pytest.approx((-2.307315454348, 0.021660109824), abs=1e-9)

    def test_mann_whitney_and_rank_biserial(self):
        r = run_between({"A": A, "B": B}, "x", "mannwhitney")
        assert r.statistic == 18.0
        assert r.p == pytest.approx(0.16786039631458505, **TOL)
        assert r.effect == pytest.approx(-0.4285714285714286, **TOL)
        assert r.df == ()

    def test_the_difference_and_its_bootstrap_interval(self):
        r = run_between({"A": A, "B": B}, "x", "welch")
        assert r.difference == pytest.approx(A.mean() - B.mean())
        low, high = r.difference_ci
        assert low < r.difference < high

    def test_permutation_is_reproducible(self):
        first = run_between({"A": A, "B": B}, "x", "permutation")
        again = run_between({"A": A, "B": B}, "x", "permutation")
        assert first.p == again.p
        assert first.difference_ci == again.difference_ci
        # Close to Welch's answer on data this well behaved.
        assert first.p == pytest.approx(0.067, abs=0.03)

    def test_a_two_group_test_refuses_three_groups(self):
        r = run_between({"A": A, "B": B, "C": C}, "x", "welch")
        assert not r.ok
        assert "two groups" in r.warnings[0]


class TestSeveralGroups:
    CELLS = {"A": A, "B": B, "C": C}

    def test_one_way_anova(self):
        r = run_between(self.CELLS, "x", "anova")
        assert r.statistic == pytest.approx(4.12870599851146, **TOL)
        assert r.df == (2.0, 24.0)
        assert r.p == pytest.approx(0.028772701951883733, **TOL)
        assert r.effect == pytest.approx(0.25598494999490373, **TOL)
        assert r.posthoc_name == "Tukey HSD"

    def test_welch_anova(self):
        r = run_between(self.CELLS, "x", "welch_anova")
        assert r.statistic == pytest.approx(5.974963959822446, **TOL)
        assert r.df == pytest.approx((2.0, 11.342202939183386), **TOL)
        assert r.p == pytest.approx(0.01689327405596058, **TOL)
        assert r.posthoc_name == "Games-Howell"

    def test_kruskal_wallis(self):
        r = run_between(self.CELLS, "x", "kruskal")
        assert r.statistic == pytest.approx(7.613604610548366, **TOL)
        assert r.p == pytest.approx(0.022219115428219315, **TOL)
        assert r.effect == pytest.approx(7.613604610548366 / 26)
        assert r.posthoc_name == "Dunn (Holm)"

    @pytest.mark.parametrize(
        "a, b, diff, t, p",
        [
            ("A", "B", -1.6222222222222236, -1.2419403543118823, 0.44085695205351405),
            ("A", "C", -3.3404040404040405, -2.867356608864776, 0.022278181867078417),
            ("B", "C", -1.718181818181817, -1.3710659270338121, 0.37143552962877135),
        ],
    )
    def test_tukey(self, a, b, diff, t, p):
        row = _pair(tukey(self.CELLS), a, b)
        assert row["diff"] == pytest.approx(diff, **TOL)
        assert row["statistic"] == pytest.approx(t, **TOL)
        # pingouin and scipy.stats.tukey_hsd agree on these to 1e-15.
        assert row["p_adj"] == pytest.approx(p, rel=1e-7)

    @pytest.mark.parametrize(
        "a, b, t, p",
        [
            ("A", "B", -2.156851455153151, 0.146319810276771),
            ("A", "C", -2.9467072855530065, 0.03341297723416825),
            ("B", "C", -1.297469055214847, 0.4173043299973618),
        ],
    )
    def test_games_howell(self, a, b, t, p):
        row = _pair(games_howell(self.CELLS), a, b)
        assert row["statistic"] == pytest.approx(t, **TOL)
        assert row["p_adj"] == pytest.approx(p, rel=1e-7)

    @pytest.mark.parametrize(
        "a, b, p, p_holm",
        [
            ("A", "B", 0.20035390989521307, 0.40070781979042613),
            ("A", "C", 0.005815153961925041, 0.017445461885775122),
            ("B", "C", 0.21899949966394483, 0.40070781979042613),
        ],
    )
    def test_dunn_is_tie_corrected_and_holm_adjusted(self, a, b, p, p_holm):
        row = _pair(dunn(self.CELLS), a, b)
        assert row["p"] == pytest.approx(p, **TOL)
        assert row["p_adj"] == pytest.approx(p_holm, **TOL)

    def test_two_groups_get_no_posthoc(self):
        assert run_between({"A": A, "B": B}, "x", "anova").posthoc is None

    def test_anova_on_two_groups_is_students_t_squared(self):
        f = run_between({"A": A, "B": B}, "x", "anova")
        t = run_between({"A": A, "B": B}, "x", "student")
        assert f.statistic == pytest.approx(t.statistic**2)
        assert f.p == pytest.approx(t.p)


class TestPaired:
    def test_paired_t(self):
        r = run_paired(BASE, STIM, ("Base", "Stim"), "x", "paired_t")
        assert r.statistic == pytest.approx(-3.3340802494058734, **TOL)
        assert r.df == (9.0,)

    def test_cohen_dz_and_its_interval(self):
        r = run_paired(BASE, STIM, ("Base", "Stim"), "x", "paired_t")
        assert r.effect == pytest.approx(-1.0543287489904811, **TOL)
        assert r.effect_ci == pytest.approx((-1.946606971655, -0.162050526326), abs=1e-9)

    def test_wilcoxon_and_matched_rank_biserial(self):
        r = run_paired(BASE, STIM, ("Base", "Stim"), "x", "wilcoxon")
        assert r.statistic == 2.0
        assert r.p == pytest.approx(0.005859375, **TOL)
        assert r.effect == pytest.approx(-0.9272727272727272, **TOL)

    def test_paired_permutation_is_reproducible(self):
        first = run_paired(BASE, STIM, test="permutation")
        assert first.p == run_paired(BASE, STIM, test="permutation").p
        assert first.p < 0.05

    def test_no_change_at_all_is_reported_not_raised(self):
        r = run_paired(BASE, BASE, test="wilcoxon")
        assert not r.ok and "zero" in r.warnings[0]

    def test_unequal_lengths_are_an_error(self):
        with pytest.raises(ValueError):
            run_paired([1, 2, 3], [1, 2])


class TestAuto:
    # Symmetric and unimodal: Shapiro-Wilk p = 0.999. (B above is bimodal,
    # p = 0.044, so auto rightly sends it to a rank test.)
    NORMAL = np.array([9.1, 9.8, 10.0, 10.3, 10.6, 10.9, 11.5, 9.6, 10.2])

    def test_a_bimodal_group_is_not_called_normal(self):
        assert run_between({"A": A, "B": B}, "x").test == "Mann-Whitney U"

    def test_normal_two_groups_get_welch(self):
        r = run_between({"A": self.NORMAL, "B": self.NORMAL + 1.5}, "x")
        assert r.test == "Welch's t-test"
        assert "Shapiro-Wilk p ≥ .05" in r.reason

    def test_a_skewed_group_gets_mann_whitney_and_names_it(self):
        skewed = np.array([1.0, 1.1, 1.0, 1.2, 1.1, 1.0, 9.0, 12.0, 30.0])
        r = run_between({"A": A, "Skewed": skewed}, "x")
        assert r.test == "Mann-Whitney U"
        assert "in Skewed" in r.reason

    def test_unequal_variances_get_welch_anova(self):
        wide = np.array([-40.0, 30.0, -25.0, 45.0, 0.0, -10.0, 20.0, 35.0, -30.0, 5.0])
        r = run_between({"A": self.NORMAL, "B": self.NORMAL + 1.5, "Wide": wide}, "x")
        assert r.test == "Welch's ANOVA"
        assert "Levene" in r.reason

    def test_equal_variances_get_anova_and_tukey(self):
        shifted = {"A": A, "A+1": A + 1, "A+2": A + 2}
        r = run_between(shifted, "x")
        assert r.test == "One-way ANOVA"
        assert r.posthoc_name == "Tukey HSD"

    def test_normal_differences_get_the_paired_t(self):
        assert run_paired(BASE, STIM).test == "Paired t-test"

    def test_skewed_differences_get_wilcoxon(self):
        stim = BASE + np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 9.0, 15.0])
        assert run_paired(BASE, stim).test == "Wilcoxon signed-rank"

    def test_small_groups_are_warned_about(self):
        r = run_between({"A": A[:4], "B": B[:4]}, "x")
        assert any("fewer than 5" in w for w in r.warnings)

    def test_a_chosen_test_carries_no_reason(self):
        assert run_between({"A": A, "B": B}, "x", "student").reason == ""


class TestRefusals:
    def test_one_group(self):
        r = run_between({"A": A}, "x")
        assert not r.ok and "two groups" in r.warnings[0]

    def test_a_group_of_one(self):
        r = run_between({"A": A, "B": B[:1]}, "x")
        assert not r.ok and "(B)" in r.warnings[0]

    def test_an_unknown_test(self):
        with pytest.raises(ValueError):
            run_between({"A": A, "B": B}, "x", "chi-by-eye")

    def test_constant_groups_give_nan_effect_not_an_exception(self):
        r = run_between({"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]}, "x", "student")
        assert math.isnan(r.effect)


class TestCorrections:
    P = [0.01, 0.04, 0.03, 0.005, 0.2, 0.04]

    def test_holm(self):
        assert adjust(self.P, "holm") == pytest.approx([0.05, 0.12, 0.12, 0.03, 0.2, 0.12])

    def test_bonferroni(self):
        expected = [0.06, 0.24, 0.18, 0.03, 1.0, 0.24]
        assert adjust(self.P, "bonferroni") == pytest.approx(expected)

    def test_benjamini_hochberg(self):
        expected = [0.03, 0.048, 0.048, 0.03, 0.2, 0.048]
        assert adjust(self.P, "fdr_bh") == pytest.approx(expected)

    def test_nan_is_kept_out_of_the_family(self):
        out = adjust([0.01, float("nan"), 0.04], "bonferroni")
        assert out[0] == pytest.approx(0.02) and math.isnan(out[1])

    def test_an_unknown_method(self):
        with pytest.raises(ValueError):
            adjust([0.1], "sidak-ish")


def _tidy() -> pd.DataFrame:
    """Two groups, two epochs; m4 has no Stim value and m6 no group."""
    rows = []
    values = {
        "m1": ("ChR2", 10.0, 14.0),
        "m2": ("ChR2", 11.0, 16.0),
        "m3": ("ChR2", 9.5, 13.0),
        "m4": ("ChR2", 10.5, None),
        "m5": ("eYFP", 10.2, 10.1),
        "m6": ("", 9.9, 10.4),
    }
    for session, (group, base, stim) in values.items():
        rows.append({"session": session, "group": group, "range": "Base", "dist": base})
        rows.append({"session": session, "group": group, "range": "Stim", "dist": stim})
    return pd.DataFrame(rows)


class TestFromTheEpochTable:
    def test_between_cells_split_by_group_with_ungrouped_last(self):
        cells, sessions, dropped = between_cells(_tidy(), "dist", "Stim")
        assert list(cells) == ["ChR2", "eYFP", UNGROUPED]
        assert sessions["ChR2"] == ["m1", "m2", "m3"]
        assert [d.session for d in dropped] == ["m4"]
        assert "Stim" in dropped[0].reason

    def test_within_drops_incomplete_animals_listwise(self):
        matrix, kept, dropped = within_matrix(_tidy(), "dist", ["Base", "Stim"])
        assert kept == ["m1", "m2", "m3", "m5", "m6"]
        assert matrix.shape == (5, 2)
        assert [d.session for d in dropped] == ["m4"]

    def test_within_one_group(self):
        _matrix, kept, _ = within_matrix(_tidy(), "dist", ["Base", "Stim"], group="ChR2")
        assert kept == ["m1", "m2", "m3"]

    def test_compare_groups_carries_who_was_dropped(self):
        r = compare_groups(_tidy(), "dist", "Base", "welch")
        assert r.n == {"ChR2": 4, "eYFP": 1, UNGROUPED: 1}
        assert not r.ok  # two-group test, three groups

    def test_compare_epochs(self):
        r = compare_epochs(_tidy(), "dist", ["Base", "Stim"], "paired_t", group="ChR2")
        assert r.n == {"Base": 3, "Stim": 3}
        assert [d.session for d in r.dropped] == ["m4"]
        assert r.difference < 0

    def test_compare_epochs_wants_two(self):
        with pytest.raises(ValueError):
            compare_epochs(_tidy(), "dist", ["Base"])


class TestReporting:
    def test_methods_sentence(self):
        text = run_between({"A": A, "B": B}, "distance", "welch").methods()
        assert text.startswith("Welch's t-test on distance, A (n = 9), B (n = 7): t(7.13) = -2.16")
        assert "p = .067" in text and "Hedges' g = -1.14 [95% CI -2.31, 0.02]" in text

    def test_integer_df_print_without_decimals(self):
        assert run_between({"A": A, "B": B}, "x", "student").statistic_text().startswith("t(14)")

    @pytest.mark.parametrize(
        "p, text", [(0.0004, "p < .001"), (0.04567, "p = .046"), (float("nan"), "p = n/a")]
    )
    def test_format_p(self, p, text):
        assert format_p(p) == text

    def test_a_failed_test_says_why(self):
        assert "not run" in run_between({"A": A}, "x").methods()
