"""Repeated-measures, Friedman and mixed ANOVA, pinned to independent references.

Reference values: pingouin 0.7.0 (``rm_anova``, ``sphericity``, ``epsilon``,
``friedman``, ``pairwise_tests``, ``mixed_anova``), computed once in a throwaway
environment on the data below. pingouin follows R's ``mauchly.test`` and takes
a mixed design's sphericity from the pooled within-group covariance, as SPSS
and afex do.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from glider.analysis.stats import (
    compare_epochs,
    compare_mixed,
    run_mixed,
    run_repeated,
    sphericity,
)

# Twelve animals, four epochs; e3 swings with each animal: sphericity fails.
SPH = np.array(
    [
        [20.06, 21.08, 22.98, 19.44],
        [20.13, 21.5, 24.13, 31.79],
        [19.44, 19.19, 22.1, 23.4],
        [18.57, 18.95, 21.15, 11.86],
        [18.42, 20.07, 22.16, 19.57],
        [17.79, 19.07, 21.05, 18.36],
        [19.17, 20.36, 22.39, 21.61],
        [22.04, 23.44, 25.73, 32.77],
        [18.09, 19.53, 22.83, 18.78],
        [18.64, 19.35, 20.83, 20.95],
        [20.35, 22.51, 24.5, 30.69],
        [20.85, 21.31, 23.79, 21.44],
    ]
)
# Ten animals, three epochs, an additive model: sphericity holds.
OK = np.array(
    [
        [11.3, 10.7, 12.1],
        [6.6, 8.8, 10.8],
        [9.9, 11.2, 11.5],
        [6.3, 6.8, 9.0],
        [9.5, 8.6, 9.8],
        [10.1, 8.8, 11.5],
        [12.4, 13.2, 12.9],
        [13.1, 13.5, 15.3],
        [5.8, 7.7, 5.8],
        [9.1, 8.8, 7.1],
    ]
)
# Mixed: 7 controls and 9 ChR2 animals; ChR2 responds in e1.
CTL = np.array(
    [
        [9.88, 9.71, 9.39],
        [8.9, 9.22, 7.37],
        [10.25, 10.86, 9.47],
        [14.75, 14.13, 15.66],
        [8.53, 8.93, 7.95],
        [8.63, 7.25, 7.82],
        [10.77, 8.78, 11.46],
    ]
)
CHR2 = np.array(
    [
        [6.23, 10.03, 6.53],
        [12.61, 13.88, 12.46],
        [7.52, 10.43, 8.03],
        [12.82, 15.1, 13.52],
        [10.28, 12.18, 10.94],
        [6.37, 10.95, 7.55],
        [12.2, 12.36, 12.69],
        [14.54, 14.87, 12.83],
        [10.21, 18.09, 9.41],
    ]
)
# Mixed: three groups (6, 8, 5) over four epochs, nothing going on.
HX = np.array(
    [
        [3.93, 5.14, 4.15, 4.54],
        [2.94, 2.84, 5.39, 3.54],
        [2.65, 2.32, 1.92, 1.85],
        [6.73, 5.8, 5.95, 6.12],
        [6.14, 5.65, 5.35, 4.4],
        [3.73, 6.06, 6.07, 4.97],
    ]
)
HY = np.array(
    [
        [6.31, 6.69, 7.69, 7.3],
        [5.03, 4.48, 6.99, 5.16],
        [6.23, 7.09, 4.6, 4.14],
        [6.64, 6.43, 6.14, 6.42],
        [3.46, 2.8, 4.15, 3.34],
        [5.63, 7.78, 7.21, 7.68],
        [2.14, 2.47, 2.13, 2.24],
        [6.49, 5.51, 6.65, 6.63],
    ]
)
HZ = np.array(
    [
        [6.59, 7.58, 8.78, 7.96],
        [2.99, 2.62, 4.07, 2.89],
        [4.13, 5.64, 4.36, 3.08],
        [4.34, 6.2, 4.91, 3.69],
        [4.04, 6.11, 5.14, 5.03],
    ]
)

RM = {
    "SPH": {
        "F": 4.569840299068531,
        "p_unc": 0.008755902565414194,
        "p_gg": 0.054035496642435794,
        "np2": 0.2935059198610998,
        "eps_gg": 0.34494665198442204,
        "eps_hf": 0.3484834304513767,
        "W": 0.0016846191600016023,
        "p_spher": 7.112047303375329e-12,
        "chi2": 62.08820876345587,
        "Q": 20.7,
        "kendall_w": 0.575,
        "friedman_p": 0.00012151040342188821,
        "pairs": [
            ("e0", "e1", -5.692842658277161, 0.0005586444587153528),
            ("e0", "e2", -15.59999478861659, 3.768342256776994e-08),
            ("e0", "e3", -1.9921394569627595, 0.21529057425689085),
            ("e1", "e2", -16.005725886292534, 3.4470823213255915e-08),
            ("e1", "e3", -1.3738547477116823, 0.39366993783402343),
            ("e2", "e3", 0.16965181282896502, 0.8683637375889692),
        ],
    },
    "OK": {
        "F": 3.0979202491807514,
        "p_unc": 0.06978886407996222,
        "p_gg": 0.07939686506940949,
        "np2": 0.25607048032826446,
        "eps_gg": 0.8656375137331306,
        "eps_hf": 1.0,
        "W": 0.8447820431355607,
        "p_spher": 0.5093058860319419,
        "chi2": 1.3494129761131404,
        "Q": 5.282051282051282,
        "kendall_w": 0.2641025641025641,
        "friedman_p": 0.071288116031728,
        "pairs": [
            ("e0", "e1", -1.0715994395536397, 0.3325584400764982),
            ("e0", "e2", -2.1945645403483645, 0.16749969478788607),
            ("e1", "e2", -1.5062050138151357, 0.3325584400764982),
        ],
    },
}

MIXED_TWO = [
    {
        "F": 1.1260673668046084,
        "df1": 1,
        "df2": 14,
        "p_unc": 0.30657974740684835,
        "np2": 0.07444548139960393,
        "p_gg": None,
        "eps": None,
        "W_spher": None,
        "p_spher": None,
    },
    {
        "F": 7.078723785962117,
        "df1": 2,
        "df2": 28,
        "p_unc": 0.00325066427677455,
        "np2": 0.33582316737203816,
        "p_gg": 0.012110729965951417,
        "eps": 0.6223684443608658,
        "W_spher": 0.393234732479185,
        "p_spher": 0.0023186604520896107,
    },
    {
        "F": 7.452743966506648,
        "df1": 2,
        "df2": 28,
        "p_unc": 0.0025411634797635734,
        "np2": 0.3474028300595175,
        "p_gg": 0.010286558313837454,
        "eps": 0.6223684443608658,
        "W_spher": 0.393234732479185,
        "p_spher": 0.0023186604520896107,
    },
]
MIXED_THREE = [
    {
        "F": 0.5454978458663371,
        "df1": 2,
        "df2": 16,
        "p_unc": 0.5899577948536606,
        "np2": 0.06383453084950545,
        "p_gg": None,
        "eps": None,
        "W_spher": None,
        "p_spher": None,
    },
    {
        "F": 2.704679796264302,
        "df1": 3,
        "df2": 48,
        "p_unc": 0.055704993628794615,
        "np2": 0.14459909636114063,
        "p_gg": 0.07529725340002384,
        "eps": 0.7413970001695399,
        "W_spher": 0.543073140390134,
        "p_spher": 0.11025454219636883,
    },
    {
        "F": 0.6625728763206146,
        "df1": 6,
        "df2": 48,
        "p_unc": 0.6800145858156408,
        "np2": 0.07648684585751377,
        "p_gg": 0.6373898140778078,
        "eps": 0.7413970001695399,
        "W_spher": 0.543073140390134,
        "p_spher": 0.11025454219636883,
    },
]

TOL = {"rel": 1e-9, "abs": 1e-12}


def _epochs(k: int) -> list[str]:
    return [f"e{j}" for j in range(k)]


@pytest.mark.parametrize("key, data", [("SPH", SPH), ("OK", OK)])
class TestRepeatedMeasuresAnova:
    def test_f_p_and_partial_eta(self, key, data):
        ref = RM[key]
        r = run_repeated(data, _epochs(data.shape[1]), "x", "rm_anova")
        assert r.statistic == pytest.approx(ref["F"], **TOL)
        assert r.details["p uncorrected"] == pytest.approx(ref["p_unc"], **TOL)
        assert r.details["p GG"] == pytest.approx(ref["p_gg"], **TOL)
        assert r.effect == pytest.approx(ref["np2"], **TOL)

    def test_sphericity(self, key, data):
        ref = RM[key]
        s = sphericity(data - data.mean(axis=0), data.shape[0] - 1)
        assert s.gg == pytest.approx(ref["eps_gg"], **TOL)
        assert s.hf == pytest.approx(ref["eps_hf"], **TOL)
        assert s.w == pytest.approx(ref["W"], **TOL)
        assert s.chi2 == pytest.approx(ref["chi2"], **TOL)
        assert s.p == pytest.approx(ref["p_spher"], rel=1e-9, abs=1e-20)

    def test_pairwise_follow_ups_are_holm_adjusted(self, key, data):
        r = run_repeated(data, _epochs(data.shape[1]), "x", "rm_anova")
        for (a, b, t, p_holm), (_, row) in zip(RM[key]["pairs"], r.posthoc.iterrows(), strict=True):
            assert (row["a"], row["b"]) == (a, b)
            assert row["statistic"] == pytest.approx(t, **TOL)
            assert row["p_adj"] == pytest.approx(p_holm, **TOL)

    def test_friedman_and_kendalls_w(self, key, data):
        ref = RM[key]
        r = run_repeated(data, _epochs(data.shape[1]), "x", "friedman")
        assert r.statistic == pytest.approx(ref["Q"], **TOL)
        assert r.effect == pytest.approx(ref["kendall_w"], **TOL)
        assert r.p == pytest.approx(ref["friedman_p"], **TOL)
        assert r.posthoc_name.startswith("Wilcoxon")


class TestSphericityDecisions:
    def test_a_violation_reports_the_corrected_p_and_df(self):
        r = run_repeated(SPH, _epochs(4), "x", "rm_anova")
        assert r.p == pytest.approx(RM["SPH"]["p_gg"], **TOL)
        assert r.df == pytest.approx((3 * RM["SPH"]["eps_gg"], 33 * RM["SPH"]["eps_gg"]))
        assert "Greenhouse-Geisser" in r.reason
        # The correction is what turns p = .009 into p = .054 here.
        assert r.details["p uncorrected"] < 0.01 < 0.05 < r.p

    def test_no_violation_reports_the_plain_test(self):
        r = run_repeated(OK, _epochs(3), "x", "rm_anova")
        assert r.p == pytest.approx(RM["OK"]["p_unc"], **TOL)
        assert r.df == (2.0, 18.0)
        assert "Greenhouse" not in r.reason

    def test_two_levels_are_always_spherical(self):
        s = sphericity(OK[:, :2] - OK[:, :2].mean(axis=0), 9)
        assert (s.w, s.gg, s.hf, s.p) == (1.0, 1.0, 1.0, 1.0)

    def test_fewer_animals_than_contrasts_cannot_test_it(self):
        few = SPH[:3]
        s = sphericity(few - few.mean(axis=0), 2)
        assert math.isnan(s.p) and not s.violated
        r = run_repeated(few, _epochs(4), "x", "rm_anova")
        assert any("sphericity" in w for w in r.warnings)


class TestRepeatedAuto:
    def test_normal_residuals_get_the_anova(self):
        r = run_repeated(OK, _epochs(3), "x")
        assert r.test == "Repeated-measures ANOVA"
        assert "residuals" in r.reason

    def test_skewed_residuals_get_friedman(self):
        skewed = OK.copy()
        skewed[0, 2] += 40.0
        r = run_repeated(skewed, _epochs(3), "x")
        assert r.test == "Friedman test"

    def test_two_epochs_belong_to_the_paired_tests(self):
        with pytest.raises(ValueError):
            run_repeated(OK[:, :2], _epochs(2))


def _check_mixed(result, reference):
    for (name, r), ref in zip(result.effects.items(), reference, strict=True):
        assert r.statistic == pytest.approx(ref["F"], **TOL), name
        assert r.details["p uncorrected"] == pytest.approx(ref["p_unc"], **TOL), name
        assert r.effect == pytest.approx(ref["np2"], **TOL), name
        if ref["eps"] is not None:
            assert r.details["p GG"] == pytest.approx(ref["p_gg"], **TOL), name
            assert result.sphericity.gg == pytest.approx(ref["eps"], **TOL)
            assert result.sphericity.w == pytest.approx(ref["W_spher"], **TOL)
            assert result.sphericity.p == pytest.approx(ref["p_spher"], **TOL)


class TestMixedAnova:
    def test_two_unbalanced_groups(self):
        result = run_mixed({"ctl": CTL, "chr2": CHR2}, _epochs(3), "x")
        _check_mixed(result, MIXED_TWO)
        assert [r.df[:2] for r in result.effects.values()][0] == (1.0, 14.0)

    def test_three_groups(self):
        result = run_mixed({"x": HX, "y": HY, "z": HZ}, _epochs(4), "v")
        _check_mixed(result, MIXED_THREE)

    def test_a_violation_corrects_the_within_effects_only(self):
        result = run_mixed({"ctl": CTL, "chr2": CHR2}, _epochs(3), "x")
        assert result.sphericity.violated
        assert result.effects["group"].p == pytest.approx(MIXED_TWO[0]["p_unc"], **TOL)
        assert result.effects["epoch"].p == pytest.approx(MIXED_TWO[1]["p_gg"], **TOL)
        assert result.effects["group × epoch"].p == pytest.approx(MIXED_TWO[2]["p_gg"], **TOL)
        assert "Greenhouse-Geisser corrected" in result.methods()

    def test_a_significant_interaction_brings_simple_effects(self):
        result = run_mixed({"ctl": CTL, "chr2": CHR2}, _epochs(3), "x")
        simple = result.simple_effects
        assert list(simple["effect"]) == ["epoch", "epoch", "group", "group", "group"]
        assert list(simple["within"]) == ["ctl", "chr2", "e0", "e1", "e2"]
        # Only ChR2 changes across epochs.
        by = simple.set_index("within")
        assert by.loc["chr2", "p_adj"] < 0.05 < by.loc["ctl", "p_adj"]
        assert (simple["p_adj"] >= simple["p"]).all()

    def test_no_interaction_no_simple_effects(self):
        result = run_mixed({"x": HX, "y": HY, "z": HZ}, _epochs(4), "v")
        assert result.simple_effects is None

    def test_the_table(self):
        table = run_mixed({"ctl": CTL, "chr2": CHR2}, _epochs(3), "x").table()
        assert list(table["effect"]) == ["group", "epoch", "group × epoch"]
        assert table["p_uncorrected"].tolist() == pytest.approx([r["p_unc"] for r in MIXED_TWO])

    def test_non_normal_residuals_are_warned_about(self):
        result = run_mixed({"ctl": CTL, "chr2": CHR2}, _epochs(3), "x")
        assert any("not normal" in w for w in result.warnings)

    def test_one_group_is_refused(self):
        with pytest.raises(ValueError):
            run_mixed({"ctl": CTL}, _epochs(3))


def _tidy() -> pd.DataFrame:
    """The two-group mixed data as the epoch table writes it, m03 missing e2."""
    rows = []
    for group, data in (("ctl", CTL), ("chr2", CHR2)):
        for i, animal in enumerate(data):
            for j, value in enumerate(animal):
                session = f"{group}{i:02d}"
                missing = session == "ctl03" and j == 2
                rows.append(
                    {
                        "session": session,
                        "group": group,
                        "range": f"e{j}",
                        "dist": float("nan") if missing else value,
                    }
                )
    return pd.DataFrame(rows)


class TestFromTheEpochTable:
    def test_compare_mixed_matches_the_matrices_and_names_who_was_dropped(self):
        result = compare_mixed(_tidy(), "dist", _epochs(3))
        assert result.n == {"chr2": 9, "ctl": 6}
        assert [d.session for d in result.dropped] == ["ctl03"]
        expected = run_mixed({"chr2": CHR2, "ctl": np.delete(CTL, 3, axis=0)}, _epochs(3), "dist")
        assert result.effects["group × epoch"].statistic == pytest.approx(
            expected.effects["group × epoch"].statistic
        )

    def test_compare_epochs_with_three_is_repeated_measures(self):
        r = compare_epochs(_tidy(), "dist", _epochs(3), "rm_anova", group="chr2")
        assert r.test == "Repeated-measures ANOVA"
        assert r.n == {"e0": 9, "e1": 9, "e2": 9}
