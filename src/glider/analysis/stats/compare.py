"""Between- and within-subject comparisons, with the follow-ups they need.

scipy supplies the distributions and the standard tests. Welch's ANOVA,
Games-Howell and Dunn's test are written out here -- scipy has none of them
before 1.15 and GLIDER supports 1.10 -- and are pinned against pingouin and
scikit-posthocs reference values in the tests.

"auto" picks the test from the data and says why in ``TestResult.reason``:

* two groups -- Welch's t when every group passes Shapiro-Wilk, else
  Mann-Whitney U. Welch rather than Student because it costs nothing when the
  variances are equal and is right when they are not;
* several groups -- one-way ANOVA + Tukey when normal with equal variances
  (Levene), Welch's ANOVA + Games-Howell when normal with unequal variances,
  Kruskal-Wallis + Dunn otherwise;
* two epochs, same animals -- paired t when the differences pass
  Shapiro-Wilk, else Wilcoxon signed-rank.
"""

from __future__ import annotations

import inspect
import itertools
import math
import warnings
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats

from glider.analysis.stats.corrections import METHODS, adjust
from glider.analysis.stats.frames import Dropped, between_cells, within_matrix
from glider.analysis.stats.result import TestResult, format_p

__all__ = [
    "BETWEEN_TESTS",
    "ONE_SAMPLE_TESTS",
    "PAIRED_TESTS",
    "compare_epochs",
    "compare_groups",
    "dunn",
    "games_howell",
    "run_between",
    "run_one_sample",
    "run_paired",
    "tukey",
]

#: The between-subject tests by key, and the name each reports under.
BETWEEN_TESTS = {
    "auto": "Auto",
    "welch": "Welch's t-test",
    "student": "Student's t-test",
    "mannwhitney": "Mann-Whitney U",
    "permutation": "Permutation test (mean difference)",
    "anova": "One-way ANOVA",
    "welch_anova": "Welch's ANOVA",
    "kruskal": "Kruskal-Wallis",
}
#: Tests that compare exactly two groups.
_TWO_ONLY = {"welch", "student", "mannwhitney", "permutation"}

#: The within-subject (two epoch) tests by key.
PAIRED_TESTS = {
    "auto": "Auto",
    "paired_t": "Paired t-test",
    "wilcoxon": "Wilcoxon signed-rank",
    "permutation": "Permutation test (paired)",
}

#: Below this, Shapiro-Wilk has too little power for "normal" to mean much.
_SMALL_N = 5
_ALPHA = 0.05
_RESAMPLES = 9999
_SEED = 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seeded() -> dict:
    """A fixed generator for scipy's resampling, under whichever name it takes.

    Resampled p values and CIs must not change between two clicks on the
    same data. ``rng`` arrived in scipy 1.15; ``random_state`` before it.
    """
    generator = np.random.default_rng(_SEED)
    params = inspect.signature(stats.bootstrap).parameters
    return {"rng": generator} if "rng" in params else {"random_state": generator}


def _shapiro(values: np.ndarray) -> float:
    """Shapiro-Wilk p, NaN under three values; 1.0 for constant data."""
    if len(values) < 3:
        return float("nan")
    if np.ptp(values) == 0:
        return 1.0
    return float(stats.shapiro(values).pvalue)


def _levene(cells: list[np.ndarray]) -> float:
    """Levene's test (median-centred, i.e. Brown-Forsythe) p, NaN if it cannot run."""
    if len(cells) < 2 or any(len(c) < 2 for c in cells):
        return float("nan")
    if all(np.ptp(c) == 0 for c in cells):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(stats.levene(*cells, center="median").pvalue)


def _t_crit(dof: float) -> float:
    return float(abs(stats.t.ppf(0.025, dof)))


def hedges_g(a: np.ndarray, b: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Hedges' g for ``a - b`` and its 95% CI (the R ``cohen.d`` / pingouin interval)."""
    n1, n2 = len(a), len(b)
    pooled = math.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    )
    if pooled == 0:
        return float("nan"), (float("nan"), float("nan"))
    g = (np.mean(a) - np.mean(b)) / pooled * (1 - 3 / (4 * (n1 + n2) - 9))
    se = math.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    crit = _t_crit(n1 + n2 - 2)
    return float(g), (float(g - crit * se), float(g + crit * se))


def cohen_dz(diff: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Cohen's d_z (mean difference over its SD) and its 95% CI."""
    n = len(diff)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return float("nan"), (float("nan"), float("nan"))
    dz = float(np.mean(diff) / sd)
    se = math.sqrt(1 / n + dz**2 / (2 * n))
    crit = _t_crit(n - 1)
    return dz, (dz - crit * se, dz + crit * se)


def _difference_ci(a, b, *, paired: bool) -> tuple[float, float] | None:
    """Percentile bootstrap 95% CI of ``mean(a) - mean(b)``; None if it cannot run."""
    if len(a) < 2 or len(b) < 2:
        return None

    def mean_diff(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = stats.bootstrap(
            (a, b),
            mean_diff,
            paired=paired,
            vectorized=True,
            n_resamples=_RESAMPLES,
            method="percentile",
            **_seeded(),
        ).confidence_interval
    low, high = float(ci.low), float(ci.high)
    return None if math.isnan(low) or math.isnan(high) else (low, high)


def _cells_warnings(n: dict[str, int]) -> list[str]:
    out = []
    tiny = [name for name, count in n.items() if count < 3]
    if tiny:
        out.append(f"n < 3 in {', '.join(tiny)}: too few animals for a dependable test")
    return out


def _failed(test: str, metric: str, n: dict[str, int], why: str, **extra) -> TestResult:
    return TestResult(
        test=test,
        metric=metric,
        statistic_name="",
        statistic=float("nan"),
        df=(),
        p=float("nan"),
        effect_name="",
        effect=float("nan"),
        effect_ci=None,
        n=n,
        warnings=[why],
        **extra,
    )


# ---------------------------------------------------------------------------
# post-hocs: one row per pair, ``a, b, diff, statistic, p, p_adj``
# ---------------------------------------------------------------------------


def _posthoc_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["a", "b", "diff", "statistic", "p", "p_adj"])


def tukey(cells: dict[str, np.ndarray]) -> pd.DataFrame:
    """Tukey-Kramer HSD. ``statistic`` is t; p is already family-wise (p_adj = p)."""
    names = list(cells)
    k = len(names)
    n_total = sum(len(v) for v in cells.values())
    dof = n_total - k
    mse = sum(((v - v.mean()) ** 2).sum() for v in cells.values()) / dof
    rows = []
    for a, b in itertools.combinations(names, 2):
        x, y = cells[a], cells[b]
        diff = float(x.mean() - y.mean())
        se = math.sqrt(mse * (1 / len(x) + 1 / len(y)))
        t = diff / se if se else float("nan")
        p = float(stats.studentized_range.sf(abs(t) * math.sqrt(2), k, dof))
        rows.append({"a": a, "b": b, "diff": diff, "statistic": t, "p": p, "p_adj": p})
    return _posthoc_frame(rows)


def games_howell(cells: dict[str, np.ndarray]) -> pd.DataFrame:
    """Games-Howell: Tukey's idea with each pair's own variances and Welch df."""
    k = len(cells)
    rows = []
    for a, b in itertools.combinations(list(cells), 2):
        x, y = cells[a], cells[b]
        va, vb = np.var(x, ddof=1) / len(x), np.var(y, ddof=1) / len(y)
        diff = float(x.mean() - y.mean())
        se = math.sqrt(va + vb)
        if se == 0:
            t = p = float("nan")
        else:
            t = diff / se
            dof = (va + vb) ** 2 / (va**2 / (len(x) - 1) + vb**2 / (len(y) - 1))
            p = float(stats.studentized_range.sf(abs(t) * math.sqrt(2), k, dof))
        rows.append({"a": a, "b": b, "diff": diff, "statistic": t, "p": p, "p_adj": p})
    return _posthoc_frame(rows)


def dunn(cells: dict[str, np.ndarray], correction: str = "holm") -> pd.DataFrame:
    """Dunn's test on the Kruskal-Wallis ranks, tie-corrected; ``diff`` is in mean ranks."""
    names = list(cells)
    pooled = np.concatenate([cells[name] for name in names])
    n_total = len(pooled)
    ranks = stats.rankdata(pooled)
    _, ties = np.unique(pooled, return_counts=True)
    tie_term = float(((ties**3) - ties).sum()) / (12 * (n_total - 1)) if n_total > 1 else 0.0
    mean_rank, start = {}, 0
    for name in names:
        size = len(cells[name])
        mean_rank[name] = float(ranks[start : start + size].mean())
        start += size
    rows = []
    for a, b in itertools.combinations(names, 2):
        diff = mean_rank[a] - mean_rank[b]
        var = (n_total * (n_total + 1) / 12 - tie_term) * (1 / len(cells[a]) + 1 / len(cells[b]))
        z = diff / math.sqrt(var) if var > 0 else float("nan")
        p = float(2 * stats.norm.sf(abs(z))) if not math.isnan(z) else float("nan")
        rows.append({"a": a, "b": b, "diff": diff, "statistic": z, "p": p})
    frame = pd.DataFrame(rows, columns=["a", "b", "diff", "statistic", "p"])
    frame["p_adj"] = adjust(frame["p"], correction)
    return frame


# ---------------------------------------------------------------------------
# between subjects
# ---------------------------------------------------------------------------


def _welch_anova(cells: list[np.ndarray]) -> tuple[float, float, float]:
    """Welch's F, its two df, for several groups with unequal variances."""
    k = len(cells)
    n = np.array([len(c) for c in cells], dtype=float)
    means = np.array([c.mean() for c in cells])
    variances = np.array([np.var(c, ddof=1) for c in cells])
    weights = n / variances
    total = weights.sum()
    grand = (weights * means).sum() / total
    numerator = (weights * (means - grand) ** 2).sum() / (k - 1)
    lam = ((1 - weights / total) ** 2 / (n - 1)).sum()
    f = numerator / (1 + 2 * (k - 2) / (k**2 - 1) * lam)
    return float(f), float(k - 1), float((k**2 - 1) / (3 * lam))


def _eta_squared(cells: list[np.ndarray]) -> float:
    pooled = np.concatenate(cells)
    total = ((pooled - pooled.mean()) ** 2).sum()
    between = sum(len(c) * (c.mean() - pooled.mean()) ** 2 for c in cells)
    return float(between / total) if total else float("nan")


def _choose_between(cells: dict[str, np.ndarray], checks: dict[str, float]) -> tuple[str, str]:
    """The test "auto" runs, and the sentence that says why."""
    shapiro = {name: checks.get(f"Shapiro {name}", float("nan")) for name in cells}
    failing = {name: p for name, p in shapiro.items() if p < _ALPHA}
    untested = [name for name, p in shapiro.items() if math.isnan(p)]
    note = f" ({', '.join(untested)} too small to check)" if untested else ""
    two = len(cells) == 2
    if failing:
        worst = min(failing, key=failing.get)
        why = f"Shapiro-Wilk {format_p(failing[worst])} in {worst}: not normal"
        return ("mannwhitney" if two else "kruskal"), why
    if two:
        return "welch", f"Shapiro-Wilk p ≥ .05 in every group{note}"
    levene = checks.get("Levene", float("nan"))
    if levene < _ALPHA:
        return "welch_anova", f"normal{note}, but Levene {format_p(levene)}: unequal variances"
    return "anova", f"normal{note}, equal variances (Levene {format_p(levene)})"


def run_between(
    cells: dict[str, np.ndarray],
    metric: str = "",
    test: str = "auto",
    *,
    correction: str = "holm",
    dropped: list[Dropped] | None = None,
) -> TestResult:
    """Compare independent groups: ``cells`` maps group -> values, one per animal."""
    if test not in BETWEEN_TESTS:
        raise ValueError(f"unknown test {test!r}; one of {sorted(BETWEEN_TESTS)}")
    if correction not in METHODS:
        raise ValueError(f"unknown correction {correction!r}; one of {sorted(METHODS)}")
    cells = {name: np.asarray(v, dtype=float) for name, v in cells.items() if len(v)}
    n = {name: len(v) for name, v in cells.items()}
    extra = {"dropped": list(dropped or [])}
    label = BETWEEN_TESTS[test]
    if len(cells) < 2:
        return _failed(label, metric, n, "needs at least two groups with data", **extra)
    if test in _TWO_ONLY and len(cells) != 2:
        return _failed(label, metric, n, f"{label} compares two groups, not {len(cells)}", **extra)
    if any(count < 2 for count in n.values()):
        small = ", ".join(name for name, count in n.items() if count < 2)
        return _failed(label, metric, n, f"needs two or more animals per group ({small})", **extra)

    checks = {f"Shapiro {name}": _shapiro(v) for name, v in cells.items()}
    checks["Levene"] = _levene(list(cells.values()))
    reason = ""
    if test == "auto":
        test, reason = _choose_between(cells, checks)
    result = _between(test, cells, metric, n, correction)
    result.reason = reason
    result.checks = checks
    result.dropped = extra["dropped"]
    result.warnings = _cells_warnings(n) + result.warnings
    if reason and test in ("welch", "anova", "welch_anova") and min(n.values()) < _SMALL_N:
        result.warnings.append(
            f"normality was checked on fewer than {_SMALL_N} animals per group, "
            "where Shapiro-Wilk rarely rejects; a rank test is the safer read"
        )
    return result


def _between(test, cells, metric, n, correction) -> TestResult:
    names = list(cells)
    values = list(cells.values())
    common = {"test": BETWEEN_TESTS[test], "metric": metric, "n": n}
    if test in _TWO_ONLY:
        a, b = values
        g, g_ci = hedges_g(a, b)
        difference = {
            "difference": float(a.mean() - b.mean()),
            "difference_ci": _difference_ci(a, b, paired=False),
        }
        if test in ("welch", "student"):
            equal = test == "student"
            res = stats.ttest_ind(a, b, equal_var=equal)
            if equal:
                dof = len(a) + len(b) - 2
            else:
                va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
                dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
            return TestResult(
                **common,
                statistic_name="t",
                statistic=float(res.statistic),
                df=(float(dof),),
                p=float(res.pvalue),
                effect_name="Hedges' g",
                effect=g,
                effect_ci=g_ci,
                **difference,
            )
        if test == "mannwhitney":
            res = stats.mannwhitneyu(a, b, alternative="two-sided")
            rbc = 2 * float(res.statistic) / (len(a) * len(b)) - 1
            return TestResult(
                **common,
                statistic_name="U",
                statistic=float(res.statistic),
                df=(),
                p=float(res.pvalue),
                effect_name="rank-biserial r",
                effect=rbc,
                effect_ci=None,
                **difference,
            )
        res = stats.permutation_test(
            (a, b),
            lambda x, y, axis=-1: np.mean(x, axis=axis) - np.mean(y, axis=axis),
            permutation_type="independent",
            vectorized=True,
            n_resamples=_RESAMPLES,
            **_seeded(),
        )
        return TestResult(
            **common,
            statistic_name="Δmean",
            statistic=float(res.statistic),
            df=(),
            p=float(res.pvalue),
            effect_name="Hedges' g",
            effect=g,
            effect_ci=g_ci,
            **difference,
        )

    k, n_total = len(values), sum(n.values())
    if test == "anova":
        res = stats.f_oneway(*values)
        stat, dfs, posthoc, name = float(res.statistic), (k - 1, n_total - k), tukey, "Tukey HSD"
        effect_name, effect = "η²", _eta_squared(values)
    elif test == "welch_anova":
        f, df1, df2 = _welch_anova(values)
        stat, dfs = f, (df1, df2)
        posthoc, name = games_howell, "Games-Howell"
        effect_name, effect = "η²", _eta_squared(values)
    else:  # kruskal
        res = stats.kruskal(*values)
        stat, dfs = float(res.statistic), (k - 1,)
        posthoc = partial(dunn, correction=correction)
        name = f"Dunn ({METHODS[correction]})"
        effect_name, effect = "ε²", stat / (n_total - 1)
    dist = stats.chi2(*dfs) if test == "kruskal" else stats.f(*dfs)
    p = float(dist.sf(stat)) if not math.isnan(stat) else float("nan")
    result = TestResult(
        **common,
        statistic_name="H" if test == "kruskal" else "F",
        statistic=stat,
        df=tuple(float(d) for d in dfs),
        p=p,
        effect_name=effect_name,
        effect=float(effect),
        effect_ci=None,
    )
    if k > 2:
        result.posthoc = posthoc(dict(zip(names, values, strict=True)))
        result.posthoc_name = name
    return result


def compare_groups(
    tidy: pd.DataFrame,
    metric: str,
    epoch: str | None = None,
    test: str = "auto",
    *,
    correction: str = "holm",
) -> TestResult:
    """Groups against each other on ``metric`` within one epoch of the tidy frame."""
    cells, _sessions, dropped = between_cells(tidy, metric, epoch)
    return run_between(cells, metric, test, correction=correction, dropped=dropped)


# ---------------------------------------------------------------------------
# within subjects: the same animals in two epochs
# ---------------------------------------------------------------------------


def _matched_rank_biserial(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if not len(nonzero):
        return float("nan")
    ranks = stats.rankdata(np.abs(nonzero))
    plus, minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return float((plus - minus) / (plus + minus))


def _location(
    values: np.ndarray,
    test: str,
    *,
    names: dict[str, str],
    metric: str,
    n: dict[str, int],
    dropped: list[Dropped],
    difference_ci,
) -> TestResult:
    """A paired or one-sample test: both are a test of ``values`` (the differences,
    or the values less ``mu``) against zero.

    ``names`` gives the test, effect and "differences" wording for the kind.
    """
    checks = {f"Shapiro {names['values']}": _shapiro(values)}
    reason = ""
    if test == "auto":
        p = checks[f"Shapiro {names['values']}"]
        if p < _ALPHA:
            test, reason = "wilcoxon", f"Shapiro-Wilk {format_p(p)} on the {names['values']}"
        else:
            test = "t"
            reason = (
                f"{names['values']} too few to check for normality"
                if math.isnan(p)
                else f"Shapiro-Wilk p ≥ .05 on the {names['values']}"
            )
    common = {
        "test": names[test],
        "metric": metric,
        "n": n,
        "difference": float(values.mean()),
        "difference_ci": difference_ci,
        "reason": reason,
        "checks": checks,
        "dropped": dropped,
    }
    d, d_ci = cohen_dz(values)
    if test == "t":
        res = stats.ttest_1samp(values, 0.0)
        result = TestResult(
            **common,
            statistic_name="t",
            statistic=float(res.statistic),
            df=(float(len(values) - 1),),
            p=float(res.pvalue),
            effect_name=names["effect"],
            effect=d,
            effect_ci=d_ci,
        )
    elif test == "wilcoxon":
        if not np.any(values):
            why = f"every {names['values'][:-1]} is zero"
            return _failed(names[test], metric, n, why, dropped=dropped, checks=checks)
        res = stats.wilcoxon(values)
        result = TestResult(
            **common,
            statistic_name="W",
            statistic=float(res.statistic),
            df=(),
            p=float(res.pvalue),
            effect_name="rank-biserial r",
            effect=_matched_rank_biserial(values),
            effect_ci=None,
        )
    else:
        # One sample with permutation_type="samples" flips signs: the same as
        # swapping each animal's two epochs.
        res = stats.permutation_test(
            (values,),
            lambda x, axis=-1: np.mean(x, axis=axis),
            permutation_type="samples",
            vectorized=True,
            n_resamples=_RESAMPLES,
            **_seeded(),
        )
        result = TestResult(
            **common,
            statistic_name="Δmean",
            statistic=float(res.statistic),
            df=(),
            p=float(res.pvalue),
            effect_name=names["effect"],
            effect=d,
            effect_ci=d_ci,
        )
    result.warnings = _cells_warnings({"animals": len(values)}) + result.warnings
    return result


_PAIRED_KIND = {
    "paired_t": "t",
    "wilcoxon": "wilcoxon",
    "permutation": "permutation",
    "auto": "auto",
}


def run_paired(
    a,
    b,
    names: tuple[str, str] = ("A", "B"),
    metric: str = "",
    test: str = "auto",
    *,
    dropped: list[Dropped] | None = None,
) -> TestResult:
    """Compare two epochs of the same animals: ``a[i]`` and ``b[i]`` are one animal."""
    if test not in PAIRED_TESTS:
        raise ValueError(f"unknown test {test!r}; one of {sorted(PAIRED_TESTS)}")
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must be the same length")
    n = {names[0]: len(a), names[1]: len(b)}
    if len(a) < 2:
        return _failed(
            PAIRED_TESTS[test], metric, n, "needs two or more animals", dropped=list(dropped or [])
        )
    return _location(
        a - b,
        _PAIRED_KIND[test],
        names={
            "values": "differences",
            "effect": "Cohen's d_z",
            "t": PAIRED_TESTS["paired_t"],
            "wilcoxon": PAIRED_TESTS["wilcoxon"],
            "permutation": PAIRED_TESTS["permutation"],
        },
        metric=metric,
        n=n,
        dropped=list(dropped or []),
        difference_ci=_difference_ci(a, b, paired=True),
    )


#: One-sample tests against a fixed value (a slope against 0, a ratio against 1).
ONE_SAMPLE_TESTS = {
    "auto": "Auto",
    "t": "One-sample t-test",
    "wilcoxon": "Wilcoxon signed-rank",
    "permutation": "Permutation test (sign-flip)",
}


def run_one_sample(
    values,
    mu: float = 0.0,
    metric: str = "",
    test: str = "auto",
    *,
    name: str = "",
    dropped: list[Dropped] | None = None,
) -> TestResult:
    """Whether ``values`` -- one per animal -- differ from ``mu``.

    ``difference`` is the mean less ``mu``, with its bootstrap 95% CI.
    """
    if test not in ONE_SAMPLE_TESTS:
        raise ValueError(f"unknown test {test!r}; one of {sorted(ONE_SAMPLE_TESTS)}")
    values = np.asarray(values, dtype=float)
    n = {name or metric or "animals": len(values)}
    if len(values) < 2:
        return _failed(
            ONE_SAMPLE_TESTS[test],
            metric,
            n,
            "needs two or more animals",
            dropped=list(dropped or []),
        )
    shifted = values - mu
    return _location(
        shifted,
        test,
        names={
            "values": "values",
            "effect": "Cohen's d",
            "t": ONE_SAMPLE_TESTS["t"],
            "wilcoxon": ONE_SAMPLE_TESTS["wilcoxon"],
            "permutation": ONE_SAMPLE_TESTS["permutation"],
        },
        metric=metric,
        n=n,
        dropped=list(dropped or []),
        difference_ci=_mean_ci(shifted),
    )


def _mean_ci(values: np.ndarray) -> tuple[float, float] | None:
    """Percentile bootstrap 95% CI of the mean; None if it cannot run."""
    if len(values) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = stats.bootstrap(
            (values,),
            np.mean,
            vectorized=True,
            n_resamples=_RESAMPLES,
            method="percentile",
            **_seeded(),
        ).confidence_interval
    low, high = float(ci.low), float(ci.high)
    return None if math.isnan(low) or math.isnan(high) else (low, high)


def compare_epochs(
    tidy: pd.DataFrame,
    metric: str,
    epochs: list[str],
    test: str = "auto",
    *,
    group: str | None = None,
) -> TestResult:
    """The same animals across epochs of the tidy frame, optionally one group only.

    Two epochs are a paired test (:data:`PAIRED_TESTS`); three or more a
    repeated-measures one (:data:`~glider.analysis.stats.repeated.REPEATED_TESTS`).
    """
    if len(epochs) < 2:
        raise ValueError("compare_epochs needs two or more epochs")
    matrix, _sessions, dropped = within_matrix(tidy, metric, list(epochs), group)
    if len(epochs) > 2:
        from glider.analysis.stats.repeated import run_repeated

        return run_repeated(matrix, list(epochs), metric, test, dropped=dropped)
    return run_paired(matrix[:, 0], matrix[:, 1], tuple(epochs), metric, test, dropped=dropped)
