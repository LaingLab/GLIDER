"""Several epochs of the same animals, alone or split by group.

* :func:`run_repeated` -- one-way repeated-measures ANOVA, or Friedman, over
  an animals x epochs matrix, with pairwise follow-ups.
* :func:`run_mixed` -- the split-plot (mixed) ANOVA: group between animals,
  epoch within them. Group, epoch, and the interaction -- the effect a
  ChR2-vs-eYFP stimulation experiment is usually about.

Sphericity follows R (``mauchly.test``, ``car::Anova``, afex) and SPSS:
Mauchly's test and the Greenhouse-Geisser and Huynh-Feldt epsilons come from
the covariance of orthonormal contrasts -- in a mixed design, the *pooled
within-group* covariance, so a difference between groups is not mistaken for
a departure from sphericity. When Mauchly rejects, the within-subject p
values reported are Greenhouse-Geisser corrected, and the result says so.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from glider.analysis.stats.compare import run_between, run_paired
from glider.analysis.stats.corrections import METHODS, adjust
from glider.analysis.stats.frames import Dropped, between_cells, within_matrix
from glider.analysis.stats.result import TestResult, format_p

__all__ = [
    "REPEATED_TESTS",
    "MixedResult",
    "Sphericity",
    "pairwise_within",
    "run_mixed",
    "compare_mixed",
    "run_repeated",
    "sphericity",
]

REPEATED_TESTS = {
    "auto": "Auto",
    "rm_anova": "Repeated-measures ANOVA",
    "friedman": "Friedman test",
}

_ALPHA = 0.05


# ---------------------------------------------------------------------------
# sphericity
# ---------------------------------------------------------------------------


def _contrasts(k: int) -> np.ndarray:
    """A (k, k-1) orthonormal basis orthogonal to the grand mean."""
    centred = np.eye(k) - 1.0 / k
    q, _ = np.linalg.qr(centred)
    return q[:, : k - 1]


@dataclass(frozen=True)
class Sphericity:
    """Mauchly's test and the two epsilon corrections for one within factor."""

    w: float
    chi2: float
    df: float
    p: float
    gg: float  # Greenhouse-Geisser epsilon
    hf: float  # Huynh-Feldt epsilon (Lecoutre's correction for mixed designs)

    @property
    def violated(self) -> bool:
        return not math.isnan(self.p) and self.p < _ALPHA


def sphericity(residuals: np.ndarray, df_resid: int) -> Sphericity:
    """Sphericity of a (subjects, k) matrix of within-group residuals.

    ``df_resid`` is n - 1 for a single group and N - groups in a mixed design.
    With two levels sphericity cannot fail: W = 1, epsilon = 1.
    """
    k = residuals.shape[1]
    d = k - 1
    if d <= 1:
        return Sphericity(1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    contrasts = residuals @ _contrasts(k)
    cov = np.cov(contrasts, rowvar=False, ddof=1)
    trace = float(np.trace(cov))
    gg = min(trace**2 / (d * float(np.trace(cov @ cov))), 1.0) if trace > 0 else 1.0
    n_plus = df_resid + 1  # n for one group; N - groups + 1 in general (Lecoutre)
    hf_denominator = d * (df_resid - d * gg)
    hf = (n_plus * d * gg - 2) / hf_denominator if hf_denominator > 0 else 1.0
    hf = float(min(hf, 1.0))
    # Mauchly's W, and its chi-square approximation with R's second-order term.
    if df_resid < d:
        # Fewer animals than contrasts: the covariance is singular.
        return Sphericity(float("nan"), float("nan"), float("nan"), float("nan"), gg, hf)
    sign, logdet = np.linalg.slogdet(cov)
    log_w = logdet - d * math.log(trace / d) if sign > 0 and trace > 0 else -math.inf
    w = math.exp(log_w)
    dof = d * (d + 1) / 2 - 1
    f = 1 - (2 * d**2 + d + 2) / (6 * d * df_resid)
    w2 = (d + 2) * (d - 1) * (d - 2) * (2 * d**3 + 6 * d**2 + 3 * k + 2)
    w2 /= 288 * (df_resid * d * f) ** 2
    chi = -df_resid * f * log_w
    p1, p2 = stats.chi2.sf(chi, dof), stats.chi2.sf(chi, dof + 4)
    return Sphericity(w, float(chi), float(dof), float(p1 + w2 * (p2 - p1)), gg, hf)


def _corrected_p(f: float, df1: float, df2: float, eps: float) -> float:
    """The F test's p on epsilon-scaled df (floored at 1, as R and pingouin do)."""
    return float(stats.f.sf(f, max(df1 * eps, 1.0), max(df2 * eps, 1.0)))


# ---------------------------------------------------------------------------
# pairwise follow-ups
# ---------------------------------------------------------------------------


def pairwise_within(
    matrix: np.ndarray, epochs: list[str], *, rank: bool = False, correction: str = "holm"
) -> pd.DataFrame:
    """Every pair of epochs, paired t (or Wilcoxon when ``rank``), adjusted together."""
    rows = []
    for i, j in itertools.combinations(range(len(epochs)), 2):
        result = run_paired(
            matrix[:, i],
            matrix[:, j],
            (epochs[i], epochs[j]),
            test="wilcoxon" if rank else "paired_t",
        )
        rows.append(
            {
                "a": epochs[i],
                "b": epochs[j],
                "diff": float(np.mean(matrix[:, i] - matrix[:, j])),
                "statistic": result.statistic,
                "p": result.p,
                "effect": result.effect,
            }
        )
    frame = pd.DataFrame(rows, columns=["a", "b", "diff", "statistic", "p", "effect"])
    frame["p_adj"] = adjust(frame["p"], correction)
    return frame


# ---------------------------------------------------------------------------
# one group, several epochs
# ---------------------------------------------------------------------------


def _rm_sums(matrix: np.ndarray) -> tuple[float, float, float]:
    """``(SS_epoch, SS_error, SS_subjects)`` of a one-way repeated-measures design."""
    n, k = matrix.shape
    grand = matrix.mean()
    ss_epoch = n * float(((matrix.mean(axis=0) - grand) ** 2).sum())
    ss_subjects = k * float(((matrix.mean(axis=1) - grand) ** 2).sum())
    ss_total = float(((matrix - grand) ** 2).sum())
    return ss_epoch, ss_total - ss_epoch - ss_subjects, ss_subjects


def _residuals(matrix: np.ndarray) -> np.ndarray:
    """What the additive subject + epoch model leaves: what normality is about."""
    return matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0) + matrix.mean()


def _shapiro(values: np.ndarray) -> float:
    values = np.ravel(values)
    if len(values) < 3 or np.ptp(values) < 1e-12 * max(1.0, float(np.abs(values).max())):
        return float("nan") if len(values) < 3 else 1.0
    return float(stats.shapiro(values).pvalue)


def run_repeated(
    matrix,
    epochs: list[str],
    metric: str = "",
    test: str = "auto",
    *,
    correction: str = "holm",
    dropped: list[Dropped] | None = None,
) -> TestResult:
    """Several epochs of the same animals: row = animal, column = epoch, complete."""
    if test not in REPEATED_TESTS:
        raise ValueError(f"unknown test {test!r}; one of {sorted(REPEATED_TESTS)}")
    if correction not in METHODS:
        raise ValueError(f"unknown correction {correction!r}; one of {sorted(METHODS)}")
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    counts = dict.fromkeys(epochs, n)
    base = {"metric": metric, "n": counts, "dropped": list(dropped or [])}
    if k < 3:
        raise ValueError("run_repeated wants three or more epochs; two are a paired test")
    if n < 2:
        return TestResult(
            test=REPEATED_TESTS[test],
            statistic_name="",
            statistic=float("nan"),
            df=(),
            p=float("nan"),
            effect_name="",
            effect=float("nan"),
            effect_ci=None,
            warnings=["needs two or more animals with every epoch"],
            **base,
        )
    ss_epoch, ss_error, _ = _rm_sums(matrix)
    df1, df2 = k - 1, (n - 1) * (k - 1)
    spher = sphericity(matrix - matrix.mean(axis=0), n - 1)
    checks = {"Shapiro residuals": _shapiro(_residuals(matrix)), "Mauchly": spher.p}
    reason = ""
    if test == "auto":
        p = checks["Shapiro residuals"]
        if p < _ALPHA:
            test, reason = "friedman", f"Shapiro-Wilk {format_p(p)} on the residuals"
        else:
            test, reason = "rm_anova", "Shapiro-Wilk p ≥ .05 on the residuals"
    warnings = []
    if n < 3:
        warnings.append("n < 3 animals: too few for a dependable test")
    if test == "rm_anova":
        f = (ss_epoch / df1) / (ss_error / df2) if ss_error > 0 else float("nan")
        p_unc = float(stats.f.sf(f, df1, df2)) if not math.isnan(f) else float("nan")
        p, dfs = p_unc, (float(df1), float(df2))
        if spher.violated:
            p = _corrected_p(f, df1, df2, spher.gg)
            dfs = (df1 * spher.gg, df2 * spher.gg)
            reason = (reason + "; " if reason else "") + (
                f"Mauchly {format_p(spher.p)}: Greenhouse-Geisser corrected (ε = {spher.gg:.3f})"
            )
        elif math.isnan(spher.p):
            warnings.append("too few animals to test sphericity; the uncorrected p assumes it")
        result = TestResult(
            test=REPEATED_TESTS[test],
            statistic_name="F",
            statistic=float(f),
            df=dfs,
            p=p,
            effect_name="partial η²",
            effect=ss_epoch / (ss_epoch + ss_error) if ss_epoch + ss_error else float("nan"),
            effect_ci=None,
            reason=reason,
            checks=checks,
            warnings=warnings,
            **base,
        )
        result.details = {
            "p uncorrected": p_unc,
            "p GG": _corrected_p(f, df1, df2, spher.gg),
            "p HF": _corrected_p(f, df1, df2, spher.hf),
            "GG ε": spher.gg,
            "HF ε": spher.hf,
            "Mauchly W": spher.w,
        }
        result.posthoc = pairwise_within(matrix, epochs, correction=correction)
        result.posthoc_name = f"Paired t-tests ({METHODS[correction]})"
        return result
    res = stats.friedmanchisquare(*matrix.T)
    chi = float(res.statistic)
    result = TestResult(
        test=REPEATED_TESTS[test],
        statistic_name="χ²",
        statistic=chi,
        df=(float(k - 1),),
        p=float(res.pvalue),
        effect_name="Kendall's W",
        effect=chi / (n * (k - 1)),
        effect_ci=None,
        reason=reason,
        checks=checks,
        warnings=warnings,
        **base,
    )
    result.posthoc = pairwise_within(matrix, epochs, rank=True, correction=correction)
    result.posthoc_name = f"Wilcoxon signed-rank ({METHODS[correction]})"
    return result


# ---------------------------------------------------------------------------
# group x epoch
# ---------------------------------------------------------------------------


@dataclass
class MixedResult:
    """A mixed ANOVA: the three effects, sphericity, and simple effects.

    ``effects`` maps ``"group"``, ``factor`` and ``"group × <factor>"`` to a
    :class:`TestResult` each; ``factor`` is ``"epoch"`` unless the caller
    names it (``"bin"`` for time bins). ``simple_effects`` is filled only when the
    interaction is significant: the epoch effect within each group, then the
    group effect within each epoch, Holm-adjusted together.
    """

    metric: str
    groups: list[str]
    epochs: list[str]
    n: dict[str, int]
    effects: dict[str, TestResult]
    sphericity: Sphericity
    simple_effects: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    factor: str = "epoch"

    def table(self) -> pd.DataFrame:
        """The ANOVA table, one row per effect."""
        rows = []
        for name, r in self.effects.items():
            rows.append(
                {
                    "effect": name,
                    "F": r.statistic,
                    "df1": r.df[0] if r.df else float("nan"),
                    "df2": r.df[1] if len(r.df) > 1 else float("nan"),
                    "p": r.p,
                    "p_uncorrected": r.details.get("p uncorrected", r.p),
                    "partial_eta2": r.effect,
                }
            )
        return pd.DataFrame(rows)

    def methods(self) -> str:
        cells = ", ".join(f"{g} (n = {n})" for g, n in self.n.items())
        parts = [
            f"{name}: {r.statistic_text()}, {format_p(r.p)}, partial η² = {r.effect:.2f}"
            for name, r in self.effects.items()
        ]
        corrected = (
            f" Greenhouse-Geisser corrected (Mauchly {format_p(self.sphericity.p)},"
            f" ε = {self.sphericity.gg:.3f})."
            if self.sphericity.violated
            else ""
        )
        return (
            f"Mixed ANOVA on {self.metric}, group × {self.factor} ({', '.join(self.epochs)}), "
            f"{cells}: " + "; ".join(parts) + "." + corrected
        )


def run_mixed(
    cells: dict[str, np.ndarray],
    epochs: list[str],
    metric: str = "",
    *,
    correction: str = "holm",
    dropped: list[Dropped] | None = None,
    factor: str = "epoch",
) -> MixedResult:
    """Group (between) x epoch (within). ``cells`` maps group -> animals x epochs matrix.

    ``factor`` names the within-subject factor in the effects and the text.
    """
    cells = {g: np.asarray(m, dtype=float) for g, m in cells.items() if len(m)}
    groups = list(cells)
    n = {g: m.shape[0] for g, m in cells.items()}
    k = len(epochs)
    if len(groups) < 2 or k < 2 or any(m.shape[1] != k for m in cells.values()):
        raise ValueError("run_mixed wants two or more groups, each an animals x epochs matrix")
    data = np.vstack([cells[g] for g in groups])
    n_total, n_groups = data.shape[0], len(groups)
    grand = data.mean()
    ss_total = float(((data - grand) ** 2).sum())
    ss_subjects = k * float(((data.mean(axis=1) - grand) ** 2).sum())
    ss_group = k * sum(n[g] * (cells[g].mean() - grand) ** 2 for g in groups)
    ss_epoch = n_total * float(((data.mean(axis=0) - grand) ** 2).sum())
    ss_cells = sum(n[g] * float(((cells[g].mean(axis=0) - grand) ** 2).sum()) for g in groups)
    ss_inter = ss_cells - ss_group - ss_epoch
    ss_err_between = ss_subjects - ss_group
    ss_err_within = ss_total - ss_subjects - ss_epoch - ss_inter
    df_group, df_err_between = n_groups - 1, n_total - n_groups
    df_epoch, df_inter = k - 1, (n_groups - 1) * (k - 1)
    df_err_within = (n_total - n_groups) * (k - 1)

    residuals = np.vstack([cells[g] - cells[g].mean(axis=0) for g in groups])
    spher = sphericity(residuals, df_err_between)
    warnings = []
    if min(n.values()) < 3:
        warnings.append("n < 3 in a group: too few animals for a dependable test")
    if math.isnan(spher.p) and k > 2:
        warnings.append("too few animals to test sphericity; the uncorrected p assumes it")
    pieces = [_residuals(m).ravel() for m in cells.values() if m.shape[0] > 1]
    p_resid = _shapiro(np.concatenate(pieces)) if pieces else float("nan")
    if p_resid < _ALPHA:
        warnings.append(
            f"residuals are not normal (Shapiro-Wilk {format_p(p_resid)}); there is no "
            "standard rank-based mixed ANOVA, so read the effects with care"
        )

    def effect(name, ss, df1, ss_err, df2, *, within: bool) -> TestResult:
        f = (ss / df1) / (ss_err / df2) if ss_err > 0 else float("nan")
        p_unc = float(stats.f.sf(f, df1, df2)) if not math.isnan(f) else float("nan")
        p, dfs = p_unc, (float(df1), float(df2))
        if within and spher.violated:
            p, dfs = _corrected_p(f, df1, df2, spher.gg), (df1 * spher.gg, df2 * spher.gg)
        result = TestResult(
            test="Mixed ANOVA",
            metric=f"{metric} ({name})",
            statistic_name="F",
            statistic=float(f),
            df=dfs,
            p=p,
            effect_name="partial η²",
            effect=ss / (ss + ss_err) if ss + ss_err else float("nan"),
            effect_ci=None,
            n=dict(n),
            checks={"Shapiro residuals": p_resid, "Mauchly": spher.p} if within else {},
        )
        result.details = {"p uncorrected": p_unc}
        if within:
            result.details.update(
                {
                    "p GG": _corrected_p(f, df1, df2, spher.gg),
                    "p HF": _corrected_p(f, df1, df2, spher.hf),
                }
            )
        return result

    effects = {
        "group": effect("group", ss_group, df_group, ss_err_between, df_err_between, within=False),
        factor: effect(factor, ss_epoch, df_epoch, ss_err_within, df_err_within, within=True),
        f"group × {factor}": effect(
            f"group × {factor}", ss_inter, df_inter, ss_err_within, df_err_within, within=True
        ),
    }
    result = MixedResult(
        metric=metric,
        groups=groups,
        epochs=list(epochs),
        n=n,
        effects=effects,
        sphericity=spher,
        warnings=warnings,
        dropped=list(dropped or []),
        factor=factor,
    )
    if effects[f"group × {factor}"].p < _ALPHA:
        result.simple_effects = _simple_effects(cells, epochs, metric, correction, factor)
    return result


def _simple_effects(cells, epochs, metric, correction, factor) -> pd.DataFrame:
    """What the interaction is made of: epoch within each group, group within each epoch."""
    rows = []
    for g, matrix in cells.items():
        if len(epochs) == 2:
            r = run_paired(matrix[:, 0], matrix[:, 1], tuple(epochs), metric, "paired_t")
        else:
            r = run_repeated(matrix, epochs, metric, "rm_anova", correction=correction)
        rows.append({"effect": factor, "within": g, "test": r.test, "result": r})
    for j, epoch in enumerate(epochs):
        column = {g: m[:, j] for g, m in cells.items()}
        r = run_between(column, metric, "welch" if len(column) == 2 else "anova")
        rows.append({"effect": "group", "within": epoch, "test": r.test, "result": r})
    frame = pd.DataFrame(
        {
            "effect": [row["effect"] for row in rows],
            "within": [row["within"] for row in rows],
            "test": [row["test"] for row in rows],
            "statistic": [row["result"].statistic_text() for row in rows],
            "p": [row["result"].p for row in rows],
        }
    )
    frame["p_adj"] = adjust(frame["p"], correction)
    return frame


def compare_mixed(
    tidy: pd.DataFrame, metric: str, epochs: list[str], *, correction: str = "holm"
) -> MixedResult:
    """Group x epoch on ``metric`` from the tidy frame; incomplete animals are dropped."""
    groups = list(between_cells(tidy[tidy["range"].isin(epochs)], metric)[0])
    cells, dropped = {}, []
    for group in groups:
        matrix, _kept, lost = within_matrix(tidy, metric, list(epochs), group=group)
        dropped += lost
        if len(matrix):
            cells[group] = matrix
    return run_mixed(cells, list(epochs), metric, correction=correction, dropped=dropped)
