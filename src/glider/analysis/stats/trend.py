"""Change across time bins: does a measure drift over the session, and do groups differ?

The input is the time-bin export's tidy frame (one row per animal per bin,
``bin, bin_start_s, bin_end_s, <metric>…``) or any tidy frame whose ``range``
column is in time order. Three readings of one question, each with its own
strength:

* **Slope per animal** -- an ordinary least-squares line through each
  animal's bins, in units per minute. The slopes are then one number per
  animal: tested against zero ("does it habituate?") and between groups
  ("does one group habituate faster?"). Uses every bin an animal has.
* **Polynomial contrasts** -- linear and quadratic trend within the
  repeated-measures design, each tested on its own error term (t² is the
  contrast's F). The quadratic answers "does it level off?".
* **First vs last bin** -- the plain habituation index: change and ratio per
  animal, the change tested as a paired comparison.

Plus the repeated-measures ANOVA across bins for one group, and the group x
bin mixed ANOVA for several, from :mod:`.repeated`.

A partial last bin (the session ended mid-bin) holds less time, so for a
total such as distance it reads as the animal slowing down -- a habituation
that is not there. Partial bins are left out unless asked for, and named.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from glider.analysis.stats.compare import run_between, run_one_sample, run_paired
from glider.analysis.stats.corrections import METHODS, adjust
from glider.analysis.stats.frames import UNGROUPED, Dropped
from glider.analysis.stats.repeated import MixedResult, run_mixed, run_repeated
from glider.analysis.stats.result import TestResult, format_p

__all__ = [
    "GroupTrend",
    "TrendResult",
    "animal_slopes",
    "compare_trend",
    "polynomial_contrasts",
    "run_trend",
    "trend_contrasts",
]

#: A bin shorter than this fraction of the full width is partial.
_PARTIAL = 0.999


# ---------------------------------------------------------------------------
# arithmetic on arrays
# ---------------------------------------------------------------------------


def polynomial_contrasts(times, degree: int = 2) -> np.ndarray:
    """Orthonormal polynomial contrast weights for (possibly unequal) ``times``.

    ``(k, degree)``: column 0 linear, column 1 quadratic. Each sums to zero,
    they are mutually orthogonal, and each is signed to rise with time (its
    own power of time), so a positive contrast means "goes up" / "curves up".
    """
    t = np.asarray(times, dtype=float)
    degree = min(degree, len(t) - 1)
    vander = np.vander(t - t.mean(), degree + 1, increasing=True)
    q, _ = np.linalg.qr(vander)
    weights = q[:, 1 : degree + 1]
    for j in range(weights.shape[1]):
        power = (t - t.mean()) ** (j + 1)
        if weights[:, j] @ (power - power.mean()) < 0:
            weights[:, j] = -weights[:, j]
    return weights


def animal_slopes(matrix, times) -> pd.DataFrame:
    """One OLS line per row of an animals x bins matrix, NaN bins skipped.

    Columns ``slope, intercept, r2, n_bins``; NaN where an animal has fewer
    than three bins (two points always lie on a line, so their slope would
    carry no information about fit).
    """
    matrix = np.asarray(matrix, dtype=float)
    t = np.asarray(times, dtype=float)
    rows = []
    for values in matrix:
        ok = np.isfinite(values)
        if ok.sum() < 3:
            rows.append(
                {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n_bins": int(ok.sum())}
            )
            continue
        x, y = t[ok], values[ok]
        if np.ptp(y) == 0:
            slope, intercept, r2 = 0.0, float(y[0]), float("nan")
        else:
            fit = stats.linregress(x, y)
            slope, intercept, r2 = float(fit.slope), float(fit.intercept), float(fit.rvalue**2)
        rows.append({"slope": slope, "intercept": intercept, "r2": r2, "n_bins": int(ok.sum())})
    return pd.DataFrame(rows, columns=["slope", "intercept", "r2", "n_bins"])


def trend_contrasts(matrix, times, *, correction: str = "holm") -> pd.DataFrame:
    """Linear and quadratic trend over a complete animals x bins matrix.

    Each contrast is the one-sample t-test of the animals' contrast scores
    against zero (Maxwell & Delaney's separate error term), so t² is the
    contrast's F on (1, n - 1) df. ``partial_eta2 = t² / (t² + df)``.
    """
    matrix = np.asarray(matrix, dtype=float)
    weights = polynomial_contrasts(times)
    names = ["linear", "quadratic"][: weights.shape[1]]
    rows = []
    n = matrix.shape[0]
    for name, w in zip(names, weights.T, strict=True):
        scores = matrix @ w
        if n < 2 or np.ptp(scores) == 0:
            t = p = float("nan")
        else:
            res = stats.ttest_1samp(scores, 0.0)
            t, p = float(res.statistic), float(res.pvalue)
        rows.append(
            {
                "contrast": name,
                "estimate": float(scores.mean()) if n else float("nan"),
                "t": t,
                "df": float(n - 1),
                "p": p,
                "partial_eta2": t**2 / (t**2 + n - 1) if not math.isnan(t) else float("nan"),
            }
        )
    frame = pd.DataFrame(rows, columns=["contrast", "estimate", "t", "df", "p", "partial_eta2"])
    frame["p_adj"] = adjust(frame["p"], correction)
    return frame


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class GroupTrend:
    """One group's trend, every reading of it."""

    group: str
    #: Slopes against zero; ``difference`` is the mean slope, per minute.
    slope_test: TestResult
    #: First bin against last, paired; ``difference`` is first minus last.
    habituation: TestResult
    #: Linear and quadratic contrasts over the complete animals; None under two bins.
    contrasts: pd.DataFrame | None = None
    #: Repeated-measures ANOVA across bins; None under three bins.
    anova: TestResult | None = None


@dataclass
class TrendResult:
    """Trend of one metric across time bins, per group and between groups."""

    metric: str
    bins: list[str]
    #: Bin midpoints, in ``time_unit``.
    times: list[float]
    time_unit: str
    #: One row per animal: ``session, group, slope, intercept, r2, n_bins,
    #: first, last, change, ratio``.
    per_animal: pd.DataFrame
    groups: dict[str, GroupTrend]
    #: Several groups only: slopes, and first-to-last change, compared between groups.
    slope_comparison: TestResult | None = None
    change_comparison: TestResult | None = None
    #: Several groups only: group x bin, complete animals.
    mixed: MixedResult | None = None
    excluded_bins: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)

    def methods(self) -> str:
        """A short paragraph: what was fitted, and the headline numbers."""
        lines = [
            f"Trend of {self.metric} across {len(self.bins)} time bins "
            f"({self.bins[0]} to {self.bins[-1]})"
            + (
                f", excluding partial bin(s) {', '.join(self.excluded_bins)}"
                if self.excluded_bins
                else ""
            )
            + "."
        ]
        for name, g in self.groups.items():
            slope = g.slope_test
            if slope.ok:
                lines.append(
                    f"{name}: mean slope {slope.difference:.3g} per {self.time_unit} "
                    f"({slope.test}, {slope.statistic_text()}, "
                    f"{format_p(slope.p)})."
                )
        if self.slope_comparison is not None and self.slope_comparison.ok:
            lines.append(f"Slopes between groups: {self.slope_comparison.methods()}")
        if self.mixed is not None:
            lines.append(self.mixed.methods())
        return " ".join(lines)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def run_trend(
    cells: dict[str, tuple[np.ndarray, list[str]]],
    bins: list[str],
    times,
    metric: str = "",
    *,
    time_unit: str = "min",
    correction: str = "holm",
    dropped: list[Dropped] | None = None,
) -> TrendResult:
    """``cells`` maps group -> (animals x bins matrix, NaN where missing; session ids)."""
    if correction not in METHODS:
        raise ValueError(f"unknown correction {correction!r}; one of {sorted(METHODS)}")
    times = [float(t) for t in times]
    if len(bins) < 2:
        raise ValueError("a trend needs two or more bins")
    dropped = list(dropped or [])
    warnings: list[str] = []
    per_animal = []
    groups: dict[str, GroupTrend] = {}
    complete: dict[str, np.ndarray] = {}
    for group, (matrix, sessions) in cells.items():
        matrix = np.asarray(matrix, dtype=float)
        slopes = animal_slopes(matrix, times)
        first, last = matrix[:, 0], matrix[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(first != 0, last / first, np.nan)
        for i, session in enumerate(sessions):
            per_animal.append(
                {
                    "session": session,
                    "group": group,
                    **slopes.iloc[i].to_dict(),
                    "first": first[i],
                    "last": last[i],
                    "change": last[i] - first[i],
                    "ratio": ratio[i],
                }
            )
            if math.isnan(slopes.iloc[i]["slope"]):
                dropped.append(Dropped(session, f"fewer than 3 bins with {metric} (no slope)"))
        has_slope = slopes["slope"].notna().to_numpy()
        slope_test = run_one_sample(
            slopes["slope"].to_numpy()[has_slope], 0.0, f"{metric} slope", name=group
        )
        ends = np.isfinite(first) & np.isfinite(last)
        habituation = run_paired(first[ends], last[ends], (bins[0], bins[-1]), metric, dropped=[])
        full = np.isfinite(matrix).all(axis=1)
        trend = GroupTrend(group, slope_test, habituation)
        if full.sum() >= 2:
            trend.contrasts = trend_contrasts(matrix[full], times, correction=correction)
            if len(bins) >= 3:
                trend.anova = run_repeated(matrix[full], list(bins), metric, correction=correction)
            complete[group] = matrix[full]
        groups[group] = trend
    frame = pd.DataFrame(per_animal)
    result = TrendResult(
        metric=metric,
        bins=list(bins),
        times=times,
        time_unit=time_unit,
        per_animal=frame,
        groups=groups,
        warnings=warnings,
        dropped=dropped,
    )
    if len(groups) >= 2:
        slopes_by = {
            g: frame.loc[(frame["group"] == g) & frame["slope"].notna(), "slope"].to_numpy()
            for g in groups
        }
        result.slope_comparison = run_between(slopes_by, f"{metric} slope", correction=correction)
        change_by = {
            g: frame.loc[(frame["group"] == g) & frame["change"].notna(), "change"].to_numpy()
            for g in groups
        }
        result.change_comparison = run_between(change_by, f"{metric} change", correction=correction)
        usable = {g: m for g, m in complete.items() if m.shape[0] >= 2}
        if len(usable) >= 2:
            result.mixed = run_mixed(
                usable, list(bins), metric, correction=correction, factor="bin"
            )
        else:
            warnings.append("too few complete animals per group for the group x bin ANOVA")
    return result


def _group_of(value) -> str:
    return UNGROUPED if value is None or str(value) in ("", "nan") else str(value)


def compare_trend(
    tidy: pd.DataFrame,
    metric: str,
    *,
    include_partial: bool = False,
    correction: str = "holm",
) -> TrendResult:
    """Trend of ``metric`` from a tidy time-bin frame.

    Bins are ordered and placed by ``bin_start_s`` / ``bin_end_s`` when the
    frame has them (the time-bin export), in minutes; otherwise ``range`` (or
    ``bin``) labels are taken in order of first appearance, one bin apart.
    """
    label = "bin" if "bin" in tidy.columns else "range"
    timed = {"bin_start_s", "bin_end_s"} <= set(tidy.columns)
    if timed:
        edges = tidy.groupby(label, sort=False)[["bin_start_s", "bin_end_s"]].first()
        edges = edges.sort_values("bin_start_s")
        widths = edges["bin_end_s"] - edges["bin_start_s"]
        partial = widths < _PARTIAL * widths.max()
        excluded = [] if include_partial else [str(b) for b in edges.index[partial]]
        edges = edges[~edges.index.isin(excluded)]
        bins = [str(b) for b in edges.index]
        times = ((edges["bin_start_s"] + edges["bin_end_s"]) / 2 / 60.0).tolist()
        unit = "min"
    else:
        bins = [str(b) for b in pd.unique(tidy[label])]
        times, unit, excluded = list(range(len(bins))), "bin", []
    rows = tidy[tidy[label].astype(str).isin(bins)]
    cells: dict[str, tuple[list, list[str]]] = {}
    for session, block in rows.groupby("session", sort=False):
        group = _group_of(block["group"].iloc[0] if "group" in block else None)
        by_bin = dict(zip(block[label].astype(str), pd.to_numeric(block[metric]), strict=True))
        values = [float(by_bin.get(b, np.nan)) for b in bins]
        matrix, sessions = cells.setdefault(group, ([], []))
        matrix.append(values)
        sessions.append(str(session))
    order = sorted(cells, key=lambda g: (g == UNGROUPED, g))
    result = run_trend(
        {g: (np.asarray(cells[g][0], dtype=float), cells[g][1]) for g in order},
        bins,
        times,
        metric,
        time_unit=unit,
        correction=correction,
    )
    result.excluded_bins = excluded
    if excluded:
        result.warnings.insert(
            0,
            f"left out partial bin {', '.join(excluded)}: it holds less time, so a total "
            "would read as a decline",
        )
    return result
