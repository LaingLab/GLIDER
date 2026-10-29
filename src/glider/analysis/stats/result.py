"""What a test returns: enough to report it, and enough to trust it."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from glider.analysis.stats.frames import Dropped

__all__ = ["TestResult", "format_p"]


def format_p(p: float) -> str:
    """APA style: exact to three places, ``< .001`` below that."""
    if p is None or math.isnan(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".", 1)


def _num(value: float, places: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{places}f}"


@dataclass
class TestResult:
    """One test on one metric.

    ``df`` is ``()`` for a test without degrees of freedom (Mann-Whitney),
    one value for t, two for F. ``effect_ci`` is ``None`` where no interval is
    computed. ``posthoc`` holds pairwise follow-ups for a several-group test:
    one row per pair with ``a, b, diff, statistic, p, p_adj``.
    """

    __test__ = False  # not a pytest class, whatever its name

    test: str
    metric: str
    statistic_name: str
    statistic: float
    df: tuple[float, ...]
    p: float
    effect_name: str
    effect: float
    effect_ci: tuple[float, float] | None
    n: dict[str, int]
    #: Mean difference (first minus second) and its bootstrap 95% CI; two-cell tests only.
    difference: float | None = None
    difference_ci: tuple[float, float] | None = None
    #: Why this test, when picked automatically; empty when the user chose it.
    reason: str = ""
    #: Assumption checks: ``"Shapiro <cell>"`` / ``"Levene"`` -> p.
    checks: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    posthoc: pd.DataFrame | None = None
    posthoc_name: str = ""
    #: Further numbers a careful reader may want: the uncorrected and
    #: epsilon-corrected p values and epsilons of a repeated-measures test.
    details: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the test produced a p value at all."""
        return not math.isnan(self.p)

    def statistic_text(self) -> str:
        """``t(12.31) = 2.41``, ``F(2, 21) = 5.10``, ``U = 12.0``."""
        if not self.df:
            return f"{self.statistic_name} = {_num(self.statistic)}"
        dfs = ", ".join(f"{d:g}" if float(d).is_integer() else f"{d:.2f}" for d in self.df)
        return f"{self.statistic_name}({dfs}) = {_num(self.statistic)}"

    def methods(self) -> str:
        """A sentence fit for a results section."""
        cells = ", ".join(f"{name} (n = {n})" for name, n in self.n.items())
        if not self.ok:
            return f"{self.test} on {self.metric} not run: {'; '.join(self.warnings)}"
        text = f"{self.test} on {self.metric}, {cells}: {self.statistic_text()}, {format_p(self.p)}"
        if self.effect_name:
            text += f", {self.effect_name} = {_num(self.effect)}"
            if self.effect_ci is not None:
                text += f" [95% CI {_num(self.effect_ci[0])}, {_num(self.effect_ci[1])}]"
        return text + "."
