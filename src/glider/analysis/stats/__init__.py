"""Statistics on the cohort's own numbers: the tests a paper needs. Qt-free.

The input is the epoch table's tidy frame (:func:`glider.analysis.epochs.tidy_frame`):
one row per animal per epoch, ``session, group, range, <metric>…``. The animal is
always the unit -- one value per animal per cell, never frames.

Every test returns a :class:`TestResult` that carries what a reader needs to
trust it: the statistic and its degrees of freedom, the exact p, an effect size
with a 95% CI, n per cell, the assumption checks, which animals were left out
and why, and -- when the test was picked automatically -- the reason it was.

* :func:`compare_groups` -- between subjects, one epoch: two groups or several.
* :func:`compare_epochs` -- within subjects: the same animals in two or more epochs.
* :func:`compare_mixed` -- group x epoch: the mixed (split-plot) ANOVA.
* :func:`compare_trend` -- change across time bins: slopes, contrasts, first vs last.
* :func:`run_between` / :func:`run_paired` -- the same on plain arrays.
* :mod:`.corrections` -- Bonferroni, Holm and Benjamini-Hochberg.
"""

from glider.analysis.stats.compare import (
    BETWEEN_TESTS,
    ONE_SAMPLE_TESTS,
    PAIRED_TESTS,
    compare_epochs,
    compare_groups,
    dunn,
    games_howell,
    run_between,
    run_one_sample,
    run_paired,
    tukey,
)
from glider.analysis.stats.corrections import adjust
from glider.analysis.stats.frames import UNGROUPED, Dropped, between_cells, within_matrix
from glider.analysis.stats.repeated import (
    REPEATED_TESTS,
    MixedResult,
    Sphericity,
    compare_mixed,
    pairwise_within,
    run_mixed,
    run_repeated,
    sphericity,
)
from glider.analysis.stats.result import TestResult
from glider.analysis.stats.trend import (
    GroupTrend,
    TrendResult,
    animal_slopes,
    compare_trend,
    polynomial_contrasts,
    run_trend,
    trend_contrasts,
)

__all__ = [
    "BETWEEN_TESTS",
    "ONE_SAMPLE_TESTS",
    "PAIRED_TESTS",
    "REPEATED_TESTS",
    "MixedResult",
    "Sphericity",
    "UNGROUPED",
    "Dropped",
    "GroupTrend",
    "TestResult",
    "TrendResult",
    "adjust",
    "animal_slopes",
    "between_cells",
    "compare_epochs",
    "compare_groups",
    "compare_mixed",
    "compare_trend",
    "dunn",
    "games_howell",
    "pairwise_within",
    "polynomial_contrasts",
    "run_between",
    "run_mixed",
    "run_one_sample",
    "run_paired",
    "run_repeated",
    "run_trend",
    "sphericity",
    "trend_contrasts",
    "tukey",
    "within_matrix",
]
