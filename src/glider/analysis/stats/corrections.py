"""Multiple-comparison corrections of a family of p values.

NaN p values (a comparison that could not be run) pass through as NaN and do
not count towards the family size.
"""

from __future__ import annotations

import numpy as np

__all__ = ["METHODS", "adjust"]

#: Accepted ``method`` names, and how a methods line names each.
METHODS = {
    "bonferroni": "Bonferroni",
    "holm": "Holm",
    "fdr_bh": "Benjamini-Hochberg",
}


def adjust(p_values, method: str = "holm") -> np.ndarray:
    """Adjusted p values, in the input order, capped at 1."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    if method not in METHODS:
        raise ValueError(f"unknown correction {method!r}; one of {sorted(METHODS)}")
    pv = p[ok]
    if method == "bonferroni":
        adjusted = pv * m
    else:
        order = np.argsort(pv, kind="stable")
        ranked = pv[order]
        if method == "holm":
            # Step-down: the k-th smallest times (m - k), never below an earlier one.
            stepped = np.maximum.accumulate(ranked * (m - np.arange(m)))
        else:
            # Step-up: the k-th smallest times m / k, never above a later one.
            stepped = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
        adjusted = np.empty(m)
        adjusted[order] = stepped
    out[ok] = np.minimum(adjusted, 1.0)
    return out
