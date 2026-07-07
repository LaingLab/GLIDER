"""Freeze/dart kinematic prior for the hybrid behavior model.

Pure functions over the windowed feature frame the base model consumes. Turns
graded freeze/dart activations into a per-class log-prior via semantic class
tags, so the same rules transfer across vocabularies (rules key off tags, not
class names). See docs/superpowers/specs/2026-07-07-hybrid-behavior-prior-fusion-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Single source of truth for the percentiles (shared with the live detector).
from glider.analysis.behavior.classify.speed_state import (
    DART_PCT_DEFAULT as DART_PCT,
)
from glider.analysis.behavior.classify.speed_state import (
    FREEZE_PCT_DEFAULT as FREEZE_PCT,
)


def prior_speed(windowed: pd.DataFrame) -> np.ndarray:
    """Scalar speed per row = mean across keypoints of the `speed_*__mean` columns.

    NaN keypoints are skipped per row (pandas `.mean` default `skipna=True`).
    """
    cols = [c for c in windowed.columns if c.startswith("speed_") and c.endswith("__mean")]
    if not cols:
        raise ValueError("no speed_*__mean columns in the windowed frame")
    return windowed[cols].mean(axis=1).to_numpy(dtype=np.float64)


def calibrate_thresholds(
    speed: np.ndarray, *, freeze_pct: float = FREEZE_PCT, dart_pct: float = DART_PCT
) -> tuple[float, float]:
    """(freeze, dart) = the freeze_pct / dart_pct percentiles of `speed` (NaNs dropped)."""
    arr = np.asarray(speed, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("no valid speed samples to calibrate from")
    return float(np.percentile(arr, freeze_pct)), float(np.percentile(arr, dart_pct))


@dataclass(frozen=True)
class Rule:
    """A kinematic rule: an activation name grades a set of tag weights."""

    name: str
    tag_weights: dict[str, float]  # e.g. {"stationary": +1.0, "locomotory": -1.0}


# The activation names `_activations` can currently produce. A rule's name must
# be one of these — it selects the kinematic activation the rule grades. This is
# the single source of truth shared by `_activations` and `KinematicPrior.__init__`
# so the two can't silently drift.
KNOWN_ACTIVATIONS: frozenset[str] = frozenset({"freeze", "dart"})


# v1 rules — freeze and dart. Adding a new rule (rhythmic / vertical /
# displacement) is NOT a data-only change today: it requires BOTH a new entry
# here AND a matching branch in `_activations` that computes its activation from
# the relevant kinematic signal (plus adding its name to KNOWN_ACTIVATIONS).
# Without the activation branch, a rule name outside KNOWN_ACTIVATIONS is
# rejected eagerly in `__init__`. The activation layer is freeze/dart-only.
DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("freeze", {"stationary": 1.0, "locomotory": -1.0}),
    Rule("dart", {"locomotory": 1.0, "stationary": -1.0}),
)


class KinematicPrior:
    """Windowed feature frame -> per-class log-prior via tag-keyed graded rules."""

    def __init__(
        self,
        tag_map: dict[str, frozenset[str]],
        rules: tuple[Rule, ...] = DEFAULT_RULES,
        *,
        freeze_pct: float = FREEZE_PCT,
        dart_pct: float = DART_PCT,
    ):
        unknown = [r.name for r in rules if r.name not in KNOWN_ACTIVATIONS]
        if unknown:
            raise ValueError(
                f"rule(s) {unknown} have no matching activation in _activations; "
                f"add a branch there first. Known activations: {sorted(KNOWN_ACTIVATIONS)}"
            )
        self.tag_map = {k: frozenset(v) for k, v in tag_map.items()}
        self.rules = rules
        self.freeze_pct, self.dart_pct = freeze_pct, dart_pct
        self._freeze_thr: float | None = None
        self._dart_thr: float | None = None
        self._scale: float | None = None

    def calibrate(self, windowed: pd.DataFrame) -> None:
        """Set freeze/dart thresholds (and activation scale) from this frame's speed."""
        speed = prior_speed(windowed)
        self._freeze_thr, self._dart_thr = calibrate_thresholds(
            speed, freeze_pct=self.freeze_pct, dart_pct=self.dart_pct
        )
        # Scale = a fraction of the freeze..dart span, so activation is graded
        # (not a cliff) across the regime. Guard against a degenerate span.
        self._scale = max((self._dart_thr - self._freeze_thr) / 6.0, 1e-6)

    def to_dict(self) -> dict:
        """Serialize all persistence-critical state (mirrors ``FeatureSpec.to_dict``).

        Owns everything a round-trip needs: the tag map, the rules (name +
        tag weights), the freeze/dart percentiles, and — if :meth:`calibrate`
        has run — the calibrated thresholds and activation scale.
        """
        return {
            "tag_map": {k: sorted(v) for k, v in self.tag_map.items()},
            "rules": [[r.name, dict(r.tag_weights)] for r in self.rules],
            "freeze_pct": float(self.freeze_pct),
            "dart_pct": float(self.dart_pct),
            "freeze_thr": self._freeze_thr,
            "dart_thr": self._dart_thr,
            "scale": self._scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> KinematicPrior:
        """Rebuild a prior from :meth:`to_dict` output, restoring calibration."""
        tag_map = {k: frozenset(v) for k, v in d["tag_map"].items()}
        rules = tuple(Rule(str(name), dict(w)) for name, w in d.get("rules", []))
        if not rules:
            rules = DEFAULT_RULES
        prior = cls(
            tag_map=tag_map,
            rules=rules,
            freeze_pct=float(d.get("freeze_pct", FREEZE_PCT)),
            dart_pct=float(d.get("dart_pct", DART_PCT)),
        )
        prior._freeze_thr = d.get("freeze_thr")
        prior._dart_thr = d.get("dart_thr")
        prior._scale = d.get("scale")
        return prior

    def _activations(self, speed: np.ndarray) -> dict[str, np.ndarray]:
        """Graded freeze/dart activation in [0,1] per row.

        A clipped-linear ramp: exactly 0 inside the neutral band (between the
        freeze and dart thresholds), ramping to 1 over `scale` past each edge.
        """
        if self._freeze_thr is None:
            raise RuntimeError("call calibrate() first")
        freeze = np.clip((self._freeze_thr - speed) / self._scale, 0.0, 1.0)  # slow
        dart = np.clip((speed - self._dart_thr) / self._scale, 0.0, 1.0)  # fast
        acts = {"freeze": freeze, "dart": dart}
        # Keep the produced keys and the advertised set in lockstep — if this
        # ever drifts from KNOWN_ACTIVATIONS, the __init__ validation would lie.
        assert set(acts) == KNOWN_ACTIVATIONS
        return acts

    def log_prior(self, windowed: pd.DataFrame, classes: list[str]) -> np.ndarray:
        """(n_rows, n_classes) additive log-prior aligned to `classes`.

        A neutral row (mid-speed) is all-zeros; the blend treats all-zeros as a
        uniform prior, so it contributes nothing there. Untagged classes stay 0.

        NaN-speed rows yield NaN prior rows; the caller (`HybridModel`) masks
        them via the `""` passthrough, so the prior is never blended for them.
        """
        speed = prior_speed(windowed)
        acts = self._activations(speed)
        n = len(speed)
        tag_lp: dict[str, np.ndarray] = {}
        for r in self.rules:
            a = acts[r.name]
            for tag, w in r.tag_weights.items():
                tag_lp[tag] = tag_lp.get(tag, np.zeros(n)) + a * w
        out = np.zeros((n, len(classes)), dtype=np.float64)
        for j, c in enumerate(classes):
            for tag in self.tag_map.get(c, frozenset()):
                if tag in tag_lp:
                    out[:, j] += tag_lp[tag]
        return out
