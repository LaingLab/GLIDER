"""Hybrid behavior model: base posterior blended with a kinematic prior.

log P_final = (1-lam)*log P_model + lam*log P_prior, argmax over the base
classes. lam=0 recovers the base model exactly. Rows the base cannot score
(NaN features / below confidence threshold) pass through as "" unblended.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from glider.analysis.behavior.model import BehaviorModel, _threshold_decision
from glider.analysis.behavior.prior import DEFAULT_RULES, KinematicPrior, Rule

_EPS = 1e-12


class HybridModel:
    def __init__(
        self,
        base: BehaviorModel,
        prior: KinematicPrior,
        lam: float,
        tag_map: dict[str, frozenset[str]],
    ):
        self.base = base
        self.prior = prior
        self.lam = float(lam)
        self.tag_map = {k: frozenset(v) for k, v in tag_map.items()}

    @property
    def classes(self) -> list[str]:
        return list(self.base.classes)

    def predict(
        self,
        windowed: pd.DataFrame,
        confidence_threshold: float = 0.0,
        class_thresholds: dict[str, float] | None = None,
    ) -> np.ndarray:
        classes = list(self.base.classifier.classes_)
        probs, valid = self.base.posteriors(windowed)  # (n_valid, k), mask
        labels = np.full(len(windowed), "", dtype=object)
        if not valid.any():
            return labels
        log_model = np.log(np.clip(probs, _EPS, None))
        if self.lam > 0.0:
            log_prior = self.prior.log_prior(windowed.loc[valid], classes)
            blended = (1.0 - self.lam) * log_model + self.lam * log_prior
        else:
            blended = log_model  # lam=0 => exactly base
        # Normalize to a proper posterior so thresholds are comparable.
        blended -= blended.max(axis=1, keepdims=True)
        p = np.exp(blended)
        p /= p.sum(axis=1, keepdims=True)
        if confidence_threshold > 0 or class_thresholds:
            preds = _threshold_decision(
                p, np.asarray(classes), confidence_threshold, class_thresholds
            )
        else:
            preds = np.asarray(classes)[np.argmax(p, axis=1)]
        labels[valid] = preds
        return labels

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "hybrid",
                "format_version": 1,
                "base": self.base,  # BehaviorModel pickles cleanly
                "lam": self.lam,
                "tag_map": {k: sorted(v) for k, v in self.tag_map.items()},
                "prior_rules": [(r.name, r.tag_weights) for r in self.prior.rules],
                "prior_freeze_thr": self.prior._freeze_thr,
                "prior_dart_thr": self.prior._dart_thr,
                "prior_scale": self.prior._scale,
                "prior_pcts": (self.prior.freeze_pct, self.prior.dart_pct),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> HybridModel:
        import joblib

        payload = joblib.load(Path(path))
        if payload.get("kind") != "hybrid":
            raise ValueError(f"{path} is not a GLIDER hybrid model bundle")
        tag_map = {k: frozenset(v) for k, v in payload["tag_map"].items()}
        rules = tuple(
            Rule(n, w)
            for n, w in payload.get("prior_rules", [(r.name, r.tag_weights) for r in DEFAULT_RULES])
        )
        fp, dp = payload["prior_pcts"]
        prior = KinematicPrior(tag_map=tag_map, rules=rules, freeze_pct=fp, dart_pct=dp)
        prior._freeze_thr = payload["prior_freeze_thr"]
        prior._dart_thr = payload["prior_dart_thr"]
        prior._scale = payload["prior_scale"]
        return cls(payload["base"], prior, payload["lam"], tag_map)
