"""Hybrid behavior model: base posterior blended with a kinematic prior.

log P_final = (1-lam)*log P_model + lam*log P_prior, argmax over the base
classes. lam=0 recovers the base model exactly. Rows the base cannot score
(NaN features / below confidence threshold) pass through as "" unblended.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from glider.analysis.behavior.model import (
    BehaviorModel,
    _threshold_decision,
    _verify_library_versions,
    capture_library_versions,
)
from glider.analysis.behavior.prior import KinematicPrior

_EPS = 1e-12


class HybridModel:
    """A supervised :class:`BehaviorModel` fused with a :class:`KinematicPrior`.

    Inference blends the base classifier's per-row posterior with the prior's
    per-class log-prior in log space::

        log P_final = (1 - lam) * log P_model + lam * log P_prior

    and decides by argmax over the base classes. ``lam`` (in ``[0, 1]``) trades
    supervised evidence against the kinematic prior: ``lam=0`` recovers the base
    model's predictions exactly; ``lam=1`` is (almost) the pure prior. Rows the
    base cannot score — NaN features (rolling window not yet filled) or, when a
    threshold is set, below-confidence rows — pass through as ``""`` and are
    never blended.
    """

    def __init__(
        self,
        base: BehaviorModel,
        prior: KinematicPrior,
        lam: float,
        tag_map: dict[str, frozenset[str]],
    ):
        """Bind a base model and a calibrated prior at blend weight ``lam``.

        ``lam`` must be in ``[0, 1]`` — an out-of-range value raises
        ``ValueError`` rather than silently producing a mis-weighted blend.
        ``tag_map`` mirrors the prior's semantic class tags; the prior owns its
        own copy for persistence, so this is retained only for introspection.
        """
        lam = float(lam)
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.base = base
        self.prior = prior
        self.lam = lam
        self.tag_map = {k: frozenset(v) for k, v in tag_map.items()}

    @property
    def classes(self) -> list[str]:
        """The classes ``predict`` decides over, in the order it uses them."""
        return list(self.base.classifier.classes_)

    def predict(
        self,
        windowed: pd.DataFrame,
        confidence_threshold: float = 0.0,
        class_thresholds: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Predict a behavior label per row of a windowed feature frame.

        Mirrors :meth:`BehaviorModel.predict`'s NaN semantics: rows with any
        NaN feature cell are emitted as ``""`` ("unknown") and never blended —
        the live-inference signal that the rolling window hasn't filled yet.
        Valid rows get the log-space blend, are renormalized to a per-row
        posterior, and decided by argmax (or, when a threshold is given, by
        :func:`_threshold_decision`).

        ``confidence_threshold`` (and ``class_thresholds``) gate on the
        *blended* posterior, and are **lam-dependent**: the per-row softmax
        renormalizes so probabilities sum to 1, but the effective sharpness of
        that distribution shifts with ``lam`` (the prior sharpens or flattens
        it). A threshold tuned against the base model (``lam=0``) therefore
        gates differently at other ``lam`` values — retune per ``lam``.
        """
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
        # Renormalize each row to sum to 1 so a threshold can be applied. NOTE:
        # this makes rows comparable to each other, NOT across lam — the blend's
        # sharpness (and thus the meaning of a fixed threshold) shifts with lam.
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
                "prior": self.prior.to_dict(),
                # The base pickles the sklearn classifier inline; record the
                # library versions so load can warn on cross-version drift,
                # mirroring BehaviorModel.save's protection.
                "library_versions": capture_library_versions(),
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
        _verify_library_versions(payload.get("library_versions"))
        prior = KinematicPrior.from_dict(payload["prior"])
        tag_map = {k: frozenset(v) for k, v in payload["tag_map"].items()}
        return cls(payload["base"], prior, payload["lam"], tag_map)
