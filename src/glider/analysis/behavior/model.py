"""Trained-model bundle: save + load + predict.

A :class:`BehaviorModel` wraps:

* the trained ``RandomForestClassifier``
* the ``FeatureSpec`` it was trained with
* the kept feature names (so we know exactly what columns to feed at
  inference)
* the rolling window length + stats
* the training fps

This is what the live inference pipeline (Part 2) loads. The file
format is a joblib pickle of one dict; we deliberately don't use a
custom binary format because (a) joblib handles the sklearn classifier
correctly without us having to think about it and (b) ``.pkl`` is what
the project plan literally asked for.

Cross-version warning
---------------------

Joblib pickles capture the exact sklearn version that produced them.
Loading on a newer / older sklearn occasionally produces a noisy
``InconsistentVersionWarning``; the bundle records the version it was
saved with so the loader can surface a friendlier message before sklearn
warns.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glider._version import __version__
from glider.analysis.behavior.features import FeatureSpec


def _threshold_decision(
    probs: np.ndarray,
    class_names: np.ndarray,
    confidence_threshold: float,
    class_thresholds: dict[str, float] | None,
) -> np.ndarray:
    """Per-class-threshold decision over a ``(n, k)`` probability matrix.

    A row fires the highest-probability class that clears its OWN
    threshold; if none clear theirs, it emits ``""`` ("unknown"). With no
    ``class_thresholds`` this is the plain global-threshold gate. Per-class
    thresholds (``{behavior: τ}``) let a clean minority class fire low
    while a noisy class sits high — something argmax + one global threshold
    can't express. Unlisted classes fall back to ``confidence_threshold``.
    """
    class_names = np.asarray(class_names)
    if class_thresholds:
        thr = np.array(
            [float(class_thresholds.get(str(c), confidence_threshold)) for c in class_names]
        )
    else:
        thr = np.full(len(class_names), float(confidence_threshold))
    above = probs >= thr
    masked = np.where(above, probs, -np.inf)
    best = np.argmax(masked, axis=1)
    return np.where(above.any(axis=1), class_names[best], "")


class BehaviorModel:
    """A trained behavior classifier with the metadata to apply it."""

    def __init__(
        self,
        classifier,
        feature_names: list[str],
        spec: FeatureSpec,
        window: int,
        stats: tuple[str, ...],
        fps: float,
        classes: list[str],
        training_summary: dict[str, Any] | None = None,
        library_versions: dict[str, str] | None = None,
        glider_version: str | None = None,
        embedding: Any | None = None,
    ):
        self.classifier = classifier
        self.feature_names = list(feature_names)
        self.spec = spec
        self.window = int(window)
        self.stats = tuple(stats)
        self.fps = float(fps)
        self.classes = list(classes)
        self.training_summary = dict(training_summary or {})
        self.library_versions = dict(library_versions or {})
        self.glider_version = glider_version or __version__
        # Optional fitted 3D feature-space embedding (EmbeddingArtifact)
        # for the live "galaxy" view. None for models trained without it
        # (and for all format_version-1 bundles).
        self.embedding = embedding

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        windowed: pd.DataFrame,
        confidence_threshold: float = 0.0,
        class_thresholds: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Predict behavior labels for each row of a windowed feature frame.

        ``windowed`` must have the columns recorded in
        :attr:`feature_names`, in that exact order. Rows with any NaN
        cell are predicted as ``""`` (i.e., "unknown") — the live-
        inference signal that the rolling window hasn't filled yet, so
        we'd rather emit a blank than a confident guess on partial data.

        ``confidence_threshold`` (in ``[0, 1]``) provides a runtime
        analog to "background" without poisoning training: rows whose
        max ``predict_proba`` is below this threshold are also emitted
        as ``""``. Use this with models trained without
        ``--with-background`` so you get a meaningful "unknown" label
        for frames the model hasn't seen examples of, instead of
        force-classifying every frame into one of the labeled
        behaviors.
        """
        if not self._is_fitted():
            raise RuntimeError("classifier is not fitted")
        missing = [c for c in self.feature_names if c not in windowed.columns]
        if missing:
            raise ValueError(
                f"input is missing {len(missing)} expected feature columns; "
                f"first few: {missing[:5]}"
            )
        # Keep the DataFrame intact through predict so sklearn sees the
        # same column names it was fit on (avoids a UserWarning about
        # missing feature names).
        df = windowed[self.feature_names]
        valid_mask = ~df.isna().any(axis=1).to_numpy()
        labels = np.full(len(df), "", dtype=object)
        if valid_mask.any():
            valid_df = df.loc[valid_mask]
            if confidence_threshold > 0 or class_thresholds:
                probs = self.classifier.predict_proba(valid_df)
                preds = _threshold_decision(
                    probs,
                    self.classifier.classes_,
                    confidence_threshold,
                    class_thresholds,
                )
            else:
                preds = self.classifier.predict(valid_df)
            labels[valid_mask] = preds
        return labels

    def posteriors(self, windowed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Class-aligned probabilities for the non-NaN rows.

        Returns ``(probs[n_valid, n_classes], valid_mask[n_rows])`` where the
        column order is ``self.classifier.classes_``. NaN feature rows (rolling
        window not yet filled) are excluded — the caller emits ``""`` for them,
        matching :meth:`predict`. The hybrid model blends these probabilities
        with a kinematic prior.
        """
        if not self._is_fitted():
            raise RuntimeError("classifier is not fitted")
        missing = [c for c in self.feature_names if c not in windowed.columns]
        if missing:
            raise ValueError(
                f"input is missing {len(missing)} expected feature columns; "
                f"first few: {missing[:5]}"
            )
        df = windowed[self.feature_names]
        valid = ~df.isna().any(axis=1).to_numpy()
        probs = (
            self.classifier.predict_proba(df.loc[valid])
            if valid.any()
            else np.empty((0, len(self.classifier.classes_)))
        )
        return probs, valid

    def predict_one(
        self,
        feature_row: np.ndarray | pd.Series,
        confidence_threshold: float = 0.0,
        class_thresholds: dict[str, float] | None = None,
    ) -> str:
        """Single-row predict for hot-path inference (live pipeline).

        Takes a 1-D array / Series of length ``len(feature_names)`` and
        returns a single behavior name (or ``""`` if any value is NaN
        or — when ``confidence_threshold > 0`` — if the model's top
        probability is below that threshold).
        """
        if not self._is_fitted():
            raise RuntimeError("classifier is not fitted")
        if isinstance(feature_row, pd.Series):
            row = feature_row.reindex(self.feature_names).to_numpy(dtype=np.float64)
        else:
            row = np.asarray(feature_row, dtype=np.float64)
            if row.shape != (len(self.feature_names),):
                raise ValueError(
                    f"feature_row shape {row.shape} != " f"({len(self.feature_names)},)"
                )
        if np.isnan(row).any():
            return ""
        # Wrap in a 1-row DataFrame so sklearn sees the feature names it
        # was fit on (avoids a UserWarning that fires on bare arrays).
        df1 = pd.DataFrame([row], columns=self.feature_names)
        if confidence_threshold > 0 or class_thresholds:
            probs = self.classifier.predict_proba(df1)
            return str(
                _threshold_decision(
                    probs,
                    self.classifier.classes_,
                    confidence_threshold,
                    class_thresholds,
                )[0]
            )
        return str(self.classifier.predict(df1)[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the model to ``path`` (joblib pickle)."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classifier": self.classifier,
            "feature_names": list(self.feature_names),
            "spec": self.spec.to_dict(),
            "window": int(self.window),
            "stats": list(self.stats),
            "fps": float(self.fps),
            "classes": list(self.classes),
            "training_summary": dict(self.training_summary),
            "library_versions": dict(self.library_versions),
            "glider_version": self.glider_version,
            "embedding": self.embedding,
            "format_version": 2,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BehaviorModel:
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "classifier" not in payload:
            raise ValueError(f"{path} doesn't look like a GLIDER behavior model bundle")
        _verify_library_versions(payload.get("library_versions"))
        return cls(
            classifier=payload["classifier"],
            feature_names=list(payload["feature_names"]),
            spec=FeatureSpec.from_dict(payload["spec"]),
            window=int(payload["window"]),
            stats=tuple(payload.get("stats", ("mean", "std", "max"))),
            fps=float(payload.get("fps", 30.0)),
            classes=list(payload.get("classes", [])),
            training_summary=payload.get("training_summary", {}),
            library_versions=payload.get("library_versions", {}),
            # Back-compat: accept legacy yolo2pose bundles (old key name).
            glider_version=payload.get("glider_version") or payload.get("yolo2pose_version"),
            embedding=payload.get("embedding"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_fitted(self) -> bool:
        return self.classifier is not None and hasattr(self.classifier, "classes_")


# Libraries whose binary pickle format we depend on.
_RECORDED_LIBS: tuple[str, ...] = ("numpy", "pandas", "scipy", "sklearn", "joblib")


def capture_library_versions() -> dict[str, str]:
    """Snapshot installed versions of the libraries we pickle from."""
    out: dict[str, str] = {}
    for name in _RECORDED_LIBS:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "not_installed"
    return out


def _verify_library_versions(recorded: dict[str, str] | None) -> None:
    """Warn on major.minor sklearn drift at load. Never raises."""
    if not recorded:
        return
    live = capture_library_versions()
    for name, saved in recorded.items():
        if saved in ("unknown", "not_installed"):
            continue
        current = live.get(name, "not_installed")
        if current in ("unknown", "not_installed"):
            warnings.warn(
                f"model was saved with {name}=={saved} but {name} is not "
                f"installed here; unpickling may fail",
                stacklevel=3,
            )
            continue
        if current == saved:
            continue
        try:
            saved_mm = tuple(int(p) for p in saved.split(".")[:2])
            cur_mm = tuple(int(p) for p in current.split(".")[:2])
        except Exception:
            continue
        if saved_mm != cur_mm:
            warnings.warn(
                f"model was saved with {name}=={saved} but this env has "
                f"{current}; behaviour may differ",
                stacklevel=3,
            )
