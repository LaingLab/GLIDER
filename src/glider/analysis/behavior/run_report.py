"""Read a training run's report folder back into something a UI can show.

:func:`glider.analysis.behavior.report.write_training_report` writes the run to
disk -- ``summary.json`` plus tidy CSVs and charts. This is the other half:
loading one back, and answering the questions a person actually asks of a
training run, rather than handing over the raw twenty-key summary dict.

Those questions are:

* Is this model any good, and *on what evidence*? A run with no held-out set
  reports a train accuracy near 1.000 that says nothing about a new animal, so
  the headline number and the warning that goes with it are computed together
  and cannot be shown apart -- see :attr:`TrainingRun.headline`.
* Which class is it failing on? That is the lowest-F1 row, and finding it by
  eye in a table of twelve behaviors is exactly the work a report should do.
* What did it confuse it *with*? A confusion matrix read row-normalized, so
  each row is "of the frames that really were X, where did they go".

Qt-free and dependency-light on purpose, in the shape of
:mod:`glider.analysis.behavior.session_view`: everything here is tested
without a display, and the widget in
:mod:`glider.gui.behavior.review_tab` only draws.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Headline", "RunReportError", "TrainingRun"]


class RunReportError(RuntimeError):
    """A report folder that cannot be read as a training run."""


#: How the headline score was arrived at, worst evidence first. The wording is
#: the caption shown under the number, so it is phrased for a reader deciding
#: whether to trust it rather than as a key.
_EVIDENCE = {
    "cross_validated": ("Mean macro F1", "averaged over folds of unseen animals", "strong"),
    "cross_session": ("Test accuracy", "measured on animals held out of training", "strong"),
    "group_shuffle": ("Test accuracy", "held-out zones from the same animals", "fair"),
    "no_holdout": ("Train accuracy", "measured on the data it was fitted to", "weak"),
}


@dataclass(frozen=True)
class Headline:
    """The one number to lead with, and how much it is worth.

    ``strength`` is ``strong`` / ``fair`` / ``weak`` and drives how the number
    is presented. A weak headline is not a small caveat: a LightGBM model fit
    without a held-out set routinely scores above 0.99 on its own training
    rows, and shown bare that reads as a finished model.
    """

    value: float | None
    label: str
    caption: str
    strength: str
    spread: float | None = None  # ± over folds, when cross-validated

    @property
    def is_trustworthy(self) -> bool:
        return self.strength != "weak"


@dataclass
class ClassMetric:
    """One behavior's precision / recall / F1 / support."""

    name: str
    precision: float | None
    recall: float | None
    f1: float | None
    support: int | None


@dataclass(frozen=True)
class _EmbeddingPoints:
    """Just enough of an embedding to draw one.

    Deliberately not an ``EmbeddingArtifact``: that carries a fitted scaler
    and reducer for projecting *new* rows, which only the live path needs.
    Reviewing a run needs the points and their labels, and nothing that
    would require the model bundle to reconstruct.
    """

    coords: Any
    labels: Any
    method: str = ""


@dataclass
class TrainingRun:
    """One training run, loaded from its report folder."""

    summary: dict[str, Any]
    path: Path | None = None
    charts: list[Path] = field(default_factory=list)

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(cls, report_dir: Path | str) -> TrainingRun:
        """Read ``summary.json`` out of a report folder.

        Accepts the folder itself or the ``summary.json`` inside it, because
        both are things a person will reasonably pick in a file dialog.
        """
        path = Path(report_dir)
        if path.is_file():
            path = path.parent
        summary_path = path / "summary.json"
        if not summary_path.exists():
            raise RunReportError(
                f"No summary.json in {path}.\n\n"
                "Pick the folder a training run wrote — it sits next to the "
                "model bundle and is named after it."
            )
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RunReportError(f"Could not read {summary_path.name}: {e}") from e
        if not isinstance(summary, dict):
            raise RunReportError(f"{summary_path.name} does not contain a training summary.")
        return cls(
            summary=summary,
            path=path,
            charts=sorted(p for p in path.glob("*.png")),
        )

    @classmethod
    def from_summary(cls, summary: Any) -> TrainingRun:
        """Wrap an in-memory summary, for a run whose report could not be written."""
        return cls(summary=dict(summary) if isinstance(summary, dict) else {})

    # -- derived ----------------------------------------------------------

    @property
    def is_cross_validated(self) -> bool:
        return bool(self.summary.get("fold_macro_f1")) or self.strategy == "cross_validated"

    @property
    def strategy(self) -> str:
        return str(self.summary.get("split_strategy") or "no_holdout")

    @property
    def headline(self) -> Headline:
        """The number to lead with, chosen by what was actually measured.

        Deliberately one method rather than a value and a separate "is it
        good" flag: the two must never be shown apart, and every caller that
        could pair them wrongly is a caller that can mislead.
        """
        label, caption, strength = _EVIDENCE.get(self.strategy, _EVIDENCE["no_holdout"])

        if self.is_cross_validated:
            folds = _floats(self.summary.get("fold_macro_f1"))
            label, caption, strength = _EVIDENCE["cross_validated"]
            return Headline(
                value=_float(self.summary.get("mean_macro_f1")),
                label=label,
                caption=caption,
                strength=strength,
                spread=_stdev(folds),
            )

        value = self.summary.get("test_accuracy")
        if not self.summary.get("test_size") or value is None:
            # No test set at all: the only number available is the fit itself.
            label, caption, strength = _EVIDENCE["no_holdout"]
            value = self.summary.get("train_accuracy")
        return Headline(value=_float(value), label=label, caption=caption, strength=strength)

    @property
    def per_class(self) -> list[ClassMetric]:
        """Per-behavior metrics, worst F1 first.

        Sorted by failure rather than alphabetically: the reason to open this
        table is to find what the model cannot do, and on a twelve-behavior
        vocabulary that row is otherwise somewhere in the middle.
        """
        raw = self.summary.get("per_class_metrics")
        if not isinstance(raw, dict):
            return []
        out = [
            ClassMetric(
                name=str(name),
                precision=_float(m.get("precision")),
                recall=_float(m.get("recall")),
                f1=_float(m.get("f1")),
                support=_int(m.get("support")),
            )
            for name, m in raw.items()
            if isinstance(m, dict)
        ]
        return sorted(out, key=lambda c: (c.f1 is None, c.f1 if c.f1 is not None else 0.0))

    @property
    def embedding(self):
        """The run's 3D embedding points, or None.

        Read from the report folder rather than the model bundle: the bundle
        carries the fitted reducer and the classifier with it, and opening
        hundreds of megabytes to draw a scatter would make browsing runs cost
        as much as loading a model.

        Returns a light stand-in with ``coords`` / ``labels`` / ``method`` —
        enough to draw, without reconstructing a fitted artifact that nothing
        here would use.
        """
        if self.path is None:
            return None
        import numpy as np

        from glider.analysis.behavior.report import EMBEDDING_FILE

        path = self.path / EMBEDDING_FILE
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return _EmbeddingPoints(
                    coords=np.asarray(data["coords"], dtype=float),
                    labels=np.asarray(data["labels"]).astype(str),
                    method=str(data["method"]) if "method" in data else "",
                )
        except Exception:  # noqa: BLE001 - a truncated file must not break browsing
            return None

    @property
    def macro_f1(self) -> float | None:
        """Unweighted mean of the per-class F1s. Pooled over folds when there
        were folds.

        Deliberately not ``mean_macro_f1`` on the cross-validated path. That
        is the mean of each fold's own macro F1, which the headline already
        reports and labels as such; returning it here made both figures the
        same number. Worse, the two disagree exactly when a class is absent
        from some fold's test set -- it scores 0 there and drags the
        mean-of-folds down, while the per-class F1s, computed once over every
        fold's pooled predictions, are unaffected. The pooled mean is the
        figure that matches this property's name and the "rare classes count
        equally" caption it is displayed under.

        Falls back to ``mean_macro_f1`` only when there is no per-class table
        to pool, so a run that measured something never shows nothing.
        """
        scores = [c.f1 for c in self.per_class if c.f1 is not None]
        if scores:
            return sum(scores) / len(scores)
        if self.is_cross_validated:
            return _float(self.summary.get("mean_macro_f1"))
        return None

    @property
    def scored_rows(self) -> int | None:
        """How many rows the reported metrics were actually measured on.

        Deliberately not ``n_rows_kept``. That counts every row assembled for
        training, and with mirror augmentation on it includes the mirrored
        copies -- which exist only to train and are excluded from scoring, so
        quoting it beside the metrics overstates the evidence roughly twofold.
        The per-class supports sum to the rows that were genuinely scored.
        """
        supports = [c.support for c in self.per_class if c.support is not None]
        return sum(supports) if supports else None

    @property
    def weakest(self) -> ClassMetric | None:
        """The behavior the model is worst at, or None when nothing was measured."""
        ranked = self.per_class
        return ranked[0] if ranked and ranked[0].f1 is not None else None

    @property
    def confusion(self) -> tuple[list[str], list[list[float]]]:
        """``(labels, matrix)`` as raw counts, or ``([], [])``."""
        block = self.summary.get("confusion_matrix")
        if not isinstance(block, dict):
            return [], []
        labels = [str(x) for x in (block.get("labels") or [])]
        matrix = block.get("matrix") or []
        if not labels or not isinstance(matrix, list) or len(matrix) != len(labels):
            return [], []
        return labels, [[_float(v) or 0.0 for v in row] for row in matrix]

    def confusion_rows_normalized(self) -> tuple[list[str], list[list[float]]]:
        """The confusion matrix as row fractions.

        Each row sums to 1 and reads "of the frames that really were X, this
        is where they went". Raw counts cannot be compared between rows when
        one behavior has fifty times the support of another, which on real
        annotation is the normal case rather than the exception.
        """
        labels, matrix = self.confusion
        normalized = []
        for row in matrix:
            total = sum(row)
            normalized.append([v / total for v in row] if total else [0.0 for _ in row])
        return labels, normalized

    @property
    def top_features(self) -> list[tuple[str, float]]:
        """``[(feature, importance)]`` descending, as written by the pipeline."""
        raw = self.summary.get("top_features")
        if not isinstance(raw, list):
            return []
        out = []
        for entry in raw:
            if isinstance(entry, dict):
                value = _float(entry.get("importance"))
                if value is not None:
                    out.append((str(entry.get("feature")), value))
        return out

    @property
    def label_counts(self) -> list[tuple[str, int]]:
        """``[(label, rows)]`` for the rows actually trained on, largest first."""
        raw = self.summary.get("kept_label_counts")
        if not isinstance(raw, dict):
            return []
        counts = [(str(k), _int(v) or 0) for k, v in raw.items()]
        return sorted(counts, key=lambda t: -t[1])

    @property
    def imbalance_ratio(self) -> float | None:
        """How many times more common the commonest class is than the rarest.

        The number behind "why is my rare behavior never predicted": past
        roughly 20:1 a model can score well overall while never firing the
        scarce class at all.
        """
        counts = [n for _label, n in self.label_counts if n > 0]
        return max(counts) / min(counts) if len(counts) > 1 else None

    @property
    def warnings(self) -> list[str]:
        """Everything about this run that should temper reading the headline.

        Collected here rather than raised at the widget, so the same list can
        be asserted in a test without a display.
        """
        out: list[str] = []
        if not self.headline.is_trustworthy:
            out.append(
                "No held-out data: the score above is measured on the rows the "
                "model was fitted to, so it says nothing about a new animal. "
                "Add holdout sessions or a within-session test split, or "
                "cross-validate."
            )
        ratio = self.imbalance_ratio
        if ratio is not None and ratio >= 20:
            rarest = self.label_counts[-1]
            out.append(
                f"Classes are imbalanced {ratio:,.0f}:1 — the rarest, "
                f"'{rarest[0]}', has {rarest[1]:,} rows. A model can score well "
                "overall while never predicting it. Try class weight 'balanced'."
            )
        weakest = self.weakest
        if weakest is not None and weakest.f1 is not None and weakest.f1 < 0.5:
            out.append(
                f"'{weakest.name}' scores F1 {weakest.f1:.2f} — the model cannot "
                "reliably tell it apart. More labelled examples of it usually "
                "helps more than any setting."
            )
        if self.is_cross_validated:
            spread = self.headline.spread
            if spread is not None and spread >= 0.05:
                out.append(
                    f"Fold-to-fold spread is ±{spread:.3f}, which is large next to "
                    "most differences between settings. Treat a small improvement "
                    "over another run as noise unless it exceeds this."
                )
        return out


# ---------------------------------------------------------------------------
# coercion helpers -- a summary is a plain dict read off disk, so every field
# is treated as untrusted rather than assumed well-formed
# ---------------------------------------------------------------------------


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _floats(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    return [f for f in (_float(v) for v in value) if f is not None]


def _stdev(values: list[float]) -> float | None:
    """Population sd, or None when there is nothing to spread."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
