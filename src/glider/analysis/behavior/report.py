"""Diagnostics report suite for a trained behavior model.

write_training_report(result, out_dir) writes loose files — summary.json, tidy
CSVs, and PNG charts — from a TrainResult or HybridTrainResult. Renders only the
artifacts the result carries (a default no-test-split model has empty
confusion/per-class metrics, so those are skipped, not errored). Charts use
matplotlib's headless object API (no pyplot / no display).
"""

from __future__ import annotations

import csv as _csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Colorblind-safe categorical palette (Okabe–Ito).
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]


def _json_default(o: Any):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _is_hybrid(result) -> bool:
    # Duck-typed to avoid importing the dataclass at runtime.
    return hasattr(result, "per_lambda_f1")


def _summary_dict(result) -> dict:
    """The dict written to summary.json for either result type."""
    if _is_hybrid(result):
        base = result.model.base
        return {
            "lam": float(result.lam),
            "per_lambda_f1": {str(k): float(v) for k, v in result.per_lambda_f1.items()},
            "n_val": int(result.n_val),
            "base_val_f1": float(result.base_val_f1),
            "classes": list(base.classes),
            "feature_names": list(base.feature_names),
            "classifier_type": type(base.classifier).__name__,
        }
    return dict(result.summary)


def _feature_importances(result) -> list[tuple[str, float]]:
    """[(feature, importance)] descending, ≤20, or [] if unavailable."""
    if _is_hybrid(result):
        clf = result.model.base.classifier
        imp = getattr(clf, "feature_importances_", None)
        if imp is None:
            return []
        names = result.model.base.feature_names
        return sorted(zip(names, (float(x) for x in imp), strict=True), key=lambda t: -t[1])[:20]
    top = result.summary.get("top_features") or []
    return [(str(d["feature"]), float(d["importance"])) for d in top]


def _write_csvs(result, out_dir: Path) -> None:
    summary = {} if _is_hybrid(result) else result.summary
    pcm = summary.get("per_class_metrics") or {}
    if pcm:
        with (out_dir / "per_class_metrics.csv").open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["class", "precision", "recall", "f1", "support"])
            for cls, m in pcm.items():
                w.writerow([cls, m["precision"], m["recall"], m["f1"], m["support"]])
    cm = summary.get("confusion_matrix") or {}
    if cm.get("matrix"):
        labels, matrix = cm["labels"], cm["matrix"]
        with (out_dir / "confusion_matrix.csv").open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["true\\pred", *labels])
            for lab, row in zip(labels, matrix, strict=True):
                w.writerow([lab, *row])
    fi = _feature_importances(result)
    if fi:
        with (out_dir / "feature_importances.csv").open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["feature", "importance"])
            w.writerows(fi)


def _new_fig(w=6.0, h=4.0) -> Figure:
    fig = Figure(figsize=(w, h), dpi=150)
    FigureCanvasAgg(fig)  # attach headless canvas
    return fig


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")


def _plot_confusion(summary, out_dir: Path) -> None:
    cm = summary.get("confusion_matrix") or {}
    if not cm.get("matrix"):
        return
    labels = cm["labels"]
    m = np.asarray(cm["matrix"], dtype=float)
    row_sums = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, row_sums, out=np.zeros_like(m), where=row_sums != 0)
    fig = _new_fig(5.5, 5.0)
    ax = fig.add_subplot(111)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    acc = summary.get("test_accuracy")
    ax.set_title("Confusion matrix" + (f"  (test acc {acc:.2f})" if acc is not None else ""))
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                int(m[i, j]),
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out_dir / "confusion_matrix.png")


def _plot_per_class(summary, out_dir: Path) -> None:
    pcm = summary.get("per_class_metrics") or {}
    if not pcm:
        return
    classes = list(pcm)
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(classes))
    width = 0.25
    fig = _new_fig(max(6.0, 1.2 * len(classes)), 4.0)
    ax = fig.add_subplot(111)
    for i, met in enumerate(metrics):
        ax.bar(
            x + (i - 1) * width,
            [pcm[c][met] for c in classes],
            width,
            label=met,
            color=_PALETTE[i],
        )
    macro_f1 = float(np.mean([pcm[c]["f1"] for c in classes]))
    ax.set_xticks(x, classes, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.legend()
    ax.set_title(f"Per-class metrics  (macro-F1 {macro_f1:.2f})")
    for xi, c in zip(x, classes, strict=True):
        ax.text(xi, -0.08, f"n={pcm[c]['support']}", ha="center", va="top", fontsize=7)
    _save(fig, out_dir / "per_class_metrics.png")


def _plot_importances(result, out_dir: Path) -> None:
    fi = _feature_importances(result)
    if not fi:
        return
    names = [n for n, _ in fi][::-1]
    vals = [v for _, v in fi][::-1]
    fig = _new_fig(6.5, max(3.0, 0.3 * len(names)))
    ax = fig.add_subplot(111)
    ax.barh(range(len(names)), vals, color=_PALETTE[0])
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("importance")
    ax.set_title(f"Top {len(names)} feature importances")
    _save(fig, out_dir / "feature_importances.png")


def _plot_class_balance(summary, out_dir: Path) -> None:
    counts = summary.get("kept_label_counts") or {}
    if not counts:
        return
    classes = list(counts)
    per_session = summary.get("per_session_label_counts") or []
    x = np.arange(len(classes))
    fig = _new_fig(max(6.0, 1.2 * len(classes)), 4.0)
    ax = fig.add_subplot(111)
    if per_session:
        # Grouped bars per count-group so per-group skew / leakage is visible.
        # Each entry is one per-session count group (doubled under mirror_augment,
        # so a "group" may be one session's mirror half rather than a whole session).
        n = len(per_session)
        width = 0.8 / n
        for si, sess in enumerate(per_session):
            ax.bar(
                x + (si - (n - 1) / 2) * width,
                [sess.get(c, 0) for c in classes],
                width,
                label=f"group {si}",
                color=_PALETTE[si % len(_PALETTE)],
            )
        ax.legend(fontsize=7)
        ax.set_title("Class balance (per session)")
    else:
        ax.bar(
            x,
            [counts[c] for c in classes],
            color=[_PALETTE[i % len(_PALETTE)] for i in range(len(classes))],
        )
        ax.set_title("Class balance")
    ax.set_xticks(x, classes, rotation=45, ha="right")
    ax.set_ylabel("kept windows")
    _save(fig, out_dir / "class_balance.png")


def _plot_lambda_sweep(result, out_dir: Path) -> None:
    per = result.per_lambda_f1
    lams = sorted(float(k) for k in per)
    vals = [per[k] for k in lams]
    fig = _new_fig(6.0, 4.0)
    ax = fig.add_subplot(111)
    ax.plot(lams, vals, "-o", color=_PALETTE[0])
    ax.axvline(
        float(result.lam),
        color=_PALETTE[3],
        linestyle="--",
        label=f"chosen λ={result.lam:g}",
    )
    ax.set_xlabel("λ (prior weight)")
    ax.set_ylabel("val macro-F1")
    ax.set_title("Hybrid λ-sweep")
    ax.legend()
    _save(fig, out_dir / "lambda_sweep.png")


def write_training_report(result, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_summary_dict(result), indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_csvs(result, out_dir)
    if _is_hybrid(result):
        _plot_importances(result, out_dir)
        _plot_lambda_sweep(result, out_dir)
    else:
        summary = result.summary
        _plot_confusion(summary, out_dir)
        _plot_per_class(summary, out_dir)
        _plot_importances(result, out_dir)
        _plot_class_balance(summary, out_dir)
    return out_dir
