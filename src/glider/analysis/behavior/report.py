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


def write_training_report(result, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_summary_dict(result), indent=2, default=_json_default)
    )
    _write_csvs(result, out_dir)
    return out_dir
