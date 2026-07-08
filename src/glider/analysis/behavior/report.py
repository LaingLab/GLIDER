"""Diagnostics report suite for a trained behavior model.

write_training_report(result, out_dir) writes loose files — summary.json, tidy
CSVs, and PNG charts — from a TrainResult or HybridTrainResult. Renders only the
artifacts the result carries (a default no-test-split model has empty
confusion/per-class metrics, so those are skipped, not errored). Charts use
matplotlib's headless object API (no pyplot / no display).
"""

from __future__ import annotations

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


def write_training_report(result, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_summary_dict(result), indent=2, default=_json_default)
    )
    return out_dir
