"""Qt-free supervised behavior-analysis core (feature extraction, training,
classification). Ported from the yolo2pose project. Heavy ML deps
(umap-learn, hdbscan, lightgbm, torch) are lazy-imported inside the functions
that need them, so importing this package stays cheap.
"""

from glider.analysis.behavior.embedding import fit_embedding
from glider.analysis.behavior.features import FeatureSpec, compute_features
from glider.analysis.behavior.labels import (
    AMBIGUOUS,
    build_label_and_group_series,
    build_label_series,
)
from glider.analysis.behavior.model import BehaviorModel
from glider.analysis.behavior.pipeline import (
    HybridTrainResult,
    LgbmReg,
    TrainResult,
    cross_validate_sessions,
    train_hybrid_model,
    train_model,
)
from glider.analysis.behavior.windowing import apply_rolling

__all__ = [
    "AMBIGUOUS",
    "BehaviorModel",
    "FeatureSpec",
    "HybridTrainResult",
    "LgbmReg",
    "TrainResult",
    "apply_rolling",
    "build_label_and_group_series",
    "build_label_series",
    "compute_features",
    "cross_validate_sessions",
    "fit_embedding",
    "train_hybrid_model",
    "train_model",
]
