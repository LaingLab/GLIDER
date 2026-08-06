"""Score an already-trained model against annotated sessions.

Cross-validation measures a *recipe* — it fits N fold models, scores them, and
discards them. Training with held-out sessions measures a model it just fitted.
Neither can answer "how does this bundle on disk do on these annotations",
which is the question a validation set exists to ask.

The metric assembly is separated from the loading and feature work so it can be
tested on hand-built label sequences, where the expected numbers are arithmetic
rather than whatever the classifier happened to do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from glider.analysis.behavior.annotations import AnnotationStore
from glider.analysis.behavior.classify.smoothing import centered_majority_vote
from glider.analysis.behavior.features import compute_features
from glider.analysis.behavior.labels import AMBIGUOUS, build_label_and_group_series
from glider.analysis.behavior.model import BehaviorModel
from glider.analysis.behavior.pipeline import (
    SessionPair,
    _clean_window_rows,
    bout_metrics,
)
from glider.analysis.behavior.trajectory import apply_trajectory_rolling
from glider.analysis.behavior.windowing import apply_rolling, apply_spectral_rolling
from glider.vision.pose.dlc import from_dlc_csv

#: A class with fewer scored frames than this is reported but kept out of the
#: macro average. Cross-validation showed why: a behaviour present in 4 of 34
#: sessions scored 0.0 in the folds that held none of it and pulled the mean
#: down by 0.06, measuring fold composition rather than the model.
DEFAULT_SUPPORT_FLOOR = 100


def summarise_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sess: np.ndarray,
    frame: np.ndarray,
    *,
    support_floor: int = DEFAULT_SUPPORT_FLOOR,
) -> dict[str, Any]:
    """Per-class, macro, confusion and bout metrics for one set of predictions.

    Scores every label appearing in either array, so a behaviour the model
    invents is charged against its own precision and a behaviour it never
    emits shows up with zero recall. Classes below *support_floor* are marked
    ``thin`` and excluded from the macro average; the exclusion is reported in
    ``thin_classes`` rather than left for the reader to infer.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    if len(y_true) == 0:
        return {
            "per_class": {},
            "macro_f1": None,
            "macro_classes": [],
            "thin_classes": [],
            "accuracy": None,
            "confusion": {"labels": [], "matrix": []},
            "bouts": {},
            "n_scored": 0,
        }

    labels = sorted({str(v) for v in y_true} | {str(v) for v in y_pred})
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true.astype(str), y_pred.astype(str), labels=labels, zero_division=0
    )

    per_class: dict[str, dict[str, Any]] = {}
    for name, p, r, f, n in zip(labels, precision, recall, f1, support, strict=True):
        per_class[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(n),
            "thin": bool(int(n) < support_floor),
        }

    scored = [n for n in labels if not per_class[n]["thin"]]
    thin = [n for n in labels if per_class[n]["thin"]]
    macro = float(np.mean([per_class[n]["f1"] for n in scored])) if scored else None

    cm = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=labels)
    return {
        "per_class": per_class,
        "macro_f1": macro,
        "macro_classes": scored,
        "thin_classes": thin,
        "support_floor": int(support_floor),
        "accuracy": float((y_true.astype(str) == y_pred.astype(str)).mean()),
        "confusion": {"labels": labels, "matrix": cm.tolist()},
        "bouts": bout_metrics(sess, frame, y_true.astype(str), y_pred.astype(str)),
        "n_scored": int(len(y_true)),
    }


def _families_in(feature_names: list[str]) -> tuple[bool, bool]:
    """``(freq, traj)`` — which optional feature families a bundle was built with.

    The flags are not stored in the bundle, but the columns name themselves:
    spectral statistics end in ``__domfreq``/``__specflat`` and trajectory
    columns start with ``traj_``. Inferring them keeps a model scoreable
    without asking the operator to remember how it was trained.
    """
    freq = any(c.endswith(("__domfreq", "__specflat")) for c in feature_names)
    traj = any(c.startswith("traj_") for c in feature_names)
    return freq, traj


def _windowed_for(model: BehaviorModel, pose_csv: Path, fps: float) -> pd.DataFrame:
    """Rebuild the exact feature frame *model* was trained on, for one session."""
    freq, traj = _families_in(model.feature_names)
    pose = from_dlc_csv(Path(pose_csv), fps=fps)
    feats = compute_features(pose, spec=model.spec)
    windowed = apply_rolling(feats, window=model.window, stats=model.stats)
    if freq:
        windowed = pd.concat([windowed, apply_spectral_rolling(feats, window=model.window)], axis=1)
    if traj:
        axis = model.spec.with_resolved_body_axis(pose.n_keypoints).body_axis
        traj_frame = apply_trajectory_rolling(pose.xy, body_axis=axis, window=model.window)
        traj_frame.index = windowed.index
        windowed = pd.concat([windowed, traj_frame], axis=1)
    return windowed


def evaluate_model(
    model_path: str | Path,
    sessions: list[SessionPair],
    *,
    support_floor: int = DEFAULT_SUPPORT_FLOOR,
    fps: float | None = None,
    smooth_window: int = 1,
) -> dict[str, Any]:
    """Score the saved bundle at *model_path* against annotated *sessions*.

    Features are rebuilt from the bundle's own ``spec``/``window``/``stats``, so
    the columns match what it was fitted on rather than whatever the caller
    would have chosen. Mirror augmentation is deliberately not applied: it is a
    training-time augmentation and its rows were never scored during training
    either.

    Only annotated, fully-populated frames are scored. Rows the rolling window
    has not filled come back from :meth:`BehaviorModel.predict` as ``""``; they
    are counted in ``n_unscored`` rather than dropped silently or charged as
    errors, because "declined to answer" is not the same as "wrong".

    ``smooth_window`` applies the offline centred vote the apply path uses, so
    a score can describe the pipeline someone will actually run rather than
    raw per-frame output. It defaults to off: an unsmoothed number is the one
    comparable to cross-validation, and silently changing what an existing
    call measures would be worse than making callers ask.
    """
    if not sessions:
        raise ValueError("evaluation needs at least one session")

    model = BehaviorModel.load(model_path)
    rate = float(fps) if fps is not None else float(model.fps)

    frames: list[pd.DataFrame] = []
    truths: list[pd.Series] = []
    sess_ids: list[np.ndarray] = []
    frame_ids: list[np.ndarray] = []
    for index, (pose_csv, ann_csv) in enumerate(sessions):
        windowed = _windowed_for(model, Path(pose_csv), rate)
        store = AnnotationStore.load_csv(Path(ann_csv))
        labels, _groups = build_label_and_group_series(
            store, n_frames=len(windowed), merge_map=None, exclude=None
        )
        frames.append(windowed)
        truths.append(labels)
        sess_ids.append(np.full(len(windowed), index, dtype=int))
        frame_ids.append(np.arange(len(windowed), dtype=int))

    x_all = pd.concat(frames, axis=0, ignore_index=True)
    y_all = pd.concat(truths, axis=0, ignore_index=True)
    sess_all = np.concatenate(sess_ids)
    frame_all = np.concatenate(frame_ids)

    missing = [c for c in model.feature_names if c not in x_all.columns]
    if missing:
        families = sorted({c.split("__")[0].split("_")[0] for c in missing})
        raise ValueError(
            f"the model expects {len(missing)} feature columns these sessions "
            f"cannot produce (families: {', '.join(families)}). The bundle was "
            f"trained with a feature set this pose data does not support."
        )

    annotated = ((y_all != "") & (y_all != AMBIGUOUS)).to_numpy()
    n_annotated = int(annotated.sum())
    # Same rule cross-validation scores under: a frame whose causal window
    # still covers the previous behavior is not a fair test of the model. Kept
    # identical to _run_cv_folds so a CV score and an evaluation score mean the
    # same thing and can be compared.
    annotated = annotated & _clean_window_rows(
        y_all.to_numpy(), sess_all, frame_all, int(model.window)
    )
    if not annotated.any():
        raise ValueError(
            "these sessions have no annotated frames to score against "
            "(every frame is unlabelled or marked unclear)"
        )

    if smooth_window > 1:
        # Smoothing needs the frames either side of each scored one, and the
        # annotated frames are scattered islands in the recording -- voting
        # over just those would pool labels seconds apart. So predict the whole
        # session, smooth along it, and only then keep the annotated rows.
        # This is also what scoring a recording actually does, which is the
        # point: the evaluation should measure the pipeline, not a fragment.
        raw = model.predict(x_all)
        predictions = np.empty(len(raw), dtype=object)
        for sid in np.unique(sess_all):
            in_session = sess_all == sid
            order = np.argsort(frame_all[in_session], kind="stable")
            idx = np.nonzero(in_session)[0][order]
            predictions[idx] = centered_majority_vote(list(raw[idx]), smooth_window)
        predictions = predictions[annotated].astype(object)
    else:
        predictions = model.predict(x_all.loc[annotated])

    y = y_all.loc[annotated].to_numpy()
    sess = sess_all[annotated]
    frame = frame_all[annotated]

    # "" is the model declining on a window it could not fill.
    answered = predictions != ""
    result = summarise_predictions(
        y[answered],
        predictions[answered],
        sess[answered],
        frame[answered],
        support_floor=support_floor,
    )
    result.update(
        {
            "model_path": str(model_path),
            "model_classes": list(model.classes),
            "sessions": [str(p) for p, _a in sessions],
            "n_sessions": len(sessions),
            "n_annotated": n_annotated,
            "n_window_contaminated": n_annotated - int(annotated.sum()),
            "n_unscored": int((~answered).sum()),
            "window": int(model.window),
            # Recorded because a smoothed score and a raw one are not
            # comparable, and a bare macro_f1 in a report gives no way to tell
            # which one you are looking at.
            "smooth_window": int(smooth_window),
            "stats": list(model.stats),
            "fps": rate,
            "n_features": len(model.feature_names),
        }
    )
    return result
