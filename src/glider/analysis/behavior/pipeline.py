"""End-to-end training orchestration.

Usage::

    result = train_model(
        sessions=[(Path("session_001_DLC.csv"), Path("session_001_annotations.csv"))],
        spec=FeatureSpec(body_axis=(0, 6)),
        window=30,
        fps=30.0,
    )
    result.model.save("behavior_model.pkl")
    print(result.summary)

What ``train_model`` does, in order:

1. For each session: load the pose CSV, compute per-frame features,
   apply ``window``-frame pandas rolling stats, build the per-frame
   label series from the annotation zones.
2. Concatenate the per-session windowed DataFrames + label series.
3. Drop rows that are unannotated, ambiguous (multi-behavior overlap),
   or contain NaN in any feature column (typically the first
   ``window-1`` rows of each session, where the rolling stat is
   undefined).
4. Optional train/test split for reporting.
5. Fit a ``RandomForestClassifier`` on the training portion.
6. Roll the result into a :class:`BehaviorModel` along with a
   diagnostic ``summary`` dict.

The function does **not** save to disk — the caller decides where to
write the bundle. This keeps the function easy to call from tests and
the live-inference CLI without side effects.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from glider.analysis.behavior.hybrid import HybridModel

from glider.analysis.behavior.annotations import AnnotationStore
from glider.analysis.behavior.benchmarks.metrics import macro_frame_f1
from glider.analysis.behavior.features import FeatureSpec, compute_features
from glider.analysis.behavior.labels import (
    AMBIGUOUS,
    build_label_and_group_series,
    split_label_counts,
)
from glider.analysis.behavior.model import BehaviorModel, capture_library_versions
from glider.analysis.behavior.trajectory import apply_trajectory_rolling
from glider.analysis.behavior.windowing import (
    DEFAULT_STATS,
    apply_rolling,
    apply_spectral_rolling,
)
from glider.vision.pose.dlc import from_dlc_csv

SessionPair = tuple[Path, Path]


@dataclass
class TrainResult:
    """Bundle returned by :func:`train_model` — model + diagnostics."""

    model: BehaviorModel
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridTrainResult:
    """Bundle returned by :func:`train_hybrid_model`.

    ``model`` is the shipped :class:`~glider.analysis.behavior.hybrid.HybridModel`
    (final base refit on all kept rows, blended at the tuned ``lam``).
    ``per_lambda_f1`` maps each grid λ to its validation macro-F1;
    ``base_val_f1`` is the base model's (λ=0) validation macro-F1.
    """

    model: HybridModel  # imported under TYPE_CHECKING to avoid an import cycle
    lam: float
    per_lambda_f1: dict[float, float]
    n_val: int
    base_val_f1: float


@dataclass(frozen=True)
class LgbmReg:
    """LightGBM regularization knobs. Ignored by the RandomForest backend.

    Defaults are mildly regularized relative to stock LightGBM
    (``min_child_samples`` 20→50, row/column subsampling 1.0→0.8,
    ``reg_lambda`` 0→1) to pull the model off the 100%-train-accuracy
    memorization regime and improve cross-session generalization. Loosen
    or tighten per-knob to trade train fit against generalization.
    """

    num_leaves: int = 31
    min_child_samples: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    reg_lambda: float = 1.0
    # Capacity limiters. Defaults are LightGBM stock so existing models
    # reproduce; lower learning_rate / cap max_depth / raise min_split_gain
    # to pull the model off the 100%-train-accuracy memorization regime.
    learning_rate: float = 0.1
    max_depth: int = -1
    min_split_gain: float = 0.0


def train_model(
    sessions: list[SessionPair],
    *,
    spec: FeatureSpec | None = None,
    window: int = 30,
    stats: tuple[str, ...] = DEFAULT_STATS,
    fps: float = 30.0,
    n_estimators: int = 200,
    random_state: int = 42,
    test_split: float = 0.0,
    class_weight: str | None = None,
    include_background: bool = False,
    background_class_name: str = "background",
    background_subsample_ratio: float = 5.0,
    holdout_sessions: list[SessionPair] | None = None,
    classifier_type: str = "lightgbm",
    mirror_augment: bool = False,
    merge_map: dict[str, str] | None = None,
    exclude: set[str] | None = None,
    lgbm_reg: LgbmReg | None = None,
    embedding: str = "none",
    freq_features: bool = False,
    traj_features: bool = False,
    motion_features: bool = False,
) -> TrainResult:
    """Fit a behavior classifier from one or more (pose, annotations) pairs.

    Parameters
    ----------
    sessions
        ``[(pose_csv_path, annotations_csv_path), ...]``. Each pose CSV
        must be DLC-formatted; each annotations CSV must be in the
        format the annotator writes.
    spec
        Feature extractor knobs. Defaults to a sensible
        :class:`FeatureSpec` (body axis = first ↔ last keypoint, all
        pairwise distances normalized).
    window
        Rolling-stat window length in frames. 30 frames @ 30 fps = 1 s.
    stats
        Rolling aggregations. Defaults to mean / std / max per the
        project plan.
    fps
        Recording frame rate, used to load the pose CSVs and recorded
        in the model bundle.
    n_estimators
        Forwarded to ``RandomForestClassifier``.
    random_state
        Forwarded to ``RandomForestClassifier`` AND used for the
        train/test split.
    test_split
        Fraction of samples to hold out for the test-accuracy report.
        ``0.0`` (default) = train on everything, no test split.
    class_weight
        Forwarded to ``RandomForestClassifier``. ``None`` (default) or
        ``"balanced"``.
    include_background
        When True, frames that aren't covered by any annotation zone
        are relabelled to ``background_class_name`` and used as
        training data instead of being dropped. This lets the live
        model emit ``background`` for "nothing of interest" instead of
        force-classifying every frame into one of the named behaviors.
        Pair with ``class_weight="balanced"`` if you don't want
        background (typically the most-frequent class) to swamp the
        annotated behaviors.
    background_class_name
        Label used when ``include_background`` is True. Defaults to
        ``"background"``.
    background_subsample_ratio
        When ``include_background`` is True, cap the background class
        at this many times the size of the largest non-background
        class. The default of 5.0 means background can be at most 5×
        the most-common annotated behavior — enough to teach the model
        what "no behavior" looks like without burying the named
        classes. Set to ``0`` to disable subsampling and keep every
        background frame (legacy behaviour; usually a bad idea on
        long videos where background outweighs labels 100:1).
    holdout_sessions
        Optional list of ``(pose_csv, annotations_csv)`` pairs to use
        as a held-out **test** set — disjoint from ``sessions``. This
        is the gold-standard generalization metric: train on session A,
        test on session B (a different mouse / day). When supplied,
        ``test_split`` is ignored.
    classifier_type
        ``"lightgbm"`` (default, ``lightgbm.LGBMClassifier``) or
        ``"rf"`` (``RandomForestClassifier``). LightGBM is
        usually a few percent better on tabular features and trains
        much faster; it is the default backend and falls back to RF
        only if not installed.
    mirror_augment
        When True, generate horizontally-mirrored copies of each
        training session (left/right keypoint pairs swapped, x-coords
        negated). Doubles the effective training set and teaches the
        model left/right invariance for free. Held-out test sessions
        are NOT mirrored — that would inflate the test number.
    lgbm_reg
        :class:`LgbmReg` regularization knobs for the LightGBM backend
        (num_leaves, min_child_samples, feature/row subsampling, L2).
        Ignored when ``classifier_type="rf"``. ``None`` uses the
        regularized defaults, which trade a little train accuracy for
        better cross-session generalization.
    exclude
        Behavior names to drop entirely from training (and any holdout).
        Their frames become unlabeled unless covered by another zone.
        Applied before ``merge_map``, so excluding a behavior overrides
        merging it. Useful for dropping a class the pose can't resolve
        (e.g. ``immobile``) so it doesn't contaminate the others.
    embedding
        Fit a 3D feature-space embedding and store it in the model bundle
        for the live "galaxy" view. ``"none"`` (default) skips it;
        ``"umap"`` (falls back to PCA if umap-learn is missing) or
        ``"pca"`` fits on the kept training rows.

    Returns
    -------
    TrainResult
        ``.model`` is the fitted :class:`BehaviorModel`; ``.summary``
        contains per-class sample counts, train/test accuracy, and the
        top-N feature importance.
    """
    if not sessions:
        raise ValueError("train_model requires at least one (pose, annotations) pair")

    spec = spec or FeatureSpec()

    # ---- 1-2. Assemble features/labels, drop unusable rows, subsample bg ----
    assembled = _assemble_and_filter(
        sessions,
        spec=spec,
        window=window,
        stats=stats,
        fps=fps,
        mirror_augment=mirror_augment,
        merge_map=merge_map,
        exclude=exclude,
        freq_features=freq_features,
        traj_features=traj_features,
        motion_features=motion_features,
        include_background=include_background,
        background_class_name=background_class_name,
        background_subsample_ratio=background_subsample_ratio,
        random_state=random_state,
    )
    x_kept, y_kept, g_kept = assembled.x_kept, assembled.y_kept, assembled.g_kept
    n_total = assembled.n_total
    n_kept = assembled.n_kept
    per_session_counts = assembled.per_session_counts
    background_subsampled_to = assembled.background_subsampled_to

    # ---- 3. Train/test split strategy ----
    # Three modes, in priority order:
    #   1. Cross-session holdout (--holdout-session): train on `sessions`,
    #      test on `holdout_sessions`. The gold-standard generalization
    #      test — different mouse / day → realistic OOD performance.
    #   2. Zone-aware within-session split (test_split > 0). Splits zones
    #      so adjacent windows don't leak across train/test, but the
    #      test set is still from the same recording.
    #   3. No holdout (test_split = 0). Train on everything; no test
    #      metrics.
    split_strategy: str
    if holdout_sessions:
        # Mirror augmentation is NOT applied to held-out sessions; the
        # test set should reflect real un-augmented data.
        x_test_all, y_test_all, _g_test, _per_test_counts = _assemble_sessions(
            holdout_sessions,
            spec=spec,
            window=window,
            stats=stats,
            fps=fps,
            mirror_augment=False,
            merge_map=merge_map,
            exclude=exclude,
            freq_features=freq_features,
            traj_features=traj_features,
            motion_features=motion_features,
        )
        # Apply the same drop logic as training.
        test_keep = (y_test_all != "") & (y_test_all != AMBIGUOUS) & ~x_test_all.isna().any(axis=1)
        # Note: if include_background is on, we don't add background to
        # the test set — the test set's labels should only be the
        # behaviors we want to evaluate. The model will still predict
        # background, but those predictions will count against recall
        # of the named class if the truth is named.
        x_train = x_kept
        y_train = y_kept
        x_test = x_test_all.loc[test_keep].reset_index(drop=True)
        y_test = y_test_all.loc[test_keep].reset_index(drop=True)
        split_strategy = "cross_session"
    elif test_split and 0.0 < test_split < 1.0:
        from sklearn.model_selection import GroupShuffleSplit

        # Background rows (group -1 after subsampling) get unique groups
        # so they can be split independently — otherwise the single
        # group -1 would force ALL of them to one side.
        eff_groups = g_kept.to_numpy().copy()
        bg_mask = eff_groups < 0
        if bg_mask.any():
            n_bg = int(bg_mask.sum())
            next_id = int(eff_groups.max()) + 1 if (eff_groups >= 0).any() else 0
            eff_groups[bg_mask] = np.arange(next_id, next_id + n_bg)

        gss = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
        train_idx, test_idx = next(gss.split(x_kept, y_kept, groups=eff_groups))
        x_train = x_kept.iloc[train_idx].reset_index(drop=True)
        x_test = x_kept.iloc[test_idx].reset_index(drop=True)
        y_train = y_kept.iloc[train_idx].reset_index(drop=True)
        y_test = y_kept.iloc[test_idx].reset_index(drop=True)
        split_strategy = "group_shuffle"
    else:
        x_train, x_test = x_kept, None
        y_train, y_test = y_kept, None
        split_strategy = "no_holdout"

    # ---- 4. Fit ----
    clf = _build_classifier(
        classifier_type=classifier_type,
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        lgbm_reg=lgbm_reg,
    )
    clf.fit(x_train, y_train)
    classifier_label = _classifier_label_for(clf)

    # ---- 5. Diagnostics ----
    train_acc = float(clf.score(x_train, y_train))
    test_acc = float(clf.score(x_test, y_test)) if x_test is not None else None

    # Per-class precision/recall/F1 + confusion matrix on the test set.
    per_class_metrics: dict[str, dict[str, float]] = {}
    confusion: dict[str, list] = {}
    if x_test is not None and len(x_test) > 0:
        from sklearn.metrics import (
            confusion_matrix as _confusion,
        )
        from sklearn.metrics import (
            precision_recall_fscore_support,
        )

        y_pred = clf.predict(x_test)
        labels_seen = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=labels_seen, zero_division=0
        )
        for lab, p, r, f, s in zip(labels_seen, precision, recall, f1, support, strict=False):
            per_class_metrics[str(lab)] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
        cm = _confusion(y_test, y_pred, labels=labels_seen)
        confusion = {
            "labels": [str(lab) for lab in labels_seen],
            "matrix": cm.tolist(),
        }

    # Feature importance (works for both RF and LightGBM).
    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_, dtype=float)
        order = np.argsort(-importances)
        top_features = [
            {
                "feature": x_kept.columns[int(i)],
                "importance": float(importances[int(i)]),
            }
            for i in order[:20]
        ]
    else:
        top_features = []

    summary = {
        "n_sessions": len(sessions),
        "n_rows_total": n_total,
        "n_rows_kept": n_kept,
        "n_rows_dropped": n_total - n_kept,
        "per_session_label_counts": per_session_counts,
        "kept_label_counts": split_label_counts(y_kept),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)) if x_test is not None else 0,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "classes": [str(c) for c in clf.classes_],
        "n_features": int(x_kept.shape[1]),
        "top_features": top_features,
        "window": int(window),
        "stats": list(stats),
        "fps": float(fps),
        "background_subsampled_to": background_subsampled_to,
        "class_weight": class_weight,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion,
        "split_strategy": split_strategy,
        "n_holdout_sessions": len(holdout_sessions) if holdout_sessions else 0,
        "classifier_type": classifier_label,
        "mirror_augment": bool(mirror_augment),
        "lgbm_reg": (
            dataclasses.asdict(lgbm_reg or LgbmReg()) if classifier_label == "lightgbm" else None
        ),
    }

    # Optional 3D embedding of the training feature space for the live
    # galaxy view. Fit on the rows the classifier actually trained on.
    embedding_artifact = None
    if embedding and embedding.lower() != "none":
        from glider.analysis.behavior import embedding as _embedding_mod

        # The embedding is a visualization add-on — never let a fit failure
        # (e.g. UMAP on a tiny/degenerate training set) abort training and
        # lose the model.
        try:
            embedding_artifact = _embedding_mod.fit_embedding(
                x_train, y_train, method=embedding, random_state=random_state
            )
        except Exception as e:  # noqa: BLE001
            import warnings

            warnings.warn(
                f"embedding fit failed ({e}); model saved without a 3D " f"embedding view",
                stacklevel=2,
            )

    model = BehaviorModel(
        classifier=clf,
        feature_names=list(x_kept.columns),
        spec=spec,
        window=window,
        stats=tuple(stats),
        fps=fps,
        classes=[str(c) for c in clf.classes_],
        training_summary=summary,
        library_versions=capture_library_versions(),
        embedding=embedding_artifact,
    )
    return TrainResult(model=model, summary=summary)


def train_hybrid_model(
    sessions: list[SessionPair],
    *,
    tag_map: dict[str, frozenset[str]],
    spec: FeatureSpec | None = None,
    window: int = 30,
    stats: tuple[str, ...] = DEFAULT_STATS,
    fps: float = 30.0,
    n_estimators: int = 200,
    random_state: int = 42,
    class_weight: str | None = None,
    lgbm_reg: LgbmReg | None = None,
    val_frac: float = 0.25,
    lam_grid: tuple[float, ...] | None = None,
    mirror_augment: bool = False,
    merge_map: dict[str, str] | None = None,
    exclude: set[str] | None = None,
    freq_features: bool = False,
    traj_features: bool = False,
    motion_features: bool = False,
) -> HybridTrainResult:
    """Train a hybrid (LightGBM base + kinematic prior) model with λ tuning.

    The blend weight ``lam`` is selected on a held-out validation split that is
    carved out **before** the λ-selection base is fit, so the base never scores
    its own training rows on validation:

    1. Assemble + filter all sessions into pooled kept rows (shared with
       :func:`train_model`).
    2. Group-shuffle split those rows into ``train_core`` / ``val`` on the zone
       group ids (``val_frac`` held out).
    3. Fit a LightGBM base on ``train_core`` only.
    4. Build a :class:`~glider.analysis.behavior.prior.KinematicPrior` and
       calibrate it **once** on the pooled kept speed (unsupervised, leakage-free).
    5. For each λ in ``lam_grid`` (default ``0.0 … 1.0`` step ``0.1``), blend the
       ``train_core`` base with the prior and score ``val`` macro-F1. Pick the
       best λ; ties resolve to the smallest λ (so λ=0 wins on no improvement).
    6. Refit the shipped base on ALL kept rows and wrap it in the final
       :class:`~glider.analysis.behavior.hybrid.HybridModel` at the tuned λ.

    LightGBM is hard-required (``require=True``); the prior needs the graded
    freeze/dart kinematics that the RandomForest fallback would not change, but
    the hybrid design commits to the gradient-boosted base.
    """
    from glider.analysis.behavior.hybrid import HybridModel
    from glider.analysis.behavior.prior import KinematicPrior

    if not sessions:
        raise ValueError("train_hybrid_model requires at least one (pose, annotations) pair")
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    spec = spec or FeatureSpec()
    if lam_grid is None:
        lam_grid = tuple(round(0.1 * i, 1) for i in range(11))

    # ---- 1. Assemble + filter (shared with train_model) ----
    assembled = _assemble_and_filter(
        sessions,
        spec=spec,
        window=window,
        stats=stats,
        fps=fps,
        mirror_augment=mirror_augment,
        merge_map=merge_map,
        exclude=exclude,
        freq_features=freq_features,
        traj_features=traj_features,
        motion_features=motion_features,
        include_background=False,
        background_class_name="background",
        background_subsample_ratio=0.0,
        random_state=random_state,
    )
    x_kept, y_kept, g_kept = assembled.x_kept, assembled.y_kept, assembled.g_kept

    # ---- 2. Split BEFORE fitting the λ-selection base ----
    from sklearn.model_selection import GroupShuffleSplit

    # Split on the zone group ids so adjacent windows from one labeled zone
    # can't leak across train_core/val. Background isn't handled here
    # (include_background=False above), so every kept row has group id >= 0 —
    # no unique-id remap for background rows is needed (unlike train_model).
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    train_idx, val_idx = next(gss.split(x_kept, y_kept, groups=g_kept.to_numpy()))
    x_train_core = x_kept.iloc[train_idx].reset_index(drop=True)
    y_train_core = y_kept.iloc[train_idx].reset_index(drop=True)
    x_val = x_kept.iloc[val_idx].reset_index(drop=True)
    y_val = y_kept.iloc[val_idx].reset_index(drop=True)

    def _fit_base(x, y) -> BehaviorModel:
        clf = _build_classifier(
            classifier_type="lightgbm",
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            lgbm_reg=lgbm_reg,
            require=True,
        )
        clf.fit(x, y)
        return BehaviorModel(
            classifier=clf,
            feature_names=list(x_kept.columns),
            spec=spec,
            window=window,
            stats=tuple(stats),
            fps=fps,
            classes=[str(c) for c in clf.classes_],
            library_versions=capture_library_versions(),
        )

    # ---- 3. train_core-only base for λ selection ----
    base_core = _fit_base(x_train_core, y_train_core)

    # ---- 4. Calibrate the prior ONCE on the pooled kept speed ----
    prior = KinematicPrior(tag_map)
    prior.calibrate(x_kept)

    classes = [str(c) for c in base_core.classifier.classes_]

    # ---- 5. Score each λ on the held-out val split ----
    gt = y_val.tolist()
    per_lambda_f1: dict[float, float] = {}
    for lam in lam_grid:
        preds = HybridModel(base_core, prior, lam, tag_map).predict(x_val)
        per_lambda_f1[float(lam)] = macro_frame_f1(gt, preds.tolist(), classes)

    # Best λ, ties → smallest λ (so λ=0 wins on no improvement).
    best_lam = max(lam_grid, key=lambda lam: (per_lambda_f1[float(lam)], -float(lam)))
    # λ=0 is exactly the base, so reuse its already-computed val F1 when the
    # grid includes 0.0 (the default); only recompute if a caller omitted it.
    if 0.0 in per_lambda_f1:
        base_val_f1 = per_lambda_f1[0.0]
    else:
        base_val_f1 = macro_frame_f1(gt, base_core.predict(x_val).tolist(), classes)

    # ---- 6. Refit the shipped base on ALL kept rows; wrap at λ* ----
    shipped_base = _fit_base(x_kept, y_kept)
    final_model = HybridModel(shipped_base, prior, best_lam, tag_map)

    return HybridTrainResult(
        model=final_model,
        lam=float(best_lam),
        per_lambda_f1=per_lambda_f1,
        n_val=int(len(x_val)),
        base_val_f1=base_val_f1,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assemble_sessions(
    sessions: list[SessionPair],
    *,
    spec: FeatureSpec,
    window: int,
    stats: tuple[str, ...],
    fps: float,
    mirror_augment: bool,
    merge_map: dict[str, str] | None = None,
    exclude: set[str] | None = None,
    freq_features: bool = False,
    traj_features: bool = False,
    motion_features: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[dict[str, int]]]:
    """Load each (pose, annotations) pair, extract features + label series.

    With ``mirror_augment=True``, every session is processed twice: once
    on the original pose and once on a horizontally-flipped copy. The
    mirrored copies inherit the same labels and zone IDs. This doubles
    the training set and teaches the model left/right invariance for
    free — useful when behaviors look the same from either side but
    your annotations skew to one direction.
    """
    if motion_features and mirror_augment:
        raise ValueError(
            "motion features are not supported with mirror augmentation yet "
            "(the source video isn't mirrored)"
        )
    xs: list[pd.DataFrame] = []
    ys: list[pd.Series] = []
    groups_per_session: list[pd.Series] = []
    per_session_counts: list[dict[str, int]] = []
    group_offset = 0
    for pose_csv, ann_csv in sessions:
        try:
            pose = from_dlc_csv(Path(pose_csv), fps=fps)
        except UnicodeDecodeError as e:
            raise ValueError(
                f"{pose_csv} is not readable as text — it looks like a "
                f"binary file (e.g. a video). The training step expects "
                f"the DLC-format pose CSV produced by GLIDER pose inference."
            ) from e
        except Exception as e:
            raise ValueError(f"failed to read DLC pose CSV {pose_csv}: {e}") from e
        store = AnnotationStore.load_csv(Path(ann_csv))

        # Bundle the original + optionally the mirrored copy.
        pose_variants = [pose]
        if mirror_augment:
            pose_variants.append(_mirror_pose(pose))

        for variant in pose_variants:
            feats = compute_features(variant, spec=spec)
            if motion_features:
                feats = _append_motion(feats, pose_csv, variant, spec)
            windowed = apply_rolling(feats, window=window, stats=stats)
            if freq_features:
                spectral = apply_spectral_rolling(feats, window=window)
                windowed = pd.concat([windowed, spectral], axis=1)
            if traj_features:
                ba = spec.with_resolved_body_axis(variant.n_keypoints).body_axis
                traj = apply_trajectory_rolling(variant.xy, body_axis=ba, window=window)
                traj.index = windowed.index
                windowed = pd.concat([windowed, traj], axis=1)
            labels, group_ids = build_label_and_group_series(
                store, n_frames=variant.n_frames, merge_map=merge_map, exclude=exclude
            )
            if len(windowed) != len(labels):
                raise RuntimeError(
                    f"windowed length {len(windowed)} != labels length "
                    f"{len(labels)} for {pose_csv}"
                )
            bumped = group_ids.copy()
            positive = bumped >= 0
            bumped[positive] = bumped[positive] + group_offset
            if positive.any():
                group_offset = int(bumped[positive].max()) + 1
            xs.append(windowed)
            ys.append(labels)
            groups_per_session.append(bumped)
            per_session_counts.append(split_label_counts(labels))

    x_all = pd.concat(xs, axis=0, ignore_index=True)
    y_all = pd.concat(ys, axis=0, ignore_index=True)
    g_all = pd.concat(groups_per_session, axis=0, ignore_index=True)
    return x_all, y_all, g_all, per_session_counts


@dataclass
class _AssembledData:
    """Kept windowed features / labels / group-ids + assembly diagnostics.

    Output of :func:`_assemble_and_filter` — the assemble → drop-mask →
    background-subsample block shared by :func:`train_model` and
    :func:`train_hybrid_model`.
    """

    x_kept: pd.DataFrame
    y_kept: pd.Series
    g_kept: pd.Series
    n_total: int
    n_kept: int
    per_session_counts: list[dict[str, int]]
    background_subsampled_to: int | None


def _assemble_and_filter(
    sessions: list[SessionPair],
    *,
    spec: FeatureSpec,
    window: int,
    stats: tuple[str, ...],
    fps: float,
    mirror_augment: bool,
    merge_map: dict[str, str] | None,
    exclude: set[str] | None,
    freq_features: bool,
    traj_features: bool,
    motion_features: bool,
    include_background: bool,
    background_class_name: str,
    background_subsample_ratio: float,
    random_state: int,
) -> _AssembledData:
    """Assemble sessions, drop unusable rows, and subsample background.

    Factors out the block both :func:`train_model` and
    :func:`train_hybrid_model` need. Rows are dropped when unannotated,
    :data:`AMBIGUOUS`, or NaN in any feature column; when
    ``include_background`` is set, unannotated frames are first promoted to
    ``background_class_name`` and (optionally) subsampled to
    ``background_subsample_ratio`` × the largest behavior class.
    """
    # ---- 1. Per-session feature + label assembly ----
    x_all, y_all, g_all, per_session_counts = _assemble_sessions(
        sessions,
        spec=spec,
        window=window,
        stats=stats,
        fps=fps,
        mirror_augment=mirror_augment,
        merge_map=merge_map,
        exclude=exclude,
        freq_features=freq_features,
        traj_features=traj_features,
        motion_features=motion_features,
    )

    # ---- 2. Optionally promote unannotated frames to a background class ----
    # Done BEFORE the drop-mask so the background label is treated as
    # ordinary supervision; AMBIGUOUS rows are still dropped because
    # we can't pick a single right answer for them.
    if include_background:
        if not background_class_name:
            raise ValueError("background_class_name must be non-empty")
        y_all = y_all.where(y_all != "", background_class_name)
    n_total = len(x_all)
    keep_mask = (y_all != "") & (y_all != AMBIGUOUS) & ~x_all.isna().any(axis=1)
    n_kept = int(keep_mask.sum())
    if n_kept < 2:
        raise ValueError(
            f"only {n_kept} usable rows after filtering; "
            f"need at least 2. Check that your annotations CSV covers "
            f"the pose CSV's frame range."
        )

    x_kept = x_all.loc[keep_mask].reset_index(drop=True)
    y_kept = y_all.loc[keep_mask].reset_index(drop=True)
    g_kept = g_all.loc[keep_mask].reset_index(drop=True)

    # ---- 2b. Subsample the background class if it dominates ----
    # On long videos most frames are unannotated, so without this the
    # background class can be 100× the largest behavior class and the
    # forest just learns "always predict background".
    background_subsampled_to: int | None = None
    if (
        include_background
        and background_subsample_ratio > 0
        and background_class_name in set(y_kept)
    ):
        bg_mask = y_kept == background_class_name
        n_bg = int(bg_mask.sum())
        non_bg_counts = y_kept[~bg_mask].value_counts()
        if len(non_bg_counts):
            largest_non_bg = int(non_bg_counts.iloc[0])
            cap = max(1, int(round(background_subsample_ratio * largest_non_bg)))
            if n_bg > cap:
                rng = np.random.default_rng(random_state)
                bg_idx = np.flatnonzero(bg_mask.to_numpy())
                keep_bg = rng.choice(bg_idx, size=cap, replace=False)
                keep_indices = np.concatenate([np.flatnonzero(~bg_mask.to_numpy()), keep_bg])
                keep_indices.sort()
                x_kept = x_kept.iloc[keep_indices].reset_index(drop=True)
                y_kept = y_kept.iloc[keep_indices].reset_index(drop=True)
                g_kept = g_kept.iloc[keep_indices].reset_index(drop=True)
                background_subsampled_to = cap

    return _AssembledData(
        x_kept=x_kept,
        y_kept=y_kept,
        g_kept=g_kept,
        n_total=n_total,
        n_kept=n_kept,
        per_session_counts=per_session_counts,
        background_subsampled_to=background_subsampled_to,
    )


def _append_motion(feats: pd.DataFrame, pose_csv: Path, pose, spec: FeatureSpec):
    """Concat per-frame egocentric motion-energy features onto ``feats``.

    Loaded (or computed + cached) from the session's source video. Joined
    BEFORE the rolling step so the motion columns pick up the same
    mean/std/max windowing as every other per-frame feature. Imported lazily
    so a plain (no ``--motion-features``) train never needs OpenCV.
    """
    from glider.analysis.behavior.motion import load_or_compute_motion, video_path_for

    ba = spec.with_resolved_body_axis(pose.n_keypoints).body_axis
    motion = load_or_compute_motion(pose_csv, video_path_for(pose_csv), pose.xy, ba)
    motion.index = feats.index
    return pd.concat([feats, motion], axis=1)


def _mirror_pose(pose):
    """Return a copy of ``pose`` with x-coords mirrored + L/R keypoints swapped.

    Detects L/R pairs by name (``left_*`` / ``right_*``). For everything
    else we just leave the keypoint in place. The x-coord flip uses the
    frame's keypoint extent as the pivot — we don't know the source
    frame width, so we mirror around the per-frame median x. This
    preserves all distances + angles + speeds (they're translation-
    invariant after the swap) and is invariant under the camera
    projection in 2D top-down setups.
    """
    from glider.vision.pose.core import PoseData

    names = list(pose.keypoint_names)
    swap_map: dict[int, int] = {}
    for i, name in enumerate(names):
        if name.startswith("left_"):
            mate = "right_" + name[len("left_") :]
            if mate in names:
                j = names.index(mate)
                swap_map[i] = j
                swap_map[j] = i
    xy = pose.xy.copy()
    confidence = pose.confidence.copy()
    # Per-frame mirror around the median x of valid keypoints.
    median_x = np.nanmedian(xy[..., 0], axis=1, keepdims=True)
    xy[..., 0] = 2.0 * median_x - xy[..., 0]
    # Apply left/right keypoint swap.
    if swap_map:
        new_xy = xy.copy()
        new_conf = confidence.copy()
        for i, j in swap_map.items():
            new_xy[:, i] = xy[:, j]
            new_conf[:, i] = confidence[:, j]
        xy, confidence = new_xy, new_conf
    return PoseData(
        xy=xy,
        confidence=confidence,
        keypoint_names=list(pose.keypoint_names),
        fps=pose.fps,
        metadata=dict(pose.metadata) if hasattr(pose, "metadata") else None,
    )


def cross_validate_sessions(
    sessions: list[SessionPair],
    *,
    spec: FeatureSpec | None = None,
    window: int = 30,
    stats: tuple[str, ...] = DEFAULT_STATS,
    fps: float = 30.0,
    n_estimators: int = 200,
    random_state: int = 42,
    class_weight: str | None = None,
    classifier_type: str = "lightgbm",
    mirror_augment: bool = False,
    merge_map: dict[str, str] | None = None,
    exclude: set[str] | None = None,
    lgbm_reg: LgbmReg | None = None,
    n_folds: int = 5,
    include_background: bool = False,
    background_class_name: str = "background",
    background_ratio: float = 5.0,
    threshold_curve: bool = False,
    freq_features: bool = False,
    traj_features: bool = False,
    motion_features: bool = False,
) -> dict[str, Any]:
    """Session-grouped K-fold cross-validation — a robust cross-session
    generalization estimate.

    Splits *whole sessions* across folds (never within a session), so
    every fold's test rows come from recordings the model never saw —
    the same realism as ``holdout_sessions`` but averaged over ``n_folds``
    splits instead of trusting a single hand-picked holdout. When
    ``n_folds`` >= number of sessions this is leave-one-session-out.

    Mirror-augmented copies (when ``mirror_augment=True``) stay with their
    parent session's fold and are used only for training; scoring is
    always on un-mirrored rows, mirroring the real training pipeline.

    With ``include_background=True``, unannotated frames are promoted to a
    background class (same as :func:`train_model`), subsampled to
    ``background_ratio`` × the largest behavior class so they can't swamp
    training or the held-out folds. The held-out folds then include
    background, so the returned ``false_alarm_rate`` measures how often the
    model false-fires a real behavior on a "nothing happening" frame — the
    metric that decides whether a background-aware detector is deployable.
    The surviving background is a representative random sample, so that
    rate is an unbiased estimate even though background is subsampled.

    Returns a dict of per-fold and aggregate accuracy / macro-F1, pooled
    per-class metrics + confusion, and (with background) the false-alarm
    rate. Does not fit or return a final model — it only measures.

    ``classifier_type`` defaults to ``"lightgbm"`` (matching
    :func:`train_model`) so CV measures the same backend that gets
    deployed; pass ``"rf"`` to evaluate the RandomForest backend instead.
    """
    if not sessions:
        raise ValueError("cross_validate_sessions requires at least one session")
    if motion_features and mirror_augment:
        raise ValueError(
            "motion features are not supported with mirror augmentation yet "
            "(the source video isn't mirrored)"
        )
    spec = spec or FeatureSpec()
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    xs: list[pd.DataFrame] = []
    ys: list[pd.Series] = []
    sess_ids: list[np.ndarray] = []
    is_mirror: list[np.ndarray] = []
    frame_ids: list[np.ndarray] = []
    for sidx, (pose_csv, ann_csv) in enumerate(sessions):
        pose = from_dlc_csv(Path(pose_csv), fps=fps)
        store = AnnotationStore.load_csv(Path(ann_csv))
        variants = [(pose, False)]
        if mirror_augment:
            variants.append((_mirror_pose(pose), True))
        for variant, mirrored in variants:
            feats = compute_features(variant, spec=spec)
            if motion_features:
                feats = _append_motion(feats, pose_csv, variant, spec)
            windowed = apply_rolling(feats, window=window, stats=stats)
            if freq_features:
                spectral = apply_spectral_rolling(feats, window=window)
                windowed = pd.concat([windowed, spectral], axis=1)
            if traj_features:
                ba = spec.with_resolved_body_axis(variant.n_keypoints).body_axis
                traj = apply_trajectory_rolling(variant.xy, body_axis=ba, window=window)
                traj.index = windowed.index
                windowed = pd.concat([windowed, traj], axis=1)
            labels, _groups = build_label_and_group_series(
                store, n_frames=variant.n_frames, merge_map=merge_map, exclude=exclude
            )
            xs.append(windowed)
            ys.append(labels)
            sess_ids.append(np.full(len(windowed), sidx, dtype=int))
            is_mirror.append(np.full(len(windowed), mirrored, dtype=bool))
            # Per-row frame index within the session — needed to stitch
            # consecutive frames back into bouts for bout-level metrics.
            frame_ids.append(np.arange(len(windowed), dtype=int))

    x_all = pd.concat(xs, axis=0, ignore_index=True)
    y_all = pd.concat(ys, axis=0, ignore_index=True)
    sess_all = np.concatenate(sess_ids)
    mirror_all = np.concatenate(is_mirror)
    frame_all_rows = np.concatenate(frame_ids)

    # Promote unannotated frames to a background class so the held-out
    # folds can measure target false-positives on "nothing" frames.
    if include_background:
        if not background_class_name:
            raise ValueError("background_class_name must be non-empty")
        y_all = y_all.where(y_all != "", background_class_name)

    keep = ((y_all != AMBIGUOUS) & ~x_all.isna().any(axis=1)).to_numpy()
    if not include_background:
        keep = keep & (y_all != "").to_numpy()
    x = x_all.loc[keep].reset_index(drop=True)
    y = y_all.loc[keep].reset_index(drop=True)
    sess = sess_all[keep]
    mirror = mirror_all[keep]
    frame = frame_all_rows[keep]

    # Subsample background to background_ratio × the largest behavior class
    # (same policy as train_model). The survivors stay a representative
    # random sample, so the false-alarm rate measured on held-out folds is
    # unbiased even though absolute background counts are capped.
    if include_background and background_ratio > 0:
        bg_mask = (y == background_class_name).to_numpy()
        non_bg = y[~bg_mask].value_counts()
        if len(non_bg) and int(bg_mask.sum()) > 0:
            cap = max(1, int(round(background_ratio * int(non_bg.iloc[0]))))
            if int(bg_mask.sum()) > cap:
                rng = np.random.default_rng(random_state)
                keep_bg = rng.choice(np.flatnonzero(bg_mask), size=cap, replace=False)
                sel = np.concatenate([np.flatnonzero(~bg_mask), keep_bg])
                sel.sort()
                x = x.iloc[sel].reset_index(drop=True)
                y = y.iloc[sel].reset_index(drop=True)
                sess = sess[sel]
                mirror = mirror[sel]
                frame = frame[sel]

    n_unique = int(len(np.unique(sess)))
    if n_unique < 2:
        raise ValueError(
            f"need at least 2 sessions with usable labels for " f"cross-validation; got {n_unique}"
        )
    n_splits = min(n_folds, n_unique)

    gkf = GroupKFold(n_splits=n_splits)
    fold_acc: list[float] = []
    fold_f1: list[float] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_sess: list[np.ndarray] = []
    all_frame: list[np.ndarray] = []
    all_proba: list[tuple[np.ndarray, list]] = []  # (proba, fold class order)
    for train_idx, test_idx in gkf.split(x, y, groups=sess):
        # Score only on un-mirrored rows of the held-out session(s).
        eval_idx = test_idx[~mirror[test_idx]]
        if len(eval_idx) == 0:
            continue
        clf = _build_classifier(
            classifier_type=classifier_type,
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            lgbm_reg=lgbm_reg,
        )
        clf.fit(x.iloc[train_idx], y.iloc[train_idx])
        y_true = y.iloc[eval_idx].to_numpy()
        y_pred = clf.predict(x.iloc[eval_idx])
        fold_acc.append(float((y_pred == y_true).mean()))
        fold_f1.append(float(f1_score(y_true, y_pred, average="macro", zero_division=0)))
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_sess.append(sess[eval_idx])
        all_frame.append(frame[eval_idx])
        if threshold_curve:
            all_proba.append((clf.predict_proba(x.iloc[eval_idx]), [str(c) for c in clf.classes_]))

    # Aggregate per-class metrics + confusion over ALL folds. GroupKFold
    # test sets are disjoint and cover every kept row exactly once, so
    # pooling predictions is a clean full-dataset cross-session estimate.
    per_class_metrics: dict[str, dict[str, float]] = {}
    confusion: dict[str, Any] = {}
    false_alarm_rate: float | None = None
    if all_true:
        from sklearn.metrics import (
            confusion_matrix as _confusion,
        )
        from sklearn.metrics import (
            precision_recall_fscore_support,
        )

        y_true_all = np.concatenate(all_true)
        y_pred_all = np.concatenate(all_pred)
        # False-alarm rate: of true-background frames, the fraction
        # predicted as some (non-background) behavior — the deployable
        # detector's headline "fires on nothing" rate.
        if include_background:
            true_bg = y_true_all == background_class_name
            if int(true_bg.sum()) > 0:
                false_alarm_rate = float((y_pred_all[true_bg] != background_class_name).mean())
        labels_seen = sorted(set(y_true_all.tolist()) | set(y_pred_all.tolist()))
        precision, recall, f1s, support = precision_recall_fscore_support(
            y_true_all, y_pred_all, labels=labels_seen, zero_division=0
        )
        for lab, p, r, f, sup in zip(labels_seen, precision, recall, f1s, support, strict=False):
            per_class_metrics[str(lab)] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(sup),
            }
        cm = _confusion(y_true_all, y_pred_all, labels=labels_seen)
        confusion = {
            "labels": [str(lab) for lab in labels_seen],
            "matrix": cm.tolist(),
        }

    # Per-class probability-threshold curves (one-vs-rest). For each class,
    # sweep the firing threshold and report recall, precision, and the
    # fires-on-rest rate (FP against confirmed non-target labels). This is
    # the operating curve you pick a deployment threshold from — built only
    # from confirmed labels, so sparse/sampled annotation can't poison it.
    threshold_curves: dict[str, list[dict[str, float]]] = {}
    tuned: dict[str, Any] = {}
    if threshold_curve and all_proba:
        col = {c: i for i, c in enumerate([str(lab) for lab in labels_seen])}
        aligned = []
        for proba, classes in all_proba:
            m = np.zeros((proba.shape[0], len(labels_seen)))
            for j, c in enumerate(classes):
                m[:, col[c]] = proba[:, j]
            aligned.append(m)
        proba_all = np.concatenate(aligned, axis=0)
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for c in (str(lab) for lab in labels_seen):
            p = proba_all[:, col[c]]
            pos = y_true_all == c
            n_pos = int(pos.sum())
            n_neg = int((~pos).sum())
            rows: list[dict[str, float]] = []
            for t in thresholds:
                fire = p >= t
                tp = int((fire & pos).sum())
                fp = int((fire & ~pos).sum())
                rows.append(
                    {
                        "threshold": t,
                        "recall": tp / n_pos if n_pos else 0.0,
                        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                        "fires_on_rest": fp / n_neg if n_neg else 0.0,
                    }
                )
            threshold_curves[c] = rows
        tuned = _tuned_threshold_eval(
            proba_all, y_true_all, [str(lab) for lab in labels_seen], thresholds
        )

    # Bout-level recall: per-frame metrics understate how many behavior
    # BOUTS the model catches, because a partially-detected bout is still a
    # detected bout once predictions are smoothed at inference. Stitch the
    # held-out predictions back into bouts and report what fraction are hit.
    bout_metrics: dict[str, dict[str, float]] = {}
    if all_true:
        bout_metrics = _bout_recall(
            np.concatenate(all_sess),
            np.concatenate(all_frame),
            y_true_all,
            y_pred_all,
        )

    acc = np.array(fold_acc, dtype=float)
    f1 = np.array(fold_f1, dtype=float)
    return {
        "n_folds": len(fold_acc),
        "n_sessions": n_unique,
        "fold_accuracies": fold_acc,
        "fold_macro_f1": fold_f1,
        "mean_accuracy": float(acc.mean()) if len(acc) else None,
        "std_accuracy": float(acc.std()) if len(acc) else None,
        "mean_macro_f1": float(f1.mean()) if len(f1) else None,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion,
        "false_alarm_rate": false_alarm_rate,
        "background_class_name": background_class_name if include_background else None,
        "threshold_curves": threshold_curves,
        "tuned_thresholds": tuned,
        "bout_metrics": bout_metrics,
        "n_rows_kept": int(len(y)),
    }


def _bout_recall(
    sess: np.ndarray,
    frame: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Bout-level detection recall, precision, and F1 per behavior.

    A *bout* is a maximal run of consecutive frames (within one session)
    sharing the same label. Two views:

    * **Recall** — over *true* bouts (a stand-in for annotated zones): a
      true bout of behavior ``X`` counts as detected at criterion ``c`` if
      ``>= c`` of its frames were predicted ``X``.
    * **Precision** — over *predicted* bouts: a predicted ``X`` bout counts
      as correct at criterion ``c`` if ``>= c`` of its frames are truly
      ``X``. This is what recall alone can't see — bouts the model
      hallucinates.
    * **F1** — harmonic mean of the two at each criterion.

    Criteria: ``any`` (≥1 frame), ``25`` (≥25%), ``50`` (majority). Uses
    argmax predictions, so it's conservative — a tuned firing threshold
    would shift these.
    """
    order = np.lexsort((frame, sess))
    s, f, t, p = sess[order], frame[order], y_true[order], y_pred[order]
    n = len(s)

    def _bouts(label: np.ndarray, other: np.ndarray) -> dict[str, list[int]]:
        """Per-behavior [count, hit_any, hit_25, hit_50], where a bout is a
        run of constant ``label`` and a 'hit' is the fraction of its frames
        whose ``other`` array equals that behavior."""
        agg: dict[str, list[int]] = {}
        i = 0
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i] and f[j] == f[j - 1] + 1 and label[j] == label[i]:
                j += 1
            beh = str(label[i])
            frac = float((other[i:j] == label[i]).mean())
            a = agg.setdefault(beh, [0, 0, 0, 0])
            a[0] += 1
            if frac > 0:
                a[1] += 1
            if frac >= 0.25:
                a[2] += 1
            if frac >= 0.5:
                a[3] += 1
            i = j
        return agg

    true_bouts = _bouts(t, p)  # recall: true runs hit by predictions
    pred_bouts = _bouts(p, t)  # precision: predicted runs that are correct

    def _f1(prec: float, rec: float) -> float:
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    out: dict[str, dict[str, float]] = {}
    for beh, a in true_bouts.items():
        nt = a[0]
        rec = {"any": a[1] / nt, "25": a[2] / nt, "50": a[3] / nt}
        pb = pred_bouts.get(beh, [0, 0, 0, 0])
        npb = pb[0]
        prec = {k: (pb[idx] / npb if npb else 0.0) for k, idx in (("any", 1), ("25", 2), ("50", 3))}
        out[beh] = {
            "n_bouts": nt,
            "n_pred_bouts": npb,
            "recall_any": rec["any"],
            "recall_25": rec["25"],
            "recall_50": rec["50"],
            "precision_any": prec["any"],
            "precision_25": prec["25"],
            "precision_50": prec["50"],
            "f1_any": _f1(prec["any"], rec["any"]),
            "f1_25": _f1(prec["25"], rec["25"]),
            "f1_50": _f1(prec["50"], rec["50"]),
        }
    return out


def _tuned_threshold_eval(
    proba_all: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
    grid: list[float],
) -> dict[str, Any]:
    """Pick the F1-maximizing one-vs-rest threshold per class, then score
    the combined decision rule against the argmax baseline.

    Combined rule: a frame fires the highest-probability class that clears
    its OWN threshold; if none clear theirs, it abstains ("unknown"). This
    lets a clean minority class (e.g. dig) fire low while a noisy class
    sits high — something one global threshold + argmax can't do.

    Returns the per-class thresholds, the tuned per-class precision/recall/
    F1, the tuned vs argmax macro-F1, and the abstain rate. NOTE: the
    thresholds are chosen on the same pooled data they're scored on, so the
    tuned macro-F1 is mildly optimistic — read it as the rebalancing
    ceiling, not a held-out number.
    """
    y_true = np.asarray(y_true)
    thr: dict[str, float] = {}
    for j, c in enumerate(labels):
        p = proba_all[:, j]
        pos = y_true == c
        n_pos = int(pos.sum())
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            fire = p >= t
            tp = int((fire & pos).sum())
            fp = int((fire & ~pos).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / n_pos if n_pos else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = float(best_t)

    lab_arr = np.array(labels)
    thr_vec = np.array([thr[c] for c in labels])
    above = proba_all >= thr_vec
    masked = np.where(above, proba_all, -np.inf)
    pred = np.where(above.any(axis=1), lab_arr[np.argmax(masked, axis=1)], "__unknown__")
    argmax_pred = lab_arr[np.argmax(proba_all, axis=1)]

    def _f1(pred_arr: np.ndarray, c: str) -> tuple[float, float, float]:
        tp = int(((pred_arr == c) & (y_true == c)).sum())
        fp = int(((pred_arr == c) & (y_true != c)).sum())
        fn = int(((pred_arr != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    per_class: dict[str, dict[str, float]] = {}
    tuned_f1s, argmax_f1s = [], []
    for c in labels:
        prec, rec, f1 = _f1(pred, c)
        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "threshold": thr[c],
        }
        tuned_f1s.append(f1)
        argmax_f1s.append(_f1(argmax_pred, c)[2])
    return {
        "thresholds": thr,
        "per_class": per_class,
        "macro_f1": float(np.mean(tuned_f1s)) if tuned_f1s else 0.0,
        "argmax_macro_f1": float(np.mean(argmax_f1s)) if argmax_f1s else 0.0,
        "abstain_rate": float((pred == "__unknown__").mean()),
    }


def _build_classifier(
    *,
    classifier_type: str,
    n_estimators: int,
    random_state: int,
    class_weight: str | None,
    lgbm_reg: LgbmReg | None = None,
    require: bool = False,
):
    """Construct an RF or LightGBM classifier. Falls back to RF if
    LightGBM is requested but not installed, unless ``require`` is True
    (used by the hybrid model, which hard-requires LightGBM)."""
    classifier_type = (classifier_type or "rf").lower()
    if classifier_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            reg = lgbm_reg or LgbmReg()
            return LGBMClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                class_weight=class_weight,
                num_leaves=reg.num_leaves,
                min_child_samples=reg.min_child_samples,
                colsample_bytree=reg.feature_fraction,
                subsample=reg.bagging_fraction,
                subsample_freq=1 if reg.bagging_fraction < 1.0 else 0,
                reg_lambda=reg.reg_lambda,
                learning_rate=reg.learning_rate,
                max_depth=reg.max_depth,
                min_split_gain=reg.min_split_gain,
                verbosity=-1,
            )
        except ImportError as e:
            if require:
                raise RuntimeError(
                    "lightgbm is required for the hybrid model; pip install lightgbm"
                ) from e
            import warnings

            warnings.warn(
                "lightgbm not installed; falling back to RandomForestClassifier. "
                "pip install lightgbm to enable.",
                stacklevel=2,
            )
            classifier_type = "rf"
    if classifier_type != "rf":
        raise ValueError(
            f"unknown classifier_type {classifier_type!r}; expected 'rf' or 'lightgbm'"
        )
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )


def _classifier_label_for(clf) -> str:
    """Friendly name for the classifier (used in the training summary)."""
    name = type(clf).__name__
    if name == "RandomForestClassifier":
        return "rf"
    if name == "LGBMClassifier":
        return "lightgbm"
    return name
