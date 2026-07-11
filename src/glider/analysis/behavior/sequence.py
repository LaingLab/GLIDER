"""1D-CNN sequence model over window-egocentric keypoint sequences.

Instead of collapsing a window into order-invariant mean/std/max
statistics (which discard the *shape* of the motion), this model
consumes the raw keypoint sequence and learns temporal patterns
directly. Each window is normalized into an egocentric frame so the net
doesn't waste capacity relearning translation/rotation/scale invariance:

* translate so the **first frame's** body center is the origin,
* rotate so the first frame's body axis points along +x,
* scale by the first frame's body length.

Crucially this is a *single* transform for the whole window (not
per-frame), so the body-center trajectory across the window is preserved
— that's the locomote-vs-sniff signal summary stats threw away.

This module is intentionally standalone: it does NOT touch BehaviorModel,
the model bundle, or the live pipeline. It exists to validate the
approach under the same session-grouped CV before any integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def egocentric_window(xy: np.ndarray, body_axis: tuple[int, int]) -> np.ndarray:
    """Normalize one ``(w, K, 2)`` window into its egocentric frame.

    Returns the same shape. Translation/rotation/scale invariant w.r.t.
    the input pose, while preserving the within-window trajectory.
    """
    return egocentric_batch(np.asarray(xy, dtype=np.float64)[None], body_axis)[0]


def egocentric_batch(windows: np.ndarray, body_axis: tuple[int, int]) -> np.ndarray:
    """Vectorized :func:`egocentric_window` over ``(m, w, K, 2)`` windows."""
    win = np.asarray(windows, dtype=np.float64)
    if win.ndim != 4 or win.shape[-1] != 2:
        raise ValueError(f"windows must be (m, w, K, 2); got {win.shape}")
    head, tail = body_axis

    first = win[:, 0, :, :]  # (m, K, 2) — reference frame per window
    center = 0.5 * (first[:, head, :] + first[:, tail, :])  # (m, 2)
    axis_vec = first[:, tail, :] - first[:, head, :]  # (m, 2)
    length = np.linalg.norm(axis_vec, axis=1)  # (m,)
    theta = np.arctan2(axis_vec[:, 1], axis_vec[:, 0])  # (m,)

    # Rotate by -theta so the body axis lands on +x.
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    # Rotation matrices (m, 2, 2).
    rot_mat = np.empty((win.shape[0], 2, 2))
    rot_mat[:, 0, 0] = cos_t
    rot_mat[:, 0, 1] = -sin_t
    rot_mat[:, 1, 0] = sin_t
    rot_mat[:, 1, 1] = cos_t

    centered = win - center[:, None, None, :]  # (m, w, K, 2)
    # Apply per-window rotation: out[..., i] = sum_j rot_mat[i, j] * centered[..., j]
    rotated = np.einsum("mij,mwkj->mwki", rot_mat, centered)

    scale = np.where(length > 0, length, np.nan)[:, None, None, None]
    return rotated / scale


# ---------------------------------------------------------------------------
# 1D-CNN model + training (torch imported lazily so the module stays light)
# ---------------------------------------------------------------------------


def _make_cnn_class():
    import torch.nn as nn

    class BehaviorCNN(nn.Module):
        """Small temporal CNN over a ``(batch, channels=K*2, time=w)`` tensor.

        Three Conv1d→BN→ReLU blocks pick up local motion patterns at
        growing receptive fields, a global average pool collapses time,
        and a linear head maps to class logits. Few params (robust on
        ~20k windows), and the channels-first layout means Conv1d
        convolves over the time axis.
        """

        def __init__(self, n_channels: int, n_classes: int, hidden: int = 64, dropout: float = 0.3):
            super().__init__()
            # Input BN normalizes each channel from running stats — lets us
            # mix unit-scaled egocentric keypoints with raw-scale engineered
            # features ("both" input) without a separate, leakage-prone scaler.
            self.input_norm = nn.BatchNorm1d(n_channels)
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x):
            h = self.features(self.input_norm(x))  # (batch, hidden, time)
            h = h.mean(dim=2)  # global average pool over time
            return self.head(h)  # (batch, n_classes)

    return BehaviorCNN


def __getattr__(name):
    # Lazily materialize BehaviorCNN so importing this module stays torch-free
    # until the model is actually used.
    if name == "BehaviorCNN":
        cls = _make_cnn_class()
        globals()["BehaviorCNN"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class FittedCNN:
    """A trained CNN + the bits needed to predict on new windows.

    Wraps the torch module with a sklearn-ish ``.predict`` / ``.predict_proba``
    over numpy ``(n, channels, w)`` arrays so the CV loop stays simple.
    """

    def __init__(self, module, classes, device, arch=None):
        self.module = module
        self.classes = np.asarray(classes)
        self.device = device
        self.arch = arch or {}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        self.module.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                xb = torch.as_tensor(x[i : i + 4096], dtype=torch.float32, device=self.device)
                logits = self.module(xb)
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(self.classes)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes[proba.argmax(axis=1)]


def train_cnn(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int | None = None,
    classes: np.ndarray | None = None,
    max_epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weight: bool = True,
    val_fraction: float = 0.15,
    patience: int = 8,
    hidden: int = 64,
    dropout: float = 0.3,
    seed: int = 42,
    device: str | None = None,
) -> FittedCNN:
    """Train a :class:`BehaviorCNN` on ``x`` ``(n, channels, w)`` / ``y``.

    ``y`` may be string labels or ints; classes are inferred (or passed via
    ``classes``). With ``val_fraction > 0`` a stratified-ish holdout drives
    early stopping (best val-loss weights are restored). ``class_weight``
    weights the loss inversely to class frequency to help rare behaviors.
    """
    import torch
    import torch.nn as nn

    # Pin determinism so a given seed reproduces exactly — CUDA conv ops are
    # nondeterministic by default, which is what made earlier CV numbers
    # unreproducible run-to-run.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)

    if classes is None:
        classes = np.unique(y)
    classes = np.asarray(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)
    if n_classes is None:
        n_classes = len(classes)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    behavior_cnn = _make_cnn_class()
    arch = {
        "n_channels": int(x.shape[1]),
        "n_classes": int(n_classes),
        "hidden": int(hidden),
        "dropout": float(dropout),
    }
    model = behavior_cnn(
        n_channels=arch["n_channels"],
        n_classes=n_classes,
        hidden=hidden,
        dropout=dropout,
    ).to(dev)

    # Train/val split for early stopping.
    n = len(x)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    x_tr = torch.as_tensor(x[tr_idx], dtype=torch.float32)
    y_tr = torch.as_tensor(y_idx[tr_idx], dtype=torch.long)

    if class_weight:
        counts = np.bincount(y_idx[tr_idx], minlength=n_classes).astype(np.float64)
        w = np.zeros(n_classes, dtype=np.float64)
        nz = counts > 0
        w[nz] = counts.sum() / (n_classes * counts[nz])
        weight = torch.as_tensor(w, dtype=torch.float32, device=dev)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_val, since_improved = None, float("inf"), 0
    for _epoch in range(max_epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(tr_idx)))
        for s in range(0, len(order), batch_size):
            bi = order[s : s + batch_size]
            xb = x_tr[bi].to(dev)
            yb = y_tr[bi].to(dev)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        if n_val > 0:
            model.eval()
            with torch.no_grad():
                xv = torch.as_tensor(x[val_idx], dtype=torch.float32, device=dev)
                yv = torch.as_tensor(y_idx[val_idx], dtype=torch.long, device=dev)
                vloss = float(criterion(model(xv), yv))
            if vloss < best_val - 1e-4:
                best_val, since_improved = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since_improved += 1
                if since_improved >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return FittedCNN(model, classes, dev, arch)


class FittedEnsemble:
    """A bag of :class:`FittedCNN` whose probabilities are averaged.

    Averaging independently-seeded nets cuts the run-to-run variance that
    plagued the single net and usually lifts accuracy a point or two — the
    cheapest robustness win for a noisy first-cut model. Exposes the same
    ``predict`` / ``predict_proba`` API so it drops into the CV loop and
    :class:`SequenceModel` unchanged.
    """

    def __init__(self, members: list[FittedCNN], classes):
        self.members = members
        self.classes = np.asarray(classes)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(x) for m in self.members], axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes[self.predict_proba(x).argmax(axis=1)]


def train_cnn_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_models: int = 5,
    seed: int = 42,
    **kwargs,
) -> FittedEnsemble:
    """Train ``n_models`` CNNs with consecutive seeds and bag them."""
    classes = kwargs.pop("classes", None)
    if classes is None:
        classes = np.unique(y)
    members = [train_cnn(x, y, classes=classes, seed=seed + i, **kwargs) for i in range(n_models)]
    return FittedEnsemble(members, classes)


# ---------------------------------------------------------------------------
# Session → egocentric sequence assembly + session-grouped CV
# ---------------------------------------------------------------------------


def _interpolate_xy(xy: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps along time, per keypoint coordinate."""
    import pandas as pd

    f, k, _ = xy.shape
    flat = xy.reshape(f, k * 2)
    filled = pd.DataFrame(flat).interpolate(limit_direction="both").to_numpy()
    return filled.reshape(f, k, 2)


def assemble_sequences(
    sessions,
    *,
    spec,
    window: int = 30,
    fps: float = 30.0,
    merge_map: dict | None = None,
    exclude: set | None = None,
    mirror_augment: bool = False,
    include_features: bool = False,
):
    """Build egocentric keypoint sequences for every labeled, finite window.

    Returns ``(x, y, sess, mirror)`` where ``x`` is ``(n, C, window)``
    float32 (``C = K*2``, or ``K*2 + n_per_frame_features`` when
    ``include_features``), ``y`` the per-window labels, ``sess`` the
    session index (for session-grouped CV), and ``mirror`` a bool flag
    (mirror-augmented rows are train-only). Mirrors the keep logic of the
    LightGBM pipeline: labelled (not ambiguous) frames whose trailing
    window is fully finite.

    With ``include_features``, the per-frame engineered features
    (distances/angles/speeds from :func:`compute_features`) are appended
    as extra channels — the "both" input. Their raw scales are handled by
    the model's input BatchNorm.
    """
    from glider.analysis.behavior.annotations import AnnotationStore
    from glider.analysis.behavior.features import compute_features
    from glider.analysis.behavior.labels import AMBIGUOUS, build_label_and_group_series
    from glider.analysis.behavior.pipeline import _mirror_pose
    from glider.vision.pose.dlc import from_dlc_csv

    xs, ys, sess_ids, mirror_flags = [], [], [], []
    for sidx, (pose_csv, ann_csv) in enumerate(sessions):
        pose = from_dlc_csv(Path(pose_csv), fps=fps)
        store = AnnotationStore.load_csv(Path(ann_csv))
        body_axis = spec.with_resolved_body_axis(pose.n_keypoints).body_axis

        variants = [(pose, False)]
        if mirror_augment:
            variants.append((_mirror_pose(pose), True))

        for variant, mirrored in variants:
            xy = _interpolate_xy(np.asarray(variant.xy, dtype=np.float64))
            feat_arr = None
            if include_features:
                fdf = compute_features(variant, spec=spec)
                feat_arr = (
                    fdf.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float64)
                )  # (F, n_feat)
            labels, _groups = build_label_and_group_series(
                store, n_frames=variant.n_frames, merge_map=merge_map, exclude=exclude
            )
            labels = labels.to_numpy()
            n = len(xy)
            # Trailing windows ending at each frame; need a full finite window.
            valid_label = (labels != "") & (labels != AMBIGUOUS)
            keep_frames = [
                i
                for i in range(window - 1, n)
                if valid_label[i] and np.isfinite(xy[i - window + 1 : i + 1]).all()
            ]
            if not keep_frames:
                continue
            wins = np.stack([xy[i - window + 1 : i + 1] for i in keep_frames])
            ego = egocentric_batch(wins, body_axis)  # (m, w, K, 2)
            m, w, kk, _ = ego.shape
            # (m, w, K, 2) → (m, K*2, w) channels-first for Conv1d over time.
            chans = ego.transpose(0, 2, 3, 1).reshape(m, kk * 2, w)
            if feat_arr is not None:
                fwins = np.stack(
                    [feat_arr[i - window + 1 : i + 1] for i in keep_frames]
                )  # (m, w, n_feat)
                fchans = fwins.transpose(0, 2, 1)  # (m, n_feat, w)
                chans = np.concatenate([chans, fchans], axis=1)
            xs.append(chans.astype(np.float32))
            ys.append(labels[keep_frames])
            sess_ids.append(np.full(m, sidx, dtype=int))
            mirror_flags.append(np.full(m, mirrored, dtype=bool))

    if not xs:
        raise ValueError("no usable labeled windows found")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    sess = np.concatenate(sess_ids)
    mirror = np.concatenate(mirror_flags)
    return x, y, sess, mirror


def cross_validate_cnn(
    sessions,
    *,
    spec,
    window: int = 30,
    fps: float = 30.0,
    merge_map: dict | None = None,
    exclude: set | None = None,
    mirror_augment: bool = False,
    include_features: bool = False,
    n_folds: int = 5,
    seed: int = 42,
    max_epochs: int = 60,
    hidden: int = 64,
    dropout: float = 0.3,
    n_ensemble: int = 1,
    progress=print,
) -> dict:
    """Session-grouped K-fold CV for the CNN — same fold structure as
    :func:`glider.analysis.behavior.cross_validate_sessions` (GroupKFold over
    sessions, mirror rows train-only, score un-mirrored test rows) so the
    accuracy is directly comparable to the LightGBM number."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import GroupKFold

    x, y, sess, mirror = assemble_sequences(
        sessions,
        spec=spec,
        window=window,
        fps=fps,
        merge_map=merge_map,
        exclude=exclude,
        mirror_augment=mirror_augment,
        include_features=include_features,
    )
    classes = np.unique(y[~mirror])
    n_unique = len(np.unique(sess))
    n_splits = min(n_folds, n_unique)
    progress(
        f"Assembled {len(x):,} windows ({int((~mirror).sum()):,} real) over "
        f"{n_unique} sessions; CNN {n_splits}-fold session-grouped CV..."
    )

    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s, all_true, all_pred = [], [], [], []
    for fold, (tr, te) in enumerate(gkf.split(x, y, groups=sess), 1):
        eval_idx = te[~mirror[te]]
        if len(eval_idx) == 0:
            continue
        if n_ensemble > 1:
            model = train_cnn_ensemble(
                x[tr],
                y[tr],
                n_models=n_ensemble,
                classes=classes,
                max_epochs=max_epochs,
                seed=seed,
                hidden=hidden,
                dropout=dropout,
            )
        else:
            model = train_cnn(
                x[tr],
                y[tr],
                classes=classes,
                max_epochs=max_epochs,
                seed=seed,
                hidden=hidden,
                dropout=dropout,
            )
        pred = model.predict(x[eval_idx])
        true = y[eval_idx]
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average="macro", labels=classes, zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        all_true.append(true)
        all_pred.append(pred)
        progress(f"  fold {fold}: acc={acc:.3f}  macro-F1={f1:.3f}")

    true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    prec, rec, f1c, supp = precision_recall_fscore_support(
        true_all, pred_all, labels=classes, zero_division=0
    )
    cm = confusion_matrix(true_all, pred_all, labels=classes)
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "per_fold_accuracy": [float(a) for a in accs],
        "classes": list(classes),
        "per_class": {
            c: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1c[i]),
                "support": int(supp[i]),
            }
            for i, c in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Deployable model wrapper: persistence + per-window / per-frame prediction
# ---------------------------------------------------------------------------

SEQ_BUNDLE_FORMAT = "glider-seq-cnn-v2"
# Back-compat: load() accepts the current GLIDER format plus the two legacy
# yolo2pose-named formats so previously-saved .pt bundles still open.
_SEQ_BUNDLE_FORMATS = (
    SEQ_BUNDLE_FORMAT,
    "yolo2pose-seq-cnn-v2",
    "yolo2pose-seq-cnn-v1",
)


def _centroid_net_displacement(xy_window, body_axis) -> float:
    """How far the keypoint centroid translated over the window, in
    body-lengths. Robust to a snout-only reach (the centroid barely moves
    when one keypoint extends), so it tracks whole-body translation —
    exactly what separates locomotion from a stationary sniff that just
    elongates the body."""
    xy = np.asarray(xy_window, dtype=np.float64)
    centroid = np.nanmean(xy, axis=1)  # (w, 2)
    net = float(np.linalg.norm(centroid[-1] - centroid[0]))
    head, tail = body_axis
    length = float(np.nanmedian(np.linalg.norm(xy[:, tail] - xy[:, head], axis=1)))
    return net / length if length > 0 else 0.0


@dataclass
class MovementGate:
    """Suppress a behavior that requires translation when the body didn't move.

    Built for the 'stationary sniff that elongates → mislabeled locomote'
    case: if the top prediction is ``gate_class`` but the centroid moved
    less than ``min_displacement`` body-lengths over the window, the frame
    is relabeled to the model's runner-up class. A purely post-hoc,
    tunable rule — no retraining.
    """

    gate_class: str
    min_displacement: float

    def relabel(self, proba, classes, xy_window, body_axis) -> str:
        order = np.argsort(proba)[::-1]
        top = str(classes[order[0]])
        if top != self.gate_class:
            return top
        if _centroid_net_displacement(xy_window, body_axis) >= self.min_displacement:
            return top
        return str(classes[order[1]]) if len(order) > 1 else top


class SequenceModel:
    """A trained CNN (or CNN ensemble) packaged for deployment.

    Holds one or more torch modules plus everything needed to turn raw
    pose into a prediction: the ``window`` length, the ``body_axis`` for
    egocentric normalization, the class list, and the architecture (so the
    net(s) can be rebuilt on load). A single net and an N-net ensemble
    share one interface — predictions average member probabilities when
    there's more than one. Predicts directly from keypoints — no feature
    extraction — via :meth:`predict_window` (one window) and
    :meth:`score_pose` (a whole recording → per-frame labels).
    """

    def __init__(self, modules, classes, window, body_axis, fps, arch, device=None, spec=None):
        import torch

        self.modules = list(modules)
        self.classes = np.asarray(classes)
        self.window = int(window)
        self.body_axis = (int(body_axis[0]), int(body_axis[1]))
        self.fps = float(fps)
        self.arch = dict(arch)
        self.spec = spec
        if not device:
            # No explicit device (the SequenceModel.load path passes none):
            # resolve inference to the best locally available accelerator
            # (CUDA > MPS > CPU) via the pose subsystem's resolver. Without this
            # a saved model runs on CPU even on Apple Silicon, where the Metal
            # GPU (MPS) is available. Training keeps its own device path so its
            # deterministic-reproducibility guarantees are unaffected.
            from glider.vision.pose.device import resolve_device

            self.device = torch.device(resolve_device(None))
        else:
            self.device = torch.device(device)
        members = []
        for m in self.modules:
            m.to(self.device).eval()
            members.append(FittedCNN(m, self.classes, self.device, self.arch))
        self._predictor = members[0] if len(members) == 1 else FittedEnsemble(members, self.classes)

    # ---- prediction -----------------------------------------------------
    def _windows_to_channels(self, windows: np.ndarray) -> np.ndarray:
        """``(m, w, K, 2)`` egocentric-normalized → ``(m, K*2, w)`` float32."""
        ego = egocentric_batch(windows, self.body_axis)
        m, w, kk, _ = ego.shape
        return ego.transpose(0, 2, 3, 1).reshape(m, kk * 2, w).astype(np.float32)

    def predict_window(self, xy_window: np.ndarray, gate=None) -> str:
        """Predict the behavior for one ``(window, K, 2)`` pose window.

        Keypoint gaps (low-confidence/occluded frames — common during
        digs) are linearly interpolated within the window first, matching
        what training does, so a brief dropout doesn't blank the whole
        window. Returns ``""`` only when a keypoint is missing for the
        *entire* window (nothing to interpolate from) — the honest
        "unknown", same contract as the LightGBM path.

        ``gate`` (a :class:`MovementGate`) optionally relabels a
        translation-requiring class (e.g. locomote) when the body didn't
        actually move.
        """
        xy = _interpolate_xy(np.asarray(xy_window, dtype=np.float64))
        chans = self._windows_to_channels(xy[None])
        if not np.isfinite(chans).all():  # whole-window gap / degenerate scale
            return ""
        if gate is None:
            return str(self._predictor.predict(chans)[0])
        proba = self._predictor.predict_proba(chans)[0]
        return gate.relabel(proba, self.classes, xy, self.body_axis)

    def score_pose(self, pose, gate=None) -> list[str]:
        """Score a whole recording → one label per frame (trailing window).

        Interpolates keypoint gaps across the whole track first (exactly as
        training does), then scores each trailing window. The first
        ``window-1`` frames, and any window still NaN after interpolation
        (a keypoint absent for its whole span), get ``""``. ``gate`` (a
        :class:`MovementGate`) is applied per window if given. Batched for
        speed.
        """
        xy = _interpolate_xy(np.asarray(pose.xy, dtype=np.float64))
        n = xy.shape[0]
        labels = [""] * n
        if n < self.window:
            return labels
        wins = np.stack([xy[i - self.window + 1 : i + 1] for i in range(self.window - 1, n)])
        chans = self._windows_to_channels(wins)  # (m, K*2, w)
        finite = np.isfinite(chans).reshape(len(chans), -1).all(axis=1)
        proba = self._predictor.predict_proba(chans)  # (m, C)
        for j, i in enumerate(range(self.window - 1, n)):
            if not finite[j]:
                continue
            if gate is None:
                labels[i] = str(self.classes[proba[j].argmax()])
            else:
                labels[i] = gate.relabel(proba[j], self.classes, wins[j], self.body_axis)
        return labels

    # ---- persistence ----------------------------------------------------
    def save(self, path) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": SEQ_BUNDLE_FORMAT,
                "state_dicts": [m.state_dict() for m in self.modules],
                "arch": self.arch,
                "classes": list(self.classes),
                "window": self.window,
                "body_axis": list(self.body_axis),
                "fps": self.fps,
                "spec": self.spec.to_dict() if self.spec is not None else None,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path) -> SequenceModel:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") not in _SEQ_BUNDLE_FORMATS:
            raise ValueError(
                f"{path} is not a CNN sequence bundle " f"(got format={payload.get('format')!r})"
            )
        arch = payload["arch"]
        # v2 stores a list of state_dicts; v1 stored a single one.
        state_dicts = payload.get("state_dicts") or [payload["state_dict"]]
        net_cls = _make_cnn_class()
        modules = []
        for sd in state_dicts:
            net = net_cls(
                n_channels=arch["n_channels"],
                n_classes=arch["n_classes"],
                hidden=arch.get("hidden", 64),
                dropout=arch.get("dropout", 0.3),
            )
            net.load_state_dict(sd)
            modules.append(net)
        spec = None
        if payload.get("spec") is not None:
            from glider.analysis.behavior.features import FeatureSpec

            spec = FeatureSpec.from_dict(payload["spec"])
        return cls(
            modules=modules,
            classes=payload["classes"],
            window=payload["window"],
            body_axis=payload["body_axis"],
            fps=payload["fps"],
            arch=arch,
            spec=spec,
        )


def train_sequence_model(
    sessions,
    *,
    spec,
    window: int = 30,
    fps: float = 30.0,
    merge_map: dict | None = None,
    exclude: set | None = None,
    mirror_augment: bool = False,
    holdout_sessions=None,
    max_epochs: int = 60,
    n_ensemble: int = 1,
    seed: int = 42,
    progress=print,
) -> tuple[SequenceModel, dict]:
    """Train a :class:`SequenceModel` on the training sessions.

    Assembles egocentric keypoint sequences, trains the CNN (or, when
    ``n_ensemble > 1``, an ensemble of independently-seeded CNNs whose
    probabilities are averaged) on all of them, and (if
    ``holdout_sessions`` is given) reports cross-session holdout metrics.
    Returns ``(model, summary)``.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    x, y, _sess, _mirror = assemble_sequences(
        sessions,
        spec=spec,
        window=window,
        fps=fps,
        merge_map=merge_map,
        exclude=exclude,
        mirror_augment=mirror_augment,
    )
    classes = np.unique(y)
    kind = "CNN" if n_ensemble <= 1 else f"{n_ensemble}-net CNN ensemble"
    progress(f"Training {kind} on {len(x):,} windows · {len(classes)} classes...")
    if n_ensemble > 1:
        fitted = train_cnn_ensemble(
            x,
            y,
            n_models=n_ensemble,
            classes=classes,
            max_epochs=max_epochs,
            seed=seed,
        )
        modules = [m.module for m in fitted.members]
        arch, device = fitted.members[0].arch, fitted.members[0].device
    else:
        fitted = train_cnn(x, y, classes=classes, max_epochs=max_epochs, seed=seed)
        modules = [fitted.module]
        arch, device = fitted.arch, fitted.device

    body_axis = spec.with_resolved_body_axis(
        # K from channel count: channels = K*2.
        x.shape[1]
        // 2
    ).body_axis
    model = SequenceModel(
        modules=modules,
        classes=classes,
        window=window,
        body_axis=body_axis,
        fps=fps,
        arch=arch,
        device=device,
        spec=spec,
    )

    summary: dict = {"classes": list(classes), "n_train_windows": int(len(x))}
    if holdout_sessions:
        x_h, y_h, _s, _m = assemble_sequences(
            holdout_sessions,
            spec=spec,
            window=window,
            fps=fps,
            merge_map=merge_map,
            exclude=exclude,
            mirror_augment=False,
        )
        pred = fitted.predict(x_h)
        acc = float(accuracy_score(y_h, pred))
        prec, rec, f1c, supp = precision_recall_fscore_support(
            y_h, pred, labels=classes, zero_division=0
        )
        summary["holdout_accuracy"] = acc
        summary["holdout_per_class"] = {
            c: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1c[i]),
                "support": int(supp[i]),
            }
            for i, c in enumerate(classes)
        }
        summary["holdout_confusion_matrix"] = confusion_matrix(y_h, pred, labels=classes).tolist()
        progress(f"  holdout accuracy: {acc:.3f} ({len(x_h):,} windows)")
    return model, summary
