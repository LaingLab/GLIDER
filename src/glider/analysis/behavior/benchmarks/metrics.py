"""Action-segmentation metrics for benchmarking against DLC2action.

Two families, because it is not certain from the DLC2action paper which "F1"
its OFT table reports — so we compute both and label them:

* **Frame-wise** — per-frame accuracy (MoF) and per-class F1 (macro over the
  behavior classes). This is the Sturman-style metric: does each frame get the
  right label? Robust to how bouts are segmented.
* **Segmental** — edit (Levenshtein) score and F1@IoU (10/25/50), the standard
  temporal-action-segmentation metrics (MS-TCN protocol). A predicted bout is a
  true positive if it overlaps a same-class ground-truth bout with IoU ≥ the
  threshold, each GT bout matched at most once. These punish over-segmentation
  (label flicker) — exactly the axis where a frame-wise classifier is expected
  to trail a temporal model.

All functions take equal-length per-frame label sequences (any hashable labels;
strings here). Frame metrics return fractions in ``[0, 1]``; segmental scores
follow the literature and return percentages in ``[0, 100]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

Segment = tuple[object, int, int]  # (label, start, end) half-open


def labels_to_segments(labels: Sequence[object]) -> list[Segment]:
    """Collapse a per-frame label sequence into ``(label, start, end)`` runs."""
    labels = list(labels)
    if not labels:
        return []
    segments: list[Segment] = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append((labels[i - 1], start, i))
            start = i
    segments.append((labels[-1], start, len(labels)))
    return segments


# --- Frame-wise ---------------------------------------------------------------


def frame_accuracy(gt: Sequence[object], pred: Sequence[object]) -> float:
    """Fraction of frames where prediction equals ground truth (MoF)."""
    g = np.asarray(gt)
    p = np.asarray(pred)
    if g.shape != p.shape:
        raise ValueError(f"gt/pred length mismatch: {g.shape} vs {p.shape}")
    if len(g) == 0:
        return 0.0
    return float(np.mean(g == p))


def per_class_frame_f1(
    gt: Sequence[object], pred: Sequence[object], classes: Sequence[object]
) -> dict[object, dict[str, float]]:
    """Per-class precision / recall / F1 / support, frame-wise (one-vs-rest)."""
    g = np.asarray(gt)
    p = np.asarray(pred)
    out: dict[object, dict[str, float]] = {}
    for c in classes:
        tp = int(np.sum((p == c) & (g == c)))
        fp = int(np.sum((p == c) & (g != c)))
        fn = int(np.sum((p != c) & (g == c)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(np.sum(g == c)),
        }
    return out


def macro_frame_f1(
    gt: Sequence[object], pred: Sequence[object], classes: Sequence[object]
) -> float:
    """Unweighted mean of per-class frame F1 over ``classes``."""
    per = per_class_frame_f1(gt, pred, classes)
    if not classes:
        return 0.0
    return float(np.mean([per[c]["f1"] for c in classes]))


# --- Segmental ----------------------------------------------------------------


def _levenshtein(a: Sequence[object], b: Sequence[object]) -> int:
    """Edit distance between two label sequences (insert/delete/substitute = 1)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def edit_score(
    gt: Sequence[object], pred: Sequence[object], *, exclude: Sequence[object] = ()
) -> float:
    """Normalised segmental edit score in ``[0, 100]`` (100 = identical order).

    Compares the collapsed sequences of segment labels (order + identity, not
    duration). ``exclude`` drops those labels (e.g. background) from both
    sequences first, so the score reflects behavior-bout ordering only.
    """
    drop = set(exclude)
    g = [lbl for lbl, _, _ in labels_to_segments(gt) if lbl not in drop]
    p = [lbl for lbl, _, _ in labels_to_segments(pred) if lbl not in drop]
    denom = max(len(g), len(p))
    if denom == 0:
        return 100.0
    return float((1.0 - _levenshtein(p, g) / denom) * 100.0)


def segmental_counts(
    gt: Sequence[object],
    pred: Sequence[object],
    classes: Sequence[object],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """``(tp, fp, fn)`` of predicted vs GT segments at ``iou_threshold``.

    A predicted segment is a true positive if some not-yet-matched ground-truth
    segment of the SAME class has IoU ≥ ``iou_threshold``. Only segments whose
    label is in ``classes`` count (background segments ignored on both sides).
    Returning raw counts lets a caller sum across videos for a micro-averaged
    dataset F1 (the MS-TCN convention).
    """
    keep = set(classes)
    gt_segs = [s for s in labels_to_segments(gt) if s[0] in keep]
    pred_segs = [s for s in labels_to_segments(pred) if s[0] in keep]
    matched = [False] * len(gt_segs)

    tp = 0
    for p_label, p_start, p_end in pred_segs:
        best_iou = 0.0
        best_j = -1
        for j, (g_label, g_start, g_end) in enumerate(gt_segs):
            if g_label != p_label or matched[j]:
                continue
            inter = max(0, min(p_end, g_end) - max(p_start, g_start))
            union = max(p_end, g_end) - min(p_start, g_start)
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched[best_j] = True

    fp = len(pred_segs) - tp
    fn = len(gt_segs) - tp
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """F1 as a percentage from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if (precision + recall) == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall) * 100.0)


def segmental_f1(
    gt: Sequence[object],
    pred: Sequence[object],
    classes: Sequence[object],
    *,
    iou_threshold: float = 0.5,
) -> float:
    """Segmental F1@IoU in ``[0, 100]`` over ``classes`` (MS-TCN protocol)."""
    tp, fp, fn = segmental_counts(gt, pred, classes, iou_threshold=iou_threshold)
    return f1_from_counts(tp, fp, fn)


# --- Bundle ------------------------------------------------------------------


@dataclass
class SegmentationMetrics:
    """All metrics for one predicted-vs-truth sequence."""

    frame_accuracy: float
    macro_f1: float  # frame-wise, mean over behavior classes (fraction)
    per_class: dict[object, dict[str, float]]
    edit: float
    f1_at: dict[int, float] = field(default_factory=dict)  # {10: .., 25: .., 50: ..}

    def as_row(self) -> dict[str, float]:
        """Flat dict for tabulation / CSV."""
        row = {
            "frame_acc": self.frame_accuracy,
            "macro_f1": self.macro_f1,
            "edit": self.edit,
        }
        for k, v in self.f1_at.items():
            row[f"f1@{k}"] = v
        return row


def evaluate(
    gt: Sequence[object],
    pred: Sequence[object],
    classes: Sequence[object],
    *,
    background: object = "background",
    iou_thresholds: Sequence[float] = (0.10, 0.25, 0.50),
) -> SegmentationMetrics:
    """Compute the full metric bundle for one video.

    ``classes`` are the behavior labels scored by the segmental / macro metrics;
    ``background`` is excluded from the edit score and segmental F1 but still
    counts as a negative in frame accuracy and per-class precision.
    """
    return SegmentationMetrics(
        frame_accuracy=frame_accuracy(gt, pred),
        macro_f1=macro_frame_f1(gt, pred, classes),
        per_class=per_class_frame_f1(gt, pred, classes),
        edit=edit_score(gt, pred, exclude=(background,)),
        f1_at={
            int(round(t * 100)): segmental_f1(gt, pred, classes, iou_threshold=t)
            for t in iou_thresholds
        },
    )
