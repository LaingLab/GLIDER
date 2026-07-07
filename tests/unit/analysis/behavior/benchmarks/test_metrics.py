"""Tests for the action-segmentation metrics."""

from __future__ import annotations

from glider.analysis.behavior.benchmarks import metrics as m

BG = "background"
CLASSES = ["Supported", "Unsupported", "Grooming"]


def seq(*runs):
    """Build a per-frame sequence from (label, count) runs."""
    out = []
    for label, n in runs:
        out.extend([label] * n)
    return out


# --- segments -----------------------------------------------------------------


def test_labels_to_segments():
    labels = seq((BG, 2), ("Supported", 3), (BG, 1))
    assert m.labels_to_segments(labels) == [
        (BG, 0, 2),
        ("Supported", 2, 5),
        (BG, 5, 6),
    ]


def test_labels_to_segments_empty():
    assert m.labels_to_segments([]) == []


# --- frame-wise ---------------------------------------------------------------


def test_frame_accuracy_perfect():
    s = seq((BG, 5), ("Supported", 5))
    assert m.frame_accuracy(s, s) == 1.0


def test_frame_accuracy_half():
    gt = seq(("Supported", 10))
    pred = seq(("Supported", 5), (BG, 5))
    assert m.frame_accuracy(gt, pred) == 0.5


def test_frame_accuracy_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError, match="length mismatch"):
        m.frame_accuracy([BG], [BG, BG])


def test_per_class_frame_f1_perfect():
    gt = seq((BG, 5), ("Supported", 5), ("Grooming", 5))
    per = m.per_class_frame_f1(gt, gt, CLASSES)
    assert per["Supported"]["f1"] == 1.0
    assert per["Grooming"]["f1"] == 1.0
    assert per["Supported"]["support"] == 5


def test_per_class_frame_f1_precision_recall():
    # GT: 10 Supported. Pred: 5 Supported (correct) + 5 Supported where GT is bg.
    gt = seq(("Supported", 10), (BG, 10))
    pred = seq(("Supported", 15), (BG, 5))
    sup = m.per_class_frame_f1(gt, pred, CLASSES)["Supported"]
    assert sup["recall"] == 1.0  # all 10 true Supported found
    assert sup["precision"] == 10 / 15  # 5 of the 15 predicted are wrong


def test_macro_frame_f1_averages_classes():
    gt = seq(("Supported", 10), ("Grooming", 10))
    pred = seq(("Supported", 10), (BG, 10))  # Grooming entirely missed
    # Supported F1 = 1.0, Grooming F1 = 0.0, Unsupported F1 = 0.0 -> mean 1/3
    assert abs(m.macro_frame_f1(gt, pred, CLASSES) - 1 / 3) < 1e-9


# --- segmental ----------------------------------------------------------------


def test_edit_score_identical():
    s = seq((BG, 5), ("Supported", 5), (BG, 5))
    assert m.edit_score(s, s, exclude=(BG,)) == 100.0


def test_edit_score_excludes_background_ordering():
    # Same behavior order (Supported then Grooming), different background;
    # excluding background makes them identical.
    gt = seq((BG, 2), ("Supported", 3), (BG, 2), ("Grooming", 3))
    pred = seq(("Supported", 3), (BG, 5), ("Grooming", 3))
    assert m.edit_score(gt, pred, exclude=(BG,)) == 100.0


def test_edit_score_missing_segment():
    gt = seq(("Supported", 3), ("Grooming", 3))  # 2 behavior segments
    pred = seq(("Supported", 6))  # 1 behavior segment -> one deletion
    # denom = 2, distance = 1 -> (1 - 1/2)*100 = 50
    assert m.edit_score(gt, pred, exclude=(BG,)) == 50.0


def test_segmental_f1_perfect_overlap():
    gt = seq((BG, 5), ("Supported", 10), (BG, 5))
    pred = seq((BG, 5), ("Supported", 10), (BG, 5))
    assert m.segmental_f1(gt, pred, CLASSES, iou_threshold=0.5) == 100.0


def test_segmental_f1_low_iou_is_false_positive():
    # Predicted bout barely overlaps the GT bout -> IoU < 0.5 -> not a TP.
    gt = seq((BG, 0), ("Supported", 10), (BG, 10))
    pred = seq((BG, 8), ("Supported", 10), (BG, 2))  # overlap 2 / union 18 ~ 0.11
    assert m.segmental_f1(gt, pred, CLASSES, iou_threshold=0.5) == 0.0
    # But at a lenient 0.10 threshold it counts.
    assert m.segmental_f1(gt, pred, CLASSES, iou_threshold=0.10) == 100.0


def test_segmental_f1_over_segmentation_penalised():
    # One true bout, predicted as three fragments: 1 TP + 2 FP -> P=1/3, R=1, F1=0.5.
    gt = seq(("Supported", 30))
    pred = seq(("Supported", 12), (BG, 1), ("Supported", 12), (BG, 1), ("Supported", 4))
    score = m.segmental_f1(gt, pred, CLASSES, iou_threshold=0.10)
    assert abs(score - 50.0) < 1e-6


def test_segmental_f1_each_gt_matched_once():
    # Two predicted bouts both overlap the same single GT bout: 1 TP + 1 FP.
    gt = seq(("Supported", 20))
    pred = seq(("Supported", 10), (BG, 0), ("Supported", 10))
    # collapses to one predicted segment actually; force a gap:
    pred = seq(("Supported", 9), (BG, 2), ("Supported", 9))
    score = m.segmental_f1(gt, pred, CLASSES, iou_threshold=0.10)
    # P = 1/2, R = 1 -> F1 = 66.7
    assert abs(score - (2 * 0.5 * 1 / 1.5) * 100) < 1e-6


# --- bundle -------------------------------------------------------------------


def test_evaluate_bundle_perfect():
    s = seq((BG, 10), ("Supported", 10), (BG, 5), ("Grooming", 5))
    result = m.evaluate(s, s, CLASSES, background=BG)
    assert result.frame_accuracy == 1.0
    assert result.macro_f1 == 1.0 / 3 * 2  # only Supported+Grooming present -> 2/3
    assert result.edit == 100.0
    assert result.f1_at[50] == 100.0
    row = result.as_row()
    assert set(row) >= {"frame_acc", "macro_f1", "edit", "f1@10", "f1@25", "f1@50"}
