"""Project annotation zones onto a per-frame label vector.

The annotator produces zones (variable-length intervals of one
behavior). The training pipeline needs a per-frame label so it can
align labels with the per-frame feature DataFrame and drop unannotated
frames.

The label convention:

* ``""`` (empty string) — frame is **unannotated**. Dropped from training.
* ``"<behavior>"`` — frame is annotated as that behavior.
* ``"__ambiguous__"`` — frame is covered by two or more zones of
  **different** behaviors. We can't pick one without losing
  information; the training pipeline drops these too and the CLI
  reports how many there were.

Same-behavior overlap is impossible at the store level (the
:class:`AnnotationStore` rejects it on add), so the only ambiguous
case is the multi-behavior overlap the annotator allows by design.
"""

from __future__ import annotations

import pandas as pd

from glider.analysis.behavior.annotations import AnnotationStore

AMBIGUOUS = "__ambiguous__"

# Reserved labels the annotator emits as "exclusion markers" — clips the
# user flagged as ambiguous or unviewable. These are deliberately
# persisted in the annotations CSV (so the sampler knows the user
# already saw them) but treated as drop signals by the training
# pipeline, since training on them would poison the model.
MULTI_BEHAVIOR = "multi-behavior"
UNCLEAR = "unclear"
RESERVED_LABELS: frozenset[str] = frozenset((MULTI_BEHAVIOR, UNCLEAR))


def build_label_series(
    store: AnnotationStore,
    n_frames: int,
    merge_map: dict[str, str] | None = None,
    exclude: set[str] | None = None,
) -> pd.Series:
    """Return a per-frame label series of length ``n_frames``.

    Each entry is one of: ``""`` (unannotated), a behavior name, or
    :data:`AMBIGUOUS` (multi-behavior overlap OR a reserved exclusion
    label from the annotator). The training pipeline drops AMBIGUOUS
    rows, so this is how ``multi-behavior`` / ``unclear`` clips are
    kept out of training.
    """
    labels, _groups = build_label_and_group_series(store, n_frames, merge_map, exclude)
    return labels


def build_label_and_group_series(
    store: AnnotationStore,
    n_frames: int,
    merge_map: dict[str, str] | None = None,
    exclude: set[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Like :func:`build_label_series` but also returns a per-frame group ID.

    The group ID is the index of the source zone in the annotation
    store (positive int) for annotated frames, or ``-1`` for
    unannotated frames. AMBIGUOUS frames keep the group ID of the
    LAST zone that touched them — they're dropped downstream anyway,
    so the value doesn't matter for training.

    Group IDs are what the training pipeline feeds to
    :class:`sklearn.model_selection.GroupShuffleSplit` so all rows
    from the same labeled zone land on the same side of the train /
    test split — preventing near-duplicate adjacent windows from
    leaking from train into test and inflating the accuracy number.

    ``merge_map`` optionally renames behaviors ``{old_name: new_name}``
    as each zone is read. The remap is applied *before* the ambiguity
    check, so frames covered by two zones that both map to the same
    merged class collapse to that class instead of being dropped as
    AMBIGUOUS. Annotations on disk are never modified.

    ``exclude`` is a set of behavior names to drop entirely: matching
    zones are skipped as if they weren't annotated, so their frames go
    unlabeled (and get dropped from training) unless another zone covers
    them. Matching is on the ORIGINAL behavior name, before ``merge_map``,
    so excluding a behavior takes precedence over merging it.
    """
    if n_frames < 0:
        raise ValueError(f"n_frames must be >= 0, got {n_frames}")
    merge_map = merge_map or {}
    exclude = exclude or frozenset()
    labels = [""] * int(n_frames)
    groups = [-1] * int(n_frames)
    for zone_idx, zone in enumerate(store):
        # Excluded behaviors are skipped wholesale — their frames stay
        # unannotated (dropped) unless another zone covers them.
        if zone.behavior in exclude:
            continue
        # Apply the merge remap up front so two merged behaviors that
        # overlap collapse to one class instead of becoming AMBIGUOUS.
        # Reserved labels are never merge_map keys, so they pass through
        # unchanged and are still caught by the RESERVED_LABELS check.
        behavior = merge_map.get(zone.behavior, zone.behavior)
        for f in range(
            max(0, zone.start_frame),
            min(n_frames, zone.end_frame),
        ):
            current = labels[f]
            if behavior in RESERVED_LABELS:
                labels[f] = AMBIGUOUS
                groups[f] = zone_idx
                continue
            if current == AMBIGUOUS:
                continue
            if current == "" or current == behavior:
                labels[f] = behavior
                groups[f] = zone_idx
            else:
                labels[f] = AMBIGUOUS
                groups[f] = zone_idx
    return (
        pd.Series(labels, name="label", dtype=object),
        pd.Series(groups, name="group", dtype=int),
    )


def split_label_counts(labels: pd.Series) -> dict:
    """Return ``{behavior_name: n, "__unannotated__": n, "__ambiguous__": n}``.

    Convenience for the CLI's training summary.
    """
    counts: dict[str, int] = {}
    for v in labels:
        key = "__unannotated__" if v == "" else v
        counts[key] = counts.get(key, 0) + 1
    return counts
