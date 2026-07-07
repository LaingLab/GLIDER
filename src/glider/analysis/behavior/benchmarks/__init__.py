"""Benchmark adapters for the GLIDER behavior classifier.

These modules convert public action-segmentation benchmarks into the
``(pose CSV, annotations CSV)`` session pairs that
:func:`glider.analysis.behavior.pipeline.train_model` consumes, so the GLIDER
model can be evaluated head-to-head against published methods (e.g. DLC2action).

Benchmark tooling, not part of the shipped runtime path — imported explicitly
by evaluation scripts and tests.
"""

from glider.analysis.behavior.benchmarks.oft import (
    OFT_ARENA_MARKERS,
    OFT_BEHAVIORS,
    OFT_BODY_AXIS,
    OFT_FPS,
    OFTSession,
    build_oft_benchmark,
    drop_keypoints,
    labels_to_store,
    leave_one_out_splits,
    list_experimenters,
    load_sturman_pose,
    oft_feature_spec,
    read_label_table,
    resolve_body_axis,
    sessions_to_pairs,
)

__all__ = [
    "OFT_ARENA_MARKERS",
    "OFT_BEHAVIORS",
    "OFT_BODY_AXIS",
    "OFT_FPS",
    "OFTSession",
    "build_oft_benchmark",
    "drop_keypoints",
    "labels_to_store",
    "leave_one_out_splits",
    "list_experimenters",
    "load_sturman_pose",
    "oft_feature_spec",
    "read_label_table",
    "resolve_body_axis",
    "sessions_to_pairs",
]
