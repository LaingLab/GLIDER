"""Active-learning clip-review annotator (GLIDER port).

Workflow
--------

1. Load a video + its DLC pose CSV.
2. Run :func:`glider.gui.behavior.annotator.sampler.propose_clips` to pick N
   diverse, distinctive clips (k-means++ over geometric + kinematic
   features).
3. Show each clip on loop in the GUI; the user labels via behavior
   chips / hotkeys, or marks the clip as ``multi-behavior`` /
   ``unclear`` / skip.
4. Persist labels to ``<video_stem>_annotations.csv``. The reserved
   ``multi-behavior`` / ``unclear`` rows stay in the CSV (so the
   sampler doesn't re-surface them on a re-run) but
   :mod:`glider.analysis.behavior.labels` treats them as drop signals so
   they can't poison the trained model.

The data model (:mod:`glider.analysis.behavior.annotations`,
:mod:`glider.analysis.behavior.vocabulary`) and the sampler are pure
Python so they're unit-testable without Qt. The UI lives in
:mod:`glider.gui.behavior.annotator.main_window`.
"""

from __future__ import annotations

from glider.gui.behavior.annotator.app import run
from glider.gui.behavior.annotator.main_window import AnnotatorWindow
from glider.gui.behavior.annotator.sampler import (
    DEFAULT_CLIP_SECONDS,
    ProposedClip,
    propose_clips,
    propose_clips_multi,
)

__all__ = [
    "DEFAULT_CLIP_SECONDS",
    "AnnotatorWindow",
    "ProposedClip",
    "propose_clips",
    "propose_clips_multi",
    "run",
]
