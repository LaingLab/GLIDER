"""Stored train/val/test split for every extracted frame.

The split lives on each :class:`~.project.FrameRecord` so it is visible in the
UI and survives top-ups. Single-video projects split per frame; with more
videos, a new video goes whole to one split so frames of one clip never
straddle train and val.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from glider.vision.training.project import FrameRecord, StudioProject

__all__ = [
    "SPLITS",
    "TARGETS",
    "assign_splits",
    "ensure_splits",
    "leak_notice",
    "leaky_videos",
    "split_counts",
]

SPLITS = ("train", "val", "test")
TARGETS = {"train": 0.7, "val": 0.2, "test": 0.1}


def _by_video(project: StudioProject, frames: list[FrameRecord]) -> dict[str, list[FrameRecord]]:
    groups: dict[str, list[FrameRecord]] = {}
    for f in frames:
        groups.setdefault(f.video, []).append(f)
    order = {k: i for i, k in enumerate(project.videos)}
    return dict(sorted(groups.items(), key=lambda kv: order.get(kv[0], len(order))))


def _random_split(project: StudioProject, frame: FrameRecord) -> str:
    r = random.Random(f"{project.seed}:{frame.name}").random()
    return "train" if r < TARGETS["train"] else "val" if r < 0.9 else "test"


def assign_splits(project: StudioProject, frames: list[FrameRecord]) -> None:
    """Give every frame in ``frames`` with no split one, in place."""
    pending = [f for f in frames if not f.split]
    for video, group in _by_video(project, pending).items():
        if len(project.videos) <= 1:
            for f in group:
                f.split = _random_split(project, f)
            continue
        known = [f.split for f in project.frames if f.video == video and f.split]
        if known:
            counts = Counter(known)
            split = max(SPLITS, key=lambda s: counts[s])  # max keeps the first on ties
        else:
            seen: dict[str, set[str]] = {}
            for f in project.frames:
                if f.split:
                    seen.setdefault(f.video, set()).add(f.split)
            held_out = any(v == {"val"} for v in seen.values())
            if not held_out:
                split = "val"
            else:
                counts = split_counts(project)
                total = sum(counts.values()) + len(group)
                split = max(SPLITS, key=lambda s: TARGETS[s] * total - counts[s])
        for f in group:
            f.split = split


def ensure_splits(project: StudioProject) -> bool:
    """Migration: split frames of a project made before splits were stored.

    A project with no stored split at all gets the old export's split for its
    labelled frames; otherwise only frames still without a split are filled and
    existing ones are never touched. True if anything changed (the caller saves).
    """
    from glider.vision.training.dataset import split_frames

    if all(f.split for f in project.frames):
        return False
    if not any(f.split for f in project.frames):  # legacy project: nothing stored yet
        labelled = [f for f in project.frames if not f.skipped and project.is_labelled(f)]
        train, val, by_video = split_frames(labelled, project.val_fraction, project.seed)
        for f in val:
            f.split = "val"
        for f in train:
            f.split = "train"
        if by_video:
            video_split = {f.video: f.split for f in labelled}
            for f in project.frames:
                if not f.split and f.video in video_split:
                    f.split = video_split[f.video]
    assign_splits(project, project.frames)  # only ever fills frames with no split
    return True


def leaky_videos(project: StudioProject) -> list[str]:
    """Videos with frames in train and in val or test."""
    seen: dict[str, set[str]] = {}
    for f in project.frames:
        if f.split:
            seen.setdefault(f.video, set()).add(f.split)
    return [k for k in project.videos if {"train"} < seen.get(k, set())]


def split_counts(project: StudioProject) -> dict[str, int]:
    counts = Counter(f.split for f in project.frames)
    return {s: counts[s] for s in SPLITS}


def leak_notice(project: StudioProject) -> str:
    """Why validation scores would be optimistic, or "" when no video leaks.

    A one-video project always splits per frame, so there is nothing to act
    on and nothing is said.
    """
    leaky = leaky_videos(project)
    if not leaky or len(project.videos) <= 1:
        return ""
    return "\n".join(
        f"{Path(project.videos[k]).name} has frames in train and in val/test \u2014 "
        "validation scores will look better than they are. Move its frames to one split."
        for k in leaky
    )
