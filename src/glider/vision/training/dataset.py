"""Turn a labelled project into something Ultralytics can train on.

The split is by video wherever it can be. Frames from one session look alike,
so a validation frame from a video the model also trained on mostly measures
memory; a held-out video measures whether the model works on the next session.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from glider.core.fileio import atomic_write_bytes
from glider.vision.training.project import FrameRecord, StudioProject
from glider.vision.training.splits import ensure_splits, leaky_videos

__all__ = ["DatasetError", "ExportResult", "export_dataset", "flip_idx", "split_frames"]


class DatasetError(ValueError):
    """The project cannot be exported yet; the message says why."""


@dataclass
class ExportResult:
    data_yaml: Path
    train: list[FrameRecord] = field(default_factory=list)
    val: list[FrameRecord] = field(default_factory=list)
    test: list[FrameRecord] = field(default_factory=list)
    #: False when a video has frames in train and in val/test, so validation
    #: frames come from a video the model also trains on. Worth telling the user.
    split_by_video: bool = True


def flip_idx(n_keypoints: int, pairs: list[tuple[int, int]]) -> list[int]:
    """Ultralytics' ``flip_idx``: where each keypoint goes when mirrored."""
    order = list(range(n_keypoints))
    for a, b in pairs:
        order[a], order[b] = b, a
    return order


def split_frames(
    frames: list[FrameRecord], val_fraction: float, seed: int
) -> tuple[list[FrameRecord], list[FrameRecord], bool]:
    """``(train, val, by_video)``.

    With two or more videos, whole videos go to validation -- shuffled by
    ``seed``, added until about ``val_fraction`` of the frames are held out,
    always leaving at least one video to train on. With one video, frames are
    split at random instead and ``by_video`` is False.
    """
    videos = sorted({f.video for f in frames})
    rng = random.Random(seed)
    if len(videos) >= 2:
        rng.shuffle(videos)
        target = max(1, round(val_fraction * len(frames)))
        held: set[str] = set()
        held_count = 0
        for video in videos[:-1]:  # never hold out the last one
            if held_count >= target:
                break
            held.add(video)
            held_count += sum(1 for f in frames if f.video == video)
        train = [f for f in frames if f.video not in held]
        val = [f for f in frames if f.video in held]
        return train, val, True
    shuffled = list(frames)
    rng.shuffle(shuffled)
    n_val = max(1, round(val_fraction * len(shuffled))) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val], False


def _yaml(value) -> str:
    # JSON is valid YAML for every value written here, and needs no PyYAML.
    return json.dumps(value)


def export_dataset(project: StudioProject) -> ExportResult:
    """Write ``train.txt``, ``val.txt``, ``test.txt`` (if any) and ``data.yaml``.

    Each labelled frame goes where its stored split says.

    Raises:
        DatasetError: Too few labelled frames to train and validate on.
    """
    labelled = [f for f in project.frames if not f.skipped and project.is_labelled(f)]
    if len(labelled) < 2:
        raise DatasetError(
            f"{len(labelled)} frame(s) labelled; label at least 2 to make a dataset "
            f"(a useful {'segmentation' if project.is_segment else 'pose'} model needs "
            "a few hundred)"
        )
    if ensure_splits(project):
        project.save()
    groups = {k: [f for f in labelled if f.split == k] for k in ("train", "val", "test")}
    for key, label in (("train", "training"), ("val", "validation")):
        if not groups[key]:
            raise DatasetError(
                f"No annotated frames in {label}: label some frames from a "
                f"{key} video, or move frames to {key}."
            )
    train, val, test = groups["train"], groups["val"], groups["test"]
    by_video = not leaky_videos(project)

    folder = project.folder
    for name, frames in (("train.txt", train), ("val.txt", val), ("test.txt", test)):
        if not frames:
            (folder / name).unlink(missing_ok=True)  # a stale list from an earlier export
            continue
        # Absolute image paths: Ultralytics finds each label by swapping
        # /images/ for /labels/ in the path, which this layout satisfies.
        lines = "".join(f"{project.image_path(f).resolve().as_posix()}\n" for f in frames)
        atomic_write_bytes(folder / name, lines.encode("utf-8"))

    yaml_lines = [
        f"# Written by GLIDER Animal Annotator for project {_yaml(project.name)}.",
        "# Regenerated on every export; edit studio.json instead.",
        f"path: {_yaml(folder.resolve().as_posix())}",
        "train: train.txt",
        "val: val.txt",
    ]
    if test:
        yaml_lines.append("test: test.txt")
    if not project.is_segment:
        k = len(project.keypoint_names)
        yaml_lines += [
            f"kpt_shape: {_yaml([k, 3])}",
            f"flip_idx: {_yaml(flip_idx(k, project.flip_pairs))}",
        ]
    yaml_lines += [
        "names:",
        *(f"  {i}: {_yaml(n)}" for i, n in enumerate(project.class_names)),
    ]
    if not project.is_segment:
        yaml_lines += [
            # Per class. Ultralytics stores these in the trained .pt, so the
            # model carries its own keypoint names instead of needing them typed.
            "kpt_names:",
            *(f"  {i}: {_yaml(project.keypoint_names)}" for i in range(len(project.class_names))),
        ]
    data_yaml = folder / "data.yaml"
    atomic_write_bytes(data_yaml, ("\n".join(yaml_lines) + "\n").encode("utf-8"))
    return ExportResult(data_yaml, train, val, test, by_video)
