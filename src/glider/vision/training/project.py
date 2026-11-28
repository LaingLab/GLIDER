"""A Animal Annotator project: frames, their labels, and what they are labelled with.

On disk it is a plain YOLO dataset plus one JSON file::

    <folder>/
      studio.json          keypoints, colours, layout, videos, frame provenance
      images/<name>.jpg    every extracted frame
      labels/<name>.txt    YOLO pose or segmentation labels, written as you label
      data.yaml, train.txt, val.txt   written by :func:`~.dataset.export_dataset`

Images and labels stay in one folder each rather than being moved between
``train/`` and ``val/``: the split is a pair of lists written at export, so it
can change (more videos, a new seed) without touching a file you labelled.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from glider.core.fileio import atomic_write_bytes
from glider.vision.training.labels import (
    PoseInstance,
    SegInstance,
    read_labels,
    read_seg_labels,
    write_labels,
    write_seg_labels,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PROJECT_FILE",
    "STUDIO_VERSION",
    "TASKS",
    "TASK_POSE",
    "TASK_SEGMENT",
    "FrameRecord",
    "StudioProject",
    "StudioProjectError",
    "flip_pairs_from_names",
]

PROJECT_FILE = "studio.json"
STUDIO_VERSION = 1

#: What a project labels: keypoints (YOLO pose) or outlines (YOLO instance
#: segmentation). Fixed when the project is made; the label files differ.
TASK_POSE = "pose"
TASK_SEGMENT = "segment"
TASKS = (TASK_POSE, TASK_SEGMENT)

#: Files Add Images treats as videos to pull frames from.
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg")
#: Files Add Images copies in as they are.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

#: Left/right spellings recognised when pairing keypoints for flip augmentation.
_SIDES = (("left", "right"), ("l", "r"))


class StudioProjectError(ValueError):
    """A studio project could not be read or made."""


def _check_class_names(names: list[str]) -> list[str]:
    names = [str(n).strip() for n in names]
    if not names or any(not n for n in names):
        raise StudioProjectError("a project needs at least one class, and names can't be blank")
    if len(set(names)) != len(names):
        raise StudioProjectError("class names must be unique")
    return names


def flip_pairs_from_names(names: list[str]) -> list[tuple[int, int]]:
    """Keypoints that swap when an image is mirrored left-to-right.

    Ultralytics mirrors images during training; without these pairs a mirrored
    left ear would still be labelled "left_ear" and the model would learn that
    sides are arbitrary. Recognises ``left_x``/``right_x``, ``x_left``/``x_right``
    and the ``l_``/``r_`` short forms, in any case.
    """
    index = {n.lower(): i for i, n in enumerate(names)}
    pairs: list[tuple[int, int]] = []
    for name, i in index.items():
        for left, right in _SIDES:
            for pattern in (rf"^{left}([_\- ])(.+)$", rf"^(.+)([_\- ]){left}$"):
                match = re.match(pattern, name)
                if not match:
                    continue
                other = (
                    f"{right}{match.group(1)}{match.group(2)}"
                    if pattern.startswith(rf"^{left}")
                    else f"{match.group(1)}{match.group(2)}{right}"
                )
                j = index.get(other)
                if j is not None and (i, j) not in pairs:
                    pairs.append((i, j))
    return pairs


@dataclass
class FrameRecord:
    """One extracted frame and where it came from.

    ``name`` is the image's file stem, unique in the project. ``video`` is the
    key of the video in :attr:`StudioProject.videos`.
    """

    name: str
    video: str
    frame: int
    skipped: bool = False
    #: "train", "val" or "test"; "" until assigned.
    split: str = ""
    #: The original file name of an imported image; "" for a video frame.
    source: str = ""

    def to_dict(self) -> dict:
        data = {"name": self.name, "video": self.video, "frame": self.frame}
        if self.skipped:
            data["skipped"] = True
        if self.split:
            data["split"] = self.split
        if self.source:
            data["source"] = self.source
        return data

    @classmethod
    def from_dict(cls, data: dict) -> FrameRecord:
        split = str(data.get("split", ""))
        return cls(
            name=str(data["name"]),
            video=str(data["video"]),
            frame=int(data["frame"]),
            skipped=bool(data.get("skipped", False)),
            split=split if split in ("train", "val", "test") else "",
            source=str(data.get("source", "")),
        )


@dataclass
class StudioProject:
    """A labelling project. Create with :meth:`create`, reopen with :meth:`load`."""

    folder: Path
    name: str
    keypoint_names: list[str]
    task: str = "pose"
    class_names: list[str] = field(default_factory=lambda: ["animal"])
    flip_pairs: list[tuple[int, int]] = field(default_factory=list)
    #: RGB per keypoint, from the skeleton template. Empty: the canvas's default.
    keypoint_colours: list[tuple[int, int, int]] = field(default_factory=list)
    #: The template the project was made from, for display only.
    template_name: str = ""
    #: Reference (x, y) per keypoint, 0-1, from the template: the shape a
    #: whole skeleton is placed in before its keypoints are adjusted.
    keypoint_layout: list[tuple[float, float]] = field(default_factory=list)
    #: Video key -> absolute path. The key prefixes every frame name from it.
    videos: dict[str, str] = field(default_factory=dict)
    frames: list[FrameRecord] = field(default_factory=list)
    val_fraction: float = 0.2
    seed: int = 0
    #: Box growth around the keypoints; see :meth:`PoseInstance.box`.
    box_padding: float = 0.2

    # ------------------------------------------------------------------
    # creating and opening
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        folder: Path | str,
        name: str,
        keypoint_names: list[str],
        *,
        colours: list[tuple[int, int, int]] | None = None,
        template_name: str = "",
        layout: list[tuple[float, float]] | None = None,
        task: str = TASK_POSE,
        class_names: list[str] | None = None,
    ) -> StudioProject:
        """Start a new project in ``folder``, which must be empty or not exist.

        A :data:`TASK_SEGMENT` project labels outlines, so ``keypoint_names``
        may be empty; ``class_names`` are the kinds of thing outlined.
        """
        folder = Path(folder)
        if (folder / PROJECT_FILE).exists():
            raise StudioProjectError(f"{folder} already holds a studio project")
        if task not in TASKS:
            raise StudioProjectError(f"unknown task {task!r}")
        if task == TASK_POSE and not keypoint_names:
            raise StudioProjectError("a pose project needs at least one keypoint")
        if len(set(keypoint_names)) != len(keypoint_names):
            raise StudioProjectError("keypoint names must be unique")
        classes = _check_class_names(class_names if class_names is not None else ["animal"])
        project = cls(
            folder=folder,
            name=name.strip() or folder.name,
            task=task,
            class_names=classes,
            keypoint_names=list(keypoint_names),
            flip_pairs=flip_pairs_from_names(keypoint_names),
            keypoint_colours=[tuple(c) for c in colours or []],
            template_name=template_name,
            keypoint_layout=[(float(x), float(y)) for x, y in layout or []],
        )
        project.images_dir.mkdir(parents=True, exist_ok=True)
        project.labels_dir.mkdir(parents=True, exist_ok=True)
        project.save()
        return project

    @classmethod
    def load(cls, folder: Path | str) -> StudioProject:
        """Open the project in ``folder``."""
        folder = Path(folder)
        path = folder / PROJECT_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise StudioProjectError(f"cannot read {path}: {e}") from e
        version = data.get("studio_version")
        if version != STUDIO_VERSION:
            raise StudioProjectError(
                f"studio_version {version!r}; this build understands {STUDIO_VERSION}"
            )
        task = str(data.get("task", TASK_POSE))
        if task not in TASKS:
            raise StudioProjectError(f"{path} is a {task!r} project; this build knows {TASKS}")
        try:
            return cls(
                folder=folder,
                name=str(data["name"]),
                task=task,
                keypoint_names=[str(n) for n in data.get("keypoint_names", [])],
                class_names=[str(n) for n in data.get("class_names", ["animal"])],
                flip_pairs=[(int(a), int(b)) for a, b in data.get("flip_pairs", [])],
                keypoint_colours=[
                    tuple(int(c) for c in rgb[:3]) for rgb in data.get("keypoint_colors", [])
                ],
                template_name=str(data.get("template", "")),
                keypoint_layout=[(float(x), float(y)) for x, y in data.get("keypoint_layout", [])],
                videos={str(k): str(v) for k, v in data.get("videos", {}).items()},
                frames=[FrameRecord.from_dict(f) for f in data.get("frames", [])],
                val_fraction=float(data.get("val_fraction", 0.2)),
                seed=int(data.get("seed", 0)),
                box_padding=float(data.get("box_padding", 0.2)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise StudioProjectError(f"malformed {path}: {e}") from e

    def to_dict(self) -> dict:
        return {
            "studio_version": STUDIO_VERSION,
            "name": self.name,
            "task": self.task,
            "keypoint_names": self.keypoint_names,
            "class_names": self.class_names,
            "flip_pairs": [list(p) for p in self.flip_pairs],
            "keypoint_colors": [list(c) for c in self.keypoint_colours],
            "template": self.template_name,
            "keypoint_layout": [[round(x, 4), round(y, 4)] for x, y in self.keypoint_layout],
            "videos": self.videos,
            "frames": [f.to_dict() for f in self.frames],
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "box_padding": self.box_padding,
        }

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        atomic_write_bytes(self.folder / PROJECT_FILE, text.encode("utf-8"))

    def apply_template(
        self,
        names: list[str],
        colours: list[tuple[int, int, int]],
        template_name: str = "",
        layout: list[tuple[float, float]] | None = None,
    ) -> None:
        """Relabel, recolour and reshape the keypoints, then save.

        Names, colours and layout can change at any time. The *number* of
        keypoints cannot once a frame is labelled: label files store keypoints
        by position, so adding or removing one would shift every label after it.

        Raises:
            StudioProjectError: The count changed with labels on disk, or the
                names are blank or repeated.
        """
        if len(set(names)) != len(names) or any(not n.strip() for n in names):
            raise StudioProjectError("keypoint names must be unique and not blank")
        if len(names) != len(self.keypoint_names) and any(self.is_labelled(f) for f in self.frames):
            raise StudioProjectError(
                f"this project has labelled frames with {len(self.keypoint_names)} keypoints; "
                "the number of keypoints can't change now (start a new project instead)"
            )
        self.keypoint_names = list(names)
        self.keypoint_colours = [tuple(c) for c in colours]
        self.flip_pairs = flip_pairs_from_names(names)
        if template_name:
            self.template_name = template_name
        if layout is not None:
            self.keypoint_layout = [(float(x), float(y)) for x, y in layout]
        self.save()

    @property
    def is_segment(self) -> bool:
        return self.task == TASK_SEGMENT

    def set_class_names(self, names: list[str]) -> None:
        """Rename or add classes, then save.

        Like keypoints, classes are stored by position in the label files, so
        once a frame is labelled a class can be renamed or added at the end
        but not removed.

        Raises:
            StudioProjectError: Names are blank or repeated, or a class would
                be removed with labels on disk.
        """
        names = _check_class_names(names)
        if len(names) < len(self.class_names) and any(self.is_labelled(f) for f in self.frames):
            raise StudioProjectError(
                "classes can't be removed once frames are labelled; rename them instead"
            )
        self.class_names = names
        self.save()

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    @property
    def images_dir(self) -> Path:
        return self.folder / "images"

    @property
    def labels_dir(self) -> Path:
        return self.folder / "labels"

    def image_path(self, frame: FrameRecord) -> Path:
        return self.images_dir / f"{frame.name}.jpg"

    def label_path(self, frame: FrameRecord) -> Path:
        return self.labels_dir / f"{frame.name}.txt"

    # ------------------------------------------------------------------
    # videos and frames
    # ------------------------------------------------------------------

    def video_key(self, video: Path | str) -> str:
        """The key for ``video``, registering it if it is new.

        Keys are the file stem, made filename-safe; a second video with the
        same stem from another folder gets ``-2``, ``-3`` and so on, so frame
        names never collide.
        """
        path = str(Path(video).resolve())
        for key, known in self.videos.items():
            if known == path:
                return key
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video).stem).strip("._") or "video"
        key, n = base, 2
        while key in self.videos:
            key, n = f"{base}-{n}", n + 1
        self.videos[key] = path
        return key

    def frame_name(self, video_key: str, frame: int) -> str:
        return f"{video_key}_f{frame:06d}"

    def has_frame(self, video_key: str, frame: int) -> bool:
        return any(f.video == video_key and f.frame == frame for f in self.frames)

    def remove_video(self, video_key: str) -> int:
        """Forget a video and its frames, deleting their images and labels.

        Returns how many frames went. Labels are work, so the caller should
        confirm before calling this on a video with labelled frames.
        """
        gone = [f for f in self.frames if f.video == video_key]
        for frame in gone:
            self.image_path(frame).unlink(missing_ok=True)
            self.label_path(frame).unlink(missing_ok=True)
        self.frames = [f for f in self.frames if f.video != video_key]
        self.videos.pop(video_key, None)
        return len(gone)

    def remove_frames(self, names: list[str] | set[str]) -> int:
        """Delete frames by name, with their images and labels. Returns how many went.

        A folder of imported images left with no images is forgotten too; a
        video stays registered, so Add Images can still top it up. The caller
        saves, and should confirm first when labelled frames are among them.
        """
        names = set(names)
        gone = [f for f in self.frames if f.name in names]
        for frame in gone:
            self.image_path(frame).unlink(missing_ok=True)
            self.label_path(frame).unlink(missing_ok=True)
        self.frames = [f for f in self.frames if f.name not in names]
        used = {f.video for f in self.frames}
        for key in [k for k in self.videos if k not in used and self.is_image_folder(k)]:
            del self.videos[key]
        return len(gone)

    def is_image_folder(self, video_key: str) -> bool:
        """True when the key is a folder of imported images, not a video."""
        path = self.videos.get(video_key, "")
        return bool(path) and Path(path).suffix.lower() not in VIDEO_SUFFIXES

    def frame_by_name(self, name: str) -> FrameRecord | None:
        return next((f for f in self.frames if f.name == name), None)

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------

    def is_labelled(self, frame: FrameRecord) -> bool:
        return self.label_path(frame).exists()

    def read_instances(self, frame: FrameRecord) -> list[PoseInstance] | list[SegInstance]:
        """The frame's labels: :class:`SegInstance` in a segmentation project."""
        if self.is_segment:
            return read_seg_labels(self.label_path(frame))
        return read_labels(self.label_path(frame), len(self.keypoint_names))

    def write_instances(
        self, frame: FrameRecord, instances: list[PoseInstance] | list[SegInstance]
    ) -> bool:
        if self.is_segment:
            return write_seg_labels(self.label_path(frame), instances)
        return write_labels(self.label_path(frame), instances, self.box_padding)

    def counts(self) -> dict[str, int]:
        """How many frames are labelled, skipped, and still to do."""
        labelled = skipped = 0
        for frame in self.frames:
            if frame.skipped:
                skipped += 1
            elif self.is_labelled(frame):
                labelled += 1
        return {
            "total": len(self.frames),
            "labelled": labelled,
            "skipped": skipped,
            "todo": len(self.frames) - labelled - skipped,
        }
