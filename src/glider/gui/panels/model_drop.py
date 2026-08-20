"""Route a path dropped on the camera panel to the slot it belongs in.

:func:`classify_drop` is pure and Qt-free: deciding *what* a dropped path is
should be testable without a running application, so the widget keeps only the
Qt event plumbing.

Classification is by shape alone — extension, or a marker file inside a folder.
Nothing here opens a model or parses a config; that happens later, on the
worker thread, because reading a ``.pt`` pulls in torch and takes seconds.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from glider.vision.pose.batch import VIDEO_EXTS
from glider.vision.pose.spec import SIDECAR_NAME

#: Files a pose model can arrive as directly.
_POSE_SUFFIXES = frozenset({".pt", ".onnx"})

#: Marker files that make a *folder* an exported pose model.
_POSE_MARKERS = (SIDECAR_NAME, "pose_cfg.yaml", "pytorch_config.yaml", "training_config.json")

_BEHAVIOR_SUFFIXES = frozenset({".pkl"})


class DropKind(Enum):
    """What a dropped path should be treated as."""

    POSE_MODEL = "pose_model"
    BEHAVIOR_MODEL = "behavior_model"
    VIDEO = "video"
    UNKNOWN = "unknown"


def classify_drop(path: str | Path) -> DropKind:
    """Classify *path* into the panel slot it belongs in.

    A folder containing only an ``.onnx`` still classifies as a pose model, so
    the drop handler can explain that its sidecar is missing. Returning UNKNOWN
    there would silently ignore the drop and leave the operator guessing.
    """
    path = Path(path)

    if path.is_dir():
        if any((path / marker).is_file() for marker in _POSE_MARKERS):
            return DropKind.POSE_MODEL
        if any(p.suffix.lower() == ".onnx" for p in path.iterdir() if p.is_file()):
            return DropKind.POSE_MODEL
        return DropKind.UNKNOWN

    if not path.is_file():
        return DropKind.UNKNOWN

    suffix = path.suffix.lower()
    if suffix in _POSE_SUFFIXES:
        return DropKind.POSE_MODEL
    if suffix in _BEHAVIOR_SUFFIXES:
        return DropKind.BEHAVIOR_MODEL
    if suffix in VIDEO_EXTS:
        return DropKind.VIDEO
    return DropKind.UNKNOWN


def paths_from_mime(mime) -> list[Path]:
    """Local filesystem paths carried by a Qt drop event's mime data.

    Remote URLs are dropped: a drag from a browser carries an ``http:`` URL that
    has no local path, and silently treating it as a filename would produce a
    confusing "no such file" much later.
    """
    if not mime.hasUrls():
        return []
    out = []
    for url in mime.urls():
        if url.isLocalFile():
            out.append(Path(url.toLocalFile()))
    return out


def route_drops(paths) -> dict[DropKind, Path]:
    """First path of each kind, keyed by kind; later duplicates are ignored.

    One drop can carry several files. Filling one slot per kind means dropping
    a model folder and a video together does the obvious thing, while dropping
    two videos does not leave the operator guessing which one won.
    """
    routed: dict[DropKind, Path] = {}
    for path in paths:
        kind = classify_drop(path)
        if kind is not DropKind.UNKNOWN and kind not in routed:
            routed[kind] = path
    return routed
