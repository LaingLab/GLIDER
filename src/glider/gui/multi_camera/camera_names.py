"""Operator-chosen names for the cameras in a rig.

``cam_3`` is an enumeration order, not a place. On an eight-arena rig the
question is always "which arena is this", and the index answers it only for
whoever plugged the cables in - and only until something is replugged.

A name is worth having only if it reaches the recordings. Eight files called
``_cam0`` through ``_cam7`` leave the same question unanswered a month later,
so :func:`slug` exists to put the name in the filename, and the recorder takes
these labels rather than deriving a number from the camera id.

Names persist in ``~/.glider/camera_labels.json``. A rig is physically stable -
arena 3 is arena 3 next week - so retyping them each session would be busywork,
and getting them wrong is how a file ends up attributed to the wrong animal.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["CameraLabels", "slug"]

#: Where labels live, unless a caller says otherwise.
LABELS_FILENAME = "camera_labels.json"

#: Longest label kept. Long enough for "arena 3 back left", short enough that a
#: filename built from eight of them stays inside a path limit.
MAX_LABEL = 48

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(label: str) -> str:
    """A label as a filename fragment.

    Windows path rules are the binding constraint, not taste: a label like
    ``arena 3 / back`` is a perfectly good thing to type and cannot appear in a
    filename. Returns ``""`` when nothing usable survives, which callers treat
    as "no label" rather than writing a file with an empty segment.
    """
    cleaned = _UNSAFE.sub("_", (label or "").strip()).strip("._-")
    return cleaned[:MAX_LABEL]


class CameraLabels:
    """The names given to each camera, remembered between sessions."""

    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path is not None else self._default_path()
        self._labels: dict[str, str] = {}
        self.load()

    @staticmethod
    def _default_path() -> Path:
        from glider.core.config import get_config

        return get_config().paths.user_config_dir / LABELS_FILENAME

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read the stored labels. A missing or unreadable file is simply no
        labels - never an error, because the window has to open regardless."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._labels = {}
            return
        if isinstance(data, dict):
            self._labels = {str(k): str(v)[:MAX_LABEL] for k, v in data.items() if str(v).strip()}

    def save(self) -> bool:
        """Write the labels. False if they could not be stored.

        Best-effort: a read-only home directory must not stop anyone recording.
        The cost of failure is retyping names, not losing data.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._labels, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError as e:
            logger.warning("could not save camera labels to %s: %s", self._path, e)
            return False
        return True

    # ------------------------------------------------------------------

    def label(self, camera_id: str) -> str:
        """The name for *camera_id*, or ``""`` when it has not been named."""
        return self._labels.get(camera_id, "")

    def display_name(self, camera_id: str) -> str:
        """What to put on the tile: the name if there is one, else the id.

        Never returns an empty string. A tile with no caption is worse than one
        captioned with the id it already had.
        """
        return self._labels.get(camera_id) or camera_id

    def set_label(self, camera_id: str, label: str) -> str:
        """Name a camera, and store it. Returns the name actually kept.

        Clearing a name is meaningful - it puts the tile back to its id - so a
        blank is stored as a removal rather than as an empty label that would
        render as nothing.
        """
        cleaned = (label or "").strip()[:MAX_LABEL]
        if cleaned:
            self._labels[camera_id] = cleaned
        else:
            self._labels.pop(camera_id, None)
        self.save()
        return cleaned

    def file_fragment(self, camera_id: str) -> str:
        """The filename fragment for this camera: its slugged name, else its id.

        This is what makes naming worth doing. Without it the operator names a
        camera in the window and still gets ``_cam5`` on disk.
        """
        return slug(self._labels.get(camera_id, "")) or slug(camera_id)

    def as_dict(self) -> dict[str, str]:
        return dict(self._labels)

    def __len__(self) -> int:
        return len(self._labels)
