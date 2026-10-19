"""Which cameras in a rig are switched on.

A rig is rarely "every camera the computer can see": a laptop's built-in
webcam enumerates alongside the arena cameras, and a sixteen-camera rig running
a four-animal session has twelve cameras spending USB bandwidth on empty
arenas. A camera switched off here is not connected at all, so it is neither
previewed nor recorded and its bandwidth goes to the cameras that matter.

What is stored is the cameras switched *off*, so a camera plugged in later
appears switched on, as every camera did before this existed. Persisted in
``~/.glider/camera_selection.json`` for the same reason the names are: arena 3
is arena 3 next week, and re-picking cameras every session is how one gets
forgotten.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["CameraSelection"]

SELECTION_FILENAME = "camera_selection.json"


class CameraSelection:
    """The cameras the operator has switched off, remembered between sessions."""

    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path is not None else self._default_path()
        self._disabled: set[str] = set()
        self.load()

    @staticmethod
    def _default_path() -> Path:
        from glider.core.config import get_config

        return get_config().paths.user_config_dir / SELECTION_FILENAME

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read the stored selection. A missing or unreadable file means every
        camera is on - never an error, because the window has to open regardless."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._disabled = set()
            return
        disabled = data.get("disabled") if isinstance(data, dict) else None
        self._disabled = {str(c) for c in disabled} if isinstance(disabled, list) else set()

    def save(self) -> bool:
        """Write the selection. False if it could not be stored.

        Best-effort, like the camera names: a read-only home directory must not
        stop anyone recording. The cost is re-picking cameras next session.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"disabled": sorted(self._disabled)}, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("could not save camera selection to %s: %s", self._path, e)
            return False
        return True

    # ------------------------------------------------------------------

    def is_enabled(self, camera_id: str) -> bool:
        return camera_id not in self._disabled

    def set_enabled(self, camera_id: str, enabled: bool) -> None:
        """Switch a camera on or off, and store it."""
        if enabled:
            self._disabled.discard(camera_id)
        else:
            self._disabled.add(camera_id)
        self.save()

    def disabled(self) -> set[str]:
        return set(self._disabled)
