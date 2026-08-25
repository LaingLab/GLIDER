"""The PoseConverter GLIDER asks about SLEAP folders.

A thin adapter over :mod:`glider_sleap.convert`, which holds the actual
conversion and deliberately imports nothing from glider so it can also be run
as a standalone script.

The split matters for one reason: ``claims`` and ``is_current`` are asked on
*every* model selection, including for folders belonging to other vendors, so
neither may import TensorFlow. Only ``convert`` does, and it does so in a child
process.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from glider_sleap import convert as convert_module

logger = logging.getLogger(__name__)

#: How long a conversion may run before it is assumed wedged. Generous: a large
#: model on a slow machine is minutes, and killing a working conversion is
#: worse than waiting.
TIMEOUT_S = 900


class SleapConverter:
    """Converts a SLEAP model folder into ONNX plus a GLIDER sidecar."""

    label = "SLEAP"

    def claims(self, folder: Path) -> bool:
        """True for a folder SLEAP wrote: a config beside a Keras checkpoint.

        Both are required. ``training_config.json`` alone is an exported folder
        with nothing left to convert, and a stray ``.h5`` alone is not a SLEAP
        model.
        """
        folder = Path(folder)
        return (folder / "training_config.json").is_file() and (
            convert_module.find_sleap_checkpoint(folder) is not None
        )

    def is_current(self, folder: Path) -> bool:
        return not convert_module.needs_conversion(folder)

    def convert(self, folder: Path) -> None:
        """Convert in a child process, and raise with a readable message.

        A subprocess rather than a call: TensorFlow costs seconds to import and
        permanently claims threads, and would then sit in the application for
        the rest of the session for the sake of a one-time job. A child gives it
        back on exit.
        """
        folder = Path(folder)
        script = Path(convert_module.__file__)
        try:
            completed = subprocess.run(
                [sys.executable, str(script), str(folder)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Converting {folder.name} took longer than "
                f"{TIMEOUT_S // 60} minutes and was stopped."
            ) from exc

        if completed.returncode != 0:
            # The child writes only the actionable message to stderr, so this is
            # a sentence about a missing dependency or an unreadable checkpoint
            # rather than a TensorFlow traceback.
            message = (completed.stderr or "").strip() or "Conversion failed."
            raise RuntimeError(message)

        logger.info("Converted SLEAP model %s", folder)

    def describe(self, folder: Path) -> dict | None:
        """The last conversion's result, if one was recorded. Diagnostics only."""
        stamp = Path(folder) / convert_module.STAMP_NAME
        try:
            return json.loads(stamp.read_text())
        except (OSError, ValueError):
            return None
