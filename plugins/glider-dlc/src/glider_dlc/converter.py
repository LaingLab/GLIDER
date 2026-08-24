"""The PoseConverter GLIDER asks about DeepLabCut folders.

A thin adapter over :mod:`glider_dlc.convert`, which holds the conversion and
runs in a different interpreter entirely -- see :mod:`glider_dlc.env`.

``claims`` and ``is_current`` are asked on *every* model selection, including
for folders belonging to other vendors, so neither may import torch or
deeplabcut. Neither does: both are path and hash work, and the module they call
into keeps its heavy imports inside functions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from glider_dlc import convert as convert_module
from glider_dlc import env as env_module

logger = logging.getLogger(__name__)


class DlcConverter:
    """Converts a DeepLabCut 3.x model folder into ONNX plus a GLIDER sidecar."""

    label = "DeepLabCut"

    def claims(self, folder: Path) -> bool:
        """True for a folder DeepLabCut 3.x wrote: a PyTorch config and a snapshot.

        Both are required. A config without weights is a description of a model
        rather than a model, and a stray ``.pt`` is more likely to be someone
        else's checkpoint than a DLC snapshot.
        """
        return convert_module.is_dlc_folder(folder)

    def is_current(self, folder: Path) -> bool:
        return convert_module.is_conversion_current(folder)

    def preflight(self, folder: Path) -> str | None:
        """What the researcher should know before saying yes, or None.

        The first conversion on a machine downloads DeepLabCut, which is over a
        gigabyte and takes as long as the lab's connection takes. Starting that
        behind a wait cursor with no warning is how a working conversion gets
        killed by someone who assumes GLIDER has hung.
        """
        if env_module.is_provisioned():
            return None
        return (
            "This is the first DeepLabCut model on this machine, so GLIDER "
            "will set up DeepLabCut first — about "
            f"{env_module.INSTALLED_SIZE_GB} GB, downloaded once and kept. "
            "Expect several minutes.\n\n"
            "It is installed in its own environment, not into GLIDER."
        )

    def convert(self, folder: Path) -> None:
        """Provision if needed, then convert in the DeepLabCut environment.

        Both steps run out of process. That is not only about import cost:
        DeepLabCut cannot be installed into GLIDER's environment at all, so
        there is no in-process version of this to fall back to.
        """
        folder = Path(folder)
        env_module.provision()
        output = env_module.run_converter(Path(convert_module.__file__), folder)
        logger.info("Converted DeepLabCut model %s: %s", folder, output.strip())

    def describe(self, folder: Path) -> dict | None:
        """The last conversion's result, if one was recorded. Diagnostics only."""
        config = convert_module.find_dlc_config(folder)
        if config is None:
            return None
        try:
            return json.loads((config.parent / convert_module.STAMP_NAME).read_text())
        except (OSError, ValueError):
            return None
