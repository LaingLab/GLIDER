"""The PoseConverter GLIDER asks about sleap-nn folders.

A thin adapter over :mod:`glider_sleap_nn.convert`, which holds the conversion
and runs in a different interpreter entirely -- see :mod:`glider_sleap_nn.env`.

``claims`` and ``is_current`` are asked on *every* model selection, including
for folders belonging to other vendors, so neither may import torch or
sleap-nn. Neither does: both are path and stat work, and the module they call
into keeps its heavy imports inside functions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from glider_sleap_nn import convert as convert_module
from glider_sleap_nn import env as env_module

logger = logging.getLogger(__name__)


class SleapNNConverter:
    """Converts a sleap-nn model folder into ONNX plus a GLIDER sidecar."""

    #: Says which SLEAP, because the answer changes what happens next: one
    #: backend converts in-process, the other downloads an environment first.
    label = "SLEAP (PyTorch)"

    def claims(self, folder: Path) -> bool:
        """True for a folder sleap-nn wrote: a YAML config and a Lightning checkpoint."""
        return convert_module.is_sleap_nn_folder(folder)

    def is_current(self, folder: Path) -> bool:
        return convert_module.is_conversion_current(folder)

    def preflight(self, folder: Path) -> str | None:
        """What the researcher should know before saying yes, or None.

        The first conversion on a machine downloads sleap-nn and torch, which
        takes as long as the lab's connection takes. Starting that behind a wait
        cursor with no warning is how a working conversion gets killed by
        someone who assumes GLIDER has hung.
        """
        if env_module.is_provisioned():
            return None
        return (
            "This is the first sleap-nn model on this machine, so GLIDER will "
            f"set up sleap-nn first — about {env_module.INSTALLED_SIZE_GB} GB, "
            "downloaded once and kept. Expect several minutes.\n\n"
            "It is installed in its own environment, not into GLIDER."
        )

    def convert(self, folder: Path) -> None:
        """Provision if needed, then convert in the sleap-nn environment.

        Both steps run out of process. That is not only about import cost:
        sleap-nn cannot be installed into GLIDER's environment at all, so there
        is no in-process version of this to fall back to.
        """
        folder = Path(folder)
        env_module.provision()
        output = env_module.run_converter(Path(convert_module.__file__), folder)
        logger.info("Converted sleap-nn model %s: %s", folder, output.strip())

    def describe(self, folder: Path) -> dict | None:
        """The last conversion's result, if one was recorded. Diagnostics only."""
        stamp = Path(folder) / convert_module.STAMP_NAME
        try:
            return json.loads(stamp.read_text())
        except (OSError, ValueError):
            return None
