"""Turning a vendor's trained model into something GLIDER can run.

GLIDER runs pose models through onnxruntime, described by a ``glider_pose.json``
sidecar. Getting from what DeepLabCut or SLEAP *writes* to that pair is a
per-vendor job, and it is the expensive part: it needs the vendor's own
framework, or at least a large one, and each vendor pins dependencies that
disagree with each other and with GLIDER.

So core does not do it. Core runs ONNX and reads sidecars; a plugin recognises
one vendor's folder and produces that pair. A lab that uses neither DeepLabCut
nor SLEAP installs neither plugin and carries neither dependency tree, which is
the point -- TensorFlow is a large thing to hand someone who tracks with YOLO.

A converter registers itself the way every other plugin component does::

    # in your plugin's __init__.py
    POSE_CONVERTERS = {"sleap": SleapConverter}

and answers three questions about a folder: is it mine, is what I produced from
it still current, and please convert it.

The contract is deliberately about *folders and files*, not about models. A
converter is asked ``claims()`` on every model selection, so it must answer from
paths alone -- importing TensorFlow to decide whether a folder is a SLEAP folder
would cost seconds on the common path where the answer is no.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Registered pose converters, ``{name: converter class}``. Written by
#: PluginManager from a plugin's ``POSE_CONVERTERS`` table or a ``glider.pose``
#: entry point, exactly as devices and nodes are registered.
POSE_CONVERTERS: dict[str, type] = {}


@runtime_checkable
class PoseConverter(Protocol):
    """What a plugin must provide to make a vendor's models loadable.

    The three methods below are required. There is one *optional* hook,
    deliberately kept off this Protocol so that adding it later cannot make an
    existing converter stop satisfying the check:

        def preflight(self, folder: Path) -> str | None

    See :func:`converter_preflight`.
    """

    #: Shown to a person while this runs, e.g. "SLEAP". Falls back to the
    #: registry name when absent.
    label: str

    def claims(self, folder: Path) -> bool:
        """Whether *folder* holds a model this converter handles.

        Must be answerable from paths alone and must not import the heavy
        framework: this is called on every model selection, including for
        folders belonging to other vendors.
        """
        ...

    def is_current(self, folder: Path) -> bool:
        """Whether a previous conversion is still valid for what is there now.

        False after the researcher retrains and drops a new checkpoint in.
        Returning True on a stale conversion is the worst answer available: the
        model runs, and quietly answers with the network they just replaced.
        """
        ...

    def convert(self, folder: Path) -> None:
        """Write ``model.onnx`` and ``glider_pose.json`` into *folder*.

        May take minutes and may need to download or build an environment.
        Raise with a message aimed at a person -- it goes on screen.
        """
        ...


def converter_preflight(converter: PoseConverter, folder: Path) -> str | None:
    """What the researcher should know before agreeing to a conversion.

    Optional, and most converters have nothing to say: a few seconds behind a
    wait cursor needs no warning. It exists for the ones where the first run
    costs something the person would want to decide about -- glider-dlc
    downloads over a gigabyte the first time -- because starting that silently
    is how a working conversion gets killed by someone who assumes GLIDER hung.

    A converter that raises here is answered with None. Third-party code
    failing to write a sentence must not stop the conversion it describes.
    """
    hook = getattr(converter, "preflight", None)
    if hook is None:
        return None
    try:
        note = hook(Path(folder))
    except Exception:
        logger.exception("Pose converter raised in preflight for %s", folder)
        return None
    return str(note) if note else None


def register_converter(name: str, converter: type) -> None:
    """Register a converter under *name* (idempotent for the same class)."""
    existing = POSE_CONVERTERS.get(name)
    if existing is converter:
        return
    if existing is not None:
        logger.warning(
            "Pose converter %r is already registered to %s; keeping the first",
            name,
            existing,
        )
        return
    POSE_CONVERTERS[name] = converter


def find_converter(folder: Path | str) -> PoseConverter | None:
    """The converter that claims *folder*, or None.

    Instantiates the class, so a converter's ``__init__`` must be cheap and
    take no arguments -- it is constructed just to be asked a question.

    When several claim the same folder the first is used and the rest are
    logged. Two converters claiming one folder means two plugins disagree about
    whose model it is, which is worth seeing rather than resolving silently.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None

    claimants: list[tuple[str, PoseConverter]] = []
    for name, cls in POSE_CONVERTERS.items():
        try:
            converter = cls()
            if converter.claims(folder):
                claimants.append((name, converter))
        except Exception:
            # A third-party converter is third-party code. One that raises must
            # not stop the others from being asked, or stop the model loading.
            logger.exception("Pose converter %r raised while inspecting %s", name, folder)

    if not claimants:
        return None
    if len(claimants) > 1:
        logger.warning(
            "%s is claimed by %d pose converters (%s); using %r",
            folder,
            len(claimants),
            ", ".join(n for n, _ in claimants),
            claimants[0][0],
        )
    return claimants[0][1]


def needs_conversion(folder: Path | str) -> PoseConverter | None:
    """The converter that both claims *folder* and has work to do, or None.

    The question the model-selection path asks: is this a folder somebody can
    handle, and does it need handling before it can be run?
    """
    converter = find_converter(folder)
    if converter is None:
        return None
    try:
        if converter.is_current(Path(folder)):
            return None
    except Exception:
        logger.exception("Pose converter raised while checking %s; assuming stale", folder)
    return converter


def converter_label(converter: PoseConverter) -> str:
    """A human-facing name for *converter*."""
    label = getattr(converter, "label", "")
    if label:
        return str(label)
    for name, cls in POSE_CONVERTERS.items():
        if isinstance(converter, cls):
            return name
    return type(converter).__name__
