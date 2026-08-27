"""Turn a sleap-nn model folder into ONNX plus the sidecar GLIDER reads.

Two halves, split by what they may import.

Everything above ``convert_sleap_nn_to_onnx`` is path and YAML work: which
folders this plugin claims, whether a conversion is stale, and what the sidecar
should say. That half is imported into GLIDER's own process and asked about
every model the operator selects, whichever vendor wrote it, so it may not
import torch or sleap-nn.

``convert_sleap_nn_to_onnx`` is the other half. It runs in the environment
:mod:`glider_sleap_nn.env` builds, keeps its heavy imports inside the function,
and is runnable as a script -- which is how the parent process calls it.

**What gets exported.** sleap-nn ships an ONNX exporter, and it is used here
rather than a hand-rolled ``torch.onnx.export``. But it is handed the
LightningModule, *not* ``SingleInstanceONNXWrapper``: the wrapper is sleap-nn's
whole inference graph and returns decoded peak coordinates, while GLIDER's
runner wants confidence maps and does its own peak extraction (see
``PoseModelSpec.output_stride`` / ``refine_window``).
``SingleInstanceLightningModule.forward`` returns exactly those confidence
maps, so exporting the module keeps both SLEAP backends producing the same
artifact and leaves GLIDER's decode path untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Written beside the ONNX, recording which checkpoint it came from. A
#: retrained model dropped into the same folder changes the digest, which is
#: what makes a stale conversion detectable rather than silently reused.
STAMP_NAME = ".glider_onnx_source.json"

SIDECAR_NAME = "glider_pose.json"
SIDECAR_SCHEMA_VERSION = 1

#: sleap-nn's own exporter defaults to 17, and its UNet needs nothing newer.
DEFAULT_OPSET = 17

#: The pair that identifies a sleap-nn folder. Classic SLEAP writes
#: ``training_config.json`` beside ``best_model.h5``; the two generations share
#: no filename, so the test is exact rather than heuristic.
CONFIG_NAME = "training_config.yaml"
CHECKPOINT_NAME = "best.ckpt"

#: The head GLIDER runs. sleap-nn writes every head type it knows into the same
#: dict and leaves the unused ones null, so the test is "is anything other than
#: this one populated" rather than a list of the ones to refuse. 0.3.3 ships
#: nine -- single_instance, centroid, centered_instance, bottomup,
#: multi_class_bottomup, multi_class_topdown, bottomup_segmentation,
#: centered_instance_segmentation, semantic_segmentation -- and a deny-list
#: would quietly start converting whatever the next release adds.
SINGLE_INSTANCE = "single_instance"

INSTALL_HINT = (
    "Converting a sleap-nn model needs sleap-nn itself -- a checkpoint is a "
    "PyTorch Lightning save, and rebuilding the network from one needs "
    "sleap-nn's own model classes. GLIDER installs it into an environment of "
    "its own the first time you select a sleap-nn model; this message means "
    "that environment is incomplete. Remove ~/.glider/envs/sleap-nn and try "
    "again, or point GLIDER_SLEAP_NN_ENV at an environment that has sleap-nn."
)


class ConversionError(RuntimeError):
    """Carries a sentence meant for a researcher: it goes on screen verbatim."""


def is_sleap_nn_folder(folder: Path | str) -> bool:
    """True for a folder sleap-nn wrote: a YAML config beside a Lightning checkpoint.

    Both are required. A config without weights describes a model rather than
    being one, and a stray ``.ckpt`` is more likely to be someone else's
    checkpoint than a sleap-nn model.
    """
    folder = Path(folder)
    return (folder / CONFIG_NAME).is_file() and (folder / CHECKPOINT_NAME).is_file()


def _stamp_for(checkpoint: Path) -> dict:
    """What the conversion came from: name, size and mtime of the checkpoint.

    Size and mtime rather than a hash of 23 MB of weights, which would be read
    on every model selection to answer a question about staleness.
    """
    stat = checkpoint.stat()
    return {
        "source": checkpoint.name,
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }


def is_conversion_current(folder: Path | str) -> bool:
    """Whether the ONNX beside the checkpoint was built from *this* checkpoint.

    False for a missing ONNX, a missing sidecar, a missing stamp, or a stamp
    describing a different file. Retrain into the same folder and this goes
    false, which is the point: a conversion that quietly answered with the
    network you just replaced would be the worst outcome available.
    """
    folder = Path(folder)
    checkpoint = folder / CHECKPOINT_NAME
    if not checkpoint.is_file():
        return False
    if not (folder / "model.onnx").is_file() or not (folder / SIDECAR_NAME).is_file():
        return False
    try:
        return json.loads((folder / STAMP_NAME).read_text()) == _stamp_for(checkpoint)
    except (OSError, ValueError):
        return False


def needs_conversion(folder: Path | str) -> bool:
    """Whether *folder* is ours and has work outstanding."""
    return is_sleap_nn_folder(folder) and not is_conversion_current(folder)


def sidecar_for_config(config: dict) -> dict:
    """The ``glider_pose.json`` body for a sleap-nn training config.

    ``divide_by_255`` is deliberately absent here. Whether the exported graph
    normalises its own input is a property of the traced graph rather than of
    the config, so only the conversion can answer it -- see
    :func:`convert_sleap_nn_to_onnx`.
    """
    model = config.get("model_config") or {}
    heads = model.get("head_configs") or {}

    populated = sorted(name for name, head in heads.items() if head)
    others = [name for name in populated if name != SINGLE_INSTANCE]
    if others:
        raise ConversionError(
            f"This sleap-nn model has a {', '.join(others)} head. GLIDER runs "
            "single-instance pose models; converting this one would produce a "
            "network GLIDER decodes as if it tracked one animal."
        )

    confmaps = (heads.get(SINGLE_INSTANCE) or {}).get("confmaps") or {}
    names = [str(n) for n in confmaps.get("part_names") or []]
    if not names:
        raise ConversionError(
            "This config declares no single-instance confidence-map head with "
            "part names, so GLIDER cannot tell what its outputs mean."
        )

    # backbone_config holds one populated entry (unet, convnext, swint) among
    # nulls, so skip the nulls rather than taking the first value.
    backbone = next((v for v in (model.get("backbone_config") or {}).values() if v), {})
    pre = (config.get("data_config") or {}).get("preprocessing") or {}
    version = config.get("sleap_nn_version")

    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "kind": "sleap",
        "onnx": "model.onnx",
        "keypoint_names": names,
        # The only thing in the folder that records which sleap-nn wrote it,
        # which is worth keeping for a package still at 0.x.
        "source_label": f"sleap_nn_{version}" if version else "sleap_nn",
        "input_layout": "NCHW",
        "color_mode": "gray" if pre.get("ensure_grayscale") else "rgb",
        "scale": float(pre.get("scale") or 1.0),
        "output_stride": float(confmaps.get("output_stride") or 1),
        "pad_to_stride": int(backbone.get("max_stride") or 1),
    }


def _needs_unit_range(onnx_path: Path, module, channels: int) -> bool:
    """Whether GLIDER must divide by 255 before feeding this graph.

    sleap-nn's ``normalize_on_gpu`` decides whether to scale by looking at the
    data -- ``elif image.max() > 1.0`` -- which is a Python branch on a tensor
    value. Tracing cannot record a branch, so the exporter froze whichever way
    its dummy input happened to send it, and torch says as much during export:

        TracerWarning: Converting a tensor to a Python boolean might cause the
        trace to be incorrect.

    That makes the answer a property of the *exported graph*, not of the module
    -- the module still has the live branch and will happily normalise input
    the frozen graph would pass straight through. So the graph is asked
    directly, with the module as the reference for what correct looks like:
    feed the graph a 0-255 frame both as-is and divided by 255, and keep
    whichever agrees with the module fed the same 0-255 frame.

    Getting this wrong produces keypoints that look entirely plausible and are
    quietly wrong, which is the worst failure available here. Hence measuring
    the artifact rather than reasoning about the source.
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(0)
    frame = (rng.random((1, channels, 64, 64), dtype=np.float32) * 255.0).astype(np.float32)
    with torch.no_grad():
        reference = module(torch.from_numpy(frame)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def disagreement(fed):
        return float(np.abs(session.run(None, {name: fed})[0] - reference).max())

    as_is = disagreement(frame)
    unit = disagreement(frame / 255.0)
    if min(as_is, unit) > 1e-3:
        raise ConversionError(
            "The exported ONNX does not reproduce the sleap-nn model under "
            f"either input scaling (0-255 differs by {as_is:.3g}, 0-1 by "
            f"{unit:.3g}). GLIDER will not write a sidecar it cannot verify."
        )
    return unit < as_is


def _in_channels(module) -> int:
    """Input channel count for a dummy trace, defaulting to RGB."""
    config = getattr(module, "config", None) or {}
    try:
        backbone = next(v for v in config["model_config"]["backbone_config"].values() if v)
        return int(backbone.get("in_channels") or 3)
    except (KeyError, TypeError, StopIteration):
        return 3


def convert_sleap_nn_to_onnx(model_dir: Path | str, *, opset: int = DEFAULT_OPSET) -> dict:
    """Convert the sleap-nn model in *model_dir* to ONNX plus a GLIDER sidecar.

    Imports torch and sleap_nn, so this runs in the environment
    :mod:`glider_sleap_nn.env` builds and never in GLIDER's own.

    Raises :class:`ConversionError` for everything a caller can act on -- a
    model shape GLIDER cannot run, an unreadable checkpoint, a missing
    environment -- rather than letting a torch traceback reach the UI.
    """
    import yaml

    model_dir = Path(model_dir)
    config = yaml.safe_load((model_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    # Refuse a shape we cannot run before importing a gigabyte of torch.
    sidecar = sidecar_for_config(config)

    try:
        import torch
        from sleap_nn.export import export_to_onnx
        from sleap_nn.export.cli import _load_lightning_model
        from sleap_nn.export.utils import (
            load_training_config,
            resolve_backbone_type,
            resolve_model_type,
        )
    except ImportError as exc:
        raise ConversionError(INSTALL_HINT) from exc

    checkpoint = model_dir / CHECKPOINT_NAME
    try:
        # sleap-nn's own loader, not a hand-rolled `load_from_checkpoint`.
        # The checkpoint carries weights but not hyperparameters, so
        # reconstructing the network needs the whole training config threaded
        # back in as roughly twenty keyword arguments -- backbone, heads,
        # scheduler, hard-keypoint mining, optimiser. Copying that list here
        # would mean re-copying it every time sleap-nn changed one.
        #
        # `_load_lightning_model` is private, which is the cost of this. The
        # `<0.4` pin in env.py is what bounds the risk, and an ImportError here
        # is reported as a plain "environment is incomplete" rather than as a
        # traceback.
        typed = load_training_config(model_dir)
        module = _load_lightning_model(
            model_type=resolve_model_type(typed),
            backbone_type=resolve_backbone_type(typed),
            cfg=typed,
            ckpt_path=checkpoint,
            device="cpu",
        )
    except Exception as exc:
        raise ConversionError(
            f"{checkpoint.name} could not be loaded as a sleap-nn single-instance "
            f"model ({exc}). GLIDER converts single-instance sleap-nn checkpoints; "
            "if this folder was written by a different sleap-nn version than the "
            "one GLIDER installed, that is the likeliest cause."
        ) from exc
    module.eval()

    pre = (config.get("data_config") or {}).get("preprocessing") or {}
    height = int(pre.get("max_height") or 512)
    width = int(pre.get("max_width") or 512)
    channels = _in_channels(module)

    onnx_path = model_dir / "model.onnx"
    try:
        export_to_onnx(
            module,
            onnx_path,
            input_shape=(1, channels, height, width),
            # float32 rather than the exporter's uint8 default: that default
            # matches sleap-nn's own inference wrappers, and this is the bare
            # confidence-map module instead.
            input_dtype=torch.float32,
            opset_version=opset,
            verify=True,
            # The one check that catches a graph which traces cleanly and is
            # numerically wrong. env.py installs onnxruntime so it cannot
            # silently degrade to a warning.
            numerical_check=True,
        )
    except Exception as exc:
        raise ConversionError(
            f"sleap-nn could not export {checkpoint.name} to ONNX: {exc}"
        ) from exc

    sidecar["divide_by_255"] = _needs_unit_range(onnx_path, module, channels)
    (model_dir / SIDECAR_NAME).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    # The stamp is written last, so an interrupted conversion leaves a folder
    # that reads as stale rather than as current and half-written.
    (model_dir / STAMP_NAME).write_text(
        json.dumps(_stamp_for(checkpoint), indent=2), encoding="utf-8"
    )
    return sidecar


def main(argv: list[str] | None = None) -> int:
    """Convert one folder, printing only an actionable sentence on failure.

    The parent process shows stderr verbatim, so a traceback here becomes a
    traceback in a dialog.
    """
    parser = argparse.ArgumentParser(description="Convert a sleap-nn model to ONNX.")
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = convert_sleap_nn_to_onnx(args.model_dir)
    except ConversionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
