"""Turn a DeepLabCut 3.x model folder into ONNX plus a GLIDER sidecar.

This module runs **inside the provisioned DeepLabCut environment**, not inside
GLIDER's, and is executed as a script by absolute path. It therefore imports
nothing from ``glider`` and nothing from ``glider_dlc`` -- the interpreter
running it has neither installed. It can also be copied into a lab's own DLC
environment and run there.

Why an environment of its own: a DeepLabCut snapshot is a bare PyTorch
``state_dict``. Loading one means rebuilding the architecture first, which needs
``deeplabcut``'s own model classes -- there is no reading it without them. And
``deeplabcut`` pulls in torch, timm, albumentations and a long tail besides,
which is not a thing to put in the dependency tree of a lab that tracks with
YOLO. So it gets its own environment, built once, and GLIDER shells into it.

Everything the sidecar records is read from the model's config or measured from
the model itself. Nothing is defaulted: a wrong stride or a wrong normalisation
does not fail, it shifts every keypoint by a constant and produces a
plausible-looking skeleton, which is the failure mode worth refusing outright.

Usage::

    python convert.py <model-folder> [--output model.onnx] [--opset 17]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: Written beside the ONNX, recording which snapshot it came from. A retrained
#: model dropped into the same folder changes the digest, which is what makes a
#: stale conversion detectable rather than silently reused.
STAMP_NAME = ".glider_onnx_source.json"

SIDECAR_NAME = "glider_pose.json"
SIDECAR_SCHEMA_VERSION = 1

#: 17 covers the ops DLC's ResNet and HRNet backbones use (including the
#: `Resize` modes timm emits) and is old enough for any onnxruntime a user is
#: likely to have.
DEFAULT_OPSET = 17

#: What ``normalize_images: true`` means, read off DeepLabCut's own transform
#: builder: albumentations' ``A.Normalize`` with ImageNet statistics, which
#: divides by 255 first. Hard-coded there too -- DLC ignores the dict form of
#: that setting -- so hard-coding it here matches what inference actually does.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#: The head GLIDER runs. DLC names its body-part head this in every 3.x config;
#: `paf`, `identity` and detector heads are multi-animal machinery GLIDER has
#: nowhere to put.
BODYPART_HEAD = "bodypart"

INSTALL_HINT = (
    "Converting a DeepLabCut model needs DeepLabCut itself -- a snapshot is a "
    "bare PyTorch state_dict, and rebuilding the network from one needs DLC's "
    "own model classes.\n\n"
    "GLIDER builds that environment for you when you select a DLC model; this "
    "message means you are running the converter by hand in an environment "
    "that does not have it. Install deeplabcut, onnx and onnxscript here, or "
    "let GLIDER do it."
)


class ConversionError(RuntimeError):
    """Something needed for a correct conversion was missing or unsupported.

    Carries a sentence meant for a researcher: it is printed to stderr and put
    on screen verbatim.
    """


# --- finding the model -------------------------------------------------------


#: How far below a selected folder a model is looked for. DeepLabCut's layout
#: puts one at ``dlc-models-pytorch/iteration-N/<task>/train/``, four levels
#: down, so five is that with room to spare. It is bounded rather than an
#: unbounded ``rglob`` because this runs on every model selection and a DLC
#: project folder also holds ``labeled-data`` with thousands of frames in it --
#: walking that to answer "is this yours?" would stall the panel.
MAX_SEARCH_DEPTH = 5


def _walk(folder: Path, name: str, depth: int = MAX_SEARCH_DEPTH):
    """Yield files called *name* at or below *folder*, breadth first, bounded."""
    level = [folder]
    for _ in range(depth + 1):
        if not level:
            return
        nxt: list[Path] = []
        for directory in level:
            candidate = directory / name
            if candidate.is_file():
                yield candidate
            try:
                nxt.extend(child for child in directory.iterdir() if child.is_dir())
            except OSError:
                continue
        level = sorted(nxt)


def find_dlc_config(folder: Path | str) -> Path | None:
    """The ``pytorch_config.yaml`` for the model in *folder*, or None.

    Path-only, so GLIDER can ask about a folder without importing anything.
    Searched at the top level first and then downwards, because a researcher
    may point at either the ``train`` folder itself or the project above it.
    """
    return next(_walk(Path(folder), "pytorch_config.yaml"), None)


def find_dlc_snapshot(folder: Path | str) -> Path | None:
    """The snapshot to convert, or None.

    Looked for **beside the config**, not anywhere under the selected folder.
    A project with two training runs in it has two configs and two sets of
    weights, and pairing one run's config with another's snapshot is not an
    error that surfaces: ``load_state_dict`` accepts it whenever the shapes
    happen to match, and the model then answers with a network nobody trained
    for this.

    ``snapshot-best-*.pt`` wins when DLC wrote one -- it is the epoch DLC itself
    selected on the validation metric, and converting a later but worse epoch
    because its number sorts higher would be a quiet downgrade. Otherwise the
    highest epoch number, numerically rather than lexically, so
    ``snapshot-100.pt`` beats ``snapshot-90.pt``.
    """
    config = find_dlc_config(folder)
    root = config.parent if config is not None else Path(folder)

    snapshots = [p for p in sorted(root.glob("snapshot-*.pt")) if p.is_file()]
    if not snapshots:
        return None

    best = [p for p in snapshots if "best" in p.stem]
    if best:
        return best[0]

    def epoch(path: Path) -> tuple[int, str]:
        digits = "".join(c for c in path.stem if c.isdigit())
        return (int(digits) if digits else -1, path.name)

    return max(snapshots, key=epoch)


def is_dlc_folder(folder: Path | str) -> bool:
    """Whether *folder* holds a DeepLabCut 3.x model this can convert."""
    return find_dlc_config(folder) is not None and find_dlc_snapshot(folder) is not None


def _looks_like_dlc2(folder: Path) -> bool:
    """A DeepLabCut 2.x (TensorFlow) folder: pose_cfg.yaml and no PyTorch config."""
    has_pose_cfg = next(_walk(folder, "pose_cfg.yaml"), None) is not None
    return has_pose_cfg and find_dlc_config(folder) is None


# --- staleness ---------------------------------------------------------------


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp_for(snapshot: Path, config: Path) -> dict:
    stat = snapshot.stat()
    return {
        "snapshot": snapshot.name,
        "config": config.name,
        "size": stat.st_size,
        "sha256": _digest(snapshot),
    }


def is_conversion_current(folder: Path | str) -> bool:
    """Whether the ONNX beside the model was made from what is there now.

    False after a retrain drops a new snapshot in. Answering True on a stale
    conversion is the worst available outcome: the model runs, and quietly
    answers with the network that was just replaced.

    Path and hash work only -- no torch, no deeplabcut.
    """
    folder = Path(folder)
    config = find_dlc_config(folder)
    snapshot = find_dlc_snapshot(folder)
    if config is None or snapshot is None:
        return False

    root = config.parent
    if not (root / "model.onnx").is_file() or not (root / SIDECAR_NAME).is_file():
        return False

    try:
        stamp = json.loads((root / STAMP_NAME).read_text())
    except (OSError, ValueError):
        return False

    # Size first: it settles the common case without reading a 200 MB file.
    if stamp.get("snapshot") != snapshot.name or stamp.get("size") != snapshot.stat().st_size:
        return False
    return stamp.get("sha256") == _digest(snapshot)


def needs_conversion(folder: Path | str) -> bool:
    """Whether *folder* is a DLC model that is not yet usable."""
    return is_dlc_folder(folder) and not is_conversion_current(folder)


# --- reading the config ------------------------------------------------------


def _require(value, what: str, where: Path):
    """Return *value*, or refuse to guess.

    Every field this guards shifts or rescales the decoded keypoints. A wrong
    one produces a skeleton that looks fine and is wrong everywhere, so a
    missing field is a hard stop.
    """
    if value is None:
        raise ConversionError(
            f"could not read {what} from {where.name}. GLIDER will not write a "
            "sidecar with a guessed value: a wrong stride or normalisation "
            "shifts every keypoint by a constant and still looks plausible."
        )
    return value


def _bodypart_head(cfg: dict, cfg_path: Path) -> dict:
    heads = cfg.get("model", {}).get("heads", {})
    head = heads.get(BODYPART_HEAD)
    if head is None:
        found = ", ".join(sorted(heads)) or "none"
        raise ConversionError(
            f"{cfg_path.name} has no '{BODYPART_HEAD}' head (found: {found}). "
            "GLIDER runs DeepLabCut's single-animal body-part models."
        )
    return head


def _normalisation(cfg: dict, cfg_path: Path) -> dict:
    """How inference preprocesses a frame, read from the model's own config.

    DeepLabCut builds this from ``data.inference``. ``normalize_images`` is
    truthy in every stock config and means albumentations' ``A.Normalize`` with
    ImageNet statistics, which divides by 255 first. Falsy means no Normalize
    at all, and ``ToTensorV2`` alone does not rescale -- the network sees raw
    0-255 floats.
    """
    inference = cfg.get("data", {}).get("inference") or {}

    if inference.get("scale_to_unit_range"):
        raise ConversionError(
            f"{cfg_path.name} sets scale_to_unit_range, which rescales each "
            "frame against its own min and max. GLIDER's sidecar describes a "
            "fixed normalisation and has no way to express a per-frame one, so "
            "this model cannot be converted rather than converted wrongly."
        )

    if inference.get("normalize_images", True):
        return {"divide_by_255": True, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return {"divide_by_255": False, "mean": None, "std": None}


def _pad_to_stride(cfg: dict) -> int:
    """The input-size multiple the backbone requires.

    HRNet configs set this to 32 through ``auto_padding``; ResNet ones leave it
    out. Feeding an HRNet a size it cannot halve cleanly does not raise -- it
    produces a heatmap a pixel off in each dimension, which is a constant
    keypoint shift.
    """
    padding = (cfg.get("data", {}).get("inference") or {}).get("auto_padding") or {}
    divisors = [
        int(padding.get("pad_height_divisor") or 1),
        int(padding.get("pad_width_divisor") or 1),
    ]
    return max(divisors)


def build_sidecar(cfg: dict, cfg_path: Path, *, onnx_name: str, label: str, stride: float) -> dict:
    """The ``glider_pose.json`` GLIDER reads, from the model's config.

    *stride* is passed in because it is measured from the network rather than
    read: see :func:`_measure_stride`.
    """
    head = _bodypart_head(cfg, cfg_path)
    predictor = head.get("predictor") or {}

    names = cfg.get("metadata", {}).get("bodyparts")
    names = _require(names, "metadata.bodyparts", cfg_path)
    if len(set(names)) != len(names):
        raise ConversionError(f"body-part names in {cfg_path.name} are not unique: {names}")

    # No locref head means peak refinement instead, which the sidecar selects
    # with a null. Reading the flag rather than assuming it: DLC can train
    # without one, and declaring a locref that isn't there fails at load.
    locref = None
    if predictor.get("location_refinement", True):
        locref = float(_require(predictor.get("locref_std"), "locref_std", cfg_path))

    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "kind": "dlc",
        "onnx": onnx_name,
        "source_label": label,
        "keypoint_names": [str(n) for n in names],
        "output_stride": float(stride),
        "locref_stdev": locref,
        "scale": 1.0,
        "input_layout": "NCHW",
        "color_mode": "rgb",
        "pad_to_stride": _pad_to_stride(cfg),
        "apply_sigmoid": bool(predictor.get("apply_sigmoid", False)),
    }
    sidecar.update(_normalisation(cfg, cfg_path))
    return sidecar


# --- converting --------------------------------------------------------------


def _load_model(cfg: dict, snapshot: Path):
    """Rebuild the network from its config and load the snapshot's weights."""
    try:
        import torch
        from deeplabcut.pose_estimation_pytorch.models import PoseModel
    except ImportError as exc:
        raise ConversionError(INSTALL_HINT) from exc

    try:
        model = PoseModel.build(cfg["model"], pretrained_backbone=False)
    except Exception as exc:
        raise ConversionError(
            f"DeepLabCut could not rebuild this model from its config ({exc}). "
            "The config and the snapshot have to come from the same training "
            "run."
        ) from exc

    try:
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
    except Exception as exc:
        raise ConversionError(f"{snapshot.name} could not be loaded ({exc}).") from exc

    model.eval()
    return model


def _wrap(model, head_name: str, with_locref: bool):
    """Present the head's outputs as an ordered tuple.

    ``PoseModel.forward`` returns a nested dict, and ONNX flattens one in
    whatever order it happens to have. GLIDER's contract is positional --
    output 0 is the heatmap, output 1 is the location refinement -- so pinning
    the order here is what makes that true by construction rather than by luck.
    """
    import torch

    class _Exported(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)[head_name]
            if with_locref:
                return out["heatmap"], out["locref"]
            return out["heatmap"]

    return _Exported().eval()


def stride_from_sizes(small: int, large: int, h_small: int, h_large: int) -> float:
    """The stride implied by two input sizes and the heatmaps they produced.

    A *difference*, not a ratio, and that is the whole point. Every DeepLabCut
    backbone checked produces ``out = in / stride + 1`` -- the head's transposed
    convolution adds one cell past the edge -- so a single measurement is wrong
    by that constant and merely converges towards the truth as the input grows:
    a ResNet at 128 px reads as a stride of 7.53, at 512 px as 7.88, and is 8.
    Subtracting two measurements cancels the constant and lands on it exactly.

    Raises:
        ConversionError: if the result is not a whole number of pixels, which
            means the relationship is not the one assumed here and every
            decoded keypoint would be placed wrongly.
    """
    if h_large <= h_small:
        raise ConversionError(
            "this model's output does not grow with its input, so GLIDER "
            "cannot work out how to map heatmap positions back to pixels."
        )

    stride = (large - small) / (h_large - h_small)
    if stride != int(stride) or stride < 1:
        raise ConversionError(
            f"this model downsamples by {stride}, which is not a whole number "
            "of pixels. GLIDER decodes keypoints against an integer stride and "
            "would place every one of them wrongly."
        )
    return float(stride)


def _heatmap_height(wrapped, size: int) -> int:
    import torch

    with torch.no_grad():
        out = wrapped(torch.zeros(1, 3, size, size))
    heatmap = out[0] if isinstance(out, tuple) else out
    return int(heatmap.shape[-2])


def _measure_stride(wrapped, pad_to_stride: int) -> float:
    """The input-to-heatmap downsampling factor, measured from the network.

    Derived rather than read, and deliberately: the effective stride is the
    backbone's stride divided by whatever the head's deconvolutions undo, both
    of which move between DLC's backbones and between DLC versions -- an HRNet
    and a ResNet trained on the same data differ by a factor of four. Running
    two tensors through settles it exactly, and this is the number every
    decoded keypoint is multiplied by.

    Both sizes are multiples of whatever the backbone demands, so neither is
    silently floored on the way down.
    """
    unit = max(pad_to_stride, 64)
    small, large = 4 * unit, 8 * unit
    return stride_from_sizes(
        small, large, _heatmap_height(wrapped, small), _heatmap_height(wrapped, large)
    )


def convert_dlc_to_onnx(
    model_dir: Path | str,
    onnx_path: Path | str | None = None,
    *,
    opset: int = DEFAULT_OPSET,
) -> dict:
    """Convert the DeepLabCut model in *model_dir*, and describe the result.

    Imports torch and deeplabcut, so call this in its own process unless you
    want them resident for the rest of the session.
    """
    model_dir = Path(model_dir)
    config = find_dlc_config(model_dir)
    if config is None:
        if _looks_like_dlc2(model_dir):
            raise ConversionError(
                f"{model_dir.name} is a DeepLabCut 2.x model (TensorFlow). "
                "GLIDER converts DeepLabCut 3.x PyTorch models; a 2.x "
                "checkpoint has to be exported by hand with "
                "tools/export_pose_onnx.py in your own DLC 2.x environment."
            )
        raise ConversionError(f"no pytorch_config.yaml found under {model_dir}.")

    snapshot = find_dlc_snapshot(model_dir)
    if snapshot is None:
        raise ConversionError(
            f"no snapshot-*.pt found under {model_dir}. A DeepLabCut config "
            "without weights is a description of a model, not a model."
        )

    try:
        import yaml
    except ImportError as exc:
        raise ConversionError(INSTALL_HINT) from exc
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    head = _bodypart_head(cfg, config)
    with_locref = bool((head.get("predictor") or {}).get("location_refinement", True))

    model = _load_model(cfg, snapshot)
    wrapped = _wrap(model, BODYPART_HEAD, with_locref)
    stride = _measure_stride(wrapped, _pad_to_stride(cfg))

    root = config.parent
    onnx_path = Path(onnx_path) if onnx_path else root / "model.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    names = ["heatmap", "locref"] if with_locref else ["heatmap"]
    # Dynamic height and width, not a fixed input size: GLIDER feeds whatever
    # the camera produces, padded up to the backbone's multiple. A model frozen
    # at one resolution would refuse every other camera in the lab.
    dynamic = {"image": {2: "height", 3: "width"}}
    dynamic.update({n: {2: "out_height", 3: "out_width"} for n in names})
    try:
        torch.onnx.export(
            wrapped,
            torch.zeros(1, 3, 256, 256),
            str(onnx_path),
            opset_version=opset,
            input_names=["image"],
            output_names=names,
            dynamic_axes=dynamic,
        )
    except Exception as exc:
        raise ConversionError(f"torch could not export {snapshot.name} to ONNX: {exc}") from exc

    sidecar = build_sidecar(
        cfg,
        config,
        onnx_name=onnx_path.name,
        label=f"dlc_{model_dir.name}",
        stride=stride,
    )
    with open(root / SIDECAR_NAME, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
        f.write("\n")

    # The stamp is written last, so an interrupted conversion leaves something
    # that reads as stale rather than something that reads as current and is
    # truncated.
    (root / STAMP_NAME).write_text(json.dumps(_stamp_for(snapshot, config), indent=2))

    return {
        "onnx": str(onnx_path),
        "sidecar": str(root / SIDECAR_NAME),
        "snapshot": snapshot.name,
        "keypoints": len(sidecar["keypoint_names"]),
        "output_stride": sidecar["output_stride"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert a DeepLabCut 3.x model to ONNX.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args(argv)

    try:
        result = convert_dlc_to_onnx(args.model_dir, args.output, opset=args.opset)
    except ConversionError as exc:
        # stderr, and only the message: the caller puts this on screen.
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
