#!/usr/bin/env python3
"""Export a DeepLabCut or SLEAP model to ONNX, with a GLIDER sidecar.

Run this **in your own DeepLabCut or SLEAP environment**, not in GLIDER's.
Converting a native checkpoint needs the framework that produced it — DLC's
model classes to rebuild a snapshot, TensorFlow plus tf2onnx for a SLEAP
SavedModel — and GLIDER deliberately does not depend on either. SLEAP in
particular is TensorFlow-pinned and has historically capped out around Python
3.10, while GLIDER targets 3.11-3.13, so the two cannot share an environment.

This script has **no glider imports** so it can be copied into that environment
on its own.

It writes two files into the output folder:

    model.onnx          the converted network
    glider_pose.json    everything GLIDER needs to run it

The sidecar is the authoritative description GLIDER reads. It is written only
when every field can be filled in from the model's own config: a silently
defaulted stride or normalisation offsets every keypoint by a constant, which
produces a plausible-looking skeleton rather than an error, and quietly
corrupts anything computed downstream.

Usage
-----
    python export_pose_onnx.py dlc   <project-or-model-dir> -o <out-dir>
    python export_pose_onnx.py sleap <sleap-model-dir>      -o <out-dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SIDECAR_NAME = "glider_pose.json"
SIDECAR_SCHEMA_VERSION = 1

#: ImageNet statistics, which DeepLabCut's PyTorch backbones normalise against.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ExportError(RuntimeError):
    """Something needed for a complete sidecar was missing."""


def _require(value, what: str, where: Path):
    """Return *value*, or refuse to guess.

    Defaulting any of these silently shifts every decoded keypoint, so a
    missing field is a hard stop rather than a warning.
    """
    if value is None:
        raise ExportError(
            f"could not read {what} from {where}. GLIDER will not write a "
            "sidecar with a guessed value: a wrong stride or normalisation "
            "offsets every keypoint without ever failing. Fill it in by hand "
            f"in {SIDECAR_NAME} if you know the correct value."
        )
    return value


def build_dlc_sidecar(cfg: dict, cfg_path: Path, onnx_name: str, label: str) -> dict:
    """Sidecar for a DeepLabCut model, from its pose_cfg/pytorch_config."""
    names = cfg.get("all_joints_names") or cfg.get("bodyparts")
    names = _require(names, "all_joints_names", cfg_path)
    if len(set(names)) != len(names):
        raise ExportError(f"body-part names in {cfg_path} are not unique: {names}")

    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "kind": "dlc",
        "onnx": onnx_name,
        "source_label": label,
        "keypoint_names": [str(n) for n in names],
        "output_stride": float(_require(cfg.get("stride"), "stride", cfg_path)),
        "locref_stdev": float(_require(cfg.get("locref_stdev"), "locref_stdev", cfg_path)),
        "scale": float(cfg.get("global_scale", 1.0)),
        "input_layout": "NCHW",
        "color_mode": "rgb",
        "divide_by_255": True,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "pad_to_stride": 1,
        "apply_sigmoid": bool(cfg.get("apply_sigmoid", False)),
    }


def build_sleap_sidecar(cfg: dict, cfg_path: Path, onnx_name: str, label: str) -> dict:
    """Sidecar for a SLEAP single-instance model, from its training_config."""
    heads = cfg.get("model", {}).get("heads", {})
    head = heads.get("single_instance")
    if head is None:
        found = ", ".join(sorted(heads)) or "none"
        raise ExportError(
            f"{cfg_path} has no single_instance head (found: {found}). GLIDER "
            "runs SLEAP single-instance models only; top-down and bottom-up "
            "models are multi-animal architectures and are not supported."
        )

    names = _require(head.get("part_names"), "part_names", cfg_path)
    if len(set(names)) != len(names):
        raise ExportError(f"part names in {cfg_path} are not unique: {names}")

    backbone = cfg.get("model", {}).get("backbone", {})
    max_stride = 1
    for arch in backbone.values():
        if isinstance(arch, dict) and "max_stride" in arch:
            max_stride = int(arch["max_stride"])
            break

    pre = cfg.get("data", {}).get("preprocessing", {})

    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "kind": "sleap",
        "onnx": onnx_name,
        "source_label": label,
        "keypoint_names": [str(n) for n in names],
        "output_stride": float(_require(head.get("output_stride"), "output_stride", cfg_path)),
        # SLEAP has no location-refinement head; null selects peak refinement.
        "locref_stdev": None,
        "scale": float(pre.get("input_scaling", 1.0)),
        "input_layout": "NHWC",
        "color_mode": "gray" if pre.get("ensure_grayscale") else "rgb",
        "divide_by_255": True,
        "mean": None,
        "std": None,
        "pad_to_stride": max_stride,
        "refine_window": 5,
    }


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ExportError("reading a DeepLabCut config needs PyYAML: pip install pyyaml") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find(root: Path, names) -> Path | None:
    for name in names:
        hit = root / name
        if hit.is_file():
            return hit
    for name in names:
        for hit in root.rglob(name):
            return hit
    return None


def export_dlc(model_dir: Path, out_dir: Path) -> Path:
    cfg_path = _find(model_dir, ["pose_cfg.yaml", "pytorch_config.yaml"])
    if cfg_path is None:
        raise ExportError(f"no pose_cfg.yaml or pytorch_config.yaml found under {model_dir}")
    cfg = _read_yaml(cfg_path)

    onnx_src = next(iter(sorted(model_dir.rglob("*.onnx"))), None)
    if onnx_src is None:
        raise ExportError(
            f"no .onnx found under {model_dir}. Export the network first, e.g.\n"
            "    import deeplabcut, torch\n"
            "    # rebuild the model from its snapshot, then:\n"
            "    torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=17)\n"
            "then re-run this command so the sidecar is written beside it."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_src, out_dir / "model.onnx")
    sidecar = build_dlc_sidecar(cfg, cfg_path, "model.onnx", f"dlc_{model_dir.name}")
    return _write_sidecar(out_dir, sidecar)


def export_sleap(model_dir: Path, out_dir: Path) -> Path:
    cfg_path = _find(model_dir, ["training_config.json"])
    if cfg_path is None:
        raise ExportError(f"no training_config.json found under {model_dir}")
    cfg = _read_json(cfg_path)

    onnx_src = next(iter(sorted(model_dir.rglob("*.onnx"))), None)
    if onnx_src is None:
        raise ExportError(
            f"no .onnx found under {model_dir}. Convert the SavedModel first:\n"
            "    sleap-export -m <model-dir> -o exported\n"
            "    python -m tf2onnx.convert --saved-model exported --output model.onnx\n"
            "then re-run this command so the sidecar is written beside it."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_src, out_dir / "model.onnx")
    sidecar = build_sleap_sidecar(cfg, cfg_path, "model.onnx", f"sleap_{model_dir.name}")
    return _write_sidecar(out_dir, sidecar)


def _write_sidecar(out_dir: Path, payload: dict) -> Path:
    path = out_dir / SIDECAR_NAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="export_pose_onnx",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="kind", required=True)

    for kind, help_text in (
        ("dlc", "a DeepLabCut project or exported-model directory"),
        ("sleap", "a SLEAP model directory (holding training_config.json)"),
    ):
        p = sub.add_parser(kind, help=help_text)
        p.add_argument("model_dir", type=Path, help=help_text)
        p.add_argument(
            "-o",
            "--out-dir",
            type=Path,
            default=None,
            help="where to write model.onnx + the sidecar (default: <model_dir>/glider)",
        )

    args = parser.parse_args(argv)
    out_dir = args.out_dir or (args.model_dir / "glider")

    try:
        if args.kind == "dlc":
            sidecar = export_dlc(args.model_dir, out_dir)
        else:
            sidecar = export_sleap(args.model_dir, out_dir)
    except ExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"wrote {sidecar.parent / 'model.onnx'}")
    print(f"wrote {sidecar}")
    print("\nDrag that folder onto GLIDER's camera panel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
