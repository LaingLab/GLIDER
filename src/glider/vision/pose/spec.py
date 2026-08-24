"""Identify a pose model on disk and describe how to run it.

Pure path/JSON/YAML work: importing this module pulls in nothing heavier than
the standard library, so the GUI can classify a dropped folder without paying
for torch or onnxruntime. The one exception is :func:`read_pose_model_meta`,
which reads a YOLO checkpoint through ultralytics — that import is deliberately
inside the function, and callers run it off the UI thread.

The ``glider_pose.json`` sidecar written by ``tools/export_pose_onnx.py`` is the
authoritative description of a model. DeepLabCut 2.x, DeepLabCut 3.x and SLEAP
each structure their own configs differently, and parsing all of them correctly
across versions is a losing game — so native configs are read only as a
best-effort fallback for hand-exported folders, and the sidecar always wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

#: Bumped when the sidecar's shape changes incompatibly.
SIDECAR_SCHEMA_VERSION = 1

SIDECAR_NAME = "glider_pose.json"

_HELPER_HINT = (
    "no GLIDER pose sidecar or recognisable DeepLabCut/SLEAP config found in "
    "{root}. A SLEAP model folder (one holding training_config.json and "
    "best_model.h5) is converted automatically when you select it, with "
    "'glider[sleap]' installed. For DeepLabCut, export the model in your own "
    f"DLC environment with tools/export_pose_onnx.py, which writes model.onnx "
    f"alongside {SIDECAR_NAME}."
)


class PoseModelError(RuntimeError):
    """A dropped or selected path is not a pose model GLIDER can run."""


@dataclass(frozen=True)
class PoseModelSpec:
    """Everything needed to load a model and map its output back to pixels."""

    kind: Literal["yolo", "dlc", "sleap"]
    model_path: Path
    root: Path
    keypoint_names: list[str] = field(default_factory=list)
    source_label: str = ""

    # ONNX only; ignored when kind == "yolo".
    input_layout: Literal["NCHW", "NHWC"] = "NCHW"
    color_mode: Literal["rgb", "gray"] = "rgb"
    scale: float = 1.0
    divide_by_255: bool = True
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    pad_to_stride: int = 1
    output_stride: float = 8.0
    locref_stdev: float | None = None
    apply_sigmoid: bool = False
    refine_window: int = 5

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoint_names)


@dataclass(frozen=True)
class PoseModelMeta:
    """What a GUI needs to show about a model before running it.

    ``keypoint_names`` is ``None`` for YOLO ``.pt`` weights: Ultralytics
    checkpoints record *class* names and a ``kpt_shape``, never body-part names.
    Callers must keep asking the operator to type those.
    """

    kind: Literal["yolo", "dlc", "sleap"]
    n_keypoints: int
    keypoint_names: list[str] | None = None


def is_onnx_model(path: str | Path) -> bool:
    """True if *path* names an ONNX model (``.onnx``, case-insensitive)."""
    return Path(path).suffix.lower() == ".onnx"


def ensure_onnxruntime() -> None:
    """Raise a clear, actionable error if onnxruntime is not importable."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required to run DeepLabCut/SLEAP pose models. "
            "Install it with: pip install 'glider[pose-onnx]'"
        ) from e


def _tuple3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    seq = [float(v) for v in value]
    if len(seq) != 3:
        raise PoseModelError(f"expected 3 values, got {len(seq)}: {value!r}")
    return (seq[0], seq[1], seq[2])


def _from_sidecar(root: Path) -> PoseModelSpec:
    raw = json.loads((root / SIDECAR_NAME).read_text())

    version = int(raw.get("schema_version", 1))
    if version > SIDECAR_SCHEMA_VERSION:
        raise PoseModelError(
            f"{root / SIDECAR_NAME} declares schema_version {version}, but this "
            f"GLIDER understands up to {SIDECAR_SCHEMA_VERSION}. Upgrade GLIDER "
            "or re-export with a matching helper."
        )

    kind = raw.get("kind")
    if kind not in ("dlc", "sleap"):
        raise PoseModelError(f"sidecar 'kind' must be 'dlc' or 'sleap'; got {kind!r}")

    onnx_path = root / raw.get("onnx", "model.onnx")
    if not onnx_path.is_file():
        raise PoseModelError(
            f"{root / SIDECAR_NAME} names {onnx_path.name}, but that file is not in {root}."
        )

    names = [str(n) for n in raw.get("keypoint_names", [])]
    if not names:
        raise PoseModelError(f"{root / SIDECAR_NAME} lists no keypoint_names.")
    if len(set(names)) != len(names):
        raise PoseModelError(f"keypoint_names must be unique; got {names}")

    return PoseModelSpec(
        kind=kind,
        model_path=onnx_path,
        root=root,
        keypoint_names=names,
        source_label=raw.get("source_label") or f"{kind}_{root.name}",
        input_layout=raw.get("input_layout", "NCHW"),
        color_mode=raw.get("color_mode", "rgb"),
        scale=float(raw.get("scale", 1.0)),
        divide_by_255=bool(raw.get("divide_by_255", True)),
        mean=_tuple3(raw.get("mean")),
        std=_tuple3(raw.get("std")),
        pad_to_stride=int(raw.get("pad_to_stride", 1)),
        output_stride=float(raw.get("output_stride", 8.0)),
        locref_stdev=(None if raw.get("locref_stdev") is None else float(raw["locref_stdev"])),
        apply_sigmoid=bool(raw.get("apply_sigmoid", False)),
        refine_window=int(raw.get("refine_window", 5)),
    )


def _sole_onnx(root: Path) -> Path:
    """The one .onnx in *root*, or an error naming the export helper."""
    candidates = sorted(p for p in root.iterdir() if is_onnx_model(p))
    if not candidates:
        raise PoseModelError(_HELPER_HINT.format(root=root))
    if len(candidates) > 1:
        raise PoseModelError(
            f"{root} holds {len(candidates)} .onnx files "
            f"({', '.join(p.name for p in candidates)}); GLIDER cannot tell which "
            f"to run. Keep one per folder, or add a {SIDECAR_NAME} naming it."
        )
    return candidates[0]


def _from_dlc_config(root: Path, cfg_path: Path) -> PoseModelSpec:
    try:
        import yaml
    except ImportError as e:
        raise PoseModelError(
            f"reading {cfg_path.name} needs PyYAML, which is not installed. "
            f"Either install it, or place a {SIDECAR_NAME} in {root} "
            "(tools/export_pose_onnx.py writes one)."
        ) from e

    raw = yaml.safe_load(cfg_path.read_text()) or {}
    names = [str(n) for n in raw.get("all_joints_names", [])]
    if not names:
        raise PoseModelError(f"{cfg_path} lists no all_joints_names.")
    if len(set(names)) != len(names):
        raise PoseModelError(f"all_joints_names must be unique; got {names}")

    return PoseModelSpec(
        kind="dlc",
        model_path=_sole_onnx(root),
        root=root,
        keypoint_names=names,
        source_label=f"dlc_{root.name}",
        # DeepLabCut's own defaults. A sidecar is the way to override them —
        # a silently-wrong stride offsets every keypoint without failing.
        output_stride=float(raw.get("stride", 8.0)),
        locref_stdev=float(raw.get("locref_stdev", 7.2831)),
        scale=float(raw.get("global_scale", 1.0)),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pad_to_stride=1,
    )


def _from_sleap_config(root: Path, cfg_path: Path) -> PoseModelSpec:
    raw = json.loads(cfg_path.read_text())
    heads = raw.get("model", {}).get("heads", {})

    head = heads.get("single_instance")
    if head is None:
        found = ", ".join(sorted(heads)) or "none"
        raise PoseModelError(
            f"{cfg_path} has no single_instance head (found: {found}). GLIDER "
            "runs SLEAP single-instance models only; top-down and bottom-up "
            "models are multi-animal architectures and are not supported."
        )

    names = [str(n) for n in head.get("part_names", [])]
    if not names:
        raise PoseModelError(f"{cfg_path} lists no part_names.")
    if len(set(names)) != len(names):
        raise PoseModelError(f"part_names must be unique; got {names}")

    backbone = raw.get("model", {}).get("backbone", {})
    max_stride = 1
    for arch in backbone.values():
        if isinstance(arch, dict) and "max_stride" in arch:
            max_stride = int(arch["max_stride"])
            break

    pre = raw.get("data", {}).get("preprocessing", {})

    return PoseModelSpec(
        kind="sleap",
        model_path=_sole_onnx(root),
        root=root,
        keypoint_names=names,
        source_label=f"sleap_{root.name}",
        output_stride=float(head.get("output_stride", 1)),
        # SLEAP has no location-refinement head; None selects peak refinement.
        locref_stdev=None,
        scale=float(pre.get("input_scaling", 1.0)),
        color_mode="gray" if bool(pre.get("ensure_grayscale", False)) else "rgb",
        pad_to_stride=max_stride,
        # SLEAP normalises uint8 to [0, 1] with no mean/std subtraction.
        divide_by_255=True,
        mean=None,
        std=None,
    )


def identify_pose_model(path: str | Path) -> PoseModelSpec:
    """Describe the pose model at *path*, or raise :class:`PoseModelError`.

    Accepts a YOLO ``.pt``, a bare ``.onnx`` (whose sidecar must sit beside it),
    or a folder holding an exported model.
    """
    path = Path(path)

    # The suffix alone identifies a YOLO checkpoint, so this deliberately does
    # not stat the file: infer_video() has always let ultralytics report a
    # missing model, and its error names the path better than a generic
    # "does not exist" would.
    if path.suffix.lower() == ".pt":
        return PoseModelSpec(
            kind="yolo",
            model_path=path,
            root=path.parent,
            source_label=f"yolo_{path.stem}",
        )

    if path.is_file():
        if is_onnx_model(path):
            if (path.parent / SIDECAR_NAME).is_file():
                return _from_sidecar(path.parent)
            raise PoseModelError(
                f"{path.name} has no {SIDECAR_NAME} beside it, so GLIDER cannot "
                "know its keypoint names, stride, or preprocessing. "
                + _HELPER_HINT.format(root=path.parent)
            )
        raise PoseModelError(f"{path.name} is not a pose model (.pt or .onnx expected).")

    if path.is_dir():
        # The sidecar always wins: it is the one description GLIDER wrote and
        # can trust across DLC 2.x, DLC 3.x and SLEAP config shapes.
        if (path / SIDECAR_NAME).is_file():
            return _from_sidecar(path)
        for cfg_name in ("pose_cfg.yaml", "pytorch_config.yaml"):
            cfg = path / cfg_name
            if cfg.is_file():
                return _from_dlc_config(path, cfg)
        sleap_cfg = path / "training_config.json"
        if sleap_cfg.is_file():
            return _from_sleap_config(path, sleap_cfg)
        raise PoseModelError(_HELPER_HINT.format(root=path))

    raise PoseModelError(f"{path} does not exist.")


def _read_yolo_kpt_count(path: Path) -> int:
    """Keypoint count off a YOLO checkpoint. Imports torch — never on the UI thread."""
    from ultralytics import YOLO

    shape = getattr(getattr(YOLO(str(path)), "model", None), "kpt_shape", None)
    return int(shape[0]) if shape else 0


def read_pose_model_meta(path: str | Path) -> PoseModelMeta | None:
    """Best-effort metadata for a model path; ``None`` when it cannot be read.

    Never raises: the callers are GUI workers that must not block the operator
    on a metadata read. Loading a ``.pt`` goes through ultralytics and takes
    seconds, so call this off the UI thread.
    """
    try:
        spec = identify_pose_model(path)
    except PoseModelError:
        return None

    if spec.kind == "yolo":
        try:
            return PoseModelMeta(kind="yolo", n_keypoints=_read_yolo_kpt_count(spec.model_path))
        except Exception:
            return None

    return PoseModelMeta(
        kind=spec.kind,
        n_keypoints=spec.n_keypoints,
        keypoint_names=list(spec.keypoint_names),
    )
