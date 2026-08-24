"""Convert a SLEAP model to ONNX, in GLIDER's own environment.

GLIDER runs pose models through onnxruntime, so a SLEAP model has to be
converted before it can be used. That conversion was documented as something
the researcher does themselves, in their SLEAP environment, with two commands
they have to know exist::

    sleap-export -m <model-dir> -o exported
    python -m tf2onnx.convert --saved-model exported --output model.onnx

Nobody does that. They point GLIDER at the folder SLEAP produced and expect it
to work.

**It can.** The documented reasoning was that converting needs the framework
that produced the model, and that SLEAP is TensorFlow-pinned and capped around
Python 3.10 while GLIDER targets 3.11-3.13, so the two cannot share an
environment. The first half conflates two different things: loading a SLEAP
checkpoint needs *TensorFlow*, not *SLEAP*. A SLEAP model is saved as an
ordinary Keras model -- ``tf.keras.models.load_model`` opens one with no SLEAP
installed at all -- and TensorFlow supports the Pythons GLIDER targets. The
version conflict is real for running SLEAP; it never applied to converting its
output.

Verified against SLEAP's own ``minimal_robot.UNet.single_instance`` test
fixture: loads without SLEAP, converts in one call, and the ONNX matches the
Keras output to 7e-7 -- floating-point noise.

TensorFlow is an optional extra (``glider[sleap]``) rather than a dependency:
it is a large install that only matters to people with SLEAP models, and
importing it costs seconds and grabs threads. So this module is designed to run
**as a subprocess** -- ``python -m glider.vision.pose.convert`` -- which keeps
TensorFlow out of the application's process entirely and lets it be reclaimed
when the child exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Written beside the ONNX so a second run can skip the work.
STAMP_NAME = ".glider_onnx_source.json"

#: ONNX opset. 13 covers the ops a SLEAP UNet uses and is old enough that any
#: onnxruntime a user is likely to have will load it.
DEFAULT_OPSET = 13

INSTALL_HINT = (
    "Converting a SLEAP model needs TensorFlow and tf2onnx, which GLIDER does "
    "not install by default because they are large and only matter to people "
    "with SLEAP models. Install them with:\n\n"
    "    pip install 'glider[sleap]'\n\n"
    "Nothing from SLEAP itself is required."
)


class ConversionError(RuntimeError):
    """A SLEAP model could not be converted to ONNX."""


@dataclass(frozen=True)
class ConversionResult:
    onnx_path: Path
    keypoint_count: int
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]


def find_sleap_checkpoint(model_dir: Path) -> Path | None:
    """The Keras checkpoint inside a SLEAP model folder, if there is one.

    SLEAP writes ``best_model.h5``; a run stopped early may leave only
    ``final_model.h5``. Both are ordinary Keras files.
    """
    for name in ("best_model.h5", "final_model.h5"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    others = sorted(model_dir.glob("*.h5"))
    return others[0] if others else None


def needs_conversion(model_dir: Path | str) -> bool:
    """Whether *model_dir* is a SLEAP folder that GLIDER cannot run yet.

    Deliberately free of any TensorFlow import: the UI asks this on every model
    selection, and importing TensorFlow to answer it would cost seconds on the
    path where the answer is almost always no.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir() or not (model_dir / "training_config.json").is_file():
        return False
    if find_sleap_checkpoint(model_dir) is None:
        return False
    onnx = sorted(model_dir.glob("*.onnx"))
    if not onnx:
        return True
    # An ONNX converted from a *previous* checkpoint is worse than none: it runs,
    # and silently answers with the network the researcher just retrained away.
    return not is_conversion_current(model_dir, onnx[0])


def _stamp_for(checkpoint: Path) -> dict:
    stat = checkpoint.stat()
    return {"source": checkpoint.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def is_conversion_current(model_dir: Path, onnx_path: Path) -> bool:
    """Whether *onnx_path* was converted from the checkpoint that is there now.

    Compares against a stamp rather than mtimes alone: a researcher who retrains
    and drops a new checkpoint into the same folder must not silently keep
    running the previous network, which would look like the retraining did
    nothing.
    """
    if not onnx_path.is_file():
        return False
    checkpoint = find_sleap_checkpoint(model_dir)
    if checkpoint is None:
        return False
    stamp_path = onnx_path.with_name(STAMP_NAME)
    try:
        stamp = json.loads(stamp_path.read_text())
    except (OSError, ValueError):
        return False
    return stamp == _stamp_for(checkpoint)


def convert_sleap_to_onnx(
    model_dir: Path | str,
    onnx_path: Path | str | None = None,
    *,
    opset: int = DEFAULT_OPSET,
) -> ConversionResult:
    """Convert the SLEAP model in *model_dir* to ONNX.

    Imports TensorFlow, so call this in a subprocess unless you want it in your
    process for good. Raises :class:`ConversionError` for everything a caller
    can act on -- a missing extra, a missing checkpoint, a model that will not
    load -- rather than letting a TensorFlow traceback reach the UI.
    """
    model_dir = Path(model_dir)
    onnx_path = Path(onnx_path) if onnx_path else model_dir / "model.onnx"

    checkpoint = find_sleap_checkpoint(model_dir)
    if checkpoint is None:
        raise ConversionError(
            f"no Keras checkpoint (best_model.h5) found in {model_dir}. This does "
            "not look like a folder SLEAP produced."
        )

    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError as exc:
        raise ConversionError(INSTALL_HINT) from exc

    try:
        model = tf.keras.models.load_model(str(checkpoint), compile=False)
    except Exception as exc:
        raise ConversionError(
            f"{checkpoint.name} could not be loaded as a Keras model ({exc}). "
            "GLIDER converts single-instance SLEAP models saved as ordinary "
            "Keras checkpoints; a model using custom layers cannot be read "
            "without SLEAP itself."
        ) from exc

    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ConversionError(
            f"{checkpoint.name} has {len(model.inputs)} inputs and "
            f"{len(model.outputs)} outputs; GLIDER runs single-input, "
            "single-output confidence-map models."
        )

    input_shape = tuple(model.inputs[0].shape)
    output_shape = tuple(model.outputs[0].shape)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        signature = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
        tf2onnx.convert.from_keras(
            model, input_signature=signature, opset=opset, output_path=str(onnx_path)
        )
    except Exception as exc:
        raise ConversionError(f"tf2onnx could not convert {checkpoint.name}: {exc}") from exc

    # The stamp is written last, so an interrupted conversion leaves a file that
    # reads as stale rather than one that reads as current and is truncated.
    onnx_path.with_name(STAMP_NAME).write_text(json.dumps(_stamp_for(checkpoint), indent=2))

    return ConversionResult(
        onnx_path=onnx_path,
        keypoint_count=int(output_shape[-1]) if output_shape[-1] is not None else 0,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point, so the conversion can run as a subprocess."""
    parser = argparse.ArgumentParser(
        prog="python -m glider.vision.pose.convert",
        description="Convert a SLEAP model folder to ONNX for GLIDER.",
    )
    parser.add_argument("model_dir", type=Path, help="the folder SLEAP produced")
    parser.add_argument("-o", "--output", type=Path, default=None, help="where to write model.onnx")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help=(
            "write the result as JSON here. TensorFlow logs to stdout without "
            "asking, so a caller parsing stdout is parsing TensorFlow's mood; "
            "a file is not."
        ),
    )
    args = parser.parse_args(argv)

    try:
        result = convert_sleap_to_onnx(args.model_dir, args.output, opset=args.opset)
    except ConversionError as exc:
        # stderr, and only the message: the caller shows this to a person.
        print(str(exc), file=sys.stderr)
        return 1

    payload = {
        "onnx_path": str(result.onnx_path),
        "keypoint_count": result.keypoint_count,
        "input_shape": [None if d is None else int(d) for d in result.input_shape],
        "output_shape": [None if d is None else int(d) for d in result.output_shape],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
