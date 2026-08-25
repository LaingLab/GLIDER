# Pose models: YOLO, DeepLabCut, and SLEAP

GLIDER runs three kinds of pose model. Ultralytics YOLO weights (`.pt`) load
directly. DeepLabCut and SLEAP models run through ONNX, converted the first time
you select one.

**Drag the model's folder onto the camera panel.** The panel routes a dropped
item by type, so a pose model, a behavior model (`.pkl`), and a video can all be
dropped the same way — together or separately.

## Install the plugin for your format

| You have | Install |
| --- | --- |
| Ultralytics YOLO `.pt` | nothing |
| SLEAP models | `pip install glider-sleap` |
| DeepLabCut 3.x models | `pip install glider-dlc` |

With the right plugin installed, selecting the folder your framework wrote is
all there is to it: GLIDER asks once, converts, and keeps the result beside the
model. Retrain and drop in a new checkpoint and it notices and reconverts — a
stale ONNX would run perfectly well while answering with the network you just
replaced.

## Why they are plugins

Converting a native checkpoint requires the framework that produced it:
DeepLabCut's model classes to rebuild a snapshot, TensorFlow to open a SLEAP
one. Between them that is several gigabytes, and it is not something to put in
the dependency tree of a lab that tracks with YOLO and has opened neither.

So GLIDER itself carries onnxruntime and nothing else, and each plugin owns one
vendor's conversion. Installing one is how you say you have that vendor's
models. `glider-dlc` goes further and carries no dependencies at all — it builds
a private DeepLabCut environment under `~/.glider/envs` the first time you need
one, and converts in that.

The division also keeps the *running* side installable everywhere it needs to
be. The same converted folder runs on a Raspberry Pi, where neither framework
would install at all.

## What GLIDER supports

| Format | Architecture | Status |
| --- | --- | --- |
| Ultralytics YOLO | pose `.pt` | Loads directly |
| DeepLabCut 3.x | single-animal (heatmaps + location refinement) | Converted on selection, with `glider-dlc` |
| DeepLabCut 2.x | TensorFlow checkpoints | Hand export only |
| SLEAP | single-instance (confidence maps) | Converted on selection, with `glider-sleap` |
| SLEAP | top-down, bottom-up | **Not supported** |
| Any | multi-animal | **Not supported** |

GLIDER's pose container is single-animal throughout, so multi-animal
architectures have nowhere to put a second subject. A SLEAP model with a
`centered_instance` or bottom-up head is rejected by name at load time rather
than silently producing one arbitrary animal's keypoints.

## Exporting by hand

Only needed for DeepLabCut 2.x, or on a machine that should download nothing.
Copy `tools/export_pose_onnx.py` into the environment where DeepLabCut or SLEAP
already works, and run it there — it imports nothing from `glider`.

### DeepLabCut

Export the network to ONNX from your DLC environment, then run the helper to
write the sidecar beside it:

```bash
python export_pose_onnx.py dlc /path/to/dlc-project -o exported-model
```

### SLEAP

SLEAP takes two steps — its own exporter produces a TensorFlow SavedModel, which
`tf2onnx` converts:

```bash
sleap-export -m /path/to/sleap-model -o exported
```

```bash
python -m tf2onnx.convert --saved-model exported --output exported/model.onnx
```

```bash
python export_pose_onnx.py sleap /path/to/sleap-model -o exported-model
```

### What you get

```
exported-model/
    model.onnx
    glider_pose.json
```

`glider_pose.json` is what GLIDER actually reads. It records the ordered body-part
names, the output stride, the location-refinement scale, and the exact
preprocessing the network expects.

The helper **refuses to write a sidecar it cannot fully populate**. A guessed
stride or normalisation shifts every decoded keypoint by a constant — which
yields a plausible-looking skeleton rather than an error, and quietly corrupts
everything computed from it. A hard stop at export time is the cheapest place to
catch that.

## Using the model

Drag `exported-model/` onto the camera panel. GLIDER detects the format, shows it
(`Pose model: exported-model (DeepLabCut, 7 kp)`), and **fills in the keypoint
names from the model's own config, in training order**, locking the field.

That last part matters more than it looks. Keypoint order is not cosmetic: with
`auto_angles`, angle features are generated from keypoint *triplets by index*,
and the names at those indices are baked into the resulting column names. Enter
the same names in a different order and those columns never appear — they arrive
as NaN, every prediction comes back blank, and nothing raises. Reading the order
from the model removes the chance to get it wrong.

YOLO checkpoints record class names and a keypoint count, never body-part names,
so a dropped `.pt` leaves the names field editable and you must still type them
in training order.

Exported folders also work anywhere a pose model is selected — batch inference
over directories of video, and the offline behavior-classification path — through
the existing pickers.

## Installing the runtime

Running an exported model needs `onnxruntime`:

```bash
pip install 'glider[pose-onnx]'
```

CPU-only by design. It covers Python 3.11–3.13 on Windows, macOS, Linux, **and**
ARM, which is what makes exported models viable on the Raspberry Pi runner.

## Verifying a model before you trust it

GLIDER's own tests prove its decoding is internally consistent. They do not
prove it matches what DeepLabCut or SLEAP would have produced for the same
video — only a comparison against real output does that.

If you rely on absolute keypoint coordinates, run the parity check once per
model with a clip you have already analysed in the source tool:

```bash
GLIDER_POSE_FIXTURES=/path/to/fixtures pytest -m pose_parity
```

The fixture layout is documented in `tests/unit/vision/pose/test_parity.py`. A
roughly constant offset across every keypoint points at a stride convention; a
consistent scale factor points at the input scaling.

## Troubleshooting

**"no GLIDER pose sidecar or recognisable config found"** — the folder has an
`.onnx` but no `glider_pose.json`. Re-run the export helper against it.

**"GLIDER cannot tell which to run"** — more than one `.onnx` in the folder. Keep
one per folder, or name the right one in the sidecar's `onnx` field.

**"the pose model emits N keypoints but the behavior model expects M"** — the
pose model and the behavior classifier were trained against different keypoint
sets. They must match.

**"keypoint names are in a different order than the behavior model was trained
on"** — the names are right but the order is not. The message prints both
orderings.

## DeepLabCut models

Point GLIDER at the folder DeepLabCut wrote — the `train` folder holding
`pytorch_config.yaml` and `snapshot-*.pt`, or the project above it.

```bash
pip install glider-dlc
```

That plugin has no dependencies of its own. The first time you select a
DeepLabCut model it builds a private Python environment under
`~/.glider/envs/deeplabcut` with `uv`, installs DeepLabCut into that, and
converts there — about 1.3 GB, downloaded once, and GLIDER's own environment is
untouched. The dialog tells you before anything starts.

Already have DeepLabCut working somewhere? Set `GLIDER_DLC_ENV` to that
virtualenv and nothing is downloaded at all.

!!! note "DeepLabCut 3.x, single animal"
    A 2.x folder is TensorFlow rather than PyTorch and says so rather than
    reporting a missing config; export it by hand. Multi-animal models — the
    ones with a `paf` or `identity` head — are reported by name rather than
    converted wrongly.

Everything in the sidecar is read from the model's own config, except the output
stride, which is measured from the network. That is deliberate: the effective
stride is the backbone's divided by whatever the head's deconvolutions undo, and
a ResNet-50 and an HRNet-w32 trained on the same data differ by a factor of
four. Nothing is defaulted — a wrong stride does not fail, it shifts every
keypoint by a constant and still draws a plausible skeleton.

## SLEAP models

Point GLIDER at the folder SLEAP produced — the one holding `training_config.json`
and `best_model.h5`. The first time you select it, GLIDER converts the model to
ONNX and keeps the result beside it; every later run reuses that.

```bash
pip install glider-sleap
```

That is a plugin, not an extra. It carries TensorFlow and tf2onnx, which is a
large thing to put in the dependency tree of a lab that tracks with YOLO and has
never opened SLEAP — installing it is how you say you have SLEAP models. GLIDER
itself carries onnxruntime and nothing else.

**It does not install SLEAP, and does not need to** — a SLEAP model is saved as
an ordinary Keras checkpoint, which TensorFlow opens on its own. (SLEAP itself
is pinned to older Pythons than GLIDER runs on, so it could not share the
environment anyway. That limit applies to *running* SLEAP, not to reading what
it wrote.)

!!! note "Single-instance models only"
    GLIDER runs SLEAP's **single_instance** models. Top-down and bottom-up
    models are multi-animal architectures with a different inference structure,
    and selecting one reports which heads it found rather than guessing.

Conversion takes a few seconds to a couple of minutes depending on the model,
and runs once. If you retrain and drop a new checkpoint into the same folder,
GLIDER notices the checkpoint changed and reconverts — a stale ONNX would run
perfectly well while answering with the network you just replaced.
