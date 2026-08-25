# glider-dlc

Run [DeepLabCut](https://deeplabcut.github.io) pose models in GLIDER.

Install this, point GLIDER at the folder DeepLabCut wrote, and it converts once
and runs. There is nothing to export beforehand and no command to remember.

```bash
uv pip install glider-dlc
```

## What it does

GLIDER runs pose models through onnxruntime, described by a `glider_pose.json`
sidecar next to the model. DeepLabCut writes a `pytorch_config.yaml` and a
`snapshot-*.pt`. This plugin is the bridge: it recognises the second shape and
produces the first.

Pick a model folder in the Camera panel — the `train` folder, or the project
above it. If it is a DeepLabCut folder that has not been converted, GLIDER asks
once, converts, and writes `model.onnx` and `glider_pose.json` beside the
snapshot. Every later run loads the ONNX directly. Retrain and drop a new
snapshot in and it notices the stamp is stale and offers to convert again — a
conversion that quietly answered with the network you just replaced would be
the worst outcome available.

## The environment it builds

DeepLabCut cannot live in GLIDER's environment. It is needed to *read* a
snapshot at all — a DLC checkpoint is a bare PyTorch `state_dict`, so rebuilding
the network from one needs DeepLabCut's own model classes — but it brings
torch, timm, albumentations and a long tail with it, about **1.3 GB installed**.

The alternative to putting it there is not making you build an environment by
hand. It is building one for you. The first time you select a DeepLabCut model,
this plugin uses `uv` to create a private Python 3.12 virtualenv under
`~/.glider/envs/deeplabcut`, installs DeepLabCut into it, and shells into that
interpreter to convert. GLIDER's own environment is untouched. The dialog tells
you the size before anything is downloaded, and it happens once per machine.

The plugin itself has **no dependencies at all**. `uv` is found on `PATH` — it
is what GLIDER's own installer already uses — and its absence is reported with
an install link rather than a crash.

Already have DeepLabCut working somewhere?

```bash
export GLIDER_DLC_ENV=/path/to/your/dlc/venv
```

Nothing is downloaded, and GLIDER converts in your environment instead.

## Scope

**DeepLabCut 3.x, single animal.** A converted model runs as one decode per
frame producing one set of keypoints, which is what the behaviour classifier
downstream expects. Multi-animal models — the ones with a `paf` or `identity`
head — are reported by name rather than converted wrongly.

**DeepLabCut 2.x is TensorFlow, not PyTorch**, and is not converted
automatically. A 2.x folder is recognised and says so, rather than reporting a
missing `pytorch_config.yaml`; export it by hand with `tools/export_pose_onnx.py`
in your own DLC 2.x environment.

## What is read, and what is refused

Everything in the sidecar comes from the model's own config or is measured from
the network. Nothing is defaulted, because none of these fail loudly when
wrong — a bad stride or normalisation shifts every keypoint by a constant and
still draws a plausible skeleton:

| Field | Source |
|---|---|
| `keypoint_names` | `metadata.bodyparts` — required, and must be unique |
| `locref_stdev` | the head's `predictor.locref_std` — required unless `location_refinement` is off |
| `mean` / `std` / `divide_by_255` | `data.inference.normalize_images`, which means ImageNet statistics |
| `pad_to_stride` | `data.inference.auto_padding` (32 for HRNet, absent for ResNet) |
| `output_stride` | **measured from the network**, see below |

The stride is measured rather than read because it is the backbone's stride
divided by whatever the head's deconvolutions undo, and those move between
backbones: a ResNet-50 and an HRNet-w32 trained on the same data differ by a
factor of four. It is measured from *two* input sizes, not one, because every
DeepLabCut head adds one cell past the edge (`out = in / stride + 1`) — a
single ratio reads a stride-8 ResNet as 7.53 at 128 px and 7.88 at 512 px,
while the difference between two sizes gives 8 exactly.

A model whose normalisation cannot be expressed in the sidecar
(`scale_to_unit_range`, which rescales each frame against its own extremes) is
refused rather than converted wrongly.

## Verified

Against DeepLabCut 3.0.1 on Python 3.12 with torch 2.13, for both backbone
families:

| | stride | ONNX vs. DeepLabCut |
|---|---|---|
| ResNet-50 | 8 | 3.1e-06 relative |
| HRNet-w32 | 2 | 1.7e-06 relative |

Both export with dynamic height and width, so any camera resolution runs
against the same file, and both correctly detect a retrained snapshot as stale.

## Development

```bash
uv pip install -e ./plugins/glider-dlc
PYTHONPATH="src;plugins/glider-dlc/src" QT_QPA_PLATFORM=offscreen pytest plugins/glider-dlc/tests/
```

The suite runs without DeepLabCut — that is the point of the plugin, so it is
also the point of the tests. Every `uv` call is faked; what is checked is the
decision-making around them, and the path and config logic GLIDER asks about
before anything is downloaded.

The one test that performs a real conversion is marked `slow` and skips
everywhere DeepLabCut is not importable, including CI. To run it, use the
provisioned interpreter:

```bash
~/.glider/envs/deeplabcut/Scripts/python -m pytest plugins/glider-dlc -m slow
```
