# glider-sleap-nn

Run [sleap-nn](https://github.com/talmolab/sleap-nn) pose models in GLIDER.

Install this, point GLIDER at the folder sleap-nn wrote, and it converts once
and runs. There is nothing to export from the SLEAP GUI beforehand.

```bash
uv pip install glider-sleap-nn
```

## Which plugin do I need?

SLEAP has two generations that write entirely different model folders, and no
filename is shared between them. Look in the folder your training wrote:

| you have | that is | install |
|---|---|---|
| `training_config.yaml` + `best.ckpt` | sleap-nn (PyTorch) | **this plugin** |
| `training_config.json` + `best_model.h5` | classic SLEAP (TensorFlow) | `glider-sleap` |

Not `labels.slp` in either case — that is the labels file, and there is no
network in it.

If you install the wrong one GLIDER says so by name rather than telling you the
folder is not a pose model.

## What it does

GLIDER runs pose models through onnxruntime, described by a `glider_pose.json`
sidecar next to the model. sleep-nn writes a Lightning checkpoint and a
`training_config.yaml`. This plugin is the bridge.

Pick a model folder in the Camera panel. If it is a sleap-nn folder that has not
been converted, GLIDER asks once, converts, and writes `model.onnx` and
`glider_pose.json` beside the checkpoint. Every later run loads the ONNX
directly. Retrain into the same folder and it notices the stamp is stale and
offers to convert again — a conversion that quietly answered with the network
you just replaced would be the worst outcome available.

## It needs sleap-nn, and installs it for you

This is the one real difference from `glider-sleap`, and it is not arbitrary.

A *classic* SLEAP model is an ordinary Keras checkpoint. TensorFlow opens it on
its own; nothing in `sleap` is involved. So `glider-sleap` can depend on
`tensorflow-cpu` directly and convert in-process.

A sleap-nn checkpoint cannot be read that way. Rebuilding the network needs
sleap-nn's own Lightning classes, and those bring torch, torchvision and
lightning with them — about 1.5 GB.

So this plugin depends on **nothing**. The first time you select a sleap-nn
model it builds a private environment under `~/.glider/envs/sleap-nn` with
`uv`, installs sleap-nn into that, and shells into it to convert. GLIDER's own
environment is untouched. It is the same arrangement `glider-dlc` uses, for the
same reason.

A lab that already has sleap-nn working can point at it and nothing is
downloaded:

```bash
export GLIDER_SLEAP_NN_ENV=/path/to/that/venv
```

## What gets exported

The confidence-map module, not sleap-nn's inference wrapper.

sleap-nn's `SingleInstanceONNXWrapper` is a whole inference graph: uint8 in,
decoded peak coordinates out. GLIDER's runner wants confidence maps and does its
own peak extraction, so exporting the wrapper would mean teaching GLIDER a
second kind of pose model. `SingleInstanceLightningModule.forward` already
returns exactly the confidence maps, so that is what is traced — which also
means both SLEAP plugins produce the same artifact and GLIDER cannot tell them
apart.

The export itself is sleap-nn's own `export_to_onnx`, including its numerical
parity check against PyTorch. That check is why `onnxruntime` is installed into
the private environment: without it the check silently degrades to a warning,
and it is the only thing that catches a graph which traces cleanly and is
numerically wrong.

Verified against a real single-instance UNet model (sleap-nn 0.3.3, 7 keypoints,
640×480): the exported ONNX matches the PyTorch module to 1.5e-07, and GLIDER's
decode picks the same keypoint pixels as sleap-nn on 20 real frames — 140 of 140
points agreeing exactly.

## Scope

Single animal. sleap-nn 0.3.3 ships nine head types; anything other than
`single_instance` is refused by name rather than converted into a network GLIDER
would decode as if it tracked one animal.

Python 3.11 to 3.13, the same range GLIDER supports. The private environment
pins its own Python, so GLIDER's interpreter does not constrain sleap-nn's.
