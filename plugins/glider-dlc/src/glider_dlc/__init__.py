"""GLIDER plugin: run DeepLabCut models.

Registers a pose converter that turns a DeepLabCut 3.x model folder into the
``model.onnx`` + ``glider_pose.json`` pair GLIDER runs. Select the folder and it
converts, once, keeping the result beside the snapshot.

This ships separately from GLIDER, and converts in an environment of its own,
for a reason that is not negotiable: a DeepLabCut snapshot is a bare PyTorch
``state_dict``, so rebuilding the network from one needs DeepLabCut's own model
classes -- and DeepLabCut brings torch, timm, albumentations and a long tail
with it, about 1.3 GB installed. GLIDER carries onnxruntime and nothing else.

So the plugin itself is small and depends on nothing. The first time a
DeepLabCut model is selected it builds a private environment under
``~/.glider/envs`` with ``uv``, installs DeepLabCut into that, and shells into
it to convert. A lab that already has DeepLabCut working can point at it with
``GLIDER_DLC_ENV`` and nothing is downloaded at all.
"""

from glider_dlc.converter import DlcConverter

__version__ = "0.1.0"

#: Read by PluginManager and registered into POSE_CONVERTERS.
POSE_CONVERTERS = {"deeplabcut": DlcConverter}

__all__ = ["POSE_CONVERTERS", "DlcConverter", "__version__"]
