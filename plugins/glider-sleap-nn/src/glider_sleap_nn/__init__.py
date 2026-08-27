"""GLIDER plugin: run sleap-nn (PyTorch SLEAP) models.

Registers a pose converter that turns a sleap-nn model folder into the
``model.onnx`` + ``glider_pose.json`` pair GLIDER runs. Select the folder and it
converts, once, keeping the result beside the checkpoint.

**This is not the same plugin as ``glider-sleap``.** SLEAP has two generations
that write entirely different folders, and no filename is shared between them:

===================  ==========================  ==========================
                     classic SLEAP               sleap-nn
===================  ==========================  ==========================
framework            TensorFlow / Keras          PyTorch / Lightning
config               ``training_config.json``    ``training_config.yaml``
weights              ``best_model.h5``           ``best.ckpt``
plugin               ``glider-sleap``            this one
===================  ==========================  ==========================

The split runs deeper than the file names. A classic SLEAP model is an ordinary
Keras checkpoint, which TensorFlow opens with nothing from ``sleap`` involved --
so ``glider-sleap`` can depend on ``tensorflow-cpu`` directly and convert
in-process. A sleap-nn checkpoint cannot be read that way: rebuilding the
network needs sleap-nn's own Lightning classes, and those bring torch with them.

So this plugin depends on nothing. The first time a sleap-nn model is selected
it builds a private environment under ``~/.glider/envs`` with ``uv``, installs
sleap-nn into that, and shells into it to convert -- the same arrangement
``glider-dlc`` uses, for the same reason. A lab that already has sleap-nn
working can point at it with ``GLIDER_SLEAP_NN_ENV`` and nothing is downloaded.
"""

from glider_sleap_nn.converter import SleapNNConverter

__version__ = "0.1.0"

#: Read by PluginManager and registered into POSE_CONVERTERS.
POSE_CONVERTERS = {"sleap_nn": SleapNNConverter}

__all__ = ["POSE_CONVERTERS", "SleapNNConverter", "__version__"]
