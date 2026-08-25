"""GLIDER plugin: run SLEAP models.

Registers a pose converter that turns the folder SLEAP wrote into the
``model.onnx`` + ``glider_pose.json`` pair GLIDER runs. Select the folder and it
converts, once, keeping the result beside the model.

This ships separately from GLIDER because of what it costs: TensorFlow, which
is a large thing to put in the dependency tree of a lab that tracks with YOLO
and has never opened SLEAP. Installing this plugin is how you say you have
SLEAP models.

It does **not** depend on sleap, and does not need to. A SLEAP model is saved
as an ordinary Keras checkpoint, which TensorFlow opens on its own. (sleap
itself is pinned to older Pythons than GLIDER runs on, so it could not share
the environment anyway -- that limit applies to *running* SLEAP, not to reading
what it wrote.)
"""

from glider_sleap.converter import SleapConverter

__version__ = "0.1.0"

#: Read by PluginManager and registered into POSE_CONVERTERS.
POSE_CONVERTERS = {"sleap": SleapConverter}

__all__ = ["POSE_CONVERTERS", "SleapConverter", "__version__"]
