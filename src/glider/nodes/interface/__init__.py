"""
Interface Nodes - Dashboard widgets for user interaction.

These nodes expose controls to the Runner UI and allow
users to interact with the experiment during execution.
"""

import logging

from glider.nodes.interface.audio_nodes import (
    AudioPlaybackNode,
    register_audio_nodes,
)
from glider.nodes.interface.display_nodes import (
    ChartNode,
    GaugeNode,
    LabelNode,
    LEDIndicatorNode,
)
from glider.nodes.interface.input_nodes import (
    ButtonNode,
    NumericInputNode,
    SliderNode,
    ToggleSwitchNode,
)
from glider.nodes.interface.video_nodes import (
    VideoPlaybackNode,
    register_video_nodes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LabelNode",
    "GaugeNode",
    "ChartNode",
    "LEDIndicatorNode",
    "ButtonNode",
    "ToggleSwitchNode",
    "SliderNode",
    "NumericInputNode",
    "AudioPlaybackNode",
    "register_audio_nodes",
    "VideoPlaybackNode",
    "register_video_nodes",
    "register_interface_nodes",
]


def register_interface_nodes(flow_engine) -> None:
    """Register input + display interface widgets with the flow engine.

    Audio and video registration live in their own ``register_audio_nodes``
    and ``register_video_nodes`` functions (legacy split — kept for
    backwards compatibility) and are still invoked from ``GliderCore``.
    """
    flow_engine.register_node("Button", ButtonNode)
    flow_engine.register_node("ToggleSwitch", ToggleSwitchNode)
    flow_engine.register_node("Slider", SliderNode)
    flow_engine.register_node("NumericInput", NumericInputNode)
    flow_engine.register_node("Label", LabelNode)
    flow_engine.register_node("Gauge", GaugeNode)
    flow_engine.register_node("Chart", ChartNode)
    flow_engine.register_node("LEDIndicator", LEDIndicatorNode)
    logger.info("Registered interface (input + display) nodes")
