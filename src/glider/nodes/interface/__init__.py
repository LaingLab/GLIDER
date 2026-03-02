"""
Interface Nodes - Dashboard widgets for user interaction.

These nodes expose controls to the Runner UI and allow
users to interact with the experiment during execution.
"""

from glider.nodes.interface.display_nodes import (
    ChartNode,
    GaugeNode,
    LabelNode,
    LEDIndicatorNode,
)
from glider.nodes.interface.audio_nodes import (
    AudioPlaybackNode,
    register_audio_nodes,
)
from glider.nodes.interface.input_nodes import (
    ButtonNode,
    NumericInputNode,
    SliderNode,
    ToggleSwitchNode,
)

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
]
