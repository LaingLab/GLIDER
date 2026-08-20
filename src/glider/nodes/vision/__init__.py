"""
Vision Nodes - Nodes for computer vision integration.
"""

from glider.nodes.vision.behavior_nodes import BehaviorInputNode, register_behavior_nodes
from glider.nodes.vision.zone_nodes import ZoneInputNode, register_zone_nodes

__all__ = [
    "BehaviorInputNode",
    "ZoneInputNode",
    "register_behavior_nodes",
    "register_zone_nodes",
]
