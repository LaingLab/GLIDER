"""
Logic Nodes - Mathematical operations, comparisons, timers, and controllers.
"""

import logging

from glider.nodes.logic.comparison_nodes import (
    InRangeNode,
    ThresholdNode,
)
from glider.nodes.logic.control_nodes import (
    PIDNode,
    ToggleNode,
)
from glider.nodes.logic.flow_nodes import (
    DelayNode,
    SequenceNode,
    TimerNode,
)
from glider.nodes.logic.math_nodes import (
    AddNode,
    ClampNode,
    DivideNode,
    MapRangeNode,
    MultiplyNode,
    SubtractNode,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AddNode",
    "SubtractNode",
    "MultiplyNode",
    "DivideNode",
    "MapRangeNode",
    "ClampNode",
    "ThresholdNode",
    "InRangeNode",
    "SequenceNode",
    "DelayNode",
    "TimerNode",
    "PIDNode",
    "ToggleNode",
    "register_math_nodes",
    "register_comparison_nodes",
    "register_logic_control_nodes",
]


def register_math_nodes(flow_engine) -> None:
    """Register arithmetic data-transform nodes."""
    flow_engine.register_node("Add", AddNode)
    flow_engine.register_node("Subtract", SubtractNode)
    flow_engine.register_node("Multiply", MultiplyNode)
    flow_engine.register_node("Divide", DivideNode)
    flow_engine.register_node("MapRange", MapRangeNode)
    flow_engine.register_node("Clamp", ClampNode)
    logger.info("Registered math nodes")


def register_comparison_nodes(flow_engine) -> None:
    """Register comparison/threshold data nodes."""
    flow_engine.register_node("Threshold", ThresholdNode)
    flow_engine.register_node("InRange", InRangeNode)
    logger.info("Registered comparison nodes")


def register_logic_control_nodes(flow_engine) -> None:
    """Register Toggle + PID controller nodes.

    Named distinctly from ``register_control_nodes`` in
    ``glider.nodes.control_nodes`` (top-level — Loop, WaitForInput) to avoid
    collision; both groups load.
    """
    flow_engine.register_node("Toggle", ToggleNode)
    flow_engine.register_node("PID", PIDNode)
    logger.info("Registered logic-control nodes (Toggle, PID)")
