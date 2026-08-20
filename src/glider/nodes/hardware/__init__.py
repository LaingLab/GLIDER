"""
Hardware Nodes - Interface with physical devices.

These nodes are proxies for physical devices and hold references
to specific hardware driver instances.
"""

import logging

from glider.nodes.hardware.analog_nodes import (
    AnalogReadNode,
    PWMWriteNode,
)
from glider.nodes.hardware.device_nodes import (
    DeviceActionNode,
    DeviceReadNode,
)
from glider.nodes.hardware.digital_nodes import (
    DigitalReadNode,
    DigitalWriteNode,
)
from glider.nodes.hardware.maimu_nodes import MaimuNode

logger = logging.getLogger(__name__)

__all__ = [
    "DigitalWriteNode",
    "DigitalReadNode",
    "AnalogReadNode",
    "PWMWriteNode",
    "DeviceActionNode",
    "DeviceReadNode",
    "MaimuNode",
    "register_hardware_nodes",
]


def register_hardware_nodes(flow_engine) -> None:
    """Register every hardware node class with the flow engine.

    Without this, the Builder library shows these nodes but
    ``flow_engine.create_node(type_str, ...)`` returns ``None`` and dropping
    one onto the canvas silently does nothing.
    """
    flow_engine.register_node("DigitalWrite", DigitalWriteNode)
    flow_engine.register_node("DigitalRead", DigitalReadNode)
    flow_engine.register_node("AnalogRead", AnalogReadNode)
    flow_engine.register_node("PWMWrite", PWMWriteNode)
    flow_engine.register_node("DeviceAction", DeviceActionNode)
    flow_engine.register_node("DeviceRead", DeviceReadNode)
    flow_engine.register_node("Maimu", MaimuNode)
    logger.info("Registered hardware nodes")
