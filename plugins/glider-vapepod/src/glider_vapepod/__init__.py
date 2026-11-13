"""GLIDER plugin for the vape box side pod.

Registers two device types, ``VapePod`` (Bluetooth) and ``VapePodUSB`` (a USB
cable), and a ``VapePod`` node that drives either. Both are declared in
the tables below and found through ``pyproject.toml``'s module-style entry
point, the same way glider-maimu registers ``Maimu``, so the plugin works
whether it was pip-installed or dropped into ``~/.glider/plugins/``.

The pod is a XIAO nRF52840 running ``firmware/vapepod/vapepod.ino``: it takes
``off``, ``pump,<percent>`` and ``puff,<ms>,<percent>`` over BLE or as lines on
its USB serial port, and pushes its status back. GLIDER ships the BLE and
serial transports; this ships the protocol.
"""

from glider_vapepod.device import (
    DEFAULT_COMMAND_CHAR_UUID,
    DEFAULT_NAME,
    DEFAULT_SERVICE_UUID,
    DEFAULT_STATUS_CHAR_UUID,
    MAX_PUFF_MS,
    VapePodDevice,
)
from glider_vapepod.node import VapePodNode
from glider_vapepod.wired import VapePodUSBDevice

__version__ = "0.1.0"

# Read by PluginManager._register_plugin_components.
DEVICE_TYPES = {"VapePod": VapePodDevice, "VapePodUSB": VapePodUSBDevice}
NODE_TYPES = {"VapePod": VapePodNode}

__all__ = [
    "DEVICE_TYPES",
    "NODE_TYPES",
    "VapePodDevice",
    "VapePodUSBDevice",
    "VapePodNode",
    "DEFAULT_NAME",
    "DEFAULT_SERVICE_UUID",
    "DEFAULT_COMMAND_CHAR_UUID",
    "DEFAULT_STATUS_CHAR_UUID",
    "MAX_PUFF_MS",
    "__version__",
]
