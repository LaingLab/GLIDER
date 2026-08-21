"""GLIDER plugin for the Maimu BLE stimulator.

Registers a ``Maimu`` device type and a ``Maimu`` node. Both are declared in
the tables below as well as in ``pyproject.toml``'s entry points; GLIDER's
PluginManager reads either, and registering the same class under the same name
twice is a no-op, so carrying both costs nothing and means the plugin works
whether it was pip-installed or dropped into ``~/.glider/plugins/``.

The device is a Nordic/Zephyr peripheral with one writable characteristic
taking ``on``, ``off``, or ``<period_ms>,<duration_s>``. It lives outside core
because it is one lab's hardware: GLIDER ships the BLE transport, and this
ships the protocol that runs over it.
"""

from glider_maimu.device import (
    DEFAULT_SERVICE_UUID,
    DEFAULT_WRITE_CHAR_UUID,
    MaimuDevice,
)
from glider_maimu.node import MaimuNode

__version__ = "0.1.0"

# Read by PluginManager._register_plugin_components.
DEVICE_TYPES = {"Maimu": MaimuDevice}
NODE_TYPES = {"Maimu": MaimuNode}

__all__ = [
    "DEVICE_TYPES",
    "NODE_TYPES",
    "MaimuDevice",
    "MaimuNode",
    "DEFAULT_SERVICE_UUID",
    "DEFAULT_WRITE_CHAR_UUID",
    "__version__",
]
