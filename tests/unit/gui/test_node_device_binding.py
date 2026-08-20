"""Picking and clearing a node's device in the properties panel.

The save path reads the *runtime* node's binding, so the properties panel and
the runtime node have to stay in step in both directions. Binding on select was
already wired; clearing left the runtime node bound, which now means the next
save writes back the device the user just removed.
"""

from types import SimpleNamespace

import pytest

from glider.gui.panels.node_editor_controller import NodeEditorController
from glider.nodes.experiment_nodes import OutputNode

pytestmark = pytest.mark.usefixtures("qtbot")


class _Device:
    id = "led1"
    name = "LED"
    device_type = "DigitalOutput"


def _controller(node, node_config, device=None):
    device = device or _Device()
    ctrl = NodeEditorController.__new__(NodeEditorController)  # skip heavy __init__
    ctrl._session_fn = lambda: SimpleNamespace(
        get_node=lambda nid: node_config,
        _mark_dirty=lambda: None,
    )
    ctrl._hardware_manager = SimpleNamespace(
        devices={"led1": device},
        get_device=lambda i: device if i == "led1" else None,
    )
    ctrl._core = SimpleNamespace(
        flow_engine=SimpleNamespace(get_node=lambda nid: node if nid == "out1" else None)
    )
    ctrl._update_properties_panel = lambda nid: None  # shadows the real method
    return ctrl


def test_selecting_a_device_binds_the_runtime_node(qtbot):
    node = OutputNode()
    config = SimpleNamespace(device_id=None)
    ctrl = _controller(node, config)

    ctrl._on_node_device_changed("out1", "led1")

    assert config.device_id == "led1"
    assert node.device is not None
    assert node.device.name == "LED"


def test_clearing_the_selection_unbinds_the_runtime_node(qtbot):
    """Otherwise the next save re-persists the device that was just cleared."""
    device = _Device()
    node = OutputNode()
    node.bind_device(device)
    config = SimpleNamespace(device_id="led1")
    ctrl = _controller(node, config, device)

    ctrl._on_node_device_changed("out1", None)

    assert config.device_id is None
    assert node.device is None


def test_clearing_survives_a_node_with_no_runtime_counterpart(qtbot):
    """A node in the session but not (yet) in the flow engine must not raise."""
    config = SimpleNamespace(device_id="led1")
    ctrl = _controller(OutputNode(), config)

    ctrl._on_node_device_changed("missing", None)

    assert config.device_id is None


def test_switching_devices_rebinds(qtbot):
    node = OutputNode()
    node.bind_device(_Device())
    config = SimpleNamespace(device_id="led1")

    other = _Device()
    other.id = "led2"
    other.name = "Other LED"
    ctrl = _controller(node, config)
    ctrl._hardware_manager = SimpleNamespace(
        devices={"led2": other}, get_device=lambda i: other if i == "led2" else None
    )

    ctrl._on_node_device_changed("out1", "led2")

    assert config.device_id == "led2"
    assert node.device.name == "Other LED"
