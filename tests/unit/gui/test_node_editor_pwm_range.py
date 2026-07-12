"""B2: the graph-editor Output node's PWM property spin takes its range from the
bound device's value_spec('set'), not a hardcoded 0-255, so a 12-bit board
(0-4095) is authorable and the editor matches what the node will actually write.

Bypasses the heavy NodeEditorController.__init__ (__new__ + fakes for the few
collaborators _update_properties_panel touches) and reads the built widget.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QSpinBox

from glider.gui.panels.node_editor_controller import NodeEditorController
from glider.hal.value_spec import KIND_WHOLE, ActionValueSpec

pytestmark = pytest.mark.usefixtures("qtbot")


class _PWM:
    device_type = "PWMOutput"
    name = "motor"

    def value_spec(self, action):
        return ActionValueSpec(KIND_WHOLE, 0, 4095) if action == "set" else None


def _controller(dev, node_config):
    ctrl = NodeEditorController.__new__(NodeEditorController)  # skip heavy __init__
    ctrl._graph_view = SimpleNamespace(
        nodes={"n1": SimpleNamespace(node_type="Output", _actual_node_type=None)}
    )
    ctrl._session_fn = lambda: SimpleNamespace(get_node=lambda nid: node_config)
    ctrl._hardware_manager = SimpleNamespace(devices={"pwm1": dev}, get_device=lambda i: dev)
    ctrl._zone_config = None
    captured: dict = {}
    ctrl._properties_dock = SimpleNamespace(setWidget=lambda w: captured.__setitem__("w", w))
    return ctrl, captured


def test_output_pwm_range_follows_value_spec(qtbot):
    dev = _PWM()
    node_config = SimpleNamespace(device_id="pwm1", state={"value": 300})
    ctrl, captured = _controller(dev, node_config)

    ctrl._update_properties_panel("n1")

    spins = [s for s in captured["w"].findChildren(QSpinBox) if s.maximum() == 4095]
    assert spins, "expected a PWM spin with max 4095 from the device's value_spec"
    # A value the old 0-255 cap would have clamped is preserved.
    assert spins[0].value() == 300


def test_output_pwm_falls_back_to_8bit_without_a_spec(qtbot):
    class _NoSpecPWM:
        device_type = "PWMOutput"
        name = "motor"

        def value_spec(self, action):
            return None

    node_config = SimpleNamespace(device_id="pwm1", state={"value": 10})
    ctrl, captured = _controller(_NoSpecPWM(), node_config)

    ctrl._update_properties_panel("n1")

    spins = [s for s in captured["w"].findChildren(QSpinBox) if s.value() == 10]
    assert spins and spins[0].maximum() == 255
