"""The Loop node's properties panel offers a "prompt for iterations at run time"
checkbox that stores a ``prompt_count`` flag. When set, running the enclosing
function from a Runner/dashboard button prompts the operator for the loop count
(one-shot) via the existing run-param mechanism (see find_run_param).

Mirrors test_node_editor_pwm_range: bypass the heavy NodeEditorController.__init__
and read the built properties widget.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QCheckBox

from glider.gui.panels.node_editor_controller import NodeEditorController

pytestmark = pytest.mark.usefixtures("qtbot")


def _loop_controller(node_config):
    ctrl = NodeEditorController.__new__(NodeEditorController)  # skip heavy __init__
    ctrl._graph_view = SimpleNamespace(
        nodes={"n1": SimpleNamespace(node_type="Loop", _actual_node_type=None)}
    )
    ctrl._session_fn = lambda: SimpleNamespace(get_node=lambda nid: node_config)
    ctrl._hardware_manager = SimpleNamespace(devices={}, get_device=lambda i: None)
    ctrl._zone_config = None
    captured: dict = {}
    ctrl._properties_dock = SimpleNamespace(setWidget=lambda w: captured.__setitem__("w", w))
    return ctrl, captured


def _prompt_checkbox(widget):
    return [c for c in widget.findChildren(QCheckBox) if "iteration" in c.text().lower()]


def test_loop_properties_include_prompt_iterations_checkbox(qtbot):
    node_config = SimpleNamespace(
        device_id=None, state={"count": 500, "delay": 0.01, "prompt_count": True}
    )
    ctrl, captured = _loop_controller(node_config)

    ctrl._update_properties_panel("n1")

    checks = _prompt_checkbox(captured["w"])
    assert checks, "expected a prompt-for-iterations checkbox on the Loop node"
    assert checks[0].isChecked(), "checkbox should reflect saved prompt_count=True"


def test_loop_prompt_checkbox_defaults_unchecked(qtbot):
    node_config = SimpleNamespace(device_id=None, state={"count": 500, "delay": 0.01})
    ctrl, captured = _loop_controller(node_config)

    ctrl._update_properties_panel("n1")

    checks = _prompt_checkbox(captured["w"])
    assert checks and not checks[0].isChecked()


def test_loop_prompt_checkbox_toggle_stores_prompt_count(qtbot):
    node_config = SimpleNamespace(device_id=None, state={"count": 500, "delay": 0.01})
    ctrl, captured = _loop_controller(node_config)
    calls = []
    ctrl._on_node_property_changed = lambda nid, key, val: calls.append((nid, key, val))

    ctrl._update_properties_panel("n1")
    checks = _prompt_checkbox(captured["w"])
    assert checks
    checks[0].setChecked(True)

    assert ("n1", "prompt_count", True) in calls
