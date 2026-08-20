"""The Maimu node's properties panel and its library entry.

Same shape as test_node_editor_pwm_range.py: bypass the heavy
NodeEditorController.__init__ and read the widgets it builds.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QComboBox, QSpinBox

from glider.gui.panels.node_editor_controller import NodeEditorController

pytestmark = pytest.mark.usefixtures("qtbot")


class _Maimu:
    device_type = "Maimu"
    name = "stim"

    @property
    def actions(self):
        return {"on": None, "off": None, "pulse": None, "write": None}


def _controller(state, dev=None):
    dev = dev or _Maimu()
    node_config = SimpleNamespace(device_id="maimu1", state=state)
    ctrl = NodeEditorController.__new__(NodeEditorController)  # skip heavy __init__
    ctrl._graph_view = SimpleNamespace(
        nodes={"n1": SimpleNamespace(node_type="Maimu", _actual_node_type=None)}
    )
    saved: list = []
    ctrl._session_fn = lambda: SimpleNamespace(
        get_node=lambda nid: node_config,
        update_node_state=lambda nid, patch: saved.append((nid, patch)),
    )
    ctrl._hardware_manager = SimpleNamespace(devices={"maimu1": dev}, get_device=lambda i: dev)
    ctrl._zone_config = None
    captured: dict = {}
    ctrl._properties_dock = SimpleNamespace(setWidget=lambda w: captured.__setitem__("w", w))
    return ctrl, captured, saved


def _widgets(captured):
    panel = captured["w"]
    spins = {s.suffix().strip(): s for s in panel.findChildren(QSpinBox)}
    mode = next(c for c in panel.findChildren(QComboBox) if c.findData("pulse") >= 0)
    return mode, spins


def test_saved_command_is_shown(qtbot):
    ctrl, captured, _ = _controller({"mode": "pulse", "period_ms": 250, "duration_s": 30})

    ctrl._update_properties_panel("n1")

    mode, spins = _widgets(captured)
    assert mode.currentData() == "pulse"
    assert spins["ms"].value() == 250
    assert spins["s"].value() == 30


def test_device_selector_offers_the_maimu(qtbot):
    ctrl, captured, _ = _controller({})

    ctrl._update_properties_panel("n1")

    combos = captured["w"].findChildren(QComboBox)
    devices = next(c for c in combos if c.findData("maimu1") >= 0)
    assert devices.currentData() == "maimu1"


def test_period_and_duration_are_greyed_out_for_on(qtbot):
    """They mean nothing outside a pulse; leaving them live would read as though
    they applied."""
    ctrl, captured, _ = _controller({"mode": "on"})

    ctrl._update_properties_panel("n1")

    _mode, spins = _widgets(captured)
    assert not spins["ms"].isEnabled()
    assert not spins["s"].isEnabled()


def test_period_and_duration_are_live_for_pulse(qtbot):
    ctrl, captured, _ = _controller({"mode": "pulse"})

    ctrl._update_properties_panel("n1")

    _mode, spins = _widgets(captured)
    assert spins["ms"].isEnabled()
    assert spins["s"].isEnabled()


def test_switching_mode_persists_and_toggles_the_fields(qtbot):
    ctrl, captured, saved = _controller({"mode": "pulse"})
    ctrl._update_properties_panel("n1")
    mode, spins = _widgets(captured)

    mode.setCurrentIndex(mode.findData("off"))

    assert ("n1", {"mode": "off"}) in saved
    assert not spins["ms"].isEnabled()


def test_editing_period_persists(qtbot):
    ctrl, captured, saved = _controller({"mode": "pulse", "period_ms": 500})
    ctrl._update_properties_panel("n1")
    _mode, spins = _widgets(captured)

    spins["ms"].setValue(125)

    assert ("n1", {"period_ms": 125}) in saved


def test_editing_duration_persists(qtbot):
    ctrl, captured, saved = _controller({"mode": "pulse", "duration_s": 10})
    ctrl._update_properties_panel("n1")
    _mode, spins = _widgets(captured)

    spins["s"].setValue(45)

    assert ("n1", {"duration_s": 45}) in saved


def test_library_offers_a_maimu_button(qtbot):
    """Registered but unreachable would be no feature at all."""
    from glider.gui.panels.node_library_panel import DraggableNodeButton, NodeLibraryPanel

    panel = NodeLibraryPanel(lambda: None, SimpleNamespace())
    qtbot.addWidget(panel)

    types = {b._node_type for b in panel.findChildren(DraggableNodeButton)}
    assert "Maimu" in types
