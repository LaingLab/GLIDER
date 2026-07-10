import pytest
from PyQt6.QtWidgets import QPushButton

from glider.gui.panels.hardware_panel import HardwarePanel

pytestmark = pytest.mark.usefixtures("qtbot")


def _panel(qtbot, mock_hardware_manager, show_add_buttons=True):
    panel = HardwarePanel(
        mock_hardware_manager,
        session_fn=lambda: None,
        run_async_fn=lambda c: None,
        show_add_buttons=show_add_buttons,
    )
    qtbot.addWidget(panel)
    return panel


def test_add_buttons_shown_by_default(qtbot, mock_hardware_manager):
    panel = _panel(qtbot, mock_hardware_manager)
    texts = [b.text() for b in panel.findChildren(QPushButton)]
    assert "+ Board" in texts
    assert "+ Device" in texts


def test_add_buttons_hidden_when_disabled(qtbot, mock_hardware_manager):
    panel = _panel(qtbot, mock_hardware_manager, show_add_buttons=False)
    assert not any(b.text() in ("+ Board", "+ Device") for b in panel.findChildren(QPushButton))
