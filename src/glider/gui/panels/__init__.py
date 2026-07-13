"""
GLIDER GUI Panels - Dock widget panels for the main window.
"""

from glider.gui.panels.camera_panel import CameraPanel, CameraPreviewWidget
from glider.gui.panels.device_control_panel import DeviceControlPanel
from glider.gui.panels.hardware_panel import HardwarePanel
from glider.gui.panels.node_editor_controller import NodeEditorController
from glider.gui.panels.node_library_panel import (
    DraggableNodeButton,
    EditableDraggableButton,
    NodeLibraryPanel,
)

__all__ = [
    "CameraPanel",
    "CameraPreviewWidget",
    "DeviceControlPanel",
    "DraggableNodeButton",
    "EditableDraggableButton",
    "HardwarePanel",
    "NodeEditorController",
    "NodeLibraryPanel",
]
