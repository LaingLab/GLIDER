"""
Runner Device Controls - Touch-optimized direct device drive grid.

Builds one control block per controllable OUTPUT device found on the
hardware manager: big ON / OFF / Toggle buttons for ``DigitalOutput``
devices, and a large slider (0-255) for ``PWMOutput`` devices. Other
device types are not rendered (deferred).

This widget is PURE UI: it does not drive any device itself. It only
emits signals; an external async handler is responsible for calling the
actual drive helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.core.hardware_manager import HardwareManager

_BUTTON_MIN_HEIGHT = 56
_SLIDER_MIN_HEIGHT = 56


class RunnerDeviceControls(QWidget):
    """Touch grid of direct device controls, built from the hardware map."""

    set_digital_requested = pyqtSignal(str, bool)
    toggle_digital_requested = pyqtSignal(str)
    set_pwm_requested = pyqtSignal(str, int)

    def __init__(self, hardware_manager: HardwareManager, parent: QWidget | None = None):
        super().__init__(parent)
        self._hardware_manager = hardware_manager

        self._buttons: dict[str, dict[str, QPushButton]] = {}
        self._sliders: dict[str, QSlider] = {}

        self.setObjectName("runnerDeviceControls")
        self._setup_ui()

    # --- UI scaffolding ---

    def _setup_ui(self) -> None:
        """Build the scroll area + content column once."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("background-color: transparent;")

        QScroller.grabGesture(
            self._scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        self._scroll.setWidget(self._content)

        layout.addWidget(self._scroll, 1)

    # --- Public API ---

    def refresh(self) -> None:
        """Rebuild control blocks from hardware_manager.devices."""
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._buttons.clear()
        self._sliders.clear()

        for dev_id, device in self._hardware_manager.devices.items():
            if device.device_type == "DigitalOutput":
                block = self._make_digital_block(dev_id, device)
                self._content_layout.addWidget(block)
            elif device.device_type == "PWMOutput":
                block = self._make_pwm_block(dev_id, device)
                self._content_layout.addWidget(block)
            # Other device types are not rendered (deferred).

        self._content_layout.addStretch(1)

    # --- Block builders ---

    def _make_digital_block(self, dev_id: str, device) -> QWidget:
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QLabel(dev_id)
        layout.addWidget(label)

        row = QHBoxLayout()
        row.setSpacing(8)

        on_btn = QPushButton("ON")
        on_btn.setMinimumHeight(_BUTTON_MIN_HEIGHT)
        on_btn.clicked.connect(lambda _=False, d=dev_id: self.set_digital_requested.emit(d, True))

        off_btn = QPushButton("OFF")
        off_btn.setMinimumHeight(_BUTTON_MIN_HEIGHT)
        off_btn.clicked.connect(lambda _=False, d=dev_id: self.set_digital_requested.emit(d, False))

        toggle_btn = QPushButton("Toggle")
        toggle_btn.setMinimumHeight(_BUTTON_MIN_HEIGHT)
        toggle_btn.clicked.connect(lambda _=False, d=dev_id: self.toggle_digital_requested.emit(d))

        row.addWidget(on_btn)
        row.addWidget(off_btn)
        row.addWidget(toggle_btn)
        layout.addLayout(row)

        self._buttons[dev_id] = {"on": on_btn, "off": off_btn, "toggle": toggle_btn}
        return block

    def _make_pwm_block(self, dev_id: str, device) -> QWidget:
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QLabel(dev_id)
        layout.addWidget(label)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 255)
        slider.setMinimumHeight(_SLIDER_MIN_HEIGHT)
        slider.valueChanged.connect(lambda value, d=dev_id: self.set_pwm_requested.emit(d, value))
        layout.addWidget(slider)

        self._sliders[dev_id] = slider
        return block
