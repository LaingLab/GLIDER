"""
Touch Widgets - Touch-optimized widgets for the Runner dashboard.

These widgets are designed for 480x800 Raspberry Pi touchscreen with
large touch targets (min 80px height) and high-contrast visuals.
"""

import sys
from collections import deque
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.nodes.base_node import GliderNode


def _get_system_font() -> str:
    """Get the system font for the current platform."""
    if sys.platform == "darwin":
        return ".AppleSystemUIFont"
    elif sys.platform == "win32":
        return "Segoe UI"
    return "DejaVu Sans"


class TouchWidgetBase(QWidget):
    """Base class for touch-optimized widgets."""

    # Signal emitted when widget value changes
    value_changed = pyqtSignal(object)

    # Minimum touch target height (80px per design doc)
    MIN_HEIGHT = 80

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._node: GliderNode | None = None
        self._label_text = ""

        self.setMinimumHeight(self.MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def bind_node(self, node: "GliderNode") -> None:
        """Bind this widget to a node for data synchronization."""
        self._node = node
        self._label_text = getattr(node, "label", node.title)
        self._on_node_bound()

    def _on_node_bound(self) -> None:
        """Called after node is bound. Override to customize."""
        pass

    def set_value(self, value: Any) -> None:
        """Set the widget value. Override in subclasses."""
        pass

    def get_value(self) -> Any:
        """Get the current widget value. Override in subclasses."""
        return None


class TouchLabel(TouchWidgetBase):
    """
    Touch-optimized label for displaying text values.

    Features large 24pt font with high contrast.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._value = ""

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the label UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # Title label
        self._title = QLabel()
        self._title.setStyleSheet("font-size: 14px; color: #888;")
        layout.addWidget(self._title)

        # Value label
        self._label = QLabel()
        self._label.setStyleSheet("font-size: 24px; font-weight: bold; color: #fff;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

    def _on_node_bound(self) -> None:
        """Update title from node."""
        self._title.setText(self._label_text)

    def set_value(self, value: Any) -> None:
        """Set the displayed text."""
        self._value = str(value)
        self._label.setText(self._value)

    def get_value(self) -> str:
        return self._value


class TouchButton(TouchWidgetBase):
    """
    Touch-optimized button with large touch target.

    Emits value_changed(True) on press, value_changed(False) on release.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the button UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self._button = QPushButton()
        self._button.setMinimumHeight(60)
        self._button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
        """)
        self._button.pressed.connect(lambda: self.value_changed.emit(True))
        self._button.released.connect(lambda: self.value_changed.emit(False))
        layout.addWidget(self._button)

    def _on_node_bound(self) -> None:
        """Update button text from node."""
        self._button.setText(self._label_text)

    def set_value(self, value: Any) -> None:
        """Set button state (for visual feedback)."""
        pass  # Button state is managed by press/release

    def get_value(self) -> bool:
        return self._button.isDown()


class TouchToggle(TouchWidgetBase):
    """
    Touch-optimized toggle switch.

    Large sliding toggle for on/off states.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._checked = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the toggle UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Label
        self._title = QLabel()
        self._title.setStyleSheet("font-size: 18px; color: #fff;")
        layout.addWidget(self._title)

        layout.addStretch()

        # Toggle button
        self._toggle = QPushButton()
        self._toggle.setCheckable(True)
        self._toggle.setMinimumSize(100, 50)
        self._toggle.setMaximumSize(100, 50)
        self._toggle.clicked.connect(self._on_toggled)
        self._update_toggle_style()
        layout.addWidget(self._toggle)

    def _on_node_bound(self) -> None:
        """Update label from node."""
        self._title.setText(self._label_text)

    def _on_toggled(self, checked: bool) -> None:
        """Handle toggle state change."""
        self._checked = checked
        self._update_toggle_style()
        self.value_changed.emit(checked)

    def _update_toggle_style(self) -> None:
        """Update toggle appearance based on state."""
        if self._checked:
            self._toggle.setStyleSheet("""
                QPushButton {
                    background-color: #2ecc71;
                    border: none;
                    border-radius: 25px;
                }
            """)
            self._toggle.setText("ON")
        else:
            self._toggle.setStyleSheet("""
                QPushButton {
                    background-color: #7f8c8d;
                    border: none;
                    border-radius: 25px;
                }
            """)
            self._toggle.setText("OFF")

    def set_value(self, value: Any) -> None:
        """Set toggle state."""
        self._checked = bool(value)
        self._toggle.setChecked(self._checked)
        self._update_toggle_style()

    def get_value(self) -> bool:
        return self._checked


class TouchSlider(TouchWidgetBase):
    """
    Touch-optimized slider with large track and handle.

    Includes value display and optional min/max labels.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._min_value = 0
        self._max_value = 100
        self._value = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the slider UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # Title and value row
        header = QHBoxLayout()
        self._title = QLabel()
        self._title.setStyleSheet("font-size: 16px; color: #888;")
        header.addWidget(self._title)

        header.addStretch()

        self._value_label = QLabel("0")
        self._value_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #fff;")
        header.addWidget(self._value_label)

        layout.addLayout(header)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimumHeight(40)
        self._slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 20px;
                background: #34495e;
                border-radius: 10px;
            }
            QSlider::handle:horizontal {
                width: 40px;
                height: 40px;
                margin: -10px 0;
                background: #3498db;
                border-radius: 20px;
            }
            QSlider::handle:horizontal:pressed {
                background: #2980b9;
            }
        """)
        self._slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self._slider)

        # Min/max labels
        range_layout = QHBoxLayout()
        self._min_label = QLabel("0")
        self._min_label.setStyleSheet("font-size: 12px; color: #666;")
        range_layout.addWidget(self._min_label)

        range_layout.addStretch()

        self._max_label = QLabel("100")
        self._max_label.setStyleSheet("font-size: 12px; color: #666;")
        range_layout.addWidget(self._max_label)

        layout.addLayout(range_layout)

    def _on_node_bound(self) -> None:
        """Update slider from node configuration."""
        self._title.setText(self._label_text)

        if self._node:
            self._min_value = getattr(self._node, "min_value", 0)
            self._max_value = getattr(self._node, "max_value", 100)
            self._slider.setRange(int(self._min_value), int(self._max_value))
            self._min_label.setText(str(self._min_value))
            self._max_label.setText(str(self._max_value))

    def _on_value_changed(self, value: int) -> None:
        """Handle slider value change."""
        self._value = value
        self._value_label.setText(str(value))
        self.value_changed.emit(value)

    def set_value(self, value: Any) -> None:
        """Set slider value."""
        self._value = int(value)
        self._slider.blockSignals(True)
        self._slider.setValue(self._value)
        self._slider.blockSignals(False)
        self._value_label.setText(str(self._value))

    def get_value(self) -> int:
        return self._value


class TouchGauge(TouchWidgetBase):
    """
    Touch-optimized gauge widget for displaying analog values.

    Circular gauge with value and unit display.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._value = 0.0
        self._min_value = 0.0
        self._max_value = 100.0
        self._unit = ""

        self.setMinimumHeight(120)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the gauge UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # Title
        self._title = QLabel()
        self._title.setStyleSheet("font-size: 14px; color: #888;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)

        # Gauge canvas
        self._gauge = GaugeCanvas()
        layout.addWidget(self._gauge, stretch=1)

    def _on_node_bound(self) -> None:
        """Update gauge from node configuration."""
        self._title.setText(self._label_text)

        if self._node:
            self._min_value = getattr(self._node, "min_value", 0.0)
            self._max_value = getattr(self._node, "max_value", 100.0)
            self._unit = getattr(self._node, "unit", "")
            self._gauge.set_range(self._min_value, self._max_value)
            self._gauge.set_unit(self._unit)

    def set_value(self, value: Any) -> None:
        """Set gauge value."""
        self._value = float(value)
        self._gauge.set_value(self._value)

    def get_value(self) -> float:
        return self._value


class GaugeCanvas(QWidget):
    """Canvas widget for drawing the gauge arc."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._value = 0.0
        self._min = 0.0
        self._max = 100.0
        self._unit = ""

        self.setMinimumSize(80, 60)

    def set_range(self, min_val: float, max_val: float) -> None:
        """Set the gauge range."""
        self._min = min_val
        self._max = max_val
        self.update()

    def set_unit(self, unit: str) -> None:
        """Set the unit label."""
        self._unit = unit
        self.update()

    def set_value(self, value: float) -> None:
        """Set the current value."""
        self._value = max(self._min, min(self._max, value))
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the gauge."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w // 2
        cy = h - 10

        # Arc dimensions
        radius = min(w // 2 - 10, h - 20)
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        # Background arc
        painter.setPen(QPen(QColor("#34495e"), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 180 * 16, -180 * 16)

        # Value arc
        if self._max > self._min:
            ratio = (self._value - self._min) / (self._max - self._min)
            angle = int(-180 * ratio * 16)
            painter.setPen(
                QPen(QColor("#3498db"), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            )
            painter.drawArc(rect, 180 * 16, angle)

        # Value text
        painter.setPen(QPen(QColor("#fff")))
        font = QFont(_get_system_font(), 16, QFont.Weight.Bold)
        painter.setFont(font)

        value_text = f"{self._value:.1f}{self._unit}"
        painter.drawText(
            QRectF(0, cy - radius // 2, w, 30), Qt.AlignmentFlag.AlignCenter, value_text
        )


class TouchChart(TouchWidgetBase):
    """
    Touch-optimized chart for time-series data.

    Rolling line chart with configurable buffer size.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._buffer_size = 100
        self._data: deque = deque(maxlen=self._buffer_size)
        self._min_value = 0.0
        self._max_value = 100.0

        self.setMinimumHeight(150)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the chart UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # Title
        self._title = QLabel()
        self._title.setStyleSheet("font-size: 14px; color: #888;")
        layout.addWidget(self._title)

        # Chart canvas
        self._chart = ChartCanvas()
        layout.addWidget(self._chart, stretch=1)

    def _on_node_bound(self) -> None:
        """Update chart from node configuration."""
        self._title.setText(self._label_text)

        if self._node:
            self._buffer_size = getattr(self._node, "buffer_size", 100)
            self._min_value = getattr(self._node, "min_value", 0.0)
            self._max_value = getattr(self._node, "max_value", 100.0)
            self._data = deque(maxlen=self._buffer_size)
            self._chart.set_range(self._min_value, self._max_value)

    def set_value(self, value: Any) -> None:
        """Add a new value to the chart."""
        self._data.append(float(value))
        self._chart.set_data(list(self._data))

    def get_value(self) -> list[float]:
        return list(self._data)


class ChartCanvas(QWidget):
    """Canvas widget for drawing the chart."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._data: list[float] = []
        self._min = 0.0
        self._max = 100.0

        self.setMinimumSize(100, 80)

    def set_range(self, min_val: float, max_val: float) -> None:
        """Set the value range."""
        self._min = min_val
        self._max = max_val
        self.update()

    def set_data(self, data: list[float]) -> None:
        """Set the chart data."""
        self._data = data
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 5

        # Background
        painter.fillRect(0, 0, w, h, QColor("#1a1a2e"))

        # Grid lines
        painter.setPen(QPen(QColor("#2d2d44"), 1))
        for i in range(1, 4):
            y = margin + (h - 2 * margin) * i // 4
            painter.drawLine(margin, y, w - margin, y)

        # Data line
        if len(self._data) < 2:
            return

        path = QPainterPath()
        range_val = self._max - self._min
        if range_val == 0:
            range_val = 1

        for i, val in enumerate(self._data):
            x = margin + (w - 2 * margin) * i / (len(self._data) - 1)
            normalized = (val - self._min) / range_val
            y = h - margin - (h - 2 * margin) * normalized

            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        painter.setPen(QPen(QColor("#3498db"), 2))
        painter.drawPath(path)


class TouchLED(TouchWidgetBase):
    """
    Touch-optimized LED indicator.

    Large colored circle that changes based on state.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._on = False
        self._on_color = QColor("#2ecc71")  # Green
        self._off_color = QColor("#7f8c8d")  # Gray

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the LED UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Label
        self._title = QLabel()
        self._title.setStyleSheet("font-size: 18px; color: #fff;")
        layout.addWidget(self._title)

        layout.addStretch()

        # LED indicator
        self._led = LEDCanvas()
        self._led.setFixedSize(50, 50)
        layout.addWidget(self._led)

    def _on_node_bound(self) -> None:
        """Update LED from node configuration."""
        self._title.setText(self._label_text)

        if self._node:
            on_color = getattr(self._node, "on_color", "#2ecc71")
            off_color = getattr(self._node, "off_color", "#7f8c8d")
            self._on_color = QColor(on_color)
            self._off_color = QColor(off_color)
            self._led.set_colors(self._on_color, self._off_color)

    def set_value(self, value: Any) -> None:
        """Set LED state."""
        self._on = bool(value)
        self._led.set_on(self._on)

    def get_value(self) -> bool:
        return self._on


class LEDCanvas(QWidget):
    """Canvas widget for drawing the LED."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._on = False
        self._on_color = QColor("#2ecc71")
        self._off_color = QColor("#7f8c8d")

    def set_colors(self, on_color: QColor, off_color: QColor) -> None:
        """Set the LED colors."""
        self._on_color = on_color
        self._off_color = off_color
        self.update()

    def set_on(self, on: bool) -> None:
        """Set the LED state."""
        self._on = on
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the LED."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        radius = min(w, h) // 2 - 2

        color = self._on_color if self._on else self._off_color

        # Outer glow when on
        if self._on:
            glow = QColor(color)
            glow.setAlpha(100)
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                w // 2 - radius - 4, h // 2 - radius - 4, (radius + 4) * 2, (radius + 4) * 2
            )

        # Main LED
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawEllipse(w // 2 - radius, h // 2 - radius, radius * 2, radius * 2)

        # Highlight
        highlight = QColor(255, 255, 255, 80)
        painter.setBrush(QBrush(highlight))
        painter.setPen(Qt.PenStyle.NoPen)
        highlight_radius = radius // 3
        painter.drawEllipse(
            w // 2 - radius // 2, h // 2 - radius // 2, highlight_radius, highlight_radius
        )


class TouchNumericInput(TouchWidgetBase):
    """
    Touch-optimized numeric input with increment/decrement buttons.

    Large +/- buttons for easy touch input.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._value = 0.0
        self._min_value = 0.0
        self._max_value = 100.0
        self._step = 1.0
        self._decimals = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the numeric input UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # Title
        self._title = QLabel()
        self._title.setStyleSheet("font-size: 14px; color: #888;")
        layout.addWidget(self._title)

        # Input row
        input_row = QHBoxLayout()

        # Decrement button
        self._dec_btn = QPushButton("-")
        self._dec_btn.setMinimumSize(60, 50)
        self._dec_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
        """)
        self._dec_btn.clicked.connect(self._decrement)
        input_row.addWidget(self._dec_btn)

        # Value display
        self._value_label = QLabel("0")
        self._value_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #fff;
            background-color: #34495e;
            border-radius: 8px;
            padding: 10px;
        """)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value_label.setMinimumWidth(100)
        input_row.addWidget(self._value_label, stretch=1)

        # Increment button
        self._inc_btn = QPushButton("+")
        self._inc_btn.setMinimumSize(60, 50)
        self._inc_btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:pressed {
                background-color: #27ae60;
            }
        """)
        self._inc_btn.clicked.connect(self._increment)
        input_row.addWidget(self._inc_btn)

        layout.addLayout(input_row)

    def _on_node_bound(self) -> None:
        """Update input from node configuration."""
        self._title.setText(self._label_text)

        if self._node:
            self._min_value = getattr(self._node, "min_value", 0.0)
            self._max_value = getattr(self._node, "max_value", 100.0)
            self._step = getattr(self._node, "step", 1.0)
            self._decimals = getattr(self._node, "decimals", 0)
            self._update_display()

    def _increment(self) -> None:
        """Increment the value."""
        new_val = min(self._max_value, self._value + self._step)
        if new_val != self._value:
            self._value = new_val
            self._update_display()
            self.value_changed.emit(self._value)

    def _decrement(self) -> None:
        """Decrement the value."""
        new_val = max(self._min_value, self._value - self._step)
        if new_val != self._value:
            self._value = new_val
            self._update_display()
            self.value_changed.emit(self._value)

    def _update_display(self) -> None:
        """Update the value display."""
        if self._decimals > 0:
            self._value_label.setText(f"{self._value:.{self._decimals}f}")
        else:
            self._value_label.setText(str(int(self._value)))

    def set_value(self, value: Any) -> None:
        """Set the input value."""
        self._value = float(value)
        self._value = max(self._min_value, min(self._max_value, self._value))
        self._update_display()

    def get_value(self) -> float:
        return self._value
