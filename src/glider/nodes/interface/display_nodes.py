"""
Display Nodes - Read-only display widgets for the dashboard.
"""

from collections import deque
from typing import Any

from glider.nodes.base_node import (
    InterfaceNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
)


class LabelNode(InterfaceNode):
    """Display a text label with a value."""

    definition = NodeDefinition(
        name="Label",
        category=NodeCategory.INTERFACE,
        description="Display a value as text",
        inputs=[
            PortDefinition(name="Value", data_type=object),
            PortDefinition(name="Format", data_type=str, default_value="{}"),
        ],
        outputs=[],
        color="#5a4a2d",
    )

    PROPERTIES_SCHEMA = [
        {
            "key": "label",
            "label": "Label",
            "type": "str",
            "default": "Value",
            "help": "Caption shown on the dashboard.",
        },
    ]

    def __init__(self):
        super().__init__()
        self._label = self.setting("label")
        self._display_text = ""

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def display_text(self) -> str:
        return self._display_text

    def update_event(self) -> None:
        value = self.get_input(0)
        format_str = self.get_input(1) or "{}"

        try:
            self._display_text = format_str.format(value)
        except Exception:
            self._display_text = str(value)

        self.notify_widget(self._display_text)

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["label"] = self._label
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._label = self.setting("label")


class GaugeNode(InterfaceNode):
    """Display a gauge/meter widget."""

    definition = NodeDefinition(
        name="Gauge",
        category=NodeCategory.INTERFACE,
        description="Display value as a gauge/meter",
        inputs=[
            PortDefinition(name="Value", data_type=float, default_value=0.0),
            PortDefinition(name="Min", data_type=float, default_value=0.0),
            PortDefinition(name="Max", data_type=float, default_value=100.0),
        ],
        outputs=[],
        color="#5a4a2d",
    )

    PROPERTIES_SCHEMA = [
        {
            "key": "label",
            "label": "Label",
            "type": "str",
            "default": "Value",
            "help": "Caption shown on the dashboard.",
        },
        {
            "key": "unit",
            "label": "Unit",
            "type": "str",
            "default": "",
            "help": "Unit text shown after the value.",
        },
        {
            "key": "warning_threshold",
            "label": "Warning at",
            "type": "float",
            "default": None,
            "help": "Value at which the gauge turns amber; null for none.",
        },
        {
            "key": "danger_threshold",
            "label": "Danger at",
            "type": "float",
            "default": None,
            "help": "Value at which the gauge turns red; null for none.",
        },
    ]

    def __init__(self):
        super().__init__()
        self._label = self.setting("label")
        self._unit = self.setting("unit")
        self._warning_threshold = self.setting("warning_threshold")
        self._danger_threshold = self.setting("danger_threshold")

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, value: str) -> None:
        self._unit = value

    def update_event(self) -> None:
        value = float(self.get_input(0) or 0)
        min_val = float(self.get_input(1) or 0)
        max_val = float(self.get_input(2) or 100)

        # Calculate percentage
        if max_val > min_val:
            percent = (value - min_val) / (max_val - min_val) * 100
        else:
            percent = 0

        self.notify_widget(
            {
                "value": value,
                "percent": percent,
                "min": min_val,
                "max": max_val,
                "unit": self._unit,
                "label": self._label,
            }
        )

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["label"] = self._label
        state["unit"] = self._unit
        state["warning_threshold"] = self._warning_threshold
        state["danger_threshold"] = self._danger_threshold
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._label = self.setting("label")
        self._unit = self.setting("unit")
        self._warning_threshold = self.setting("warning_threshold")
        self._danger_threshold = self.setting("danger_threshold")


class ChartNode(InterfaceNode):
    """Display a real-time chart/graph."""

    definition = NodeDefinition(
        name="Chart",
        category=NodeCategory.INTERFACE,
        description="Display real-time chart of values",
        inputs=[
            PortDefinition(name="Value", data_type=float, default_value=0.0),
        ],
        outputs=[],
        color="#5a4a2d",
    )

    PROPERTIES_SCHEMA = [
        {
            "key": "label",
            "label": "Label",
            "type": "str",
            "default": "Chart",
            "help": "Caption shown on the dashboard.",
        },
        {
            "key": "max_points",
            "label": "Points",
            "type": "int",
            "default": 100,
            "min": 1,
            "help": "How many recent values the chart keeps.",
        },
        {
            "key": "y_min",
            "label": "Y min",
            "type": "float",
            "default": None,
            "help": "Fixed lower axis bound; null to autoscale.",
        },
        {
            "key": "y_max",
            "label": "Y max",
            "type": "float",
            "default": None,
            "help": "Fixed upper axis bound; null to autoscale.",
        },
    ]

    def __init__(self):
        super().__init__()
        self._label = self.setting("label")
        self._max_points = self.setting("max_points")
        self._data: deque = deque(maxlen=self._max_points)
        self._y_min = self.setting("y_min")
        self._y_max = self.setting("y_max")

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def max_points(self) -> int:
        return self._max_points

    @max_points.setter
    def max_points(self, value: int) -> None:
        self._max_points = value
        self._data = deque(list(self._data), maxlen=value)

    @property
    def data(self) -> list[float]:
        return list(self._data)

    def clear_data(self) -> None:
        """Clear the chart data."""
        self._data.clear()

    def update_event(self) -> None:
        value = float(self.get_input(0) or 0)
        self._data.append(value)

        self.notify_widget(
            {
                "data": list(self._data),
                "label": self._label,
                "y_min": self._y_min,
                "y_max": self._y_max,
            }
        )

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["label"] = self._label
        state["max_points"] = self._max_points
        state["y_min"] = self._y_min
        state["y_max"] = self._y_max
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._label = self.setting("label")
        self._max_points = self.setting("max_points")
        self._y_min = self.setting("y_min")
        self._y_max = self.setting("y_max")


class LEDIndicatorNode(InterfaceNode):
    """Display an LED indicator."""

    definition = NodeDefinition(
        name="LED Indicator",
        category=NodeCategory.INTERFACE,
        description="Display an on/off LED indicator",
        inputs=[
            PortDefinition(name="State", data_type=bool, default_value=False),
        ],
        outputs=[],
        color="#5a4a2d",
    )

    PROPERTIES_SCHEMA = [
        {
            "key": "label",
            "label": "Label",
            "type": "str",
            "default": "LED",
            "help": "Caption shown on the dashboard.",
        },
        {
            "key": "on_color",
            "label": "On colour",
            "type": "str",
            "default": "#00ff00",
            "help": "Colour when on, as #rrggbb.",
        },
        {
            "key": "off_color",
            "label": "Off colour",
            "type": "str",
            "default": "#333333",
            "help": "Colour when off, as #rrggbb.",
        },
    ]

    def __init__(self):
        super().__init__()
        self._label = self.setting("label")
        self._on_color = self.setting("on_color")  # Green
        self._off_color = self.setting("off_color")  # Dark gray

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def on_color(self) -> str:
        return self._on_color

    @on_color.setter
    def on_color(self, value: str) -> None:
        self._on_color = value

    @property
    def off_color(self) -> str:
        return self._off_color

    @off_color.setter
    def off_color(self, value: str) -> None:
        self._off_color = value

    def update_event(self) -> None:
        state = bool(self.get_input(0))
        color = self._on_color if state else self._off_color

        self.notify_widget(
            {
                "state": state,
                "color": color,
                "label": self._label,
            }
        )

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["label"] = self._label
        state["on_color"] = self._on_color
        state["off_color"] = self._off_color
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._label = self.setting("label")
        self._on_color = self.setting("on_color")
        self._off_color = self.setting("off_color")
