"""
Widget Factory - Creates dashboard widgets for interface nodes.
"""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from glider.nodes.base_node import GliderNode


class WidgetFactory:
    """Factory for creating dashboard widgets from interface nodes."""

    @staticmethod
    def create_widget(node: "GliderNode") -> QWidget | None:
        """
        Create a dashboard widget for a node.

        Args:
            node: The node to create a widget for

        Returns:
            Created widget or None if not supported
        """
        from glider.gui.widgets.touch_widgets import (
            TouchButton,
            TouchChart,
            TouchGauge,
            TouchLabel,
            TouchLED,
            TouchNumericInput,
            TouchSlider,
            TouchToggle,
        )

        node_type = type(node).__name__

        # Map node types to widget classes
        widget_map = {
            "LabelNode": TouchLabel,
            "ButtonNode": TouchButton,
            "ToggleSwitchNode": TouchToggle,
            "SliderNode": TouchSlider,
            "GaugeNode": TouchGauge,
            "ChartNode": TouchChart,
            "LEDIndicatorNode": TouchLED,
            "NumericInputNode": TouchNumericInput,
            "AnalogReadNode": TouchLabel,  # Analog value displays as label
        }

        widget_class = widget_map.get(node_type)
        if widget_class:
            widget = widget_class()
            widget.bind_node(node)

            # Special handling for nodes with get_display_value
            if hasattr(node, "get_display_value"):
                # Create update callback to use get_display_value
                def update_from_node(*args):
                    display_val = node.get_display_value()
                    widget.set_value(display_val)

                # Register callback with node
                if hasattr(node, "register_update_callback"):
                    node.register_update_callback(update_from_node)

            return widget

        return None
