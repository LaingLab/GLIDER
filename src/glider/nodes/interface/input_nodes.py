"""
Input Nodes - Interactive input widgets for the dashboard.
"""

from typing import Any

from glider.nodes.base_node import (
    ExecNode,
    InterfaceNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)


class ButtonNode(ExecNode, InterfaceNode):
    """A clickable button that triggers execution."""

    definition = NodeDefinition(
        name="Button",
        category=NodeCategory.INTERFACE,
        description="Clickable button that triggers execution",
        inputs=[],
        outputs=[
            PortDefinition(name="Pressed", port_type=PortType.EXEC),
        ],
        color="#5a4a2d",
    )

    def __init__(self):
        ExecNode.__init__(self)
        self._visible_in_runner = True
        self._label = "Button"
        self._press_count = 0

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    def press(self) -> None:
        """Called when button is pressed."""
        self._press_count += 1
        self.exec_output(0)

    async def execute(self) -> None:
        """Execute button press."""
        self.press()

    def update_event(self) -> None:
        """Buttons don't update on input changes."""
        pass

    def get_state(self) -> dict[str, Any]:
        state = ExecNode.get_state(self)
        state["label"] = self._label
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        ExecNode.set_state(self, state)
        self._label = state.get("label", "Button")


class ToggleSwitchNode(InterfaceNode):
    """A toggle switch for on/off control."""

    definition = NodeDefinition(
        name="Toggle Switch",
        category=NodeCategory.INTERFACE,
        description="Toggle switch for on/off control",
        inputs=[],
        outputs=[
            PortDefinition(name="State", data_type=bool),
            PortDefinition(name="Changed", port_type=PortType.EXEC),
        ],
        color="#5a4a2d",
    )

    def __init__(self):
        super().__init__()
        self._label = "Switch"
        self._switch_state = False

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def state(self) -> bool:
        return self._switch_state

    def toggle(self) -> None:
        """Toggle the switch state."""
        self._switch_state = not self._switch_state
        self.set_output(0, self._switch_state)
        # Trigger changed exec output
        for callback in getattr(self, "_exec_callbacks", []):
            callback(1)

    def set_state_value(self, value: bool) -> None:
        """Set the switch state directly."""
        if value != self._switch_state:
            self._switch_state = value
            self.set_output(0, self._switch_state)

    def update_event(self) -> None:
        """Toggle switches update from UI interactions, not inputs."""
        pass

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["label"] = self._label
        state["switch_state"] = self._switch_state
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._label = state.get("label", "Switch")
        self._switch_state = state.get("switch_state", False)


class SliderNode(InterfaceNode):
    """A slider for continuous value input."""

    definition = NodeDefinition(
        name="Slider",
        category=NodeCategory.INTERFACE,
        description="Slider for continuous value input",
        inputs=[],
        outputs=[
            PortDefinition(name="Value", data_type=float),
            PortDefinition(name="Changed", port_type=PortType.EXEC),
        ],
        color="#5a4a2d",
    )

    def __init__(self):
        super().__init__()
        self._label = "Slider"
        self._value = 0.0
        self._min_value = 0.0
        self._max_value = 100.0
        self._step = 1.0

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def value(self) -> float:
        return self._value

    @property
    def min_value(self) -> float:
        return self._min_value

    @min_value.setter
    def min_value(self, value: float) -> None:
        self._min_value = value

    @property
    def max_value(self) -> float:
        return self._max_value

    @max_value.setter
    def max_value(self, value: float) -> None:
        self._max_value = value

    @property
    def step(self) -> float:
        return self._step

    @step.setter
    def step(self, value: float) -> None:
        self._step = value

    def set_value(self, value: float) -> None:
        """Set the slider value."""
        # Clamp to range
        value = max(self._min_value, min(self._max_value, value))
        if value != self._value:
            self._value = value
            self.set_output(0, self._value)

    def update_event(self) -> None:
        """Sliders update from UI interactions, not inputs."""
        pass

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["label"] = self._label
        state["slider_value"] = self._value
        state["min_value"] = self._min_value
        state["max_value"] = self._max_value
        state["step"] = self._step
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._label = state.get("label", "Slider")
        self._value = state.get("slider_value", 0.0)
        self._min_value = state.get("min_value", 0.0)
        self._max_value = state.get("max_value", 100.0)
        self._step = state.get("step", 1.0)


class NumericInputNode(InterfaceNode):
    """A numeric input field with optional keypad."""

    definition = NodeDefinition(
        name="Numeric Input",
        category=NodeCategory.INTERFACE,
        description="Numeric input with virtual keypad support",
        inputs=[],
        outputs=[
            PortDefinition(name="Value", data_type=float),
            PortDefinition(name="Submitted", port_type=PortType.EXEC),
        ],
        color="#5a4a2d",
    )

    def __init__(self):
        super().__init__()
        self._label = "Input"
        self._value = 0.0
        self._min_value = None
        self._max_value = None
        self._decimal_places = 2

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        """Set the input value."""
        # Apply constraints
        if self._min_value is not None:
            value = max(self._min_value, value)
        if self._max_value is not None:
            value = min(self._max_value, value)

        # Round to decimal places
        value = round(value, self._decimal_places)

        self._value = value
        self.set_output(0, self._value)

    def submit(self) -> None:
        """Called when value is submitted (e.g., Enter key)."""
        self.set_output(0, self._value)
        # Trigger submitted exec output
        for callback in getattr(self, "_exec_callbacks", []):
            callback(1)

    def update_event(self) -> None:
        """Numeric inputs update from UI interactions, not inputs."""
        pass

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["label"] = self._label
        state["numeric_value"] = self._value
        state["min_value"] = self._min_value
        state["max_value"] = self._max_value
        state["decimal_places"] = self._decimal_places
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._label = state.get("label", "Input")
        self._value = state.get("numeric_value", 0.0)
        self._min_value = state.get("min_value")
        self._max_value = state.get("max_value")
        self._decimal_places = state.get("decimal_places", 2)
