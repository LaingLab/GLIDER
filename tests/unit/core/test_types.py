"""
Tests for glider.core.types module.

Tests the type enums and their utility methods.
"""

import pytest

from glider.core.types import DeviceType, NodeType
from glider.core.experiment_session import SessionState


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_type_values(self):
        """Test that NodeType enum has expected values."""
        assert NodeType.START_EXPERIMENT.value == "StartExperiment"
        assert NodeType.END_EXPERIMENT.value == "EndExperiment"
        assert NodeType.DELAY.value == "Delay"
        assert NodeType.LOOP.value == "Loop"
        assert NodeType.OUTPUT.value == "Output"
        assert NodeType.INPUT.value == "Input"

    def test_from_string_exact_match(self):
        """Test from_string with exact matches."""
        assert NodeType.from_string("StartExperiment") == NodeType.START_EXPERIMENT
        assert NodeType.from_string("Delay") == NodeType.DELAY
        assert NodeType.from_string("Loop") == NodeType.LOOP
        assert NodeType.from_string("Output") == NodeType.OUTPUT

    def test_from_string_normalized(self):
        """Test from_string with normalized input."""
        # Should handle spaces
        assert NodeType.from_string("Start Experiment") == NodeType.START_EXPERIMENT
        # Should handle 'Node' suffix removal
        assert NodeType.from_string("DelayNode") == NodeType.DELAY

    def test_from_string_invalid(self):
        """Test from_string with invalid input raises ValueError."""
        with pytest.raises(ValueError, match="Unknown node type"):
            NodeType.from_string("InvalidNodeType")

        with pytest.raises(ValueError):
            NodeType.from_string("")

    def test_from_string_safe_valid(self):
        """Test from_string_safe returns enum for valid input."""
        assert NodeType.from_string_safe("Delay") == NodeType.DELAY
        assert NodeType.from_string_safe("Loop") == NodeType.LOOP

    def test_from_string_safe_invalid(self):
        """Test from_string_safe returns None for invalid input."""
        assert NodeType.from_string_safe("InvalidType") is None
        assert NodeType.from_string_safe("") is None
        assert NodeType.from_string_safe("NotARealNode") is None

    def test_all_node_types_have_values(self):
        """Test that all NodeType members have non-empty string values."""
        for node_type in NodeType:
            assert isinstance(node_type.value, str)
            assert len(node_type.value) > 0

    def test_hardware_node_types(self):
        """Test hardware-related node types exist."""
        assert NodeType.DIGITAL_WRITE.value == "DigitalWrite"
        assert NodeType.DIGITAL_READ.value == "DigitalRead"
        assert NodeType.ANALOG_READ.value == "AnalogRead"
        assert NodeType.PWM_WRITE.value == "PWMWrite"

    def test_logic_node_types(self):
        """Test logic node types exist."""
        assert NodeType.ADD.value == "Add"
        assert NodeType.SUBTRACT.value == "Subtract"
        assert NodeType.MULTIPLY.value == "Multiply"
        assert NodeType.DIVIDE.value == "Divide"
        assert NodeType.THRESHOLD.value == "Threshold"

    def test_interface_node_types(self):
        """Test interface node types exist."""
        assert NodeType.LABEL.value == "Label"
        assert NodeType.BUTTON.value == "Button"
        assert NodeType.SLIDER.value == "Slider"
        assert NodeType.GAUGE.value == "Gauge"


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_device_type_values(self):
        """Test that DeviceType enum has expected values."""
        assert DeviceType.DIGITAL_OUTPUT.value == "DigitalOutput"
        assert DeviceType.DIGITAL_INPUT.value == "DigitalInput"
        assert DeviceType.ANALOG_INPUT.value == "AnalogInput"
        assert DeviceType.PWM_OUTPUT.value == "PWMOutput"
        assert DeviceType.SERVO.value == "Servo"
        assert DeviceType.STEPPER.value == "Stepper"

    def test_from_string_valid(self):
        """Test from_string with valid input."""
        assert DeviceType.from_string("DigitalOutput") == DeviceType.DIGITAL_OUTPUT
        assert DeviceType.from_string("AnalogInput") == DeviceType.ANALOG_INPUT
        assert DeviceType.from_string("Servo") == DeviceType.SERVO

    def test_from_string_invalid(self):
        """Test from_string with invalid input raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device type"):
            DeviceType.from_string("InvalidDevice")

    def test_from_string_safe_valid(self):
        """Test from_string_safe returns enum for valid input."""
        assert DeviceType.from_string_safe("DigitalOutput") == DeviceType.DIGITAL_OUTPUT
        assert DeviceType.from_string_safe("Servo") == DeviceType.SERVO

    def test_from_string_safe_invalid(self):
        """Test from_string_safe returns None for invalid input."""
        assert DeviceType.from_string_safe("InvalidDevice") is None
        assert DeviceType.from_string_safe("") is None

    def test_is_input_property(self):
        """Test is_input property."""
        assert DeviceType.DIGITAL_INPUT.is_input is True
        assert DeviceType.ANALOG_INPUT.is_input is True
        assert DeviceType.DIGITAL_OUTPUT.is_input is False
        assert DeviceType.PWM_OUTPUT.is_input is False
        assert DeviceType.SERVO.is_input is False

    def test_is_output_property(self):
        """Test is_output property."""
        assert DeviceType.DIGITAL_OUTPUT.is_output is True
        assert DeviceType.PWM_OUTPUT.is_output is True
        assert DeviceType.SERVO.is_output is True
        assert DeviceType.STEPPER.is_output is True
        assert DeviceType.DIGITAL_INPUT.is_output is False
        assert DeviceType.ANALOG_INPUT.is_output is False

    def test_is_analog_property(self):
        """Test is_analog property."""
        assert DeviceType.ANALOG_INPUT.is_analog is True
        assert DeviceType.PWM_OUTPUT.is_analog is True
        assert DeviceType.DIGITAL_INPUT.is_analog is False
        assert DeviceType.DIGITAL_OUTPUT.is_analog is False

    def test_input_output_mutually_exclusive(self):
        """Test that a device is either input or output, not both."""
        for device_type in DeviceType:
            # A device cannot be both input and output
            assert not (device_type.is_input and device_type.is_output)


class TestSessionState:
    """Tests for SessionState enum."""

    def test_session_state_values(self):
        """Test that SessionState enum has expected members."""
        assert SessionState.IDLE is not None
        assert SessionState.INITIALIZING is not None
        assert SessionState.READY is not None
        assert SessionState.RUNNING is not None
        assert SessionState.PAUSED is not None
        assert SessionState.STOPPING is not None
        assert SessionState.ERROR is not None

    def test_is_active_property(self):
        """Test is_active property."""
        assert SessionState.RUNNING.is_active is True
        assert SessionState.IDLE.is_active is False
        assert SessionState.PAUSED.is_active is False
        assert SessionState.ERROR.is_active is False

    def test_can_start_property(self):
        """Test can_start property."""
        assert SessionState.IDLE.can_start is True
        assert SessionState.READY.can_start is True
        assert SessionState.RUNNING.can_start is False
        assert SessionState.PAUSED.can_start is False
        assert SessionState.ERROR.can_start is False

    def test_can_stop_property(self):
        """Test can_stop property."""
        assert SessionState.RUNNING.can_stop is True
        assert SessionState.PAUSED.can_stop is True
        assert SessionState.ERROR.can_stop is True
        assert SessionState.IDLE.can_stop is False
        assert SessionState.READY.can_stop is False

    def test_state_transitions_logic(self):
        """Test that state transition logic is consistent."""
        # If can_start is True, is_active should be False
        for state in SessionState:
            if state.can_start:
                assert not state.is_active

        # If is_active is True, can_stop should be True
        for state in SessionState:
            if state.is_active:
                assert state.can_stop
