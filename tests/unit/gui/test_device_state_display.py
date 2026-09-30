"""The live-state pill shows every output type, not only digital ones."""

from types import SimpleNamespace

from glider.gui.device_status import device_state_display


def _board(pwm_bits=8, analog_bits=10):
    return SimpleNamespace(
        capabilities=SimpleNamespace(pwm_resolution=pwm_bits, analog_resolution=analog_bits)
    )


def test_pwm_shows_its_duty_cycle():
    pwm = SimpleNamespace(device_type="PWMOutput", _value=255, _board=_board())
    assert device_state_display(pwm)[0] == "100%"
    pwm._value = 128
    assert device_state_display(pwm)[0] == "50%"
    pwm._value = 0
    assert device_state_display(pwm)[0] == "0%"


def test_servo_shows_its_angle():
    servo = SimpleNamespace(device_type="Servo", _angle=90)
    assert device_state_display(servo)[0] == "90°"


def test_digital_and_analog_unchanged():
    assert (
        device_state_display(SimpleNamespace(device_type="DigitalOutput", _state=True))[0] == "HIGH"
    )
    assert (
        device_state_display(SimpleNamespace(device_type="DigitalOutput", _state=None))[0] == "---"
    )
    analog = SimpleNamespace(device_type="AnalogInput", _last_value=1023, _board=_board())
    assert device_state_display(analog)[0] == "1023\n5.00V"
