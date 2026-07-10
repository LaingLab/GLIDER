"""UI-free direct device-drive calls, shared by the desktop DeviceControlPanel
and the Runner live-run device controls. No Qt, no dialogs -- callers own
presentation and error surfacing.
"""

from __future__ import annotations


def _first_pin(device) -> int:
    return list(device.pins.values())[0] if getattr(device, "pins", None) else 0


async def set_digital(device, value: bool) -> None:
    if device is None:
        raise ValueError("no device")
    if hasattr(device, "set_state"):
        await device.set_state(value)
    elif hasattr(device, "turn_on") and hasattr(device, "turn_off"):
        await (device.turn_on() if value else device.turn_off())
    else:
        await device.board.write_digital(_first_pin(device), value)


async def toggle_digital(device) -> None:
    if device is None:
        raise ValueError("no device")
    if hasattr(device, "toggle"):
        await device.toggle()
    elif hasattr(device, "state") and hasattr(device, "set_state"):
        await device.set_state(not device.state)
    else:
        raise AttributeError("device does not support toggle")


async def set_pwm(device, value: int) -> None:
    if device is None:
        raise ValueError("no device")
    if hasattr(device, "set_value"):
        await device.set_value(value)
    else:
        await device.board.write_analog(_first_pin(device), value)


async def read_input(device) -> str:
    """Read an input device once and return a display-ready string.

    Faithful port of DeviceControlPanel._read_input_once's display logic,
    stripped of Qt/status-label side effects. Callers own presentation and
    error surfacing.
    """
    if device is None:
        raise ValueError("no device")

    if not getattr(device, "_initialized", False):
        await device.initialize()

    device_type = getattr(device, "device_type", "")

    if device_type == "DigitalInput":
        value = await device.read()
        return "HIGH (1)" if value else "LOW (0)"
    elif device_type == "AnalogInput":
        raw_value = await device.read()
        if hasattr(device, "read_voltage"):
            voltage = await device.read_voltage()
        else:
            voltage = (raw_value / 1023.0) * getattr(device, "_reference_voltage", 5.0)
        return f"{raw_value}\n{voltage:.2f}V"
    elif device_type == "ADS1115":
        channel = device._config.settings.get("channel", 0)
        raw_value = await device.read(channel)
        voltage = await device.read_voltage(channel)
        return f"{raw_value}\n{voltage:.3f}V"
    elif device_type == "GenericI2C":
        value = await device.read()
        return f"{value}\n0x{value:X}"
    else:
        raise ValueError(f"unsupported device type: {device_type}")
