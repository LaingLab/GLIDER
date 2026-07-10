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
