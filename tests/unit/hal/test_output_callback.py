"""Tests for the BaseBoard output-callback mechanism.

The new ``register_output_callback`` hook is used by DeviceEventLogger
to capture write_digital/write_analog/write_servo events without
instrumenting every Device class. These tests exercise the mechanism
through MockBoard, which fires the new path from its async writes.
"""

import pytest

from glider.hal.base_board import PinType
from glider.hal.mock_board import MockBoard


@pytest.mark.asyncio
async def test_output_callback_fires_on_write_digital():
    """Writing a digital value should fan out via the output-callback path."""
    board = MockBoard()
    captured: list[tuple[int, PinType, object]] = []

    def cb(pin, pin_type, value):
        captured.append((pin, pin_type, value))

    board.register_output_callback(cb)
    await board.write_digital(13, True)
    await board.write_digital(13, False)

    assert captured == [(13, PinType.DIGITAL, True), (13, PinType.DIGITAL, False)]


@pytest.mark.asyncio
async def test_output_callback_fires_on_write_analog():
    """PWM writes should fire the output-callback with PinType.PWM."""
    board = MockBoard()
    captured: list[tuple[int, PinType, object]] = []
    board.register_output_callback(lambda p, t, v: captured.append((p, t, v)))

    await board.write_analog(9, 128)

    assert captured == [(9, PinType.PWM, 128)]


@pytest.mark.asyncio
async def test_unregister_output_callback_silences_further_events():
    """After unregister, no further output rows should appear."""
    board = MockBoard()
    captured: list[tuple] = []

    def cb(pin, pin_type, value):
        captured.append((pin, pin_type, value))

    board.register_output_callback(cb)
    await board.write_digital(7, True)
    assert len(captured) == 1

    board.unregister_output_callback(cb)
    await board.write_digital(7, False)
    assert len(captured) == 1  # second write did not fire the cb


@pytest.mark.asyncio
async def test_output_callback_exception_does_not_break_writes():
    """A failing callback must not propagate up the write path."""
    board = MockBoard()

    def boom(pin, pin_type, value):
        raise RuntimeError("intentional")

    board.register_output_callback(boom)
    # Must not raise — writes are user-visible operations that we never
    # want callback registration to break.
    await board.write_digital(2, True)
    assert board.get_pin_state(2) is True


@pytest.mark.asyncio
async def test_register_output_callback_is_idempotent():
    """Registering the same callback twice should not double-fire."""
    board = MockBoard()
    count = 0

    def cb(pin, pin_type, value):
        nonlocal count
        count += 1

    board.register_output_callback(cb)
    board.register_output_callback(cb)
    await board.write_digital(5, True)

    assert count == 1
