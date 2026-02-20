"""
Minimal test: does telemetrix-aio's analog callback work at all?
This mirrors how GLIDER uses telemetrix-aio, without any of GLIDER's wrapper code.

Run: python test_analog_aio.py
"""

import asyncio
import time


async def analog_callback(data):
    """Async callback - same style GLIDER uses."""
    pin = data[1]
    value = data[2]
    print(f"[ASYNC] Pin A{pin} Value: {value}")


async def main():
    from telemetrix_aio import telemetrix_aio

    print("Connecting via telemetrix-aio...")
    board = telemetrix_aio.TelemetrixAIO(autostart=False, sleep_tune=0.0001)
    await board.start_aio()
    print("Connected!")

    await board.set_pin_mode_analog_input(0, differential=1, callback=analog_callback)
    print("Analog pin 0 configured. Waiting for callbacks...")

    try:
        while True:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        await board.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
