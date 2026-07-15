#!/usr/bin/env python3
"""
GLIDER Hardware Latency Test

Measures end-to-end latency through the GLIDER HAL:
  Pi-to-Pi:       Pi GPIO output -> Pi GPIO input loopback
  Arduino-to-Pi:  Arduino digital write -> Pi GPIO read
  Pi-to-Arduino:  Pi GPIO write -> Arduino input-change report (ROUND TRIP)

Measurement endpoints:
  The clock starts immediately before the GLIDER HAL write call, so every
  result includes Python/asyncio driver overhead. This is intentional: the
  measured quantity is end-to-end HAL latency as experienced by a GLIDER
  flow, not bare electrical pin-to-pin latency.

  Pi-to-Pi / Arduino-to-Pi: the clock stops when a busy-poll on the Pi
  input pin first observes the rising edge.

  Pi-to-Arduino: the Pi cannot observe the Arduino's input register
  directly, so the clock stops when the Arduino's input-change report
  arrives back at the host and the HAL pin callback fires. This is a
  ROUND TRIP (HAL command -> Pi pin -> Arduino input scan -> serial
  report -> host callback) and is an upper bound on the one-way
  Pi-to-Arduino latency. Measuring true one-way latency requires
  external instrumentation (e.g. a logic analyzer across both pins).

Each test runs a number of unrecorded warm-up trials first (--warmup) so
lazy initialization does not contaminate the statistics. Trials that time
out are excluded from the statistics and reported separately.

Wiring required:
  Pi-to-Pi:       Pi GPIO19 (output) --> wire --> Pi GPIO26 (input)
  Arduino-to-Pi:  Arduino D7 (output) --> wire --> Pi GPIO13 (input)
  Pi-to-Arduino:  Pi GPIO6 (output)  --> wire --> Arduino D8 (input)
  Common ground between Pi and Arduino.

Usage:
  python tests/latency_test.py
  python tests/latency_test.py --trials 500 --arduino-port /dev/ttyUSB0
  python tests/latency_test.py --tests pi-to-pi arduino-to-pi
  python tests/latency_test.py --tests pi-to-arduino
"""

import argparse
import asyncio
import csv
import platform
import statistics
import sys
import threading
import time

# Per-trial detection timeout. A trial that exceeds this is dropped from
# the statistics and counted separately.
DETECT_TIMEOUT_S = 1.0


def find_arduino_port() -> str | None:
    """Auto-detect Arduino serial port."""
    from serial.tools import list_ports

    arduino_ids = {
        (0x2341, None),  # Arduino SA
        (0x2A03, None),  # Arduino.org
        (0x1A86, 0x7523),  # CH340
        (0x0403, 0x6001),  # FTDI FT232
        (0x10C4, 0xEA60),  # CP210x
    }

    for port_info in list_ports.comports():
        if port_info.vid is None:
            continue
        for vid, pid in arduino_ids:
            if port_info.vid == vid and (pid is None or port_info.pid == pid):
                print(f"Auto-detected Arduino on {port_info.device}")
                return port_info.device
        desc = (port_info.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "cp210" in desc or "ft232" in desc:
            print(f"Auto-detected Arduino on {port_info.device}")
            return port_info.device

    return None


def _busy_wait_for_high(detector) -> bool:
    """Busy-poll a gpiozero input until it reads high. Returns False on timeout."""
    deadline = time.perf_counter_ns() + int(DETECT_TIMEOUT_S * 1e9)
    while not detector.value:
        if time.perf_counter_ns() > deadline:
            return False
    return True


async def run_pi_to_pi(num_trials, output_pin, input_pin, warmup):
    """Pi GPIO output -> Pi GPIO input loopback."""
    from gpiozero import DigitalInputDevice

    from glider.hal.base_board import PinMode, PinType
    from glider.hal.boards.pi_gpio_board import PiGPIOBoard

    board = PiGPIOBoard()
    detector = None
    results = []
    timeouts = 0

    try:
        if not await board.connect():
            print("ERROR: Failed to connect PiGPIOBoard")
            return results, timeouts

        await board.set_pin_mode(output_pin, PinMode.OUTPUT, PinType.DIGITAL)
        detector = DigitalInputDevice(input_pin, pull_up=False)
        await board.write_digital(output_pin, False)
        await asyncio.sleep(0.05)

        print(f"Running {num_trials} Pi-to-Pi trials (GPIO{output_pin} -> GPIO{input_pin})...")

        for i in range(-warmup, num_trials):
            await board.write_digital(output_pin, False)
            await asyncio.sleep(0.001)

            start = time.perf_counter_ns()
            await board.write_digital(output_pin, True)
            detected = _busy_wait_for_high(detector)
            end = time.perf_counter_ns()

            await asyncio.sleep(0.005)
            if i < 0:  # warm-up trial, discard
                continue
            if not detected:
                timeouts += 1
                print(f"  WARNING: Trial {i + 1} timed out")
                continue

            results.append((end - start) / 1_000_000)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_trials}")

    finally:
        try:
            await board.write_digital(output_pin, False)
        except Exception:
            pass
        if detector:
            detector.close()
        await board.disconnect()

    return results, timeouts


async def run_arduino_to_pi(num_trials, arduino_pin, pi_input_pin, arduino_board, warmup):
    """Arduino digital output -> Pi GPIO input."""
    from gpiozero import DigitalInputDevice

    from glider.hal.base_board import PinMode, PinType

    detector = None
    results = []
    timeouts = 0

    try:
        await arduino_board.set_pin_mode(arduino_pin, PinMode.OUTPUT, PinType.DIGITAL)
        detector = DigitalInputDevice(pi_input_pin, pull_up=False)
        await arduino_board.write_digital(arduino_pin, False)
        await asyncio.sleep(0.05)

        print(
            f"Running {num_trials} Arduino-to-Pi trials "
            f"(D{arduino_pin} -> GPIO{pi_input_pin})..."
        )

        for i in range(-warmup, num_trials):
            await arduino_board.write_digital(arduino_pin, False)
            await asyncio.sleep(0.005)

            start = time.perf_counter_ns()
            await arduino_board.write_digital(arduino_pin, True)
            detected = _busy_wait_for_high(detector)
            end = time.perf_counter_ns()

            await asyncio.sleep(0.005)
            if i < 0:  # warm-up trial, discard
                continue
            if not detected:
                timeouts += 1
                print(f"  WARNING: Trial {i + 1} timed out")
                continue

            results.append((end - start) / 1_000_000)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_trials}")

    finally:
        try:
            await arduino_board.write_digital(arduino_pin, False)
        except Exception:
            pass
        if detector:
            detector.close()

    return results, timeouts


async def run_pi_to_arduino(num_trials, pi_output_pin, arduino_input_pin, arduino_board, warmup):
    """Pi GPIO output -> Arduino input-change report (round trip).

    The HAL marshals the Arduino's input-change callback onto this event
    loop, so the wait for detection must not block the loop: a plain
    threading.Event.wait() on the loop thread would deadlock every trial,
    because the callback could never be delivered while we block. The wait
    therefore runs in an executor thread while the loop stays free to
    dispatch callbacks.
    """
    from glider.hal.base_board import PinMode, PinType
    from glider.hal.boards.pi_gpio_board import PiGPIOBoard

    pi_board = PiGPIOBoard()
    results = []
    timeouts = 0
    detected = threading.Event()

    def on_change(pin, value):
        if value:
            detected.set()

    try:
        if not await pi_board.connect():
            print("ERROR: Failed to connect PiGPIOBoard")
            return results, timeouts

        loop = asyncio.get_running_loop()
        await pi_board.set_pin_mode(pi_output_pin, PinMode.OUTPUT, PinType.DIGITAL)
        await arduino_board.set_pin_mode(arduino_input_pin, PinMode.INPUT, PinType.DIGITAL)
        arduino_board.register_callback(arduino_input_pin, on_change)
        await pi_board.write_digital(pi_output_pin, False)
        await asyncio.sleep(0.05)

        print(
            f"Running {num_trials} Pi-to-Arduino round-trip trials "
            f"(GPIO{pi_output_pin} -> D{arduino_input_pin})..."
        )

        for i in range(-warmup, num_trials):
            await pi_board.write_digital(pi_output_pin, False)
            # Let the falling-edge report and any straggling callbacks from
            # the previous trial drain through the event loop before arming
            # the detector, so a stale rising-edge report cannot end the
            # next trial early.
            await asyncio.sleep(0.02)
            detected.clear()

            start = time.perf_counter_ns()
            await pi_board.write_digital(pi_output_pin, True)
            ok = await loop.run_in_executor(None, detected.wait, DETECT_TIMEOUT_S)
            end = time.perf_counter_ns()

            if i < 0:  # warm-up trial, discard
                continue
            if not ok:
                timeouts += 1
                print(f"  WARNING: Trial {i + 1} timed out")
                continue

            results.append((end - start) / 1_000_000)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_trials}")

    finally:
        try:
            await pi_board.write_digital(pi_output_pin, False)
        except Exception:
            pass
        try:
            arduino_board.unregister_callback(arduino_input_pin, on_change)
        except Exception:
            pass
        await pi_board.disconnect()

    return results, timeouts


def print_stats(results, label, timeouts=0):
    """Print summary statistics for latency measurements."""
    if not results:
        suffix = f" ({timeouts} trials timed out.)" if timeouts else ""
        print(f"\n=== {label} ===\n  No results.{suffix}")
        return

    s = sorted(results)
    n = len(s)

    print(f"\n=== {label} ===")
    print(f"  Trials:   {n}")
    if timeouts:
        print(f"  Timeouts: {timeouts} (excluded from statistics)")
    print(f"  Mean:     {statistics.mean(results):.3f} ms")
    if n > 1:
        print(f"  Std Dev:  {statistics.stdev(results):.3f} ms")
    print(f"  Min:      {min(results):.3f} ms")
    print(f"  Max:      {max(results):.3f} ms")
    print(f"  Median:   {statistics.median(results):.3f} ms")
    print(f"  95th pct: {s[max(0, int(n * 0.95) - 1)]:.3f} ms")
    print(f"  99th pct: {s[max(0, int(n * 0.99) - 1)]:.3f} ms")


def save_csv(results_dict, path):
    """Save results to CSV. results_dict maps column names to lists of floats."""
    if not any(results_dict.values()):
        return

    headers = ["trial"] + list(results_dict.keys())
    columns = list(results_dict.values())
    max_len = max(len(col) for col in columns)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(max_len):
            row = [i + 1]
            for col in columns:
                row.append(f"{col[i]:.6f}" if i < len(col) else "")
            writer.writerow(row)

    print(f"\nResults saved to {path}")


async def main():
    parser = argparse.ArgumentParser(
        description="GLIDER Hardware Latency Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--trials", type=int, default=1000, help="Trials per test (default: 1000)")
    parser.add_argument(
        "--warmup", type=int, default=10, help="Unrecorded warm-up trials per test (default: 10)"
    )
    parser.add_argument("--arduino-port", type=str, default=None, help="Arduino serial port")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["pi-to-pi", "arduino-to-pi", "pi-to-arduino"],
        default=None,
        help="Tests to run (default: all)",
    )
    parser.add_argument("--output", type=str, default="latency_results.csv", help="CSV output path")
    parser.add_argument("--pi-output-pin", type=int, default=19)
    parser.add_argument("--pi-input-pin", type=int, default=26)
    parser.add_argument("--arduino-output-pin", type=int, default=7)
    parser.add_argument("--atp-input-pin", type=int, default=13)
    parser.add_argument("--pta-output-pin", type=int, default=6)
    parser.add_argument("--pta-input-pin", type=int, default=8)
    args = parser.parse_args()

    # Record the measurement environment for reporting alongside results.
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print(f"Trials: {args.trials}, warm-up: {args.warmup}\n")

    tests = set(args.tests) if args.tests else {"pi-to-pi", "arduino-to-pi", "pi-to-arduino"}
    results = {}

    # Connect Arduino once if any Arduino test is selected
    arduino_board = None
    if tests & {"arduino-to-pi", "pi-to-arduino"}:
        from glider.hal.boards.telemetrix_board import TelemetrixBoard

        port = args.arduino_port or find_arduino_port()
        if not port:
            print("ERROR: No Arduino found. Use --arduino-port.")
            tests -= {"arduino-to-pi", "pi-to-arduino"}
        else:
            arduino_board = TelemetrixBoard(port=port)
            if not await arduino_board.connect():
                print("ERROR: Failed to connect to Arduino")
                tests -= {"arduino-to-pi", "pi-to-arduino"}
                arduino_board = None

    try:
        if "pi-to-pi" in tests:
            r, t = await run_pi_to_pi(
                args.trials, args.pi_output_pin, args.pi_input_pin, args.warmup
            )
            results["pi_to_pi_ms"] = r
            print_stats(r, "Pi-to-Pi Latency", t)

        if "arduino-to-pi" in tests and arduino_board:
            r, t = await run_arduino_to_pi(
                args.trials, args.arduino_output_pin, args.atp_input_pin, arduino_board, args.warmup
            )
            results["arduino_to_pi_ms"] = r
            print_stats(r, "Arduino-to-Pi Latency", t)

        if "pi-to-arduino" in tests and arduino_board:
            r, t = await run_pi_to_arduino(
                args.trials, args.pta_output_pin, args.pta_input_pin, arduino_board, args.warmup
            )
            results["pi_to_arduino_roundtrip_ms"] = r
            print_stats(r, "Pi-to-Arduino Round-Trip Latency (command -> host notification)", t)

    finally:
        if arduino_board:
            await arduino_board.disconnect()

    if results:
        save_csv(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
