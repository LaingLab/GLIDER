#!/usr/bin/env python3
"""
GLIDER Hardware Latency Test

Measures end-to-end latency through the GLIDER HAL:
  Pi-to-Pi:       Pi GPIO output -> Pi GPIO input loopback
  Arduino-to-Pi:  Arduino digital write -> Pi GPIO read
  Pi-to-Arduino:  Pi GPIO write -> Arduino digital read

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
import statistics
import threading
import time


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


async def run_pi_to_pi(num_trials, output_pin, input_pin):
    """Pi GPIO output -> Pi GPIO input loopback."""
    from gpiozero import DigitalInputDevice

    from glider.hal.base_board import PinMode, PinType
    from glider.hal.boards.pi_gpio_board import PiGPIOBoard

    board = PiGPIOBoard()
    detector = None
    results = []

    try:
        if not await board.connect():
            print("ERROR: Failed to connect PiGPIOBoard")
            return results

        await board.set_pin_mode(output_pin, PinMode.OUTPUT, PinType.DIGITAL)
        detector = DigitalInputDevice(input_pin, pull_up=False)
        await board.write_digital(output_pin, False)
        time.sleep(0.05)

        print(f"Running {num_trials} Pi-to-Pi trials (GPIO{output_pin} -> GPIO{input_pin})...")

        for i in range(num_trials):
            await board.write_digital(output_pin, False)
            time.sleep(0.001)

            start = time.perf_counter_ns()
            await board.write_digital(output_pin, True)
            while not detector.value:
                pass
            end = time.perf_counter_ns()

            results.append((end - start) / 1_000_000)
            time.sleep(0.005)

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

    return results


async def run_arduino_to_pi(num_trials, arduino_pin, pi_input_pin, arduino_board):
    """Arduino digital output -> Pi GPIO input."""
    from gpiozero import DigitalInputDevice

    from glider.hal.base_board import PinMode, PinType

    detector = None
    results = []

    try:
        await arduino_board.set_pin_mode(arduino_pin, PinMode.OUTPUT, PinType.DIGITAL)
        detector = DigitalInputDevice(pi_input_pin, pull_up=False)
        await arduino_board.write_digital(arduino_pin, False)
        time.sleep(0.05)

        print(
            f"Running {num_trials} Arduino-to-Pi trials "
            f"(D{arduino_pin} -> GPIO{pi_input_pin})..."
        )

        for i in range(num_trials):
            await arduino_board.write_digital(arduino_pin, False)
            time.sleep(0.005)

            start = time.perf_counter_ns()
            await arduino_board.write_digital(arduino_pin, True)
            while not detector.value:
                pass
            end = time.perf_counter_ns()

            results.append((end - start) / 1_000_000)
            time.sleep(0.005)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_trials}")

    finally:
        try:
            await arduino_board.write_digital(arduino_pin, False)
        except Exception:
            pass
        if detector:
            detector.close()

    return results


async def run_pi_to_arduino(num_trials, pi_output_pin, arduino_input_pin, arduino_board):
    """Pi GPIO output -> Arduino digital input (detected via telemetrix callback)."""
    from glider.hal.base_board import PinMode, PinType
    from glider.hal.boards.pi_gpio_board import PiGPIOBoard

    pi_board = PiGPIOBoard()
    results = []
    detected = threading.Event()

    def on_change(pin, value):
        if value:
            detected.set()

    try:
        if not await pi_board.connect():
            print("ERROR: Failed to connect PiGPIOBoard")
            return results

        await pi_board.set_pin_mode(pi_output_pin, PinMode.OUTPUT, PinType.DIGITAL)
        await arduino_board.set_pin_mode(arduino_input_pin, PinMode.INPUT, PinType.DIGITAL)
        arduino_board.register_callback(arduino_input_pin, on_change)
        await pi_board.write_digital(pi_output_pin, False)
        time.sleep(0.05)

        print(
            f"Running {num_trials} Pi-to-Arduino trials "
            f"(GPIO{pi_output_pin} -> D{arduino_input_pin})..."
        )

        for i in range(num_trials):
            await pi_board.write_digital(pi_output_pin, False)
            time.sleep(0.005)
            detected.clear()

            start = time.perf_counter_ns()
            await pi_board.write_digital(pi_output_pin, True)
            if not detected.wait(timeout=1.0):
                print(f"  WARNING: Trial {i + 1} timed out")
                continue
            end = time.perf_counter_ns()

            results.append((end - start) / 1_000_000)
            time.sleep(0.005)

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

    return results


def print_stats(results, label):
    """Print summary statistics for latency measurements."""
    if not results:
        print(f"\n=== {label} ===\n  No results.")
        return

    s = sorted(results)
    n = len(s)

    print(f"\n=== {label} ===")
    print(f"  Trials:   {n}")
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
            r = await run_pi_to_pi(args.trials, args.pi_output_pin, args.pi_input_pin)
            results["pi_to_pi_ms"] = r
            print_stats(r, "Pi-to-Pi Latency")

        if "arduino-to-pi" in tests and arduino_board:
            r = await run_arduino_to_pi(
                args.trials, args.arduino_output_pin, args.atp_input_pin, arduino_board
            )
            results["arduino_to_pi_ms"] = r
            print_stats(r, "Arduino-to-Pi Latency")

        if "pi-to-arduino" in tests and arduino_board:
            r = await run_pi_to_arduino(
                args.trials, args.pta_output_pin, args.pta_input_pin, arduino_board
            )
            results["pi_to_arduino_ms"] = r
            print_stats(r, "Pi-to-Arduino Latency")

    finally:
        if arduino_board:
            await arduino_board.disconnect()

    if results:
        save_csv(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
