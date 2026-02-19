#!/usr/bin/env python3
"""
GLIDER Hardware Latency Test

Measures end-to-end latency of the GLIDER hardware stack using a loopback
wiring approach. Compares Pi GPIO baseline, Arduino-to-Pi, and Pi-to-Arduino
latencies through the GLIDER HAL.

Wiring required:
  Pi-to-Pi:         Pi GPIO19 (output) --> wire --> Pi GPIO26 (input)
  Arduino-to-Pi:    Arduino D7 (output) --> wire --> Pi GPIO13 (input)
  Pi-to-Arduino:    Pi GPIO6 (output)  --> wire --> Arduino D8 (input)
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


async def measure_pi_gpio_latency(num_trials: int, output_pin: int, input_pin: int) -> list[float]:
    """Measure Pi GPIO output latency through the GLIDER HAL.

    Uses PiGPIOBoard (HAL) for the output side and raw gpiozero
    DigitalInputDevice for detection to isolate HAL overhead.

    Returns list of latencies in milliseconds.
    """
    from gpiozero import DigitalInputDevice

    from glider.hal.base_board import PinMode, PinType
    from glider.hal.boards.pi_gpio_board import PiGPIOBoard

    board = PiGPIOBoard()
    detector = None
    results = []

    try:
        # Connect and configure output pin via HAL
        connected = await board.connect()
        if not connected:
            print("ERROR: Failed to connect PiGPIOBoard")
            return []

        await board.set_pin_mode(output_pin, PinMode.OUTPUT, PinType.DIGITAL)

        # Raw gpiozero input for detection (not through HAL)
        detector = DigitalInputDevice(input_pin, pull_up=False)

        # Ensure output starts LOW
        await board.write_digital(output_pin, False)
        time.sleep(0.05)

        print(
            f"Running {num_trials} Pi GPIO trials (output=GPIO{output_pin}, input=GPIO{input_pin})..."
        )

        for i in range(num_trials):
            # Ensure LOW and settled
            await board.write_digital(output_pin, False)
            time.sleep(0.001)

            # Measure: write HIGH through HAL, poll until detected
            start_ns = time.perf_counter_ns()
            await board.write_digital(output_pin, True)

            while not detector.value:
                pass

            end_ns = time.perf_counter_ns()

            delta_ms = (end_ns - start_ns) / 1_000_000
            results.append(delta_ms)

            # Brief pause between trials
            time.sleep(0.005)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_trials} trials complete")

    finally:
        # Cleanup
        try:
            await board.write_digital(output_pin, False)
        except Exception:
            pass
        if detector is not None:
            detector.close()
        await board.disconnect()

    return results


def find_arduino_port() -> str | None:
    """Auto-detect Arduino serial port by scanning USB devices.

    This avoids telemetrix-aio's _find_arduino which has no read timeout
    and can hang indefinitely on non-Arduino serial devices.
    """
    from serial.tools import list_ports

    arduino_ids = {
        (0x2341, None),  # Arduino SA
        (0x2A03, None),  # Arduino.org
        (0x1A86, 0x7523),  # CH340 (common Arduino clone chip)
        (0x0403, 0x6001),  # FTDI FT232
        (0x10C4, 0xEA60),  # CP210x (common Arduino clone chip)
    }

    for port_info in list_ports.comports():
        if port_info.vid is None:
            continue
        # Match by vendor ID (any product from that vendor)
        for vid, pid in arduino_ids:
            if port_info.vid == vid and (pid is None or port_info.pid == pid):
                print(
                    f"Auto-detected Arduino on {port_info.device} "
                    f"(VID:0x{port_info.vid:04X} PID:0x{port_info.pid:04X})"
                )
                return port_info.device
        # Fallback: match common Arduino descriptions
        desc = (port_info.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "cp210" in desc or "ft232" in desc:
            print(f"Auto-detected Arduino on {port_info.device} ({port_info.description})")
            return port_info.device

    return None


async def measure_arduino_latency(
    num_trials: int, arduino_pin: int, input_pin: int, port: str | None
) -> list[float]:
    """Measure Arduino output latency through the GLIDER HAL (telemetrix).

    Uses TelemetrixBoard (HAL) for the Arduino output side and raw gpiozero
    DigitalInputDevice on the Pi for detection.

    Returns list of latencies in milliseconds.
    """
    from gpiozero import DigitalInputDevice

    from glider.hal.base_board import PinMode, PinType
    from glider.hal.boards.telemetrix_board import TelemetrixBoard

    # Auto-detect port if not specified to avoid telemetrix's _find_arduino
    # which can hang indefinitely on non-Arduino serial devices
    if port is None:
        port = find_arduino_port()
        if port is None:
            print(
                "ERROR: Could not auto-detect Arduino port. "
                "Use --arduino-port to specify it manually."
            )
            return []

    board = TelemetrixBoard(port=port)
    detector = None
    results = []

    try:
        # Connect to Arduino via telemetrix
        connected = await board.connect()
        if not connected:
            print("ERROR: Failed to connect TelemetrixBoard")
            return []

        await board.set_pin_mode(arduino_pin, PinMode.OUTPUT, PinType.DIGITAL)

        # Raw gpiozero input for detection on Pi side
        detector = DigitalInputDevice(input_pin, pull_up=False)

        # Ensure output starts LOW
        await board.write_digital(arduino_pin, False)
        time.sleep(0.05)

        print(
            f"Running {num_trials} Arduino trials "
            f"(Arduino D{arduino_pin} -> Pi GPIO{input_pin})..."
        )

        for i in range(num_trials):
            # Ensure LOW and settled
            await board.write_digital(arduino_pin, False)
            time.sleep(0.005)

            # Measure: write HIGH through HAL, poll until detected on Pi
            start_ns = time.perf_counter_ns()
            await board.write_digital(arduino_pin, True)

            while not detector.value:
                pass

            end_ns = time.perf_counter_ns()

            delta_ms = (end_ns - start_ns) / 1_000_000
            results.append(delta_ms)

            # Brief pause between trials
            time.sleep(0.005)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_trials} trials complete")

    finally:
        # Cleanup
        try:
            await board.write_digital(arduino_pin, False)
        except Exception:
            pass
        if detector is not None:
            detector.close()
        await board.disconnect()

    return results


async def measure_pi_to_arduino_latency(
    num_trials: int, pi_output_pin: int, arduino_input_pin: int, port: str | None
) -> list[float]:
    """Measure Pi GPIO -> Arduino input latency through the GLIDER HAL.

    Uses PiGPIOBoard (HAL) for the Pi output side and TelemetrixBoard (HAL)
    for Arduino input detection via digital callback.

    Returns list of latencies in milliseconds.
    """
    from glider.hal.base_board import PinMode, PinType
    from glider.hal.boards.pi_gpio_board import PiGPIOBoard
    from glider.hal.boards.telemetrix_board import TelemetrixBoard

    if port is None:
        port = find_arduino_port()
        if port is None:
            print(
                "ERROR: Could not auto-detect Arduino port. "
                "Use --arduino-port to specify it manually."
            )
            return []

    pi_board = PiGPIOBoard()
    arduino_board = TelemetrixBoard(port=port)
    results = []

    try:
        # Connect both boards
        if not await pi_board.connect():
            print("ERROR: Failed to connect PiGPIOBoard")
            return []
        if not await arduino_board.connect():
            print("ERROR: Failed to connect TelemetrixBoard")
            await pi_board.disconnect()
            return []

        # Configure pins
        await pi_board.set_pin_mode(pi_output_pin, PinMode.OUTPUT, PinType.DIGITAL)
        await arduino_board.set_pin_mode(arduino_input_pin, PinMode.INPUT, PinType.DIGITAL)

        # Thread-safe event for detecting Arduino callback
        # (telemetrix runs in a separate thread)
        detected = threading.Event()

        def on_change(pin, value):
            if value:
                detected.set()

        arduino_board.register_callback(arduino_input_pin, on_change)

        # Ensure output starts LOW
        await pi_board.write_digital(pi_output_pin, False)
        time.sleep(0.05)

        print(
            f"Running {num_trials} Pi->Arduino trials "
            f"(Pi GPIO{pi_output_pin} -> Arduino D{arduino_input_pin})..."
        )

        for i in range(num_trials):
            # Ensure LOW and settled
            await pi_board.write_digital(pi_output_pin, False)
            time.sleep(0.005)
            detected.clear()

            # Measure: write HIGH on Pi, wait for Arduino to report the change
            start_ns = time.perf_counter_ns()
            await pi_board.write_digital(pi_output_pin, True)

            if not detected.wait(timeout=1.0):
                print(f"  WARNING: Trial {i + 1} timed out waiting for Arduino callback")
                continue

            end_ns = time.perf_counter_ns()

            delta_ms = (end_ns - start_ns) / 1_000_000
            results.append(delta_ms)

            # Brief pause between trials
            time.sleep(0.005)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_trials} trials complete")

    finally:
        # Cleanup
        try:
            await pi_board.write_digital(pi_output_pin, False)
        except Exception:
            pass
        try:
            arduino_board.unregister_callback(arduino_input_pin, on_change)
        except Exception:
            pass
        await arduino_board.disconnect()
        await pi_board.disconnect()

    return results


def print_statistics(results: list[float], label: str) -> None:
    """Print summary statistics for a set of latency measurements."""
    if not results:
        print(f"\n=== {label} ===")
        print("  No results collected.")
        return

    results_sorted = sorted(results)
    n = len(results_sorted)
    p95_idx = int(n * 0.95) - 1
    p99_idx = int(n * 0.99) - 1

    print(f"\n=== {label} ===")
    print(f"  Trials:     {n}")
    print(f"  Mean:       {statistics.mean(results):.3f} ms")
    print(f"  Std Dev:    {statistics.stdev(results):.3f} ms" if n > 1 else "  Std Dev:    N/A")
    print(f"  Min:        {min(results):.3f} ms")
    print(f"  Max:        {max(results):.3f} ms")
    print(f"  Median:     {statistics.median(results):.3f} ms")
    print(f"  95th pct:   {results_sorted[max(0, p95_idx)]:.3f} ms")
    print(f"  99th pct:   {results_sorted[max(0, p99_idx)]:.3f} ms")


def save_csv(
    pi_results: list[float] | None,
    arduino_to_pi_results: list[float] | None,
    pi_to_arduino_results: list[float] | None,
    path: str,
) -> None:
    """Save raw trial data to CSV for external analysis."""
    max_len = max(
        len(pi_results or []),
        len(arduino_to_pi_results or []),
        len(pi_to_arduino_results or []),
    )
    if max_len == 0:
        print("No data to save.")
        return

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "pi_to_pi_ms", "arduino_to_pi_ms", "pi_to_arduino_ms"])
        for i in range(max_len):
            pi_val = f"{pi_results[i]:.6f}" if pi_results and i < len(pi_results) else ""
            atp_val = (
                f"{arduino_to_pi_results[i]:.6f}"
                if arduino_to_pi_results and i < len(arduino_to_pi_results)
                else ""
            )
            pta_val = (
                f"{pi_to_arduino_results[i]:.6f}"
                if pi_to_arduino_results and i < len(pi_to_arduino_results)
                else ""
            )
            writer.writerow([i + 1, pi_val, atp_val, pta_val])

    print(f"\nResults saved to {path}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="GLIDER Hardware Latency Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--trials", type=int, default=1000, help="Number of trials per test (default: 1000)"
    )
    parser.add_argument(
        "--arduino-port",
        type=str,
        default=None,
        help="Arduino serial port (default: auto-detect)",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["pi-to-pi", "arduino-to-pi", "pi-to-arduino"],
        default=None,
        help="Tests to run (default: all). Options: pi-to-pi, arduino-to-pi, pi-to-arduino",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="latency_results.csv",
        help="CSV output path (default: latency_results.csv)",
    )
    # Pi-to-Pi pins
    parser.add_argument(
        "--pi-output-pin",
        type=int,
        default=19,
        help="Pi GPIO output pin for Pi-to-Pi (default: 19)",
    )
    parser.add_argument(
        "--pi-input-pin", type=int, default=26, help="Pi GPIO input pin for Pi-to-Pi (default: 26)"
    )
    # Arduino-to-Pi pins
    parser.add_argument(
        "--arduino-output-pin",
        type=int,
        default=7,
        help="Arduino digital output pin for Arduino-to-Pi (default: 7)",
    )
    parser.add_argument(
        "--atp-input-pin",
        type=int,
        default=13,
        help="Pi GPIO input pin for Arduino-to-Pi (default: 13)",
    )
    # Pi-to-Arduino pins
    parser.add_argument(
        "--pta-output-pin",
        type=int,
        default=6,
        help="Pi GPIO output pin for Pi-to-Arduino (default: 6)",
    )
    parser.add_argument(
        "--pta-input-pin",
        type=int,
        default=8,
        help="Arduino digital input pin for Pi-to-Arduino (default: 8)",
    )
    args = parser.parse_args()

    all_tests = {"pi-to-pi", "arduino-to-pi", "pi-to-arduino"}
    tests_to_run = set(args.tests) if args.tests else all_tests

    pi_results = None
    arduino_to_pi_results = None
    pi_to_arduino_results = None

    # Pi-to-Pi baseline test
    if "pi-to-pi" in tests_to_run:
        try:
            pi_results = await measure_pi_gpio_latency(
                args.trials, args.pi_output_pin, args.pi_input_pin
            )
            print_statistics(pi_results, "Pi-to-Pi Latency (GLIDER HAL)")
        except Exception as e:
            print(f"\nPi-to-Pi test failed: {e}")

    # Arduino-to-Pi test
    if "arduino-to-pi" in tests_to_run:
        try:
            arduino_to_pi_results = await measure_arduino_latency(
                args.trials, args.arduino_output_pin, args.atp_input_pin, args.arduino_port
            )
            print_statistics(
                arduino_to_pi_results,
                "Arduino-to-Pi Latency (GLIDER HAL -> Telemetrix -> USB -> Pi GPIO)",
            )
        except Exception as e:
            print(f"\nArduino-to-Pi test failed: {e}")

    # Pi-to-Arduino test
    if "pi-to-arduino" in tests_to_run:
        try:
            pi_to_arduino_results = await measure_pi_to_arduino_latency(
                args.trials, args.pta_output_pin, args.pta_input_pin, args.arduino_port
            )
            print_statistics(
                pi_to_arduino_results,
                "Pi-to-Arduino Latency (Pi GPIO -> wire -> Telemetrix callback)",
            )
        except Exception as e:
            print(f"\nPi-to-Arduino test failed: {e}")

    # Save CSV
    if pi_results or arduino_to_pi_results or pi_to_arduino_results:
        save_csv(pi_results, arduino_to_pi_results, pi_to_arduino_results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
