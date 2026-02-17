#!/usr/bin/env python3
"""
GLIDER Hardware Latency Test

Measures end-to-end latency of the GLIDER hardware stack using a loopback
wiring approach. Compares Pi GPIO baseline latency against Arduino (telemetrix)
serial stack latency.

Wiring required:
  Pi GPIO baseline:  Pi GPIO17 (output) --> wire --> Pi GPIO27 (input)
  Arduino loopback:  Arduino D7 (output) --> wire --> Pi GPIO22 (input)
  Common ground between Pi and Arduino.

Usage:
  python tests/latency_test.py
  python tests/latency_test.py --trials 500 --arduino-port /dev/ttyUSB0
  python tests/latency_test.py --pi-only
  python tests/latency_test.py --arduino-only
"""

import argparse
import asyncio
import csv
import statistics
import sys
import time


async def measure_pi_gpio_latency(
    num_trials: int, output_pin: int, input_pin: int
) -> list[float]:
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

        print(f"Running {num_trials} Pi GPIO trials (output=GPIO{output_pin}, input=GPIO{input_pin})...")

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
    arduino_results: list[float] | None,
    path: str,
) -> None:
    """Save raw trial data to CSV for external analysis."""
    max_len = max(len(pi_results or []), len(arduino_results or []))
    if max_len == 0:
        print("No data to save.")
        return

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "pi_gpio_ms", "arduino_ms"])
        for i in range(max_len):
            pi_val = f"{pi_results[i]:.6f}" if pi_results and i < len(pi_results) else ""
            ard_val = (
                f"{arduino_results[i]:.6f}"
                if arduino_results and i < len(arduino_results)
                else ""
            )
            writer.writerow([i + 1, pi_val, ard_val])

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
    parser.add_argument("--pi-only", action="store_true", help="Skip Arduino test")
    parser.add_argument("--arduino-only", action="store_true", help="Skip Pi GPIO test")
    parser.add_argument(
        "--output",
        type=str,
        default="latency_results.csv",
        help="CSV output path (default: latency_results.csv)",
    )
    parser.add_argument(
        "--pi-output-pin", type=int, default=17, help="Pi GPIO output pin (default: 17)"
    )
    parser.add_argument(
        "--pi-input-pin", type=int, default=27, help="Pi GPIO input pin (default: 27)"
    )
    parser.add_argument(
        "--arduino-output-pin",
        type=int,
        default=7,
        help="Arduino digital output pin (default: 7)",
    )
    parser.add_argument(
        "--arduino-input-pin",
        type=int,
        default=22,
        help="Pi GPIO input pin for Arduino loopback (default: 22)",
    )
    args = parser.parse_args()

    if args.pi_only and args.arduino_only:
        print("ERROR: Cannot specify both --pi-only and --arduino-only")
        sys.exit(1)

    pi_results = None
    arduino_results = None

    # Pi GPIO baseline test
    if not args.arduino_only:
        try:
            pi_results = await measure_pi_gpio_latency(
                args.trials, args.pi_output_pin, args.pi_input_pin
            )
            print_statistics(pi_results, "Pi GPIO Latency (GLIDER HAL)")
        except Exception as e:
            print(f"\nPi GPIO test failed: {e}")

    # Arduino loopback test
    if not args.pi_only:
        try:
            arduino_results = await measure_arduino_latency(
                args.trials, args.arduino_output_pin, args.arduino_input_pin, args.arduino_port
            )
            print_statistics(
                arduino_results,
                "Arduino Loopback Latency (GLIDER HAL -> Telemetrix -> USB)",
            )
        except Exception as e:
            print(f"\nArduino test failed: {e}")

    # Save CSV
    if pi_results or arduino_results:
        save_csv(pi_results, arduino_results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
