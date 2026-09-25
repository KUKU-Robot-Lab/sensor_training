import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime

try:
    import serial
except ImportError:
    serial = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from serial_autodetect import resolve_port


DEFAULT_PORT = "COM10"  # fallback only; auto-detected in main() when possible
LOADCELL_PORT_VID_PID = {(0x067B, 0x2303)}  # Prolific USB-RS232 adapter
LOADCELL_PORT_KEYWORDS = ("prolific", "usb-to-serial", "usb serial")
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 1.0
DEFAULT_ENCODING = "ascii"
VALUE_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def default_output_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(base_dir, "loadcell data")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"loadcell_{timestamp}.csv")


def parse_kg(raw_line):
    match = VALUE_PATTERN.search(raw_line)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def open_serial(args):
    if serial is None:
        raise RuntimeError(
            "pyserial is not installed. Install it with: python -m pip install pyserial"
        )

    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=args.timeout,
    )
    ser.reset_input_buffer()
    return ser


def log_loadcell(args):
    output_path = os.path.abspath(args.output or default_output_path())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ser = None
    rows_written = 0
    start_perf = time.perf_counter()
    next_status_at = start_perf

    try:
        ser = open_serial(args)
        print(
            f"Loadcell connected on {args.port} "
            f"({args.baud} baud, 8N1)."
        )
        print(f"Saving CSV: {output_path}")
        print("Press Ctrl+C to stop.")

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["elapsed_s", "kg"])

            while True:
                elapsed_s = time.perf_counter() - start_perf
                if args.duration_sec is not None and elapsed_s >= args.duration_sec:
                    break

                raw_bytes = ser.readline()
                if not raw_bytes:
                    continue

                raw_line = raw_bytes.decode(args.encoding, errors="replace").strip()
                if not raw_line:
                    continue

                kg = parse_kg(raw_line)
                if kg is None:
                    continue

                elapsed_s = time.perf_counter() - start_perf
                writer.writerow([f"{elapsed_s:.6f}", f"{kg:.6f}"])
                rows_written += 1

                now = time.perf_counter()
                if now >= next_status_at:
                    print(f"\r[REC] rows:{rows_written} kg:{kg: .3f}", end="")
                    next_status_at = now + 0.2

    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        if ser is not None and ser.is_open:
            ser.close()

    print(f"\nLoadcell logging complete: {output_path}")
    print(f"Rows written: {rows_written}")


def run_self_test():
    cases = {
        "0.000 kg": 0.0,
        "ST,+000.125kg": 0.125,
        "US,-001.234 kg": -1.234,
        "no numeric value": None,
    }
    for raw_line, expected in cases.items():
        actual = parse_kg(raw_line)
        if actual != expected:
            raise AssertionError(
                f"parse_kg({raw_line!r}) returned {actual!r}, expected {expected!r}"
            )
    print("Self-test passed.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Log loadcell indicator values from an RS232/USB serial port."
    )
    parser.add_argument(
        "--port",
        default=None,
        help=f"Serial port. Default: auto-detect, falling back to the LOADCELL_PORT "
        f"env var, then {DEFAULT_PORT}.",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Baud rate, default {DEFAULT_BAUD}.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Serial read timeout in seconds.")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional automatic stop time in seconds.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help=f"Serial text encoding, default {DEFAULT_ENCODING}.")
    parser.add_argument("--self-test", action="store_true", help="Run parser self-test and exit.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    args.port = resolve_port(
        "Loadcell", args.port, os.environ.get("LOADCELL_PORT"), DEFAULT_PORT,
        LOADCELL_PORT_VID_PID, LOADCELL_PORT_KEYWORDS,
    )

    try:
        log_loadcell(args)
    except Exception as exc:
        print(f"Loadcell logger error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
